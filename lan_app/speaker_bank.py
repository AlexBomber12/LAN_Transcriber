from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from .config import AppSettings
from .db import (
    SPEAKER_REVIEW_STATE_SYSTEM_SUGGESTED,
    create_voice_sample,
    merge_voice_profiles,
    set_speaker_assignment,
)

DEFAULT_ASSIGNMENT_THRESHOLD = 0.75
DEFAULT_SAMPLE_ATTACH_THRESHOLD = 0.85


def _clamp_score(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(parsed, 1.0))


def _normalise_candidate_matches(
    candidate_matches: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not candidate_matches:
        return []
    best_by_profile_id: dict[int, dict[str, Any]] = {}
    for row in candidate_matches:
        if not isinstance(row, dict):
            continue
        try:
            voice_profile_id = int(row.get("voice_profile_id"))
        except (TypeError, ValueError):
            continue
        normalized = {
            "voice_profile_id": voice_profile_id,
            "score": round(
                _clamp_score(row.get("score", row.get("confidence"))),
                4,
            ),
        }
        display_name = str(row.get("display_name") or "").strip()
        if display_name:
            normalized["display_name"] = display_name
        existing = best_by_profile_id.get(voice_profile_id)
        if existing is None or tuple(
            normalized.get(key, "") for key in ("score", "display_name")
        ) > tuple(existing.get(key, "") for key in ("score", "display_name")):
            best_by_profile_id[voice_profile_id] = normalized
    return sorted(
        best_by_profile_id.values(),
        key=lambda item: (
            -float(item.get("score") or 0.0),
            str(item.get("display_name") or ""),
            int(item["voice_profile_id"]),
        ),
    )


def _normalise_speaker_rows(
    diarized_speakers: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in diarized_speakers:
        if not isinstance(row, dict):
            continue
        diar_speaker_label = str(
            row.get("diar_speaker_label") or row.get("speaker") or ""
        ).strip()
        if not diar_speaker_label:
            raise ValueError("diar_speaker_label is required")
        normalized.append(
            {
                "diar_speaker_label": diar_speaker_label,
                "candidate_matches": _normalise_candidate_matches(
                    row.get("candidate_matches")
                ),
            }
        )
    return normalized


def _assignment_tiebreak(assignment: tuple[int | None, ...]) -> tuple[int, ...]:
    return tuple(value if value is not None else 1_000_000_000 for value in assignment)


def _is_better_result(
    candidate_total: float,
    candidate_count: int,
    candidate_assignment: tuple[int | None, ...],
    best_total: float,
    best_count: int,
    best_assignment: tuple[int | None, ...],
) -> bool:
    rounded_candidate_total = round(candidate_total, 6)
    rounded_best_total = round(best_total, 6)
    if rounded_candidate_total != rounded_best_total:
        return rounded_candidate_total > rounded_best_total
    if candidate_count != best_count:
        return candidate_count > best_count
    return _assignment_tiebreak(candidate_assignment) < _assignment_tiebreak(
        best_assignment
    )


def resolve_one_to_one_assignments(
    diarized_speakers: Sequence[dict[str, Any]],
    *,
    min_confidence: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> list[dict[str, Any]]:
    normalized_rows = _normalise_speaker_rows(diarized_speakers)
    if not normalized_rows:
        return []

    threshold = _clamp_score(min_confidence)
    eligible_profile_ids = sorted(
        {
            int(candidate["voice_profile_id"])
            for row in normalized_rows
            for candidate in row["candidate_matches"]
            if float(candidate.get("score") or 0.0) >= threshold
        }
    )
    profile_bits = {
        voice_profile_id: 1 << index
        for index, voice_profile_id in enumerate(eligible_profile_ids)
    }
    eligible_candidates = [
        [
            candidate
            for candidate in row["candidate_matches"]
            if float(candidate.get("score") or 0.0) >= threshold
        ]
        for row in normalized_rows
    ]

    @lru_cache(maxsize=None)
    def _best_assignment(
        index: int,
        used_mask: int,
    ) -> tuple[float, int, tuple[int | None, ...]]:
        if index >= len(normalized_rows):
            return 0.0, 0, ()

        next_total, next_count, next_assignment = _best_assignment(index + 1, used_mask)
        best_total = next_total
        best_count = next_count
        best_assignment_rows = (None,) + next_assignment

        for candidate in eligible_candidates[index]:
            voice_profile_id = int(candidate["voice_profile_id"])
            bit = profile_bits[voice_profile_id]
            if used_mask & bit:
                continue
            next_total, next_count, next_assignment = _best_assignment(
                index + 1,
                used_mask | bit,
            )
            candidate_total = float(candidate["score"]) + next_total
            candidate_count = next_count + 1
            candidate_assignment = (voice_profile_id,) + next_assignment
            if _is_better_result(
                candidate_total,
                candidate_count,
                candidate_assignment,
                best_total,
                best_count,
                best_assignment_rows,
            ):
                best_total = candidate_total
                best_count = candidate_count
                best_assignment_rows = candidate_assignment

        return best_total, best_count, best_assignment_rows

    _, _, selected_profile_ids = _best_assignment(0, 0)
    decisions: list[dict[str, Any]] = []
    for row, selected_profile_id in zip(normalized_rows, selected_profile_ids, strict=True):
        selected_candidate = next(
            (
                candidate
                for candidate in row["candidate_matches"]
                if int(candidate["voice_profile_id"]) == selected_profile_id
            ),
            None,
        )
        best_score = (
            float(row["candidate_matches"][0]["score"])
            if row["candidate_matches"]
            else 0.0
        )
        decisions.append(
            {
                "diar_speaker_label": row["diar_speaker_label"],
                "voice_profile_id": selected_profile_id,
                "confidence": (
                    float(selected_candidate["score"])
                    if selected_candidate is not None
                    else best_score
                ),
                "low_confidence": selected_profile_id is None and bool(
                    row["candidate_matches"]
                ),
                "candidate_matches": row["candidate_matches"],
            }
        )
    return decisions


def assign_speakers_to_recording(
    recording_id: str,
    diarized_speakers: Sequence[dict[str, Any]],
    *,
    min_confidence: float = DEFAULT_ASSIGNMENT_THRESHOLD,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    decisions = resolve_one_to_one_assignments(
        diarized_speakers,
        min_confidence=min_confidence,
    )
    persisted: list[dict[str, Any]] = []
    for decision in decisions:
        resolved_review_state = (
            SPEAKER_REVIEW_STATE_SYSTEM_SUGGESTED
            if decision["voice_profile_id"] is not None or decision["candidate_matches"]
            else None
        )
        row = set_speaker_assignment(
            recording_id=recording_id,
            diar_speaker_label=str(decision["diar_speaker_label"]),
            voice_profile_id=decision["voice_profile_id"],
            confidence=float(decision["confidence"]),
            candidate_matches=decision["candidate_matches"],
            low_confidence=bool(decision["low_confidence"]),
            review_state=resolved_review_state,
            settings=settings,
        )
        persisted.append(
            row
            or {
                "recording_id": recording_id,
                "voice_profile_name": "",
                **decision,
            }
        )
    return persisted


def register_voice_sample(
    *,
    snippet_path: str,
    candidate_matches: Sequence[dict[str, Any]] | None = None,
    attach_threshold: float = DEFAULT_SAMPLE_ATTACH_THRESHOLD,
    recording_id: str | None = None,
    diar_speaker_label: str | None = None,
    sample_source: str = "speaker-bank",
    sample_start_sec: float | None = None,
    sample_end_sec: float | None = None,
    embedding: Sequence[float] | None = None,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    normalized_candidates = _normalise_candidate_matches(candidate_matches)
    threshold = _clamp_score(attach_threshold)
    best_candidate = normalized_candidates[0] if normalized_candidates else None
    selected_profile_id = (
        int(best_candidate["voice_profile_id"])
        if best_candidate is not None and float(best_candidate["score"]) >= threshold
        else None
    )
    confidence = (
        float(best_candidate["score"])
        if best_candidate is not None
        else None
    )
    return create_voice_sample(
        voice_profile_id=selected_profile_id,
        snippet_path=snippet_path,
        recording_id=recording_id,
        diar_speaker_label=diar_speaker_label,
        sample_source=sample_source,
        sample_start_sec=sample_start_sec,
        sample_end_sec=sample_end_sec,
        embedding=embedding,
        candidate_matches=normalized_candidates,
        needs_review=selected_profile_id is None,
        confidence=confidence,
        settings=settings,
    )


def merge_canonical_speakers(
    source_profile_id: int,
    target_profile_id: int,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    return merge_voice_profiles(
        source_profile_id,
        target_profile_id,
        settings=settings,
    )


__all__ = [
    "DEFAULT_ASSIGNMENT_THRESHOLD",
    "DEFAULT_SAMPLE_ATTACH_THRESHOLD",
    "resolve_one_to_one_assignments",
    "assign_speakers_to_recording",
    "register_voice_sample",
    "merge_canonical_speakers",
]
