from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from lan_transcriber.utils import normalise_language_code, safe_float

DEFAULT_DIALOG_MIN_SPEAKERS = 2
DEFAULT_DIALOG_MAX_SPEAKERS = 2
DEFAULT_MEETING_MIN_SPEAKERS = 2
DEFAULT_MEETING_MAX_SPEAKERS = 6
DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS = 20.0
DEFAULT_DIALOG_RETRY_MIN_TURNS = 6
DEFAULT_DIARIZATION_MERGE_GAP_SECONDS = 0.5
DEFAULT_DIARIZATION_MIN_TURN_SECONDS = 0.5


@dataclass(frozen=True)
class SpeakerTurnSmoothingResult:
    turns: list[dict[str, Any]]
    adjacent_merges: int
    micro_turn_absorptions: int
    turn_count_before: int
    turn_count_after: int
    speaker_count_before: int
    speaker_count_after: int


def profile_default_speaker_hints(profile: str) -> tuple[int | None, int | None]:
    normalized = str(profile or "auto").strip().lower()
    if normalized == "dialog":
        return DEFAULT_DIALOG_MIN_SPEAKERS, DEFAULT_DIALOG_MAX_SPEAKERS
    if normalized == "meeting":
        return DEFAULT_MEETING_MIN_SPEAKERS, DEFAULT_MEETING_MAX_SPEAKERS
    return None, None


def annotation_speaker_count(diarization: Any) -> int:
    if diarization is None or not hasattr(diarization, "itertracks"):
        return 0
    speakers: set[str] = set()
    for item in diarization.itertracks(yield_label=True):
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        label = item[1] if len(item) == 2 else item[-1]
        speaker = str(label).strip()
        if speaker:
            speakers.add(speaker)
    return len(speakers)


def should_retry_dialog(
    *,
    profile: str,
    min_speakers: int | None,
    max_speakers: int | None,
    detected_speaker_count: int,
    speech_turn_count: int,
    duration_sec: float | None,
    min_turns: int,
    min_duration_seconds: float,
) -> bool:
    normalized_profile = str(profile or "auto").strip().lower()
    if min_speakers is not None and min_speakers > 2:
        return False
    if max_speakers is not None and max_speakers != 2:
        return False
    if normalized_profile != "dialog" and max_speakers != 2:
        return False
    if detected_speaker_count != 1:
        return False
    if speech_turn_count < max(int(min_turns), 1):
        return False
    return safe_float(duration_sec, default=0.0) >= max(
        safe_float(min_duration_seconds, default=0.0),
        0.0,
    )


def _normalise_turns(turns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in turns:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        start = safe_float(row.get("start"), default=0.0)
        end = safe_float(row.get("end"), default=start)
        if end < start:
            end = start
        speaker = str(row.get("speaker") or "S1").strip() or "S1"
        payload: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
            "text": text,
        }
        language = normalise_language_code(row.get("language"))
        if language:
            payload["language"] = language
        out.append(payload)
    out.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return out


def _turn_duration(turn: dict[str, Any]) -> float:
    start = safe_float(turn.get("start"), default=0.0)
    end = safe_float(turn.get("end"), default=start)
    return max(end - start, 0.0)


def _turn_gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return safe_float(right.get("start"), default=0.0) - safe_float(
        left.get("end"),
        default=0.0,
    )


def _join_text(left: str, right: str) -> str:
    return " ".join(part for part in (left.strip(), right.strip()) if part).strip()


def _merge_turns(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    speaker: str | None = None,
) -> dict[str, Any]:
    merged_speaker = speaker or str(left.get("speaker") or "S1")
    payload: dict[str, Any] = {
        "start": round(
            min(
                safe_float(left.get("start"), default=0.0),
                safe_float(right.get("start"), default=0.0),
            ),
            3,
        ),
        "end": round(
            max(
                safe_float(left.get("end"), default=0.0),
                safe_float(right.get("end"), default=0.0),
            ),
            3,
        ),
        "speaker": merged_speaker,
        "text": _join_text(str(left.get("text") or ""), str(right.get("text") or "")),
    }
    language = normalise_language_code(left.get("language")) or normalise_language_code(
        right.get("language")
    )
    if language:
        payload["language"] = language
    return payload


def _merge_adjacent_same_speaker(
    turns: Sequence[dict[str, Any]],
    *,
    gap_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    if not turns:
        return [], 0

    merged: list[dict[str, Any]] = [dict(turns[0])]
    merge_count = 0
    for turn in turns[1:]:
        previous = merged[-1]
        if (
            str(previous.get("speaker")) == str(turn.get("speaker"))
            and _turn_gap(previous, turn) <= gap_threshold
        ):
            merged[-1] = _merge_turns(previous, turn)
            merge_count += 1
            continue
        merged.append(dict(turn))
    return merged, merge_count


def _should_absorb_micro_turn(
    previous: dict[str, Any],
    current: dict[str, Any],
    following: dict[str, Any],
    *,
    max_duration: float,
    gap_threshold: float,
) -> bool:
    if max_duration <= 0.0:
        return False
    if str(previous.get("speaker")) != str(following.get("speaker")):
        return False
    if str(current.get("speaker")) == str(previous.get("speaker")):
        return False
    if _turn_duration(current) >= max_duration:
        return False
    if _turn_duration(previous) < max_duration or _turn_duration(following) < max_duration:
        return False
    return _turn_gap(previous, current) <= gap_threshold and _turn_gap(
        current,
        following,
    ) <= gap_threshold


def _absorb_micro_turns(
    turns: Sequence[dict[str, Any]],
    *,
    max_duration: float,
    gap_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    collapsed = [dict(turn) for turn in turns]
    absorbed = 0
    idx = 1
    while idx < len(collapsed) - 1:
        previous = collapsed[idx - 1]
        current = collapsed[idx]
        following = collapsed[idx + 1]
        if _should_absorb_micro_turn(
            previous,
            current,
            following,
            max_duration=max_duration,
            gap_threshold=gap_threshold,
        ):
            merged = _merge_turns(previous, current, speaker=str(previous["speaker"]))
            merged = _merge_turns(merged, following, speaker=str(previous["speaker"]))
            collapsed[idx - 1 : idx + 2] = [merged]
            absorbed += 1
            idx = max(idx - 1, 1)
            continue
        idx += 1
    return collapsed, absorbed


def smooth_speaker_turns(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    merge_gap_seconds: float = DEFAULT_DIARIZATION_MERGE_GAP_SECONDS,
    min_turn_seconds: float = DEFAULT_DIARIZATION_MIN_TURN_SECONDS,
) -> SpeakerTurnSmoothingResult:
    turns = _normalise_turns(speaker_turns)
    if not turns:
        return SpeakerTurnSmoothingResult(
            turns=[],
            adjacent_merges=0,
            micro_turn_absorptions=0,
            turn_count_before=0,
            turn_count_after=0,
            speaker_count_before=0,
            speaker_count_after=0,
        )

    safe_gap = max(safe_float(merge_gap_seconds, default=0.0), 0.0)
    safe_min_turn = max(safe_float(min_turn_seconds, default=0.0), 0.0)
    turn_count_before = len(turns)
    speaker_count_before = len({str(turn["speaker"]) for turn in turns})

    merged_turns, adjacent_merges = _merge_adjacent_same_speaker(
        turns,
        gap_threshold=safe_gap,
    )
    absorbed_turns, micro_turn_absorptions = _absorb_micro_turns(
        merged_turns,
        max_duration=safe_min_turn,
        gap_threshold=safe_gap,
    )
    final_turns, extra_adjacent_merges = _merge_adjacent_same_speaker(
        absorbed_turns,
        gap_threshold=safe_gap,
    )

    return SpeakerTurnSmoothingResult(
        turns=final_turns,
        adjacent_merges=adjacent_merges + extra_adjacent_merges,
        micro_turn_absorptions=micro_turn_absorptions,
        turn_count_before=turn_count_before,
        turn_count_after=len(final_turns),
        speaker_count_before=speaker_count_before,
        speaker_count_after=len({str(turn["speaker"]) for turn in final_turns}),
    )


__all__ = [
    "DEFAULT_DIALOG_MAX_SPEAKERS",
    "DEFAULT_DIALOG_MIN_SPEAKERS",
    "DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS",
    "DEFAULT_DIALOG_RETRY_MIN_TURNS",
    "DEFAULT_DIARIZATION_MERGE_GAP_SECONDS",
    "DEFAULT_DIARIZATION_MIN_TURN_SECONDS",
    "DEFAULT_MEETING_MAX_SPEAKERS",
    "DEFAULT_MEETING_MIN_SPEAKERS",
    "SpeakerTurnSmoothingResult",
    "annotation_speaker_count",
    "profile_default_speaker_hints",
    "should_retry_dialog",
    "smooth_speaker_turns",
]
