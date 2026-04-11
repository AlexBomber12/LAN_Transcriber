from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from lan_transcriber.utils import normalise_language_code, safe_float

from .speaker_turns import _decapitalize_join, _diarization_segments

DEFAULT_DIALOG_MIN_SPEAKERS = 2
DEFAULT_DIALOG_MAX_SPEAKERS = 2
DEFAULT_MEETING_MIN_SPEAKERS = 2
DEFAULT_MEETING_MAX_SPEAKERS = 6
DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS = 20.0
DEFAULT_DIALOG_RETRY_MIN_TURNS = 6
DEFAULT_DIARIZATION_MERGE_GAP_SECONDS = 0.5
DEFAULT_DIARIZATION_MIN_TURN_SECONDS = 0.5
DEFAULT_DIARIZATION_FLICKER_MIN_SECONDS = 3.0
DEFAULT_DIARIZATION_FLICKER_MAX_CONSECUTIVE = 2
AUTO_PROFILE_DEFAULT_INITIAL_PROFILE = "meeting"
AUTO_PROFILE_DIALOG_MAX_SPEAKERS = 4
AUTO_PROFILE_DIALOG_MIN_TOP_TWO_COVERAGE = 0.85
AUTO_PROFILE_DIALOG_MAX_LOW_MASS_COVERAGE = 0.12
AUTO_PROFILE_DIALOG_MAX_LOW_MASS_SHARE_PER_SPEAKER = 0.12
AUTO_PROFILE_DIALOG_MIN_DOMINANT_TURNS = 3
AUTO_PROFILE_DIALOG_MIN_ALTERNATION_RATIO = 0.55
AUTO_PROFILE_DIALOG_MAX_OVERLAP_RATIO = 0.2
AUTO_PROFILE_DIALOG_RETRY_SCORE_MIN_DELTA = 0.05
AUTO_PROFILE_DIALOG_MIN_ACCEPTABLE_TOP_TWO_COVERAGE = 0.75


@dataclass(frozen=True)
class DiarizationProfileMetrics:
    speaker_count: int = 0
    total_speech_seconds: float = 0.0
    top_speakers: tuple[tuple[str, float], ...] = ()
    top_two_coverage: float = 0.0
    low_mass_speaker_count: int = 0
    low_mass_seconds: float = 0.0
    low_mass_coverage: float = 0.0
    dominant_turn_count: int = 0
    dominant_alternation_ratio: float = 0.0
    overlap_seconds: float = 0.0
    overlap_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        dominant_speakers = [speaker for speaker, _seconds in self.top_speakers[:2]]
        return {
            "speaker_count": self.speaker_count,
            "total_speech_seconds": self.total_speech_seconds,
            "top_speakers": [
                {"speaker": speaker, "seconds": seconds}
                for speaker, seconds in self.top_speakers
            ],
            "dominant_speakers": dominant_speakers,
            "top_two_coverage": self.top_two_coverage,
            "low_mass_speaker_count": self.low_mass_speaker_count,
            "low_mass_seconds": self.low_mass_seconds,
            "low_mass_coverage": self.low_mass_coverage,
            "dominant_turn_count": self.dominant_turn_count,
            "dominant_alternation_ratio": self.dominant_alternation_ratio,
            "overlap_seconds": self.overlap_seconds,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass(frozen=True)
class DiarizationProfileDecision:
    selected_profile: Literal["dialog", "meeting"]
    reason: str
    metrics: DiarizationProfileMetrics
    dialog_score: float


@dataclass(frozen=True)
class DialogRetrySelection:
    selected_result: Literal["initial_pass", "dialog_retry"]
    winner_reason: str
    initial_score: float
    retry_score: float


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
    return len(
        {
            str(row.get("speaker") or "").strip()
            for row in _diarization_segments(diarization)
            if str(row.get("speaker") or "").strip()
        }
    )


def _segment_duration(row: dict[str, Any]) -> float:
    start = safe_float(row.get("start"), default=0.0)
    end = safe_float(row.get("end"), default=start)
    return max(end - start, 0.0)


def _rounded_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return round(numerator / denominator, 4)


def _dominant_sequence(
    segments: Sequence[dict[str, Any]],
    dominant_speakers: set[str],
) -> list[str]:
    out: list[str] = []
    for row in segments:
        speaker = str(row.get("speaker") or "").strip()
        if speaker in dominant_speakers and _segment_duration(row) > 0.0:
            out.append(speaker)
    return out


def _overlap_seconds(segments: Sequence[dict[str, Any]]) -> float:
    total = 0.0
    previous_end = 0.0
    for row in segments:
        start = safe_float(row.get("start"), default=0.0)
        end = safe_float(row.get("end"), default=start)
        if end <= start:
            continue
        overlap = min(previous_end, end) - start
        if overlap > 0.0:
            total += overlap
        previous_end = max(previous_end, end)
    return round(total, 4)


def diarization_profile_metrics(diarization: Any) -> DiarizationProfileMetrics:
    segments = _diarization_segments(diarization)
    if not segments:
        return DiarizationProfileMetrics()

    speaker_durations: dict[str, float] = {}
    for row in segments:
        speaker = str(row.get("speaker") or "").strip()
        if not speaker:
            continue
        duration = _segment_duration(row)
        if duration <= 0.0:
            continue
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration

    if not speaker_durations:
        return DiarizationProfileMetrics()

    ordered_speakers = tuple(
        sorted(
            (
                (speaker, round(seconds, 4))
                for speaker, seconds in speaker_durations.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    total_speech_seconds = round(sum(seconds for _speaker, seconds in ordered_speakers), 4)
    top_two_seconds = round(sum(seconds for _speaker, seconds in ordered_speakers[:2]), 4)
    low_mass_speakers = tuple(
        (speaker, seconds)
        for speaker, seconds in ordered_speakers[2:]
        if _rounded_ratio(seconds, total_speech_seconds)
        <= AUTO_PROFILE_DIALOG_MAX_LOW_MASS_SHARE_PER_SPEAKER
    )
    low_mass_seconds = round(
        sum(seconds for _speaker, seconds in low_mass_speakers),
        4,
    )
    dominant_speakers = {speaker for speaker, _seconds in ordered_speakers[:2]}
    dominant_sequence = _dominant_sequence(segments, dominant_speakers)
    dominant_turn_count = len(dominant_sequence)
    alternations = sum(
        1
        for left, right in zip(dominant_sequence, dominant_sequence[1:])
        if left != right
    )
    overlap_seconds = _overlap_seconds(segments)

    return DiarizationProfileMetrics(
        speaker_count=len(ordered_speakers),
        total_speech_seconds=total_speech_seconds,
        top_speakers=ordered_speakers,
        top_two_coverage=_rounded_ratio(top_two_seconds, total_speech_seconds),
        low_mass_speaker_count=len(low_mass_speakers),
        low_mass_seconds=low_mass_seconds,
        low_mass_coverage=_rounded_ratio(low_mass_seconds, total_speech_seconds),
        dominant_turn_count=dominant_turn_count,
        dominant_alternation_ratio=_rounded_ratio(
            float(alternations),
            float(max(dominant_turn_count - 1, 1)),
        ),
        overlap_seconds=overlap_seconds,
        overlap_ratio=_rounded_ratio(overlap_seconds, total_speech_seconds),
    )


def _is_dialog_like_multispeaker(metrics: DiarizationProfileMetrics) -> bool:
    extra_speakers = max(metrics.speaker_count - 2, 0)
    return (
        2 <= metrics.speaker_count <= AUTO_PROFILE_DIALOG_MAX_SPEAKERS
        and metrics.top_two_coverage >= AUTO_PROFILE_DIALOG_MIN_TOP_TWO_COVERAGE
        and metrics.low_mass_speaker_count >= extra_speakers
        and metrics.low_mass_coverage <= AUTO_PROFILE_DIALOG_MAX_LOW_MASS_COVERAGE
        and metrics.dominant_turn_count >= AUTO_PROFILE_DIALOG_MIN_DOMINANT_TURNS
        and metrics.dominant_alternation_ratio >= AUTO_PROFILE_DIALOG_MIN_ALTERNATION_RATIO
        and metrics.overlap_ratio <= AUTO_PROFILE_DIALOG_MAX_OVERLAP_RATIO
    )


def _dialog_score(metrics: DiarizationProfileMetrics) -> float:
    if metrics.speaker_count == 2:
        speaker_bonus = 0.6
    elif metrics.speaker_count == 1:
        speaker_bonus = 0.15
    else:
        speaker_bonus = max(0.0, 0.45 - 0.15 * max(metrics.speaker_count - 2, 0))
    score = (
        speaker_bonus
        + metrics.top_two_coverage
        + metrics.dominant_alternation_ratio
        - metrics.low_mass_coverage
        - metrics.overlap_ratio
    )
    return round(score, 4)


def classify_diarization_profile(
    diarization: Any,
    *,
    speech_turn_count: int,
    duration_sec: float | None,
    min_turns: int,
    min_duration_seconds: float,
) -> DiarizationProfileDecision:
    metrics = diarization_profile_metrics(diarization)
    enough_turns = speech_turn_count >= max(int(min_turns), 1)
    enough_duration = safe_float(duration_sec, default=0.0) >= max(
        safe_float(min_duration_seconds, default=0.0),
        0.0,
    )
    dialog_score = _dialog_score(metrics)
    dialog_like_multispeaker = _is_dialog_like_multispeaker(metrics)

    if metrics.speaker_count == 1 and enough_turns and enough_duration:
        return DiarizationProfileDecision(
            selected_profile="dialog",
            reason="single_speaker_long_recording",
            metrics=metrics,
            dialog_score=dialog_score,
        )
    if dialog_like_multispeaker and enough_turns and enough_duration:
        return DiarizationProfileDecision(
            selected_profile="dialog",
            reason="dominant_pair_dialog_like",
            metrics=metrics,
            dialog_score=dialog_score,
        )

    if metrics.speaker_count == 0:
        reason = "no_valid_speakers"
    elif metrics.speaker_count == 1:
        reason = "single_speaker_short_recording"
    elif dialog_like_multispeaker and not enough_turns:
        reason = "dialog_like_below_min_turns"
    elif dialog_like_multispeaker and not enough_duration:
        reason = "dialog_like_below_min_duration"
    elif metrics.speaker_count > AUTO_PROFILE_DIALOG_MAX_SPEAKERS:
        reason = "too_many_speakers"
    elif metrics.low_mass_speaker_count < max(metrics.speaker_count - 2, 0):
        reason = "non_tiny_extra_speakers"
    elif metrics.top_two_coverage < AUTO_PROFILE_DIALOG_MIN_TOP_TWO_COVERAGE:
        reason = "low_top_two_coverage"
    elif metrics.low_mass_coverage > AUTO_PROFILE_DIALOG_MAX_LOW_MASS_COVERAGE:
        reason = "too_much_low_mass_speech"
    elif metrics.dominant_turn_count < AUTO_PROFILE_DIALOG_MIN_DOMINANT_TURNS:
        reason = "insufficient_dominant_turns"
    elif metrics.dominant_alternation_ratio < AUTO_PROFILE_DIALOG_MIN_ALTERNATION_RATIO:
        reason = "low_turn_alternation"
    else:
        reason = "high_overlap"
    return DiarizationProfileDecision(
        selected_profile="meeting",
        reason=reason,
        metrics=metrics,
        dialog_score=dialog_score,
    )


def choose_dialog_retry_winner(
    initial: DiarizationProfileDecision,
    retry: DiarizationProfileDecision,
) -> DialogRetrySelection:
    if retry.selected_profile != "dialog":
        return DialogRetrySelection(
            selected_result="initial_pass",
            winner_reason="dialog_retry_not_dialog_like",
            initial_score=initial.dialog_score,
            retry_score=retry.dialog_score,
        )
    if retry.metrics.speaker_count < 2:
        return DialogRetrySelection(
            selected_result="initial_pass",
            winner_reason="dialog_retry_single_speaker",
            initial_score=initial.dialog_score,
            retry_score=retry.dialog_score,
        )
    if retry.metrics.dominant_turn_count < 2:
        return DialogRetrySelection(
            selected_result="initial_pass",
            winner_reason="dialog_retry_pathological_turns",
            initial_score=initial.dialog_score,
            retry_score=retry.dialog_score,
        )
    if retry.metrics.top_two_coverage < AUTO_PROFILE_DIALOG_MIN_ACCEPTABLE_TOP_TWO_COVERAGE:
        return DialogRetrySelection(
            selected_result="initial_pass",
            winner_reason="dialog_retry_low_top_two_coverage",
            initial_score=initial.dialog_score,
            retry_score=retry.dialog_score,
        )
    if retry.dialog_score <= (
        initial.dialog_score + AUTO_PROFILE_DIALOG_RETRY_SCORE_MIN_DELTA
    ):
        return DialogRetrySelection(
            selected_result="initial_pass",
            winner_reason="dialog_retry_not_better",
            initial_score=initial.dialog_score,
            retry_score=retry.dialog_score,
        )
    return DialogRetrySelection(
        selected_result="dialog_retry",
        winner_reason="dialog_retry_improved_dialog_score",
        initial_score=initial.dialog_score,
        retry_score=retry.dialog_score,
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
    return _decapitalize_join(left.strip(), right.strip())


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


def _segment_distance(
    segment: dict[str, Any],
    other: dict[str, Any],
) -> float:
    seg_start = safe_float(segment.get("start"), default=0.0)
    seg_end = safe_float(segment.get("end"), default=seg_start)
    other_start = safe_float(other.get("start"), default=0.0)
    other_end = safe_float(other.get("end"), default=other_start)
    if other_end < seg_start:
        return seg_start - other_end
    if other_start > seg_end:
        return other_start - seg_end
    return 0.0


def filter_flickering_speakers(
    diar_segments: list[dict[str, Any]],
    *,
    min_total_seconds: float = DEFAULT_DIARIZATION_FLICKER_MIN_SECONDS,
    max_consecutive_segments: int = DEFAULT_DIARIZATION_FLICKER_MAX_CONSECUTIVE,
) -> list[dict[str, Any]]:
    """Reassign brief, isolated diarization speakers to nearest stable speaker.

    A speaker is considered a "flicker" speaker when both conditions hold:
    - the speaker's total speech duration across ``diar_segments`` is strictly
      less than ``min_total_seconds``;
    - the speaker never appears in more than ``max_consecutive_segments``
      consecutive diarization segments (i.e. they only show up as brief
      isolated bursts surrounded by other speakers).

    Each diarization segment whose speaker is a flicker is relabelled with the
    speaker of the closest non-flicker segment by time proximity. If every
    speaker in the input is a flicker speaker (e.g. very short recording with
    only tiny bursts), the original list is returned unchanged so no transcript
    content is lost.

    The returned list is sorted by start time.
    """

    if not diar_segments:
        return []

    sorted_segments = sorted(
        (dict(seg) for seg in diar_segments),
        key=lambda row: (
            safe_float(row.get("start"), default=0.0),
            safe_float(row.get("end"), default=0.0),
            str(row.get("speaker") or ""),
        ),
    )

    totals: dict[str, float] = {}
    for seg in sorted_segments:
        speaker = str(seg.get("speaker") or "")
        totals[speaker] = totals.get(speaker, 0.0) + _segment_duration(seg)

    max_run: dict[str, int] = {}
    current_speaker: str | None = None
    current_run = 0
    for seg in sorted_segments:
        speaker = str(seg.get("speaker") or "")
        if speaker == current_speaker:
            current_run += 1
        else:
            current_speaker = speaker
            current_run = 1
        if current_run > max_run.get(speaker, 0):
            max_run[speaker] = current_run

    safe_min_total = max(safe_float(min_total_seconds, default=0.0), 0.0)
    safe_max_consecutive = max(int(max_consecutive_segments), 0)

    flicker_speakers = {
        speaker
        for speaker, total in totals.items()
        if total < safe_min_total
        and max_run.get(speaker, 0) <= safe_max_consecutive
    }

    if not flicker_speakers:
        return sorted_segments

    non_flicker_segments = [
        seg
        for seg in sorted_segments
        if str(seg.get("speaker") or "") not in flicker_speakers
    ]
    if not non_flicker_segments:
        return sorted_segments

    cleaned: list[dict[str, Any]] = []
    for seg in sorted_segments:
        speaker = str(seg.get("speaker") or "")
        if speaker in flicker_speakers:
            nearest = min(
                non_flicker_segments,
                key=lambda candidate: (
                    _segment_distance(seg, candidate),
                    safe_float(candidate.get("start"), default=0.0),
                ),
            )
            new_segment = dict(seg)
            new_segment["speaker"] = str(nearest.get("speaker") or "")
            cleaned.append(new_segment)
        else:
            cleaned.append(dict(seg))

    cleaned.sort(
        key=lambda row: (
            safe_float(row.get("start"), default=0.0),
            safe_float(row.get("end"), default=0.0),
            str(row.get("speaker") or ""),
        )
    )
    return cleaned


__all__ = [
    "AUTO_PROFILE_DEFAULT_INITIAL_PROFILE",
    "AUTO_PROFILE_DIALOG_MAX_OVERLAP_RATIO",
    "AUTO_PROFILE_DIALOG_MAX_SPEAKERS",
    "AUTO_PROFILE_DIALOG_MAX_LOW_MASS_COVERAGE",
    "AUTO_PROFILE_DIALOG_MAX_LOW_MASS_SHARE_PER_SPEAKER",
    "AUTO_PROFILE_DIALOG_MIN_ACCEPTABLE_TOP_TWO_COVERAGE",
    "AUTO_PROFILE_DIALOG_MIN_ALTERNATION_RATIO",
    "AUTO_PROFILE_DIALOG_MIN_DOMINANT_TURNS",
    "AUTO_PROFILE_DIALOG_MIN_TOP_TWO_COVERAGE",
    "AUTO_PROFILE_DIALOG_RETRY_SCORE_MIN_DELTA",
    "DEFAULT_DIALOG_MAX_SPEAKERS",
    "DEFAULT_DIALOG_MIN_SPEAKERS",
    "DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS",
    "DEFAULT_DIALOG_RETRY_MIN_TURNS",
    "DEFAULT_DIARIZATION_FLICKER_MAX_CONSECUTIVE",
    "DEFAULT_DIARIZATION_FLICKER_MIN_SECONDS",
    "DEFAULT_DIARIZATION_MERGE_GAP_SECONDS",
    "DEFAULT_DIARIZATION_MIN_TURN_SECONDS",
    "DEFAULT_MEETING_MAX_SPEAKERS",
    "DEFAULT_MEETING_MIN_SPEAKERS",
    "DialogRetrySelection",
    "DiarizationProfileDecision",
    "DiarizationProfileMetrics",
    "SpeakerTurnSmoothingResult",
    "annotation_speaker_count",
    "choose_dialog_retry_winner",
    "classify_diarization_profile",
    "diarization_profile_metrics",
    "filter_flickering_speakers",
    "profile_default_speaker_hints",
    "smooth_speaker_turns",
]
