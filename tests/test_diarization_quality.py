from __future__ import annotations

from types import SimpleNamespace

import pytest

from lan_transcriber.pipeline_steps.diarization_quality import (
    AUTO_PROFILE_DIALOG_MIN_ACCEPTABLE_TOP_TWO_COVERAGE,
    DiarizationProfileDecision,
    DiarizationProfileMetrics,
    SpeakerTurnSmoothingResult,
    _merge_adjacent_same_speaker,
    _rounded_ratio,
    _should_absorb_micro_turn,
    annotation_speaker_count,
    choose_dialog_retry_winner,
    classify_diarization_profile,
    diarization_profile_metrics,
    profile_default_speaker_hints,
    smooth_speaker_turns,
)


def _annotation(*items):
    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            for item in items:
                yield item

    return _Annotation()


def _annotation_from_segments(*segments: tuple[float, float, str]):
    rows = []
    for index, (start, end, speaker) in enumerate(segments, start=1):
        segment = SimpleNamespace(start=start, end=end)
        if index % 2:
            rows.append((segment, speaker))
        else:
            rows.append((segment, f"track-{index}", speaker))
    return _annotation(*rows)


def _classify(
    diarization,
    *,
    speech_turn_count: int = 8,
    duration_sec: float = 30.0,
    min_turns: int = 4,
    min_duration_seconds: float = 20.0,
) -> DiarizationProfileDecision:
    return classify_diarization_profile(
        diarization,
        speech_turn_count=speech_turn_count,
        duration_sec=duration_sec,
        min_turns=min_turns,
        min_duration_seconds=min_duration_seconds,
    )


def _decision(
    profile: str,
    score: float,
    *,
    speaker_count: int = 2,
    top_two_coverage: float = 1.0,
    dominant_turn_count: int = 4,
) -> DiarizationProfileDecision:
    top_speakers = (("S1", 5.0), ("S2", 5.0))
    if speaker_count == 1:
        top_speakers = (("S1", 10.0),)
    metrics = DiarizationProfileMetrics(
        speaker_count=speaker_count,
        total_speech_seconds=10.0,
        top_speakers=top_speakers,
        top_two_coverage=top_two_coverage,
        low_mass_speaker_count=max(speaker_count - 2, 0),
        low_mass_seconds=0.0,
        low_mass_coverage=0.0,
        dominant_turn_count=dominant_turn_count,
        dominant_alternation_ratio=0.8 if dominant_turn_count > 1 else 0.0,
        overlap_seconds=0.0,
        overlap_ratio=0.0,
    )
    return DiarizationProfileDecision(
        selected_profile=profile,
        reason="test",
        metrics=metrics,
        dialog_score=score,
    )


def test_profile_default_speaker_hints_and_annotation_count_cover_known_shapes():
    assert profile_default_speaker_hints("dialog") == (2, 2)
    assert profile_default_speaker_hints("meeting") == (2, 6)
    assert profile_default_speaker_hints("auto") == (None, None)
    assert profile_default_speaker_hints("unexpected") == (None, None)

    segment = SimpleNamespace(start=0.0, end=1.0)
    annotation = _annotation(
        "bad",
        (segment, ""),
        (segment, "S1"),
        (segment, "track-1", "S2"),
    )

    assert annotation_speaker_count(None) == 0
    assert annotation_speaker_count(object()) == 0
    assert annotation_speaker_count(annotation) == 2


def test_diarization_profile_metrics_capture_dominant_speakers_low_mass_and_overlap():
    metrics = diarization_profile_metrics(
        _annotation_from_segments(
            (0.0, 2.0, "S1"),
            (2.0, 4.0, "S2"),
            (4.0, 6.0, "S1"),
            (6.0, 8.0, "S2"),
            (8.0, 8.5, "S3"),
            (8.25, 8.75, "S2"),
        )
    )

    assert metrics.speaker_count == 3
    assert metrics.top_speakers == (("S2", 4.5), ("S1", 4.0), ("S3", 0.5))
    assert metrics.top_two_coverage == 0.9444
    assert metrics.low_mass_speaker_count == 1
    assert metrics.low_mass_coverage == 0.0556
    assert metrics.dominant_turn_count == 5
    assert metrics.dominant_alternation_ratio == 0.75
    assert metrics.overlap_seconds == 0.25
    assert metrics.overlap_ratio == 0.0278
    assert metrics.as_dict()["dominant_speakers"] == ["S2", "S1"]


def test_diarization_profile_metrics_handle_malformed_segments_and_zero_denominators():
    assert _rounded_ratio(1.0, 0.0) == 0.0

    malformed_only = diarization_profile_metrics(
        _annotation(
            (SimpleNamespace(start=0.0, end=1.0), " "),
            (SimpleNamespace(start=2.0, end=1.0), "S1"),
        )
    )
    assert malformed_only == DiarizationProfileMetrics()

    with_zero_duration = diarization_profile_metrics(
        _annotation(
            (SimpleNamespace(start=0.0, end=1.0), "S1"),
            (SimpleNamespace(start=2.0, end=1.0), "track-2", "S2"),
        )
    )
    assert with_zero_duration.speaker_count == 1
    assert with_zero_duration.overlap_seconds == 0.0


def test_classify_diarization_profile_identifies_dialog_candidates():
    single_speaker = _classify(
        _annotation_from_segments((0.0, 25.0, "S1")),
        speech_turn_count=8,
        duration_sec=25.0,
    )
    assert single_speaker.selected_profile == "dialog"
    assert single_speaker.reason == "single_speaker_long_recording"

    dominant_pair = _classify(
        _annotation_from_segments(
            (0.0, 1.0, "S1"),
            (1.0, 2.0, "S2"),
            (2.0, 3.0, "S1"),
            (3.0, 4.0, "S2"),
            (4.0, 5.0, "S1"),
            (5.0, 6.0, "S2"),
            (6.0, 6.2, "S3"),
        ),
        speech_turn_count=10,
        duration_sec=30.0,
    )
    assert dominant_pair.selected_profile == "dialog"
    assert dominant_pair.reason == "dominant_pair_dialog_like"
    assert dominant_pair.metrics.low_mass_speaker_count == 1


@pytest.mark.parametrize(
    ("diarization", "speech_turn_count", "duration_sec", "expected_reason"),
    [
        (object(), 8, 30.0, "no_valid_speakers"),
        (_annotation_from_segments((0.0, 10.0, "S1")), 2, 10.0, "single_speaker_short_recording"),
        (
            _annotation_from_segments(
                (0.0, 1.0, "S1"),
                (1.0, 2.0, "S2"),
                (2.0, 3.0, "S3"),
                (3.0, 4.0, "S4"),
                (4.0, 5.0, "S5"),
            ),
            10,
            30.0,
            "too_many_speakers",
        ),
        (
            _annotation_from_segments(
                (0.0, 2.0, "S1"),
                (2.0, 4.0, "S2"),
                (4.0, 6.0, "S1"),
                (6.0, 8.0, "S2"),
                (8.0, 10.0, "S3"),
            ),
            10,
            30.0,
            "non_tiny_extra_speakers",
        ),
        (
            _annotation_from_segments(
                (0.0, 1.5, "S1"),
                (1.5, 3.0, "S2"),
                (3.0, 4.5, "S1"),
                (4.5, 6.0, "S2"),
                (6.0, 6.7, "S3"),
                (6.7, 7.4, "S4"),
            ),
            10,
            30.0,
            "low_top_two_coverage",
        ),
        (
            _annotation_from_segments(
                (0.0, 2.0, "S1"),
                (2.0, 4.0, "S2"),
                (4.0, 6.0, "S1"),
                (6.0, 8.0, "S2"),
                (8.0, 8.65, "S3"),
                (8.65, 9.3, "S4"),
            ),
            10,
            30.0,
            "too_much_low_mass_speech",
        ),
        (
            _annotation_from_segments(
                (0.0, 4.0, "S1"),
                (4.0, 8.0, "S2"),
            ),
            10,
            30.0,
            "insufficient_dominant_turns",
        ),
        (
            _annotation_from_segments(
                (0.0, 2.0, "S1"),
                (2.0, 4.0, "S1"),
                (4.0, 6.0, "S1"),
                (6.0, 8.0, "S2"),
            ),
            10,
            30.0,
            "low_turn_alternation",
        ),
        (
            _annotation_from_segments(
                (0.0, 2.0, "S1"),
                (1.0, 3.0, "S2"),
                (3.0, 5.0, "S1"),
                (4.0, 6.0, "S2"),
            ),
            10,
            30.0,
            "high_overlap",
        ),
    ],
)
def test_classify_diarization_profile_reports_meeting_reasons(
    diarization,
    speech_turn_count: int,
    duration_sec: float,
    expected_reason: str,
):
    decision = _classify(
        diarization,
        speech_turn_count=speech_turn_count,
        duration_sec=duration_sec,
    )

    assert decision.selected_profile == "meeting"
    assert decision.reason == expected_reason


def test_choose_dialog_retry_winner_covers_rejections_and_success():
    initial = _decision("dialog", 1.0)

    assert choose_dialog_retry_winner(
        initial,
        _decision("meeting", 1.3),
    ).winner_reason == "dialog_retry_not_dialog_like"
    assert choose_dialog_retry_winner(
        initial,
        _decision("dialog", 1.3, speaker_count=1),
    ).winner_reason == "dialog_retry_single_speaker"
    assert choose_dialog_retry_winner(
        initial,
        _decision("dialog", 1.3, dominant_turn_count=1),
    ).winner_reason == "dialog_retry_pathological_turns"
    assert choose_dialog_retry_winner(
        initial,
        _decision(
            "dialog",
            1.3,
            top_two_coverage=AUTO_PROFILE_DIALOG_MIN_ACCEPTABLE_TOP_TWO_COVERAGE - 0.01,
        ),
    ).winner_reason == "dialog_retry_low_top_two_coverage"
    assert choose_dialog_retry_winner(
        initial,
        _decision("dialog", 1.04),
    ).winner_reason == "dialog_retry_not_better"

    accepted = choose_dialog_retry_winner(
        initial,
        _decision("dialog", 1.2),
    )
    assert accepted.selected_result == "dialog_retry"
    assert accepted.winner_reason == "dialog_retry_improved_dialog_score"


def test_smooth_speaker_turns_merges_adjacent_same_speaker_and_normalises_rows():
    result = smooth_speaker_turns(
        [
            "bad",
            {"text": "   "},
            {
                "start": 2.0,
                "end": 1.0,
                "speaker": "S1",
                "text": "hello",
                "language": "EN",
            },
            {"start": 2.2, "end": 2.8, "speaker": "S1", "text": "again"},
            {"start": 4.0, "end": 4.2, "speaker": "S2", "text": "next", "language": "fr"},
        ],
        merge_gap_seconds=0.3,
        min_turn_seconds=0.0,
    )

    assert isinstance(result, SpeakerTurnSmoothingResult)
    assert result.adjacent_merges == 1
    assert result.micro_turn_absorptions == 0
    assert result.turn_count_before == 3
    assert result.turn_count_after == 2
    assert result.speaker_count_before == 2
    assert result.speaker_count_after == 2
    assert result.turns[0] == {
        "start": 2.0,
        "end": 2.8,
        "speaker": "S1",
        "text": "hello again",
        "language": "en",
    }


def test_smooth_speaker_turns_absorbs_micro_turns_but_preserves_real_changes():
    absorbed = smooth_speaker_turns(
        [
            {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "alpha"},
            {"start": 1.0, "end": 1.2, "speaker": "S2", "text": "noise"},
            {"start": 1.2, "end": 2.3, "speaker": "S1", "text": "omega"},
        ],
        merge_gap_seconds=0.2,
        min_turn_seconds=0.5,
    )
    assert absorbed.micro_turn_absorptions == 1
    assert absorbed.turns == [
        {"start": 0.0, "end": 2.3, "speaker": "S1", "text": "alpha noise omega"}
    ]

    preserved = smooth_speaker_turns(
        [
            {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "hello"},
            {"start": 1.0, "end": 1.7, "speaker": "S2", "text": "real interruption"},
            {"start": 1.7, "end": 2.6, "speaker": "S1", "text": "continue"},
            {"start": 3.5, "end": 4.0, "speaker": "S1", "text": "later"},
        ],
        merge_gap_seconds=0.2,
        min_turn_seconds=0.5,
    )
    assert preserved.micro_turn_absorptions == 0
    assert preserved.turn_count_after == 4
    assert [turn["speaker"] for turn in preserved.turns] == ["S1", "S2", "S1", "S1"]


def test_smooth_speaker_turns_handles_empty_input_and_private_guards():
    result = smooth_speaker_turns([])

    assert result == SpeakerTurnSmoothingResult(
        turns=[],
        adjacent_merges=0,
        micro_turn_absorptions=0,
        turn_count_before=0,
        turn_count_after=0,
        speaker_count_before=0,
        speaker_count_after=0,
    )
    assert _merge_adjacent_same_speaker([], gap_threshold=0.1) == ([], 0)
    assert not _should_absorb_micro_turn(
        {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "left"},
        {"start": 1.0, "end": 1.1, "speaker": "S2", "text": "tiny"},
        {"start": 1.1, "end": 2.0, "speaker": "S1", "text": "right"},
        max_duration=0.0,
        gap_threshold=0.1,
    )
    assert not _should_absorb_micro_turn(
        {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "left"},
        {"start": 1.0, "end": 1.1, "speaker": "S1", "text": "same"},
        {"start": 1.1, "end": 2.0, "speaker": "S1", "text": "right"},
        max_duration=0.5,
        gap_threshold=0.1,
    )
    assert not _should_absorb_micro_turn(
        {"start": 0.0, "end": 0.2, "speaker": "S1", "text": "short"},
        {"start": 0.2, "end": 0.3, "speaker": "S2", "text": "tiny"},
        {"start": 0.3, "end": 1.0, "speaker": "S1", "text": "right"},
        max_duration=0.5,
        gap_threshold=0.1,
    )
