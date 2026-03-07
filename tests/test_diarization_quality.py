from __future__ import annotations

from types import SimpleNamespace

from lan_transcriber.pipeline_steps.diarization_quality import (
    SpeakerTurnSmoothingResult,
    _merge_adjacent_same_speaker,
    _should_absorb_micro_turn,
    annotation_speaker_count,
    profile_default_speaker_hints,
    should_retry_dialog,
    smooth_speaker_turns,
)


def _annotation(*items):
    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            for item in items:
                yield item

    return _Annotation()


def test_profile_default_speaker_hints_cover_known_profiles():
    assert profile_default_speaker_hints("dialog") == (2, 2)
    assert profile_default_speaker_hints("meeting") == (2, 6)
    assert profile_default_speaker_hints("auto") == (None, None)
    assert profile_default_speaker_hints("unexpected") == (None, None)


def test_annotation_speaker_count_ignores_invalid_tracks_and_counts_multiple_shapes():
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


def test_should_retry_dialog_requires_dialog_like_mode_single_speaker_and_plausible_audio():
    assert not should_retry_dialog(
        profile="auto",
        max_speakers=None,
        detected_speaker_count=1,
        speech_turn_count=8,
        duration_sec=45.0,
        min_turns=4,
        min_duration_seconds=20.0,
    )
    assert not should_retry_dialog(
        profile="dialog",
        max_speakers=2,
        detected_speaker_count=2,
        speech_turn_count=8,
        duration_sec=45.0,
        min_turns=4,
        min_duration_seconds=20.0,
    )
    assert not should_retry_dialog(
        profile="dialog",
        max_speakers=2,
        detected_speaker_count=1,
        speech_turn_count=3,
        duration_sec=45.0,
        min_turns=4,
        min_duration_seconds=20.0,
    )
    assert not should_retry_dialog(
        profile="dialog",
        max_speakers=2,
        detected_speaker_count=1,
        speech_turn_count=8,
        duration_sec=10.0,
        min_turns=4,
        min_duration_seconds=20.0,
    )
    assert not should_retry_dialog(
        profile="dialog",
        max_speakers=4,
        detected_speaker_count=1,
        speech_turn_count=8,
        duration_sec=45.0,
        min_turns=4,
        min_duration_seconds=20.0,
    )
    assert should_retry_dialog(
        profile="meeting",
        max_speakers=2,
        detected_speaker_count=1,
        speech_turn_count=8,
        duration_sec=45.0,
        min_turns=4,
        min_duration_seconds=20.0,
    )


def test_smooth_speaker_turns_merges_adjacent_same_speaker_and_normalises_rows():
    result = smooth_speaker_turns(
        [
            "bad",
            {"text": "   "},
            {"start": 2.0, "end": 1.0, "speaker": "S1", "text": "hello", "language": "EN"},
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


def test_smooth_speaker_turns_handles_empty_input():
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


def test_private_diarization_quality_helpers_cover_remaining_guard_paths():
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
