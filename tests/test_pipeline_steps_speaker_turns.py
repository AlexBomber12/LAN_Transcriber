from __future__ import annotations

from lan_transcriber.pipeline_steps.speaker_turns import (
    build_speaker_turns,
    count_interruptions,
    normalise_asr_segments,
)


def test_normalise_asr_segments_keeps_word_timestamps():
    rows = [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "hello team",
            "words": [{"start": 0.0, "end": 0.4, "word": "hello"}],
            "language": "EN-us",
            "language_confidence": 0.91,
            "language_source": "chunk_hint",
            "language_uncertain": True,
            "language_conflict": False,
            "language_hint": "en",
            "language_hint_applied": True,
        }
    ]
    payload = normalise_asr_segments(rows)
    assert payload[0]["words"][0]["word"] == "hello"
    assert payload[0]["language"] == "EN-us"
    assert payload[0]["language_confidence"] == 0.91
    assert payload[0]["language_source"] == "chunk_hint"
    assert payload[0]["language_uncertain"] is True
    assert payload[0]["language_conflict"] is False
    assert payload[0]["language_hint"] == "en"
    assert payload[0]["language_hint_applied"] is True


def test_build_speaker_turns_assigns_speakers_from_diarization():
    asr_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.6, "word": "hello"}],
            "language": "en",
        },
        {
            "start": 2.0,
            "end": 3.0,
            "text": "status",
            "words": [{"start": 2.0, "end": 2.4, "word": "status"}],
            "language": "en",
        },
    ]
    diar_segments = [
        {"start": 0.0, "end": 1.5, "speaker": "S1"},
        {"start": 1.5, "end": 3.5, "speaker": "S2"},
    ]

    turns = build_speaker_turns(asr_segments, diar_segments, default_language="en")

    assert turns[0]["speaker"] == "S1"
    assert turns[-1]["speaker"] == "S2"


def test_count_interruptions_small_synthetic_case():
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "S1", "text": "long turn"},
        {"start": 3.7, "end": 5.0, "speaker": "S2", "text": "interrupts"},
        {"start": 5.1, "end": 6.0, "speaker": "S1", "text": "continues"},
    ]

    stats = count_interruptions(turns, overlap_threshold=0.2)

    assert stats["total"] == 1
    assert stats["done"]["S2"] == 1
    assert stats["received"]["S1"] == 1
