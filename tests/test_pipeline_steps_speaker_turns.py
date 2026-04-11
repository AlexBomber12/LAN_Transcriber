from __future__ import annotations

from lan_transcriber.pipeline_steps.speaker_turns import (
    DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC,
    DEFAULT_SPEAKER_TURN_MIN_WORDS,
    DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC,
    _decapitalize_join,
    build_speaker_turns,
    count_interruptions,
    merge_short_turns,
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


def test_merge_gap_default():
    """Two single-word turns from the same speaker with a 2.5s gap should merge.

    With the legacy 1.0s threshold these stayed separate; with the new 4.0s
    default they collapse into a single turn.
    """
    asr_segments = [
        {
            "start": 0.0,
            "end": 0.5,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello"}],
        },
        {
            "start": 3.0,
            "end": 3.5,
            "text": "world",
            "words": [{"start": 3.0, "end": 3.5, "word": "world"}],
        },
    ]
    diar_segments = [{"start": 0.0, "end": 5.0, "speaker": "S1"}]

    turns = build_speaker_turns(asr_segments, diar_segments, default_language=None)

    assert DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC == 4.0
    assert len(turns) == 1
    assert turns[0]["speaker"] == "S1"
    assert turns[0]["text"] == "hello world"


def test_merge_gap_exceeded():
    """Same speaker with a gap above the default threshold remains split."""
    asr_segments = [
        {
            "start": 0.0,
            "end": 0.5,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello"}],
        },
        {
            "start": 6.5,
            "end": 7.0,
            "text": "world",
            "words": [{"start": 6.5, "end": 7.0, "word": "world"}],
        },
    ]
    diar_segments = [{"start": 0.0, "end": 8.0, "speaker": "S1"}]

    turns = build_speaker_turns(asr_segments, diar_segments, default_language=None)

    assert len(turns) == 2
    assert turns[0]["text"] == "hello"
    assert turns[1]["text"] == "world"


def test_short_turn_merged_into_previous():
    """A short follow-up turn from the same speaker merges into the previous turn."""
    long_text = " ".join(f"word{i}" for i in range(20))
    short_text = "ok thanks bye"
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "S1", "text": long_text},
        {"start": 6.0, "end": 6.5, "speaker": "S1", "text": short_text},
    ]

    merged = merge_short_turns(turns, min_words=DEFAULT_SPEAKER_TURN_MIN_WORDS)

    assert len(merged) == 1
    assert merged[0]["speaker"] == "S1"
    assert merged[0]["text"] == f"{long_text} {short_text}"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 6.5


def test_short_turn_merged_into_next():
    """A short opening turn merges forward into a same-speaker following turn."""
    long_text = " ".join(f"word{i}" for i in range(20))
    turns = [
        {"start": 0.0, "end": 0.5, "speaker": "S1", "text": "hi"},
        {"start": 1.0, "end": 5.0, "speaker": "S1", "text": long_text},
    ]

    merged = merge_short_turns(turns, min_words=DEFAULT_SPEAKER_TURN_MIN_WORDS)

    assert len(merged) == 1
    assert merged[0]["speaker"] == "S1"
    assert merged[0]["start"] == 0.0
    assert merged[0]["text"] == f"hi {long_text}"


def test_short_turn_different_speakers_kept():
    """A short turn between two different speakers is preserved as-is."""
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "hello there partner"},
        {"start": 1.2, "end": 1.5, "speaker": "S2", "text": "yes please"},
        {"start": 1.7, "end": 3.0, "speaker": "S3", "text": "all right then continue please"},
    ]

    merged = merge_short_turns(turns, min_words=DEFAULT_SPEAKER_TURN_MIN_WORDS)

    assert len(merged) == 3
    assert merged[1]["speaker"] == "S2"
    assert merged[1]["text"] == "yes please"


def test_merge_short_turns_empty():
    assert merge_short_turns([]) == []


def test_merge_short_turns_default_gap_exceeds_base_merge_gap():
    """The post-pass default must exceed the base merge gap to do real work.

    After ``build_speaker_turns`` runs with the base ``merge_gap_sec``, every
    surviving same-speaker pair has a gap strictly greater than that base
    threshold. The short-turn post-pass therefore needs a strictly larger
    default in order to ever fire on real pipeline output. This test pins both
    invariants and exercises the case where ``build_speaker_turns`` left a
    short same-speaker continuation split because the gap exceeded the base
    merge threshold, and the post-pass still folds it back together.
    """
    assert DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC > DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC

    long_text = " ".join(f"word{i}" for i in range(20))
    asr_segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": long_text,
            "words": [
                {"start": float(i) * 0.25, "end": float(i) * 0.25 + 0.2, "word": f"word{i}"}
                for i in range(20)
            ],
        },
        {
            "start": 11.0,
            "end": 11.5,
            "text": "ok",
            "words": [{"start": 11.0, "end": 11.5, "word": "ok"}],
        },
    ]
    diar_segments = [{"start": 0.0, "end": 12.0, "speaker": "S1"}]

    base_turns = build_speaker_turns(
        asr_segments,
        diar_segments,
        default_language=None,
    )

    # build_speaker_turns left these split because the inter-turn gap (~6s)
    # exceeds the base merge_gap_sec default (4s).
    assert len(base_turns) == 2
    assert base_turns[0]["speaker"] == base_turns[1]["speaker"] == "S1"

    merged = merge_short_turns(base_turns)
    assert len(merged) == 1
    assert merged[0]["text"].endswith("ok")
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 11.5


def test_decapitalize_mid_sentence():
    """A capitalized word mid-sentence (no preceding sentence punct) is lowercased."""
    assert (
        _decapitalize_join("энергии. Он может", "Понизить")
        == "энергии. Он может понизить"
    )


def test_keep_after_period():
    """After sentence-ending punctuation the appended capital is preserved."""
    assert (
        _decapitalize_join("решение.", "Но мы понимаем")
        == "решение. Но мы понимаем"
    )


def test_keep_after_exclamation():
    assert _decapitalize_join("стоп!", "Вперёд") == "стоп! Вперёд"


def test_keep_after_question():
    assert _decapitalize_join("правда?", "Да") == "правда? Да"


def test_keep_acronym():
    """All-caps acronyms longer than one char are preserved."""
    assert _decapitalize_join("стандарту", "ISO") == "стандарту ISO"
    assert _decapitalize_join("use the", "API today") == "use the API today"


def test_keep_acronym_leading_compound():
    """Acronym-leading compounds (e.g. ``API-based``, ``NASA's``) are preserved.

    The acronym check must look at the leading run of alphabetic characters
    rather than the whole word, otherwise common compounds with lowercase
    suffixes would be incorrectly rewritten to ``aPI-based`` / ``nASA's``.
    """
    assert (
        _decapitalize_join("use an", "API-based approach")
        == "use an API-based approach"
    )
    assert (
        _decapitalize_join("that is", "NASA's mission")
        == "that is NASA's mission"
    )
    # Three-letter acronym with a suffix.
    assert (
        _decapitalize_join("use a", "JSON-encoded payload")
        == "use a JSON-encoded payload"
    )


def test_keep_english_I():
    """The English pronoun ``I`` is preserved as capitalized."""
    assert _decapitalize_join("think", "I will") == "think I will"


def test_keep_already_lowercase():
    """An already-lowercase appended fragment is passed through unchanged."""
    assert _decapitalize_join("просто", "уходить") == "просто уходить"


def test_empty_strings():
    """Empty existing or appended text falls back to a plain strip-join."""
    assert _decapitalize_join("", "Hello") == "Hello"
    assert _decapitalize_join("text", "") == "text"
    assert _decapitalize_join("", "") == ""


def test_decapitalize_single_capital_letter_lowercased():
    """A single capital letter (not ``I``) mid-sentence is still lowercased."""
    # Russian "А" (single Cyrillic capital letter) should be lowercased.
    assert _decapitalize_join("ну", "А дальше") == "ну а дальше"


def test_real_example_build_speaker_turns_decapitalizes_false_starts():
    """End-to-end: segment-boundary-capitalized words are decapitalized on merge."""
    # Simulate ASR output split into segments where each segment starts with
    # a capitalized word even though it is a continuation of the previous
    # sentence. The full paragraph has no sentence-ending punctuation between
    # the boundaries, so the merged transcript must not contain the false
    # capitals.
    asr_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Компрессор работает на",
            "words": [
                {"start": 0.0, "end": 0.3, "word": "Компрессор"},
                {"start": 0.3, "end": 0.6, "word": "работает"},
                {"start": 0.6, "end": 1.0, "word": "на"},
            ],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "Половину мощности когда",
            "words": [
                {"start": 1.0, "end": 1.3, "word": "Половину"},
                {"start": 1.3, "end": 1.6, "word": "мощности"},
                {"start": 1.6, "end": 2.0, "word": "когда"},
            ],
        },
        {
            "start": 2.0,
            "end": 3.0,
            "text": "Нагрузка падает.",
            "words": [
                {"start": 2.0, "end": 2.4, "word": "Нагрузка"},
                {"start": 2.4, "end": 3.0, "word": "падает."},
            ],
        },
        {
            "start": 3.0,
            "end": 4.0,
            "text": "Потом снова",
            "words": [
                {"start": 3.0, "end": 3.5, "word": "Потом"},
                {"start": 3.5, "end": 4.0, "word": "снова"},
            ],
        },
    ]
    diar_segments = [{"start": 0.0, "end": 5.0, "speaker": "S1"}]
    turns = build_speaker_turns(asr_segments, diar_segments, default_language=None)
    assert len(turns) == 1
    text = turns[0]["text"]
    # Turn start remains capitalized.
    assert text.startswith("Компрессор ")
    # Mid-sentence boundaries are lowercased.
    assert "половину" in text
    assert "когда" in text
    # After the period, capitalization resumes.
    assert "падает. Потом" in text
    # No false capitals remain in the middle of the paragraph.
    assert "Половину" not in text
    assert "Нагрузка" not in text


def test_build_speaker_turns_preserves_mid_segment_proper_noun():
    """A proper noun mid-segment is not lowercased when the preceding word is
    part of the same ASR segment (no false-boundary risk)."""
    asr_segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello John how are you",
            "words": [
                {"start": 0.0, "end": 0.3, "word": "hello"},
                {"start": 0.3, "end": 0.6, "word": "John"},
                {"start": 0.6, "end": 0.9, "word": "how"},
                {"start": 0.9, "end": 1.2, "word": "are"},
                {"start": 1.2, "end": 1.6, "word": "you"},
            ],
        },
    ]
    diar_segments = [{"start": 0.0, "end": 3.0, "speaker": "S1"}]
    turns = build_speaker_turns(asr_segments, diar_segments, default_language=None)
    assert len(turns) == 1
    # "John" must remain capitalized — the previous word is inside the same
    # ASR segment so this is not a segment-boundary merge.
    assert turns[0]["text"] == "hello John how are you"


def test_build_speaker_turns_preserves_acronym_leading_compound_at_boundary():
    """An acronym-leading compound at a real segment boundary is preserved."""
    asr_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "we chose an",
            "words": [
                {"start": 0.0, "end": 0.2, "word": "we"},
                {"start": 0.2, "end": 0.5, "word": "chose"},
                {"start": 0.5, "end": 0.8, "word": "an"},
            ],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "API-based approach",
            "words": [
                {"start": 1.0, "end": 1.4, "word": "API-based"},
                {"start": 1.4, "end": 1.8, "word": "approach"},
            ],
        },
    ]
    diar_segments = [{"start": 0.0, "end": 3.0, "speaker": "S1"}]
    turns = build_speaker_turns(asr_segments, diar_segments, default_language=None)
    assert len(turns) == 1
    assert turns[0]["text"] == "we chose an API-based approach"


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
