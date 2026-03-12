from __future__ import annotations

import json

import pytest

from lan_transcriber import llm_chunking


def test_split_transcript_for_llm_validates_and_handles_short_text() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        llm_chunking.plan_transcript_chunks("hello", max_chars=0, overlap_chars=0)

    with pytest.raises(ValueError, match="overlap_chars"):
        llm_chunking.plan_transcript_chunks("hello", max_chars=10, overlap_chars=-1)

    assert llm_chunking.split_transcript_for_llm("   ", max_chars=20, overlap_chars=0) == []
    assert llm_chunking.split_transcript_for_llm("single line", max_chars=20, overlap_chars=0) == [
        "single line"
    ]


def test_split_transcript_for_llm_is_deterministic_and_keeps_overlap_on_line_boundaries() -> None:
    text = "line one\nline two\nline three\nline four"

    chunks = llm_chunking.plan_transcript_chunks(text, max_chars=17, overlap_chars=12)
    repeated = llm_chunking.plan_transcript_chunks(text, max_chars=17, overlap_chars=12)

    assert [chunk.text for chunk in repeated] == [chunk.text for chunk in chunks]
    assert [chunk.base_text for chunk in chunks] == ["line one\nline two", "line three", "line four"]
    assert chunks[1].overlap_prefix == "line two"
    assert chunks[1].text.startswith("line two\nline three")
    assert chunks[0].plan_payload()["chunk_total"] == 3


def test_plan_transcript_chunks_splits_long_sentences_and_words() -> None:
    text = "This sentence is much too long for the current limit. Tiny.\n" + ("x" * 25)
    chunks = llm_chunking.plan_transcript_chunks(text, max_chars=10, overlap_chars=25)

    assert len(chunks) >= 5
    assert all(chunk.text for chunk in chunks)
    assert all(len(chunk.base_text) <= 10 for chunk in chunks)
    assert any(chunk.overlap_prefix for chunk in chunks[1:])


def test_internal_split_helpers_cover_blank_short_and_sentence_edges() -> None:
    assert llm_chunking._split_words_to_fit("", max_chars=5) == []
    assert llm_chunking._split_words_to_fit("tiny", max_chars=10) == ["tiny"]
    assert llm_chunking._split_words_to_fit("ok " + ("x" * 25), max_chars=10) == [
        "ok",
        "xxxxxxxxxx",
        "xxxxxxxxxx",
        "xxxxx",
    ]
    assert llm_chunking._split_unit_to_fit("   ", max_chars=10) == []

    flushed = llm_chunking._split_unit_to_fit(
        "short. this sentence is definitely longer than ten chars. next.",
        max_chars=10,
    )
    assert flushed[0] == "short."
    assert flushed[-1] == "next."

    emptied = llm_chunking._split_unit_to_fit(
        "short. this sentence is definitely longer than ten chars.",
        max_chars=10,
    )
    assert emptied[0] == "short."

    overflow = llm_chunking._split_unit_to_fit("one. two three. four.", max_chars=10)
    assert overflow == ["one.", "two three.", "four."]


def test_tail_text_and_plan_transcript_chunks_empty_units_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert llm_chunking._tail_text("", max_chars=5) == ""
    assert llm_chunking._tail_text("hello\n   ", max_chars=5) == "o\n   "
    assert llm_chunking._tail_text("prefix-without-newline", max_chars=5) == "wline"

    monkeypatch.setattr(llm_chunking, "_split_unit_to_fit", lambda *_args, **_kwargs: [])
    assert llm_chunking.plan_transcript_chunks("first line\n\nsecond line", max_chars=10, overlap_chars=0) == []


def test_build_compact_transcript_merges_adjacent_turns_and_drops_noise() -> None:
    compact = llm_chunking.build_compact_transcript(
        [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.8, "text": "  hello   there  "},
            {"speaker": "SPEAKER_00", "start": 1.2, "end": 1.9, "text": "again"},
            {"speaker": "SPEAKER_00", "start": 3.5, "end": 4.0, "text": "Later update"},
            {"speaker": "SPEAKER_01", "start": 4.1, "end": 4.3, "text": "..."},
            {"speaker": "SPEAKER_01", "start": 4.5, "end": 5.0, "text": "   replied clearly   "},
        ],
        aliases={"SPEAKER_00": "Alex", "SPEAKER_01": "Priya"},
    )

    assert compact.source_turn_count == 5
    assert compact.compact_turn_count == 3
    assert compact.source_chars > compact.compact_chars
    assert compact.text.splitlines() == [
        "Alex: hello there again",
        "Alex: Later update",
        "Priya: replied clearly",
    ]
    assert compact.artifact_payload()["speaker_mapping"] == [
        {
            "compact_label": "Alex",
            "original_speaker": "SPEAKER_00",
            "display_name": "Alex",
        },
        {
            "compact_label": "Priya",
            "original_speaker": "SPEAKER_01",
            "display_name": "Priya",
        },
    ]
    assert compact.prompt_speaker_mapping() == [
        {"label": "Alex", "speaker": "Alex", "original_speaker": "SPEAKER_00"},
        {"label": "Priya", "speaker": "Priya", "original_speaker": "SPEAKER_01"},
    ]


def test_build_compact_transcript_uses_stable_reversible_speaker_mapping() -> None:
    compact = llm_chunking.build_compact_transcript(
        [
            {"speaker": "SPEAKER_05", "start": 0.0, "end": 0.8, "text": "First update"},
            {"speaker": "SPEAKER_99", "start": 1.0, "end": 1.4, "text": "Second update"},
            {"speaker": "SPEAKER_05", "start": 2.0, "end": 2.4, "text": "Third update"},
        ]
    )

    assert compact.text.splitlines() == [
        "S1: First update",
        "S2: Second update",
        "S1: Third update",
    ]
    assert compact.prompt_speaker_mapping() == [
        {"label": "S1", "speaker": "SPEAKER_05"},
        {"label": "S2", "speaker": "SPEAKER_99"},
    ]
    assert compact.artifact_payload()["speaker_mapping"] == [
        {
            "compact_label": "S1",
            "original_speaker": "SPEAKER_05",
            "display_name": "SPEAKER_05",
        },
        {
            "compact_label": "S2",
            "original_speaker": "SPEAKER_99",
            "display_name": "SPEAKER_99",
        },
    ]


def test_build_compact_transcript_ignores_noise_only_speakers_when_assigning_labels() -> None:
    compact = llm_chunking.build_compact_transcript(
        [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 0.2, "text": "..."},
            {"speaker": "SPEAKER_02", "start": 0.3, "end": 1.0, "text": "Real update"},
            {"speaker": "SPEAKER_03", "start": 1.1, "end": 1.8, "text": "Second speaker"},
        ]
    )

    assert compact.text.splitlines() == [
        "S1: Real update",
        "S2: Second speaker",
    ]
    assert compact.prompt_speaker_mapping() == [
        {"label": "S1", "speaker": "SPEAKER_02"},
        {"label": "S2", "speaker": "SPEAKER_03"},
    ]


def test_build_compact_transcript_rejects_noise_only_rows() -> None:
    with pytest.raises(ValueError, match="produced no usable content"):
        llm_chunking.build_compact_transcript(
            [
                {"speaker": "S1", "start": 0.0, "end": 0.1, "text": "..."},
                {"speaker": "S2", "start": 0.2, "end": 0.3, "text": "???!"},
            ]
        )


def test_compact_helpers_cover_edge_cases_and_validation() -> None:
    assert not llm_chunking._is_concise_speaker_label("")
    assert not llm_chunking._is_concise_speaker_label("SPEAKER_02")
    assert not llm_chunking._is_concise_speaker_label("This Alias Is Too Long")
    assert not llm_chunking._is_concise_speaker_label("a b c")

    with pytest.raises(ValueError, match="produced no usable content"):
        llm_chunking.build_compact_transcript([{"speaker": "S1", "start": 0.0, "end": 0.1, "text": "   "}])

    compact = llm_chunking.build_compact_transcript(
        [{"speaker": "Alex", "start": 0.0, "end": 1.0, "text": "hello"}]
    )
    assert compact.prompt_speaker_mapping() == []

    split = llm_chunking._split_compact_unit_to_fit(  # noqa: SLF001
        llm_chunking.CompactTranscriptUnit(
            speaker_label="LONG",
            original_speaker="LONG",
            display_name="LONG",
            start_seconds=0.0,
            end_seconds=1.0,
            text="alpha beta",
            source_turn_count=1,
        ),
        max_chars=3,
    )
    assert split
    assert all(len(item.text) <= 3 for item in split)

    prefixed = llm_chunking._split_compact_unit_to_fit(  # noqa: SLF001
        llm_chunking.CompactTranscriptUnit(
            speaker_label="S1",
            original_speaker="S1",
            display_name="S1",
            start_seconds=0.0,
            end_seconds=1.0,
            text="alpha beta gamma",
            source_turn_count=1,
        ),
        max_chars=8,
    )
    assert prefixed
    assert all(item.text.startswith("S1: ") for item in prefixed)
    assert all(len(item.text) <= 8 for item in prefixed)

    empty_compact = llm_chunking.CompactTranscript(
        text="   ",
        source_chars=0,
        compact_chars=0,
        merge_gap_seconds=llm_chunking.DEFAULT_COMPACT_MERGE_GAP_SECONDS,
        source_turn_count=0,
        compact_turn_count=0,
        speaker_mapping=(),
        units=(),
    )
    assert llm_chunking.plan_compact_transcript_chunks(empty_compact, max_chars=10, overlap_chars=0) == []

    with pytest.raises(ValueError, match="max_chars"):
        llm_chunking.plan_compact_transcript_chunks(compact, max_chars=0, overlap_chars=0)
    with pytest.raises(ValueError, match="overlap_chars"):
        llm_chunking.plan_compact_transcript_chunks(compact, max_chars=10, overlap_chars=-1)


def test_build_compact_transcript_flushes_blank_current_and_compact_plan_empty_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_chunking, "_has_substantive_content", lambda _text: True)
    monkeypatch.setattr(
        llm_chunking,
        "_normalise_source_turns",
        lambda *_a, **_k: [
            {
                "speaker": "S1",
                "display_name": "S1",
                "start": 0.0,
                "end": 1.0,
                "text": "   ",
            }
        ],
    )
    with pytest.raises(ValueError, match="produced no usable content"):
        llm_chunking.build_compact_transcript([{"speaker": "S1", "text": "ignored"}])

    compact = llm_chunking.CompactTranscript(
        text="S1: hello",
        source_chars=9,
        compact_chars=9,
        merge_gap_seconds=llm_chunking.DEFAULT_COMPACT_MERGE_GAP_SECONDS,
        source_turn_count=1,
        compact_turn_count=1,
        speaker_mapping=(),
        units=(
            llm_chunking.CompactTranscriptUnit(
                speaker_label="S1",
                original_speaker="S1",
                display_name="S1",
                start_seconds=0.0,
                end_seconds=1.0,
                text="hello",
                source_turn_count=1,
            ),
        ),
    )
    monkeypatch.setattr(llm_chunking, "_split_compact_unit_to_fit", lambda *_a, **_k: [])
    assert llm_chunking.plan_compact_transcript_chunks(compact, max_chars=10, overlap_chars=0) == []


def test_plan_compact_transcript_chunks_records_effective_sizes_and_time_ranges() -> None:
    compact = llm_chunking.build_compact_transcript(
        [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "alpha beta gamma delta"},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0, "text": "epsilon zeta eta theta"},
            {"speaker": "SPEAKER_00", "start": 4.0, "end": 5.0, "text": "iota kappa lambda mu"},
        ]
    )

    chunks = llm_chunking.plan_compact_transcript_chunks(compact, max_chars=28, overlap_chars=8)

    assert len(chunks) >= 2
    assert all("[" not in chunk.text for chunk in chunks)
    assert chunks[0].plan_payload()["time_range"] == {"start_seconds": 0.0, "end_seconds": 1.0}
    assert chunks[1].plan_payload()["effective_chars"] >= chunks[1].plan_payload()["base_chars"]
    assert chunks[1].plan_payload()["overlap_chars"] == len(chunks[1].overlap_prefix)


def test_build_chunk_prompt_and_parse_chunk_extract() -> None:
    chunk = llm_chunking.TranscriptChunk(
        index=2,
        total=4,
        text="Alex: review blockers",
        base_text="Alex: review blockers",
        start_seconds=12.0,
        end_seconds=18.0,
    )
    system_prompt, user_prompt = llm_chunking.build_chunk_prompt(
        chunk,
        target_summary_language="es",
        calendar_title="Roadmap Review",
        calendar_attendees=["Alex", "Priya"],
        speaker_mapping=[{"label": "S1", "speaker": "Alex"}],
    )
    prompt_payload = json.loads(user_prompt)

    assert "strict JSON" in system_prompt
    assert prompt_payload["chunk"] == {
        "index": 2,
        "total": 4,
        "id": "2",
        "time_range": {"start_seconds": 12.0, "end_seconds": 18.0},
    }
    assert prompt_payload["calendar"]["title"] == "Roadmap Review"
    assert prompt_payload["calendar"]["attendees"] == ["Alex", "Priya"]
    assert prompt_payload["speaker_mapping"] == [{"label": "S1", "speaker": "Alex"}]

    parsed = llm_chunking.parse_chunk_extract(
        """```json
        {
          "topic": "Weekly sync",
          "summary": "- first\\n- second",
          "decisions": ["Ship Friday"],
          "action_items": [{"task": "Send notes", "owner": "Alex", "confidence": 1.5}],
          "emotional_summary": "Focused",
          "questions": {
            "types": {"status": 1},
            "extracted": ["Is QA done?"]
          }
        }
        ```"""
    )

    assert parsed["topic_candidates"] == ["Weekly sync"]
    assert parsed["summary_bullets"] == ["first", "second"]
    assert parsed["decisions"] == ["Ship Friday"]
    assert parsed["action_items"] == [
        {"task": "Send notes", "owner": "Alex", "deadline": None, "confidence": 1.0}
    ]
    assert parsed["emotional_cues"] == ["Focused"]
    assert parsed["questions"]["total_count"] == 1
    assert parsed["questions"]["types"]["status"] == 1
    assert parsed["questions"]["extracted"] == ["Is QA done?"]


def test_parse_chunk_extract_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="json_object_not_found"):
        llm_chunking.parse_chunk_extract("plain text only")


def test_internal_parse_and_dedupe_helpers_cover_edge_shapes() -> None:
    assert llm_chunking._extract_json_dict("") is None
    assert llm_chunking._extract_json_dict('{"topic":"A"}') == {"topic": "A"}
    assert llm_chunking._extract_json_dict("[1]") is None

    actions = llm_chunking._normalise_action_items(
        ["", "Task"] + [{"task": f"T{index}"} for index in range(40)] + [{"task": ""}]
    )
    assert actions[0]["task"] == "Task"
    assert len(actions) == 30

    assert llm_chunking._normalise_questions("not-a-dict") == {
        "total_count": 0,
        "types": {
            "open": 0,
            "yes_no": 0,
            "clarification": 0,
            "status": 0,
            "decision_seeking": 0,
        },
        "extracted": [],
    }

    assert llm_chunking._dedupe_text_items(["", "A", "a", "B", "C"], max_items=2) == ["A", "B"]

    deduped_actions = llm_chunking._dedupe_action_items(
        [{"task": ""}] + [{"task": f"T{index}"} for index in range(40)]
    )
    assert deduped_actions[0]["task"] == "T0"
    assert len(deduped_actions) == 30


def test_merge_chunk_results_and_build_merge_prompt_dedupe_inputs() -> None:
    merged = llm_chunking.merge_chunk_results(
        [
            {
                "topic_candidates": ["Weekly sync"],
                "summary_bullets": ["Reviewed blockers"],
                "decisions": ["Ship Friday"],
                "action_items": [{"task": "Send notes", "owner": "Alex"}],
                "emotional_cues": ["Focused"],
                "questions": {
                    "total_count": 1,
                    "types": {"status": 1},
                    "extracted": ["Is QA done?"],
                },
            },
            {
                "chunk_index": 2,
                "chunk_total": 2,
                "topic_candidates": ["Weekly sync"],
                "summary_bullets": ["Reviewed blockers", "Confirmed scope"],
                "decisions": ["Ship Friday"],
                "action_items": [{"task": "Send notes", "owner": "Alex"}],
                "emotional_cues": ["Focused", "Optimistic"],
                "questions": {
                    "total_count": 1,
                    "types": {"status": 1},
                    "extracted": ["Is QA done?"],
                },
            },
        ]
    )

    assert merged["chunk_count"] == 2
    assert merged["topic_candidates"] == ["Weekly sync"]
    assert merged["summary_bullets"] == ["Reviewed blockers", "Confirmed scope"]
    assert merged["decisions"] == ["Ship Friday"]
    assert merged["action_items"] == [
        {"task": "Send notes", "owner": "Alex", "deadline": None, "confidence": 0.5}
    ]
    assert merged["emotional_cues"] == ["Focused", "Optimistic"]
    assert merged["questions"]["total_count_hint"] == 2
    assert merged["questions"]["type_hints"]["status"] == 2
    assert merged["chunks"][0]["chunk_index"] == 1
    assert merged["chunks"][1]["chunk_total"] == 2

    system_prompt, user_prompt = llm_chunking.build_merge_prompt(
        merged,
        target_summary_language="en",
        calendar_title="Roadmap Review",
        calendar_attendees=["Alex"],
    )
    prompt_payload = json.loads(user_prompt)

    assert "overlapping transcript chunks" in system_prompt
    assert prompt_payload["calendar"]["title"] == "Roadmap Review"
    assert prompt_payload["merge_input"]["chunk_count"] == 2
    assert "overlap" in prompt_payload["overlap_warning"].lower()
