from __future__ import annotations

from pathlib import Path

import pytest

import lan_app.conversation_metrics as conversation_metrics
import lan_app.routing as routing
from lan_app.config import AppSettings
from lan_app.db import create_recording, init_db


def _cfg(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )


def test_conversation_normalise_turns_merge_and_interruptions_edge_cases():
    normalized = conversation_metrics._normalise_turns(  # noqa: SLF001
        [
            "not-a-dict",
            {"text": "   "},
            {"start": 5.0, "end": 4.0, "speaker": " ", "text": "Hello", "language": "ENG"},
            {"start": 1.0, "end": 1.2, "speaker": "S2", "text": "Hi"},
        ]
    )
    assert [row["speaker"] for row in normalized] == ["S2", "S1"]
    assert normalized[1]["start"] == 5.0
    assert normalized[1]["end"] == 5.0
    assert normalized[1]["language"] == "en"

    merged = conversation_metrics.merge_speaker_turns(
        [
            {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "one"},
            {"start": 1.1, "end": 2.0, "speaker": "S1", "text": "two", "language": "de"},
        ],
        gap_threshold=0.2,
    )
    assert merged[0]["language"] == "de"
    assert conversation_metrics.merge_speaker_turns([], gap_threshold=0.0) == []

    interruption_stats = conversation_metrics.count_interruptions(
        [{"start": 1.0, "end": 1.0, "speaker": "S1", "text": "zero-duration"}],
        overlap_threshold=0.1,
    )
    assert interruption_stats["total"] == 0
    assert interruption_stats["done"]["S1"] == 0


def test_conversation_fallback_from_transcript_segments_text_and_empty():
    from_segments = conversation_metrics._fallback_speaker_turns_from_transcript(  # noqa: SLF001
        {
            "segments": [
                "skip",
                {"text": " "},
                {"start": 3.0, "end": 2.0, "text": "Segment from transcript", "language": "en-US"},
                {"start": 1.0, "end": 2.0, "text": "Second segment"},
            ]
        }
    )
    assert from_segments[0] == {
        "start": 3.0,
        "end": 3.0,
        "speaker": "S1",
        "text": "Segment from transcript",
        "language": "en",
    }
    assert from_segments[1] == {
        "start": 1.0,
        "end": 2.0,
        "speaker": "S1",
        "text": "Second segment",
    }

    from_empty_segments = conversation_metrics._fallback_speaker_turns_from_transcript(  # noqa: SLF001
        {"segments": [{"text": "   "}, "skip"], "text": "Fallback text from empty segments"}
    )
    assert from_empty_segments == [
        {"start": 0.0, "end": 0.0, "speaker": "S1", "text": "Fallback text from empty segments"}
    ]

    from_text = conversation_metrics._fallback_speaker_turns_from_transcript({"text": "Fallback text"})  # noqa: SLF001
    assert from_text == [{"start": 0.0, "end": 0.0, "speaker": "S1", "text": "Fallback text"}]
    assert conversation_metrics._fallback_speaker_turns_from_transcript({"text": "   "}) == []  # noqa: SLF001


def test_conversation_internal_normalizers_and_allocation():
    assert conversation_metrics._normalise_text_items("alpha\nbeta\n", max_items=5) == [  # noqa: SLF001
        "alpha",
        "beta",
    ]
    assert conversation_metrics._normalise_text_items(  # noqa: SLF001
        ["- alpha", " ", "- ", "beta", "gamma"],
        max_items=2,
    ) == ["alpha", "beta"]

    actions = conversation_metrics._normalise_action_items(  # noqa: SLF001
        [
            "skip",
            {},
            {"task": "   "},
            {"task": "Ship", "owner": "", "deadline": " "},
            {"task": "Docs", "owner": "Alex", "deadline": "2026-03-01"},
        ]
    )
    assert actions == [
        {"task": "Ship", "owner": None, "deadline": None},
        {"task": "Docs", "owner": "Alex", "deadline": "2026-03-01"},
    ]

    question_types = conversation_metrics._normalise_question_types({"open": "bad", "yes_no": None})  # noqa: SLF001
    assert question_types["open"] == 0
    assert question_types["yes_no"] == 0

    assert conversation_metrics._allocate_by_weight(0, {"a": 1}) == {"a": 0}  # noqa: SLF001
    assert conversation_metrics._allocate_by_weight(3, {"a": 0, "b": 0}) == {"a": 2, "b": 1}  # noqa: SLF001
    assert conversation_metrics._allocate_by_weight(5, {"a": 2, "b": 1}) == {"a": 3, "b": 2}  # noqa: SLF001


def test_conversation_question_assignment_llm_fallback_paths():
    counts, total_questions, source, question_types = conversation_metrics._question_counts_by_speaker(  # noqa: SLF001
        merged_turns=[
            {"speaker": "S1", "text": "Status update?"},
            {"speaker": "S2", "text": "Everything is stable"},
        ],
        summary_payload={
            "questions": {
                "total_count": "invalid",
                "types": {"open": "invalid"},
                "extracted": ["??", ":", "unmatched keyword phrase"],
            }
        },
    )
    assert source == "llm"
    assert total_questions == 3
    assert sum(counts.values()) == 3
    assert question_types["open"] == 0


def test_conversation_infer_role_hint_branches():
    assert (
        conversation_metrics._infer_role_hint(  # noqa: SLF001
            airtime_share=0.1,
            turns=1,
            questions_count=0,
            decision_task_signals=0,
            avg_turn_duration_sec=1.0,
            terminology_density=0.0,
            avg_turns=3.0,
            avg_turn_duration=2.0,
            meeting_questions=3,
        )
        == "Passive"
    )
    assert (
        conversation_metrics._infer_role_hint(  # noqa: SLF001
            airtime_share=0.2,
            turns=6,
            questions_count=4,
            decision_task_signals=0,
            avg_turn_duration_sec=2.0,
            terminology_density=0.01,
            avg_turns=4.0,
            avg_turn_duration=2.0,
            meeting_questions=10,
        )
        == "Facilitator"
    )
    assert (
        conversation_metrics._infer_role_hint(  # noqa: SLF001
            airtime_share=0.2,
            turns=2,
            questions_count=0,
            decision_task_signals=0,
            avg_turn_duration_sec=6.0,
            terminology_density=0.1,
            avg_turns=3.0,
            avg_turn_duration=3.0,
            meeting_questions=0,
        )
        == "Expert"
    )


def test_conversation_build_uses_transcript_fallback_with_zero_airtime():
    payload = conversation_metrics.build_conversation_metrics(
        transcript_payload={"text": "Only transcript text fallback"},
        summary_payload={},
        speaker_turns=[],
    )
    assert payload["meeting"]["participants_count"] == 1
    assert payload["meeting"]["total_speech_time_seconds"] == 0.0
    assert payload["participants"][0]["airtime_share"] == 0.0


def test_conversation_json_loaders_handle_missing_invalid_and_wrong_types(tmp_path: Path):
    assert conversation_metrics._load_json_dict(tmp_path / "missing.json") == {}  # noqa: SLF001
    assert conversation_metrics._load_json_list(tmp_path / "missing-list.json") == []  # noqa: SLF001

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert conversation_metrics._load_json_dict(bad_json) == {}  # noqa: SLF001
    assert conversation_metrics._load_json_list(bad_json) == []  # noqa: SLF001

    wrong_dict = tmp_path / "wrong-dict.json"
    wrong_dict.write_text("[]", encoding="utf-8")
    assert conversation_metrics._load_json_dict(wrong_dict) == {}  # noqa: SLF001

    wrong_list = tmp_path / "wrong-list.json"
    wrong_list.write_text("{}", encoding="utf-8")
    assert conversation_metrics._load_json_list(wrong_list) == []  # noqa: SLF001


def test_routing_refresh_raises_for_missing_recording(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    with pytest.raises(KeyError):
        routing.refresh_recording_routing("missing-recording", settings=cfg)


def test_routing_refresh_skips_invalid_rows_and_keeps_status_without_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _cfg(tmp_path)
    captured: dict[str, object] = {}
    project_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        routing,
        "get_recording",
        lambda *_args, **_kwargs: {
            "status": "Queued",
            "project_id": None,
            "project_assignment_source": None,
        },
    )
    monkeypatch.setattr(
        routing,
        "list_projects",
        lambda **_kwargs: [{"id": "bad"}, {"id": 1, "name": "Alpha"}],
    )
    monkeypatch.setattr(
        routing,
        "list_project_keyword_weights",
        lambda **_kwargs: [
            {"project_id": "x", "keyword": "cal:alpha", "weight": 1},
            {"project_id": 1, "keyword": " ", "weight": 1},
            {"project_id": 1, "keyword": "cal:alpha", "weight": 3},
        ],
    )
    monkeypatch.setattr(
        routing,
        "_build_routing_signals",
        lambda *_args, **_kwargs: routing.RoutingSignals([], [], [], []),
    )
    monkeypatch.setattr(
        routing,
        "set_recording_routing_suggestion",
        lambda recording_id, **kwargs: captured.update({"recording_id": recording_id, **kwargs}),
    )
    monkeypatch.setattr(routing, "count_routing_training_examples", lambda **_kwargs: 4)
    monkeypatch.setattr(
        routing,
        "set_recording_project",
        lambda *args, **kwargs: project_updates.append((args, kwargs)),
    )

    decision = routing.refresh_recording_routing(
        "rec-routing-edge-1",
        settings=cfg,
        apply_workflow=False,
    )

    assert decision["suggested_project_id"] is None
    assert decision["status_after_routing"] == "Queued"
    assert decision["training_examples_total"] == 4
    assert project_updates == []
    assert captured["recording_id"] == "rec-routing-edge-1"


def test_routing_train_raises_for_missing_recording_and_project(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)

    with pytest.raises(KeyError):
        routing.train_routing_from_manual_selection("missing-recording", 123, settings=cfg)

    create_recording(
        "rec-routing-train-missing-project",
        source="drive",
        source_filename="meeting.mp3",
        settings=cfg,
    )
    with pytest.raises(KeyError):
        routing.train_routing_from_manual_selection(
            "rec-routing-train-missing-project",
            999,
            settings=cfg,
        )


def test_routing_private_helpers_cover_scoring_rationale_signals_and_calendar_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _cfg(tmp_path)
    original_selected_calendar_candidate = routing._selected_calendar_candidate  # noqa: SLF001

    zero_score = routing._project_score(  # noqa: SLF001
        project={"name": "Alpha"},
        signals=routing.RoutingSignals([], [], [], []),
        learned_weights={},
    )
    assert zero_score["score"] == 0.0

    fallback_component = routing._component_score(  # noqa: SLF001
        tokens=["roadmap"],
        keyword_prefix="cal",
        learned_weights={},
        fallback_tokens={"roadmap"},
        fallback_score=0.4,
    )
    assert fallback_component["matched_count"] == 1
    assert fallback_component["score"] == 0.4
    fallback_zero_component = routing._component_score(  # noqa: SLF001
        tokens=["roadmap"],
        keyword_prefix="cal",
        learned_weights={},
        fallback_tokens={"roadmap"},
        fallback_score=0.0,
    )
    assert fallback_zero_component["matched_count"] == 0

    rationale = routing._build_rationale(  # noqa: SLF001
        top={
            "project_name": "Roadmap",
            "score": 0.5,
            "components": {
                "calendar_subject": {},
                "participants": {},
                "llm_tags": {},
                "voice_profiles": {},
            },
        },
        runner_up=None,
        confidence=0.2,
        threshold=0.5,
        signals=routing.RoutingSignals([], [], [], []),
    )
    assert all("Runner-up:" not in row for row in rationale)
    assert any("No calendar subject tokens" in row for row in rationale)
    assert any("No LLM tags/keywords" in row for row in rationale)
    assert any("No assigned voice profiles" in row for row in rationale)
    rationale_with_voice_profiles = routing._build_rationale(  # noqa: SLF001
        top={
            "project_name": "Roadmap",
            "score": 0.5,
            "components": {
                "calendar_subject": {},
                "participants": {},
                "llm_tags": {},
                "voice_profiles": {},
            },
        },
        runner_up=None,
        confidence=0.2,
        threshold=0.5,
        signals=routing.RoutingSignals([], [], [], [7]),
    )
    assert all("No assigned voice profiles" not in row for row in rationale_with_voice_profiles)

    monkeypatch.setattr(
        routing,
        "_selected_calendar_candidate",
        lambda *_args, **_kwargs: {
            "subject": "Roadmap sync",
            "organizer": "Alex",
            "attendees": ["Priya"],
        },
    )
    monkeypatch.setattr(routing, "_llm_keywords", lambda *_args, **_kwargs: {"planning"})
    monkeypatch.setattr(
        routing,
        "list_speaker_assignments",
        lambda *_args, **_kwargs: [
            {"voice_profile_id": None},
            {"voice_profile_id": "bad"},
            {"voice_profile_id": "7"},
        ],
    )
    signals = routing._build_routing_signals("rec-routing-signals-1", settings=cfg)  # noqa: SLF001
    assert signals.voice_profile_ids == [7]
    assert "roadmap" in signals.calendar_subject_tokens

    monkeypatch.setattr(
        routing,
        "_selected_calendar_candidate",
        original_selected_calendar_candidate,
    )

    monkeypatch.setattr(
        routing,
        "get_calendar_match",
        lambda *_args, **_kwargs: {"selected_event_id": "evt-1", "candidates_json": {}},
    )
    assert routing._selected_calendar_candidate("rec-routing-cal-1", settings=cfg) == {}  # noqa: SLF001

    monkeypatch.setattr(
        routing,
        "get_calendar_match",
        lambda *_args, **_kwargs: {
            "selected_event_id": "evt-1",
            "candidates_json": ["not-a-dict", {"event_id": "evt-2", "subject": "Other"}],
        },
    )
    assert routing._selected_calendar_candidate("rec-routing-cal-2", settings=cfg) == {}  # noqa: SLF001


def test_routing_text_json_and_token_helpers(tmp_path: Path):
    assert routing._action_item_keywords("not-a-list") == []  # noqa: SLF001
    keywords = routing._action_item_keywords(  # noqa: SLF001
        [
            "skip",
            {},
            {"task": "Ship", "owner": "Alex", "deadline": "2026-03-01"},
        ]
    )
    assert keywords == ["Ship", "Alex", "2026-03-01"]

    assert routing._text_list([" ", " Roadmap ", 7]) == ["Roadmap", "7"]  # noqa: SLF001

    bad_json = tmp_path / "bad-routing.json"
    bad_json.write_text("{", encoding="utf-8")
    assert routing._load_json_dict(bad_json) == {}  # noqa: SLF001

    wrong_type_json = tmp_path / "wrong-type-routing.json"
    wrong_type_json.write_text("[]", encoding="utf-8")
    assert routing._load_json_dict(wrong_type_json) == {}  # noqa: SLF001

    tokens = routing._tokenize("a roadmap and b")  # noqa: SLF001
    assert "roadmap" in tokens
    assert "a" not in tokens
