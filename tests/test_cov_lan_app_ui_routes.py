from __future__ import annotations

from datetime import timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from fastapi.testclient import TestClient
import pytest

from lan_app import api, ui_routes
from lan_app.calendar.service import CalendarSyncError
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FAILED,
    RECORDING_STATUS_READY,
)
from lan_app.db import (
    create_calendar_source,
    create_job,
    create_recording,
    init_db,
)
from lan_app.jobs import DuplicateRecordingJobError


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AppSettings, TestClient]:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    return cfg, TestClient(api.app, follow_redirects=False)


def _seed_recording(cfg: AppSettings, recording_id: str = "rec-cov-ui-routes-1") -> str:
    create_recording(
        recording_id,
        source="upload",
        source_filename=f"{recording_id}.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    return recording_id


def test_language_and_json_helper_edge_paths() -> None:
    class _NoStrip(str):
        def strip(self) -> "_NoStrip":  # type: ignore[override]
            return self

    class _BulletOnly:
        def __str__(self) -> _NoStrip:
            return _NoStrip("- ")

    assert ui_routes._language_display_name("unknown") == "Unknown"  # noqa: SLF001

    with pytest.raises(ValueError, match="target must be a language code"):
        ui_routes._parse_language_form_value("??", field_name="target")  # noqa: SLF001

    warning = ui_routes._recording_recovery_warning(  # noqa: SLF001
        [{"error": "stuck job recovered", "finished_at": "2026-01-02T03:04:05Z"}]
    )
    assert warning and "2026-01-02 04:04:05 CET" in warning

    warning_no_ts = ui_routes._recording_recovery_warning(  # noqa: SLF001
        [{"error": "stuck job recovered"}]
    )
    assert warning_no_ts and "recovered from a stuck job" in warning_no_ts

    assert ui_routes._normalise_text_items(  # noqa: SLF001
        ["", "  ", _BulletOnly(), "- first", "second", "third"],
        max_items=2,
    ) == [
        "first",
        "second",
    ]


def test_display_helpers_cover_timezone_duration_and_prepare_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)

    assert ui_routes._pipeline_stage_label("precheck") == "Sanitize & Precheck"  # noqa: SLF001
    assert ui_routes._pipeline_stage_label("llm_chunk_2_of_5") == "LLM Chunk 2 of 5"  # noqa: SLF001
    assert ui_routes._pipeline_stage_label("llm_chunk_bad") == "Llm Chunk Bad"  # noqa: SLF001
    assert ui_routes._pipeline_stage_label("custom_stage") == "Custom Stage"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(None) == "—"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(0) == "—"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(2.0) == "2s"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(2.345) == "2.35s"  # noqa: SLF001
    assert ui_routes._format_local_timestamp("") == "—"  # noqa: SLF001
    assert ui_routes._format_local_timestamp("bad-timestamp") == "bad-timestamp"  # noqa: SLF001
    assert "CET" in ui_routes._format_local_timestamp("2026-01-10T10:00:00Z")  # noqa: SLF001

    observed_update: dict[str, object] = {}
    monkeypatch.setattr(
        ui_routes,
        "_recording_audio_candidates",
        lambda *_a, **_k: [Path("/tmp/fake.wav")],
    )
    monkeypatch.setattr(ui_routes, "_probe_duration_seconds", lambda *_a, **_k: 3.5)
    monkeypatch.setattr(
        ui_routes,
        "set_recording_duration",
        lambda recording_id, duration_sec, *, settings=None: observed_update.update(
            {"recording_id": recording_id, "duration_sec": duration_sec}
        )
        or True,
    )

    prepared = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {
            "id": "rec-helper-1",
            "duration_sec": None,
            "captured_at": "2026-01-10T10:00:00Z",
            "created_at": "",
            "updated_at": "bad",
            "pipeline_updated_at": "2026-01-10T10:05:00Z",
            "review_reason_text": "  Needs a closer look.  ",
        },
        settings=cfg,
    )
    assert observed_update == {"recording_id": "rec-helper-1", "duration_sec": 3.5}
    assert prepared["duration_display"] == "3.50s"
    assert prepared["captured_at_display"].endswith("CET")
    assert prepared["created_at_display"] == "—"
    assert prepared["updated_at_display"] == "bad"
    assert prepared["pipeline_updated_at_display"].endswith("CET")
    assert prepared["review_reason_text_display"] == "Needs a closer look."

    observed_update.clear()
    prepared_existing_duration = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {
            "id": "rec-helper-2",
            "duration_sec": 1.0,
            "captured_at": None,
            "created_at": None,
            "updated_at": None,
            "pipeline_updated_at": None,
            "review_reason_text": None,
        },
        settings=cfg,
    )
    assert observed_update == {}
    assert prepared_existing_duration["duration_display"] == "1s"

    monkeypatch.setattr(
        ui_routes,
        "ZoneInfo",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ui_routes.ZoneInfoNotFoundError("missing tzdata")
        ),
        raising=False,
    )
    assert ui_routes._display_timezone() is timezone.utc  # noqa: SLF001


def test_load_json_and_chunk_helpers_cover_error_paths(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert ui_routes._load_json_dict(broken) == {}  # noqa: SLF001
    assert ui_routes._load_json_list(broken) == []  # noqa: SLF001

    as_list = tmp_path / "list.json"
    as_list.write_text("[1,2]", encoding="utf-8")
    assert ui_routes._load_json_dict(as_list) == {}  # noqa: SLF001
    as_dict = tmp_path / "dict.json"
    as_dict.write_text('{"k": 1}', encoding="utf-8")
    assert ui_routes._load_json_list(as_dict) == []  # noqa: SLF001

    assert ui_routes._chunk_text_for_turns("   ") == []  # noqa: SLF001
    # Long single word forces split path and leaves no trailing buffered chunk.
    chunks = ui_routes._chunk_text_for_turns("abcdefghij", chunk_size=4)  # noqa: SLF001
    assert chunks == ["abcd", "efgh", "ij"]


def test_summary_context_and_metrics_merge_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-summary-helpers"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "summary": "- bullet-1\n\nbullet-2",
                "action_items": [
                    "bad-row",
                    {},
                    {"task": ""},
                    {"task": "task-1", "owner": " ", "deadline": " ", "confidence": "bad"},
                ],
                "questions": {
                    "types": {"open": "NaN"},
                    "total_count": "NaN",
                    "extracted": "Q1\n\nQ2",
                },
            }
        ),
        encoding="utf-8",
    )
    summary = ui_routes._summary_context(recording_id, cfg)  # noqa: SLF001
    assert summary["summary_bullets"][:2] == ["bullet-1", "bullet-2"]
    assert summary["action_items"][0]["confidence"] == 0.5
    assert summary["questions"]["total_count"] == 2

    (derived / "summary.json").write_text(
        json.dumps({"questions": {"types": "bad", "total_count": "x", "extracted": []}}),
        encoding="utf-8",
    )
    summary_no_types = ui_routes._summary_context(recording_id, cfg)  # noqa: SLF001
    assert summary_no_types["questions"]["types"]["open"] == 0

    (derived / "metrics.json").write_text(
        json.dumps(
            {
                "meeting": {
                    "total_interruptions": "7",
                    "total_questions": "bad",
                    "actionability_ratio": "bad",
                },
                "participants": [
                    {"speaker": "S1", "airtime_seconds": "10.2", "turns": "3"},
                    {"speaker": "", "airtime_seconds": 12},
                    {"speaker": "S2", "airtime_seconds": "6.4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ui_routes, "get_meeting_metrics", lambda *_a, **_k: {"json": "bad"})
    monkeypatch.setattr(
        ui_routes,
        "list_participant_metrics",
        lambda *_a, **_k: [
            {"json": "bad", "diar_speaker_label": "Sbad"},
            {"json": {"speaker": ""}, "diar_speaker_label": ""},
            {"json": {"speaker": "S1", "questions_count": "x"}, "diar_speaker_label": "S1"},
        ],
    )
    metrics = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert metrics["meeting"]["total_interruptions"] == 7
    assert metrics["meeting"]["total_questions"] == 0
    assert [row["speaker"] for row in metrics["participants"]] == ["S1", "S2"]

    # Cover additional artifact branches: no meeting dict and participants payload not list.
    (derived / "metrics.json").write_text(
        json.dumps({"meeting": "bad", "participants": "bad"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui_routes, "list_participant_metrics", lambda *_a, **_k: [])
    metrics_no_participants = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert metrics_no_participants["participants"] == []

    # Cover participant skip path in final rendering loop.
    (derived / "metrics.json").write_text(
        json.dumps({"participants": [{"speaker": ""}, {"speaker": "S9", "airtime_seconds": 1}]}),
        encoding="utf-8",
    )
    metrics_skip_blank = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert [row["speaker"] for row in metrics_skip_blank["participants"]] == ["S9"]

    # Force the merge path where an existing participant row is skipped due to empty speaker.
    (derived / "metrics.json").write_text(
        json.dumps({"participants": [{"speaker": "S1", "airtime_seconds": 2.0}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ui_routes,
        "list_participant_metrics",
        lambda *_a, **_k: [{"json": {"speaker": "S1", "airtime_seconds": 1.0}, "diar_speaker_label": "S1"}],
    )
    real_str = str
    seen: dict[str, int] = {"count": 0}

    def _stateful_str(value: object) -> str:
        seen["count"] += 1
        # First conversion keeps S1 during participant ingestion; second triggers line-346 skip.
        if seen["count"] == 2:
            return ""
        return real_str(value)

    monkeypatch.setattr(ui_routes, "str", _stateful_str, raising=False)
    merged = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert merged["participants"]


def test_fallback_turns_and_path_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    turns = ui_routes._fallback_speaker_turns_from_transcript(  # noqa: SLF001
        {
            "segments": [
                "skip",
                {"text": "   "},
                {"text": "hello", "start": "bad", "end": "bad"},
            ]
        }
    )
    assert turns == [
        {"start": 0.0, "end": 0.0, "speaker": "S1", "text": "hello", "language": None}
    ]

    fallback = ui_routes._fallback_speaker_turns_from_transcript({"text": "hello world"})  # noqa: SLF001
    assert fallback and fallback[0]["speaker"] == "S1"

    fallback_with_empty_segments = ui_routes._fallback_speaker_turns_from_transcript(  # noqa: SLF001
        {"segments": ["skip", {"text": "  "}], "text": "from transcript"}
    )
    assert fallback_with_empty_segments and fallback_with_empty_segments[0]["text"] == "from transcript"

    def _boom_resolve(_self: Path) -> Path:
        raise OSError("resolve-failed")

    monkeypatch.setattr(Path, "resolve", _boom_resolve)
    assert ui_routes._safe_path(tmp_path / "x", root=tmp_path) is None  # noqa: SLF001


def test_audio_snippet_helpers_and_speakers_context_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-speakers-helpers"
    derived = cfg.recordings_root / recording_id / "derived"
    snippets_dir = derived / "snippets" / "S1"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (snippets_dir / "clip.txt").write_text("x", encoding="utf-8")
    (snippets_dir / "clip.wav").write_bytes(b"wav")
    (snippets_dir / "subdir").mkdir()

    assert ui_routes._safe_path(tmp_path / "x", root=tmp_path / "other") is None  # noqa: SLF001
    assert ui_routes._safe_audio_path(tmp_path / "clip.mp3", root=tmp_path) is None  # noqa: SLF001
    assert ui_routes._speaker_snippet_files(recording_id, "S1", settings=cfg) == [  # noqa: SLF001
        snippets_dir / "clip.wav"
    ]
    assert ui_routes._as_data_relative_path(Path("/etc/passwd"), settings=cfg) is None  # noqa: SLF001

    (derived / "transcript.json").write_text(json.dumps({"text": "fallback only"}), encoding="utf-8")
    (derived / "speaker_turns.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        ui_routes,
        "list_speaker_assignments",
        lambda *_a, **_k: [{"diar_speaker_label": "S1", "voice_profile_id": "x"}],
    )
    monkeypatch.setattr(ui_routes, "list_voice_profiles", lambda *_a, **_k: [])
    fallback_ctx = ui_routes._speakers_tab_context(recording_id, cfg)  # noqa: SLF001
    assert fallback_ctx["speaker_rows"]

    (derived / "speaker_turns.json").write_text(
        json.dumps([{"speaker": "S1", "start": "bad", "end": "bad", "text": "sample"}]),
        encoding="utf-8",
    )
    parsed_ctx = ui_routes._speakers_tab_context(recording_id, cfg)  # noqa: SLF001
    assert parsed_ctx["speaker_rows"][0]["duration_sec"] == 0.0
    assert parsed_ctx["speaker_rows"][0]["voice_profile_id"] is None


def test_project_language_and_resummarize_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-proj-lang-helpers"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    assert ui_routes._as_int("bad") is None  # noqa: SLF001
    assert ui_routes._as_int(None) is None  # noqa: SLF001

    monkeypatch.setattr(
        ui_routes,
        "refresh_recording_routing",
        lambda *_a, **_k: {
            "suggested_project_id": None,
            "suggested_project_name": "",
            "confidence": "bad",
            "rationale": "bad",
            "threshold": 0.9,
        },
    )
    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda *_a, **_k: {
            "project_id": "5",
            "suggested_project_id": "9",
            "suggested_project_name": "Fallback Name",
            "routing_confidence": "bad",
            "routing_rationale_json": [" reason "],
        },
    )
    monkeypatch.setattr(ui_routes, "list_projects", lambda *_a, **_k: [{"id": 5, "name": "Roadmap"}])
    monkeypatch.setattr(
        ui_routes,
        "count_routing_training_examples",
        lambda *_a, **_k: 3,
    )
    project_ctx = ui_routes._project_tab_context(recording_id, {"project_id": 5}, cfg)  # noqa: SLF001
    assert project_ctx["selected_project_name"] == "Roadmap"
    assert project_ctx["suggested_project_name"] == "Fallback Name"
    assert project_ctx["rationale"] == ["reason"]
    assert project_ctx["confidence"] == 0.0

    monkeypatch.setattr(
        ui_routes,
        "refresh_recording_routing",
        lambda *_a, **_k: {
            "suggested_project_id": 7,
            "suggested_project_name": "Decision Name",
            "confidence": 0.8,
            "rationale": ["decision"],
            "threshold": 0.9,
        },
    )
    monkeypatch.setattr(
        ui_routes,
        "list_projects",
        lambda *_a, **_k: [{"id": 1, "name": "A"}, {"id": 5, "name": "Roadmap"}],
    )
    decision_ctx = ui_routes._project_tab_context(recording_id, {"project_id": 5}, cfg)  # noqa: SLF001
    assert decision_ctx["suggested_project_id"] == 7
    assert decision_ctx["suggested_project_name"] == "Decision Name"

    options = ui_routes._language_options(  # noqa: SLF001
        distribution_codes=["unknown", "en"],
        target_summary_language="fr",
        transcript_language_override="de",
    )
    assert any(row["code"] == "fr" for row in options)
    assert any(row["code"] == "de" for row in options)
    duplicate_options = ui_routes._language_options(  # noqa: SLF001
        distribution_codes=["en", "en"],
        target_summary_language=None,
        transcript_language_override=None,
    )
    assert duplicate_options[0]["code"] == "en"

    transcript_payload = {
        "language_distribution": {"en": "bad", "fr": 22.5},
        "language_spans": ["skip", {"start": "bad", "end": "bad", "lang": "en"}],
    }
    (derived / "transcript.json").write_text(json.dumps(transcript_payload), encoding="utf-8")
    (derived / "summary.json").write_text("{}", encoding="utf-8")
    lang_ctx = ui_routes._language_tab_context(recording_id, {}, cfg)  # noqa: SLF001
    assert lang_ctx["distribution"][0]["code"] == "fr"
    assert lang_ctx["spans"] == []

    with pytest.raises(ValueError, match="No transcript.json"):
        ui_routes._resummarize_recording(recording_id="missing", settings=cfg, target_summary_language="en")  # noqa: SLF001

    (cfg.recordings_root / "empty" / "derived").mkdir(parents=True, exist_ok=True)
    (cfg.recordings_root / "empty" / "derived" / "transcript.json").write_text(
        json.dumps({"text": ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Transcript text is empty"):
        ui_routes._resummarize_recording(recording_id="empty", settings=cfg, target_summary_language="en")  # noqa: SLF001


def test_resummarize_recording_default_turn_and_attendees_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-resummarize-edge"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hello world",
                "calendar_attendees": [" Alex ", " ", "Priya"],
            }
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text(json.dumps({"friendly": "bad"}), encoding="utf-8")

    monkeypatch.setattr(ui_routes, "_fallback_speaker_turns_from_transcript", lambda *_a, **_k: [])
    monkeypatch.setattr(
        ui_routes,
        "PipelineSettings",
        lambda **kwargs: type("S", (), {"llm_model": "test-model", **kwargs})(),
    )
    prompts_seen: dict[str, Any] = {}

    def _prompts(turns: list[dict[str, Any]], *_a: Any, **kwargs: Any) -> tuple[str, str]:
        prompts_seen["turns"] = turns
        prompts_seen["attendees"] = kwargs.get("calendar_attendees")
        return "sys", "usr"

    class _FakeLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"content": '{"summary":"ok"}'}

    writes: dict[Path, dict[str, Any]] = {}

    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        writes[path] = dict(payload)

    monkeypatch.setattr(ui_routes, "build_structured_summary_prompts", _prompts)
    monkeypatch.setattr(ui_routes, "LLMClient", lambda: _FakeLLM())
    monkeypatch.setattr(
        ui_routes,
        "build_summary_payload",
        lambda **_k: {"summary": "ok", "friendly": 0},
    )
    monkeypatch.setattr(ui_routes, "atomic_write_json", _write_json)
    monkeypatch.setattr(ui_routes, "refresh_recording_metrics", lambda *_a, **_k: None)

    ui_routes._resummarize_recording(recording_id, settings=cfg, target_summary_language=None)  # noqa: SLF001
    assert prompts_seen["turns"][0]["speaker"] == "S1"
    assert prompts_seen["attendees"] == ["Alex", "Priya"]
    assert any(path.name == "summary.json" for path in writes)


def test_resummarize_recording_uses_existing_speaker_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-resummarize-existing-turns"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(json.dumps({"text": "hello"}), encoding="utf-8")
    (derived / "speaker_turns.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "speaker": "S1", "text": "hello"}]),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        ui_routes,
        "PipelineSettings",
        lambda **kwargs: type("S", (), {"llm_model": "test-model", **kwargs})(),
    )
    monkeypatch.setattr(ui_routes, "build_structured_summary_prompts", lambda *_a, **_k: ("sys", "usr"))

    class _FakeLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"content": '{"summary":"ok"}'}

    monkeypatch.setattr(ui_routes, "LLMClient", lambda: _FakeLLM())
    monkeypatch.setattr(
        ui_routes,
        "build_summary_payload",
        lambda **_k: {"summary": "ok", "friendly": 1},
    )
    monkeypatch.setattr(ui_routes, "atomic_write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(ui_routes, "refresh_recording_metrics", lambda *_a, **_k: None)

    ui_routes._resummarize_recording(recording_id, settings=cfg, target_summary_language="en")  # noqa: SLF001


def test_datetime_and_calendar_parse_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="from is required"):
        ui_routes._parse_iso_datetime("", field_name="from")  # noqa: SLF001
    with pytest.raises(ValueError, match="from must be ISO-8601 datetime"):
        ui_routes._parse_iso_datetime("bad", field_name="from")  # noqa: SLF001
    parsed = ui_routes._parse_iso_datetime("2026-02-01T10:00:00", field_name="from")  # noqa: SLF001
    assert parsed.tzinfo is timezone.utc
    parsed_tz = ui_routes._parse_iso_datetime("2026-02-01T10:00:00Z", field_name="from")  # noqa: SLF001
    assert parsed_tz.tzinfo is timezone.utc

    with pytest.raises(ValueError, match="to is required"):
        ui_routes._parse_ymd_date("", field_name="to")  # noqa: SLF001
    with pytest.raises(ValueError, match="to must be YYYY-MM-DD"):
        ui_routes._parse_ymd_date("2026/02/01", field_name="to")  # noqa: SLF001

    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ui_routes, "list_calendar_sources", lambda *_a, **_k: [])
    monkeypatch.setattr(ui_routes, "list_calendar_events", lambda *_a, **_k: [])
    with pytest.raises(ValueError, match="to must be after from"):
        ui_routes._calendar_page_data(  # noqa: SLF001
            date_from="2026-02-10",
            date_to="2026-02-01",
            source_id=None,
            settings=cfg,
        )


def test_auth_route_edge_paths(client: tuple[AppSettings, TestClient], monkeypatch: pytest.MonkeyPatch) -> None:
    _cfg, c = client
    monkeypatch.setattr(ui_routes, "auth_enabled", lambda *_a, **_k: False)
    assert c.get("/ui").headers["location"] == "/"
    assert c.get("/ui/login").headers["location"] == "/"
    assert c.post("/ui/login", data={"token": "x"}).headers["location"] == "/"
    assert c.get("/ui/logout").headers["location"] == "/"

    monkeypatch.setattr(ui_routes, "auth_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(ui_routes, "request_is_authenticated", lambda *_a, **_k: True)
    assert c.get("/ui/login?next=/queue").headers["location"] == "/queue"

    monkeypatch.setattr(ui_routes, "expected_bearer_token", lambda *_a, **_k: "expected")
    bad_login = c.post("/ui/login", data={"token": "bad", "next": "/ui"})
    assert bad_login.status_code == 401
    assert "Invalid token." in bad_login.text
    monkeypatch.setattr(ui_routes, "request_is_authenticated", lambda *_a, **_k: False)
    login_page = c.get("/ui/login?next=/ui")
    assert login_page.status_code == 200
    assert "Invalid token." not in login_page.text

    assert c.get("/ui/logout").headers["location"] == "/ui/login"


def test_recording_progress_export_and_snippet_not_found_paths(
    client: tuple[AppSettings, TestClient],
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-snippet-missing-1")
    assert c.get("/ui/recordings/missing/progress").status_code == 404
    assert c.get("/ui/recordings/missing/export.zip").status_code == 404
    assert c.get("/ui/recordings/missing/snippets/S1/a.wav").status_code == 404
    assert c.get(f"/ui/recordings/{recording_id}/snippets/S1/missing.wav").status_code == 404


def test_assign_speaker_validation_and_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg)
    base = f"/ui/recordings/{recording_id}/speakers/assign"

    assert c.post(
        "/ui/recordings/missing/speakers/assign",
        data={"diar_speaker_label": "S1", "voice_profile_id": ""},
    ).status_code == 404
    assert c.post(base, data={"diar_speaker_label": " ", "voice_profile_id": ""}).status_code == 422
    assert c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "bad"}).status_code == 422

    monkeypatch.setattr(
        ui_routes,
        "set_speaker_assignment",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.IntegrityError("missing")),
    )
    assert c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"}).status_code == 404

    monkeypatch.setattr(
        ui_routes,
        "set_speaker_assignment",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad assignment")),
    )
    failed = c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"})
    assert failed.status_code == 422
    assert "bad assignment" in failed.text

    monkeypatch.setattr(ui_routes, "set_speaker_assignment", lambda *_a, **_k: None)
    ok = c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": ""})
    assert ok.status_code == 303


def test_create_and_assign_speaker_validation_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-create-assign-1")
    base = f"/ui/recordings/{recording_id}/speakers/create-and-assign"

    assert c.post(
        "/ui/recordings/missing/speakers/create-and-assign",
        data={"diar_speaker_label": "S1", "display_name": "Alex"},
    ).status_code == 404
    assert c.post(base, data={"diar_speaker_label": " ", "display_name": "A"}).status_code == 422
    assert c.post(base, data={"diar_speaker_label": "S1", "display_name": " "}).status_code == 422

    monkeypatch.setattr(ui_routes, "create_voice_profile", lambda *_a, **_k: {"id": "bad"})
    resp = c.post(base, data={"diar_speaker_label": "S1", "display_name": "Alex"})
    assert resp.status_code == 503


def test_add_speaker_sample_validation_and_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-add-sample-1")
    base = f"/ui/recordings/{recording_id}/speakers/add-sample"

    assert c.post(
        "/ui/recordings/missing/speakers/add-sample",
        data={"diar_speaker_label": "S1", "voice_profile_id": "1"},
    ).status_code == 404
    assert c.post(base, data={"diar_speaker_label": " ", "voice_profile_id": "1"}).status_code == 422
    assert c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": ""}).status_code == 422
    assert c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "bad"}).status_code == 422
    assert c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"}).status_code == 422

    rel_bad = tmp_path / "outside.wav"
    rel_bad.write_bytes(b"wav")
    monkeypatch.setattr(ui_routes, "_speaker_snippet_files", lambda *_a, **_k: [rel_bad])
    monkeypatch.setattr(ui_routes, "_as_data_relative_path", lambda *_a, **_k: None)
    assert c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"}).status_code == 422

    monkeypatch.setattr(ui_routes, "_as_data_relative_path", lambda *_a, **_k: "recordings/x.wav")
    monkeypatch.setattr(
        ui_routes,
        "create_voice_sample",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.IntegrityError("missing")),
    )
    assert c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"}).status_code == 404

    monkeypatch.setattr(
        ui_routes,
        "create_voice_sample",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad snippet")),
    )
    bad_value = c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"})
    assert bad_value.status_code == 422
    assert "bad snippet" in bad_value.text


def test_set_recording_project_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-project-errors-1")
    base = f"/ui/recordings/{recording_id}/project"

    assert c.post("/ui/recordings/missing/project", data={}).status_code == 404
    assert c.post(base, data={"project_id": "bad"}).status_code == 422

    monkeypatch.setattr(
        ui_routes,
        "set_recording_project",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.IntegrityError("missing")),
    )
    assert c.post(base, data={"project_id": "1"}).status_code == 404

    monkeypatch.setattr(ui_routes, "set_recording_project", lambda *_a, **_k: False)
    assert c.post(base, data={"project_id": "1"}).status_code == 404

    monkeypatch.setattr(ui_routes, "set_recording_project", lambda *_a, **_k: True)
    monkeypatch.setattr(
        ui_routes,
        "train_routing_from_manual_selection",
        lambda *_a, **_k: (_ for _ in ()).throw(KeyError("missing")),
    )
    failed_train = c.post(base, data={"project_id": "1", "train_routing": "yes"})
    assert failed_train.status_code == 404

    monkeypatch.setattr(ui_routes, "set_recording_project", lambda *_a, **_k: True)
    monkeypatch.setattr(ui_routes, "refresh_recording_routing", lambda *_a, **_k: {"ok": True})
    no_project = c.post(base, data={"project_id": "", "train_routing": "yes"})
    assert no_project.status_code == 303


def test_voices_page_and_audio_route_edge_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg, c = client
    monkeypatch.setattr(ui_routes, "list_voice_profiles", lambda *_a, **_k: [{"id": "bad"}])
    monkeypatch.setattr(
        ui_routes,
        "list_voice_samples",
        lambda *_a, **_k: [{"id": 1, "voice_profile_id": "bad"}, {"id": 2, "voice_profile_id": "1"}],
    )
    assert c.get("/voices").status_code == 200

    monkeypatch.setattr(ui_routes, "get_voice_sample", lambda *_a, **_k: None)
    assert c.get("/ui/voice-samples/1/audio").status_code == 404

    monkeypatch.setattr(ui_routes, "get_voice_sample", lambda *_a, **_k: {"snippet_path": "  "})
    assert c.get("/ui/voice-samples/2/audio").status_code == 404

    missing_rel = {"snippet_path": "recordings/missing.wav"}
    monkeypatch.setattr(ui_routes, "get_voice_sample", lambda *_a, **_k: missing_rel)
    assert c.get("/ui/voice-samples/3/audio").status_code == 404

    good_rel = cfg.data_root / "recordings" / "sample.wav"
    good_rel.parent.mkdir(parents=True, exist_ok=True)
    good_rel.write_bytes(b"wav")
    monkeypatch.setattr(
        ui_routes,
        "get_voice_sample",
        lambda *_a, **_k: {"snippet_path": "recordings/sample.wav"},
    )
    ok = c.get("/ui/voice-samples/4/audio")
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("audio/wav")

    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"wav")
    monkeypatch.setattr(
        ui_routes,
        "get_voice_sample",
        lambda *_a, **_k: {"snippet_path": str(outside)},
    )
    assert c.get("/ui/voice-samples/5/audio").status_code == 404


def test_calendar_route_edge_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    bad_range = c.get("/calendars?from=2026-02-10&to=2026-02-01")
    assert bad_range.status_code == 200
    assert "to must be after from" in bad_range.text

    created = c.post(
        "/calendars/sources",
        data={"name": "ICS URL", "kind": "url", "url": "https://example.com/team.ics"},
    )
    assert created.status_code == 303
    assert created.headers["location"].startswith("/calendars?source_id=")

    bad_create = c.post(
        "/calendars/sources",
        data={"name": "Bad URL", "kind": "url", "url": "not-a-url"},
    )
    assert bad_create.status_code == 303
    assert "error=" in bad_create.headers["location"]

    assert c.post("/calendars/sources/999/sync").status_code == 404

    source = create_calendar_source(
        name="sync-source",
        kind="file",
        file_ics="BEGIN:VCALENDAR\nEND:VCALENDAR",
        settings=cfg,
    )
    source_id = int(source["id"])

    async def _raise_sync(*_args: Any, **_kwargs: Any) -> Any:
        raise CalendarSyncError("sync boom")

    monkeypatch.setattr(ui_routes, "run_in_threadpool", _raise_sync)
    failed_sync = c.post(f"/calendars/sources/{source_id}/sync")
    assert failed_sync.status_code == 303
    assert "error=" in failed_sync.headers["location"]


def test_queue_action_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-queue-errors-1")

    monkeypatch.setattr(
        ui_routes,
        "enqueue_recording_job",
        lambda *_a, **_k: (_ for _ in ()).throw(
            DuplicateRecordingJobError(recording_id=recording_id, job_id="job-dup")
        ),
    )
    requeue = c.post(f"/ui/recordings/{recording_id}/requeue")
    assert requeue.status_code == 409
    assert "already active" in requeue.text

    assert c.post("/ui/recordings/missing/jobs/job-1/retry").status_code == 404
    assert c.post(f"/ui/recordings/{recording_id}/jobs/missing/retry").status_code == 404

    create_job(
        "job-failed-ui-cov",
        recording_id=recording_id,
        job_type="precheck",
        status=JOB_STATUS_FAILED,
        settings=cfg,
    )

    retry_dup = c.post(f"/ui/recordings/{recording_id}/jobs/job-failed-ui-cov/retry")
    assert retry_dup.status_code == 409

    monkeypatch.setattr(
        ui_routes,
        "enqueue_recording_job",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("queue-down")),
    )
    retry_failed = c.post(f"/ui/recordings/{recording_id}/jobs/job-failed-ui-cov/retry")
    assert retry_failed.status_code == 503


def test_language_action_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-lang-errors-1")

    assert c.post("/ui/recordings/missing/language/settings").status_code == 404

    monkeypatch.setattr(
        ui_routes,
        "_save_language_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad language")),
    )
    bad_settings = c.post(
        f"/ui/recordings/{recording_id}/language/settings",
        data={"target_summary_language": "xx"},
    )
    assert bad_settings.status_code == 422

    monkeypatch.setattr(ui_routes, "_save_language_settings", lambda *_a, **_k: ("en", None))
    ok_settings = c.post(f"/ui/recordings/{recording_id}/language/settings")
    assert ok_settings.status_code == 303

    assert c.post("/ui/recordings/missing/language/resummarize").status_code == 404
    bad_resummarize = c.post(f"/ui/recordings/{recording_id}/language/resummarize")
    assert bad_resummarize.status_code == 422

    monkeypatch.setattr(
        ui_routes,
        "_save_language_settings",
        lambda *_a, **_k: (None, None),
    )

    async def _raise_resummarize(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("summarize down")

    monkeypatch.setattr(ui_routes, "run_in_threadpool", _raise_resummarize)
    re_summary_error = c.post(f"/ui/recordings/{recording_id}/language/resummarize")
    assert re_summary_error.status_code == 503

    assert c.post("/ui/recordings/missing/language/retranscribe").status_code == 404
    monkeypatch.setattr(
        ui_routes,
        "_save_language_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad override")),
    )
    retr_bad = c.post(f"/ui/recordings/{recording_id}/language/retranscribe")
    assert retr_bad.status_code == 422

    monkeypatch.setattr(ui_routes, "_save_language_settings", lambda *_a, **_k: (None, None))
    monkeypatch.setattr(
        ui_routes,
        "enqueue_recording_job",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("enqueue down")),
    )
    retr_err = c.post(f"/ui/recordings/{recording_id}/language/retranscribe")
    assert retr_err.status_code == 503
