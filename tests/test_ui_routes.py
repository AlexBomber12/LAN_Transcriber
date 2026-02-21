"""Tests for the server-rendered HTML UI routes (PR-UI-SHELL-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lan_app import api, ui_routes
from lan_app.config import AppSettings
from lan_app.db import (
    create_job,
    create_recording,
    create_project,
    create_voice_profile,
    get_recording,
    init_db,
    list_projects,
    list_voice_profiles,
)
from lan_app.constants import (
    JOB_TYPE_PRECHECK,
    RECORDING_STATUS_READY,
)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    return TestClient(api.app, follow_redirects=True)


@pytest.fixture()
def seeded_client(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-1",
        source="drive",
        source_filename="meeting.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    create_job(
        "job-ui-1",
        recording_id="rec-ui-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )
    return TestClient(api.app, follow_redirects=True)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_empty(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Dashboard" in r.text
    assert "LAN Transcriber" in r.text


def test_dashboard_with_data(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-dash-1", source="drive", source_filename="a.mp3", settings=cfg)
    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/")
    assert r.status_code == 200
    assert "rec-dash-1" in r.text or "a.mp3" in r.text


# ---------------------------------------------------------------------------
# Recordings list
# ---------------------------------------------------------------------------


def test_recordings_empty(client):
    r = client.get("/recordings")
    assert r.status_code == 200
    assert "Recordings" in r.text


def test_recordings_with_data(seeded_client):
    r = seeded_client.get("/recordings")
    assert r.status_code == 200
    assert "meeting.mp3" in r.text


def test_recordings_status_filter(seeded_client):
    r = seeded_client.get("/recordings?status=Ready")
    assert r.status_code == 200
    assert "meeting.mp3" in r.text


def test_recordings_invalid_status_filter_shows_all(seeded_client):
    r = seeded_client.get("/recordings?status=InvalidStatus")
    assert r.status_code == 200
    assert "meeting.mp3" in r.text


def test_recordings_pagination(seeded_client):
    r = seeded_client.get("/recordings?limit=1&offset=0")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Recording detail
# ---------------------------------------------------------------------------


def test_recording_detail_overview(seeded_client):
    r = seeded_client.get("/recordings/rec-ui-1")
    assert r.status_code == 200
    assert "rec-ui-1" in r.text
    assert "meeting.mp3" in r.text
    assert "Ready" in r.text


def test_recording_detail_log_tab(seeded_client):
    r = seeded_client.get("/recordings/rec-ui-1?tab=log")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_recording_detail_calendar_tab(seeded_client):
    r = seeded_client.get("/recordings/rec-ui-1?tab=calendar")
    assert r.status_code == 200
    assert "Candidate Events" in r.text
    assert "Save selection" in r.text


def test_recording_detail_placeholder_tabs(seeded_client):
    for tab in ("project", "speakers"):
        r = seeded_client.get(f"/recordings/rec-ui-1?tab={tab}")
        assert r.status_code == 200
        assert "placeholder" in r.text.lower() or "available after" in r.text.lower()


def test_recording_detail_metrics_tab_uses_summary_payload(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-metrics-tab-1",
        source="drive",
        source_filename="metrics.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-metrics-tab-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Weekly sync",
                "summary_bullets": ["Reviewed blockers"],
                "decisions": ["Ship on Friday"],
                "action_items": [
                    {"task": "Send notes", "owner": "Alex", "deadline": "2026-02-23", "confidence": 0.9}
                ],
                "emotional_summary": "Focused.",
                "questions": {
                    "total_count": 1,
                    "types": {
                        "open": 0,
                        "yes_no": 0,
                        "clarification": 0,
                        "status": 1,
                        "decision_seeking": 0,
                    },
                    "extracted": ["Is QA complete?"],
                },
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-metrics-tab-1?tab=metrics")
    assert r.status_code == 200
    assert "Decisions" in r.text
    assert "Ship on Friday" in r.text
    assert "Send notes" in r.text
    assert "Is QA complete?" in r.text


def test_recording_detail_overview_shows_topic_and_emotional_summary(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-overview-summary-1",
        source="drive",
        source_filename="overview.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-overview-summary-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Quarterly roadmap",
                "summary_bullets": ["Roadmap reviewed"],
                "summary": "- Roadmap reviewed",
                "emotional_summary": "Calm and constructive.",
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-overview-summary-1")
    assert r.status_code == 200
    assert "Quarterly roadmap" in r.text
    assert "Roadmap reviewed" in r.text
    assert "Calm and constructive." in r.text


def test_recording_detail_language_tab_renders_spans(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-tab-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-lang-tab-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hello hola",
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "es",
                "language_distribution": {"en": 40.0, "es": 60.0},
                "language_spans": [
                    {"start": 0.0, "end": 1.0, "lang": "en"},
                    {"start": 1.0, "end": 2.0, "lang": "es"},
                ],
            }
        ),
        encoding="utf-8",
    )
    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-lang-tab-1?tab=language")
    assert r.status_code == 200
    assert "Language Distribution" in r.text
    assert "Language Spans" in r.text
    assert "Re-summarize (LLM only)" in r.text


def test_recording_detail_language_tab_keeps_auto_target_selected_when_unset(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-auto-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-lang-auto-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hola equipo",
                "language": {"detected": "es", "confidence": 0.95},
                "dominant_language": "es",
                "target_summary_language": "es",
                "language_distribution": {"es": 100.0},
                "language_spans": [{"start": 0.0, "end": 2.0, "lang": "es"}],
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-lang-auto-1?tab=language")
    assert r.status_code == 200
    target_select = r.text.split('id="target_summary_language"', 1)[1].split("</select>", 1)[0]
    assert 'value="" selected>Auto (dominant language)</option>' in target_select
    assert 'value="es" selected' not in target_select
    assert "Spanish (es)" in r.text


def test_recording_detail_not_found(client):
    r = client.get("/recordings/nonexistent-id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def test_projects_empty(client):
    r = client.get("/projects")
    assert r.status_code == 200
    assert "Projects" in r.text


def test_projects_create_and_list(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)

    r = c.post("/projects", data={"name": "AlphaProject"})
    assert r.status_code == 200
    assert "AlphaProject" in r.text

    projects = list_projects(settings=cfg)
    assert any(p["name"] == "AlphaProject" for p in projects)


def test_projects_create_blank_name_ignored(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)
    c.post("/projects", data={"name": "   "})
    assert list_projects(settings=cfg) == []


def test_projects_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    proj = create_project("ToDelete", settings=cfg)
    c = TestClient(api.app, follow_redirects=True)
    r = c.post(f"/projects/{proj['id']}/delete")
    assert r.status_code == 200
    assert list_projects(settings=cfg) == []


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


def test_voices_empty(client):
    r = client.get("/voices")
    assert r.status_code == 200
    assert "Voice Profiles" in r.text


def test_voices_create_and_list(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)

    r = c.post("/voices", data={"display_name": "Alice Smith", "notes": "sales team"})
    assert r.status_code == 200
    assert "Alice Smith" in r.text

    profiles = list_voice_profiles(settings=cfg)
    assert any(v["display_name"] == "Alice Smith" for v in profiles)


def test_voices_create_blank_name_ignored(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)
    c.post("/voices", data={"display_name": "  ", "notes": ""})
    assert list_voice_profiles(settings=cfg) == []


def test_voices_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    vp = create_voice_profile("Bob", settings=cfg)
    c = TestClient(api.app, follow_redirects=True)
    r = c.post(f"/voices/{vp['id']}/delete")
    assert r.status_code == 200
    assert list_voice_profiles(settings=cfg) == []


# ---------------------------------------------------------------------------
# Queue / Jobs
# ---------------------------------------------------------------------------


def test_queue_empty(client):
    r = client.get("/queue")
    assert r.status_code == 200
    assert "Queue" in r.text


def test_queue_with_data(seeded_client):
    r = seeded_client.get("/queue")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_queue_status_filter(seeded_client):
    r = seeded_client.get("/queue?status=queued")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_queue_recording_filter(seeded_client):
    r = seeded_client.get("/queue?recording_id=rec-ui-1")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_queue_invalid_status_shows_all(seeded_client):
    r = seeded_client.get("/queue?status=bogus")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def test_connections(client):
    r = client.get("/connections")
    assert r.status_code == 200
    assert "Google Drive" in r.text
    assert "Microsoft Graph" in r.text


# ---------------------------------------------------------------------------
# Inline action endpoints
# ---------------------------------------------------------------------------


def test_ui_action_quarantine(seeded_client):
    r = seeded_client.post("/ui/recordings/rec-ui-1/quarantine")
    assert r.status_code == 200


def test_ui_action_quarantine_not_found(client):
    r = client.post("/ui/recordings/no-such-rec/quarantine")
    assert r.status_code == 404


def test_ui_action_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-del-1", source="drive", source_filename="x.mp3", settings=cfg)

    def _fake_purge(recording_id: str, *, settings=None) -> int:
        return 0

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", _fake_purge)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-del-1/delete")
    assert r.status_code in (200, 307, 302)


def test_ui_action_delete_not_found(client):
    r = client.post("/ui/recordings/no-such-rec/delete")
    assert r.status_code == 404


def test_ui_action_requeue(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-rq-1", source="drive", source_filename="y.mp3", settings=cfg)

    def _fake_enqueue(recording_id: str, *, settings=None, job_type=JOB_TYPE_PRECHECK):
        return None

    monkeypatch.setattr(ui_routes, "enqueue_recording_job", _fake_enqueue)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-rq-1/requeue")
    assert r.status_code in (200, 307, 302)


def test_ui_action_requeue_not_found(client):
    r = client.post("/ui/recordings/no-such-rec/requeue")
    assert r.status_code == 404


def test_ui_action_requeue_failure_returns_503(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-rqf-1", source="drive", source_filename="z.mp3", settings=cfg)

    def _fail_enqueue(recording_id: str, *, settings=None, job_type=JOB_TYPE_PRECHECK):
        raise RuntimeError("redis down")

    monkeypatch.setattr(ui_routes, "enqueue_recording_job", _fail_enqueue)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-rqf-1/requeue")
    assert r.status_code == 503
    assert "redis down" in r.text


def test_ui_language_resummarize_uses_target_language_override(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-rsum-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    derived = cfg.recordings_root / "rec-lang-rsum-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hello team and hola equipo",
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
                "language_distribution": {"en": 55.0, "es": 45.0},
                "language_spans": [
                    {"start": 0.0, "end": 2.0, "lang": "en"},
                    {"start": 2.0, "end": 4.0, "lang": "es"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text(
        json.dumps({"friendly": 0, "model": "llama3:8b", "summary": "- old"}),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    async def _fake_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ):
        captured["system_prompt"] = system_prompt
        captured["model"] = model or ""
        return {
            "content": json.dumps(
                {
                    "topic": "Resumen semanal",
                    "summary_bullets": ["Bloqueadores revisados."],
                    "decisions": ["Publicar el viernes."],
                    "action_items": [
                        {
                            "task": "Enviar notas",
                            "owner": "Alex",
                            "deadline": "2026-02-23",
                            "confidence": 0.85,
                        }
                    ],
                    "emotional_summary": "Enfoque positivo.",
                    "questions": {
                        "total_count": 1,
                        "types": {
                            "open": 0,
                            "yes_no": 0,
                            "clarification": 1,
                            "status": 0,
                            "decision_seeking": 0,
                        },
                        "extracted": ["Quien valida QA?"],
                    },
                }
            )
        }

    monkeypatch.setattr(ui_routes.LLMClient, "generate", _fake_generate)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-lang-rsum-1/language/resummarize",
        data={
            "target_summary_language": "es",
            "transcript_language_override": "en",
        },
    )
    assert r.status_code == 303

    recording = get_recording("rec-lang-rsum-1", settings=cfg)
    assert recording is not None
    assert recording["target_summary_language"] == "es"
    assert recording["language_override"] == "en"

    summary_payload = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["summary_bullets"] == ["Bloqueadores revisados."]
    assert summary_payload["topic"] == "Resumen semanal"
    assert summary_payload["target_summary_language"] == "es"
    assert "in Spanish." in captured["system_prompt"]


def test_ui_language_retranscribe_enqueues_precheck_and_saves_overrides(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-rtr-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    called: dict[str, str] = {}

    def _fake_enqueue(recording_id: str, *, settings=None, job_type=JOB_TYPE_PRECHECK):
        called["recording_id"] = recording_id
        called["job_type"] = job_type
        return None

    monkeypatch.setattr(ui_routes, "enqueue_recording_job", _fake_enqueue)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-lang-rtr-1/language/retranscribe",
        data={
            "target_summary_language": "es",
            "transcript_language_override": "en",
        },
    )
    assert r.status_code == 303
    assert called["recording_id"] == "rec-lang-rtr-1"
    assert called["job_type"] == JOB_TYPE_PRECHECK

    recording = get_recording("rec-lang-rtr-1", settings=cfg)
    assert recording is not None
    assert recording["target_summary_language"] == "es"
    assert recording["language_override"] == "en"


def test_ui_action_delete_purge_failure_returns_503(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-pf-1", source="drive", source_filename="pf.mp3", settings=cfg)

    def _fail_purge(recording_id: str, *, settings=None) -> int:
        raise RuntimeError("redis down")

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", _fail_purge)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-pf-1/delete")
    assert r.status_code == 503
    assert "redis down" in r.text


def test_ui_action_delete_removes_disk_artifacts(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-disk-1", source="drive", source_filename="w.mp3", settings=cfg)
    rec_dir = cfg.recordings_root / "rec-disk-1"
    rec_dir.mkdir(parents=True)
    (rec_dir / "audio.mp3").write_text("fake")

    def _fake_purge(recording_id: str, *, settings=None) -> int:
        return 0

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", _fake_purge)
    c = TestClient(api.app, follow_redirects=False)
    c.post("/ui/recordings/rec-disk-1/delete")
    assert not rec_dir.exists()


def test_projects_duplicate_name_handled(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_project("DupeTest", settings=cfg)
    c = TestClient(api.app, follow_redirects=True)
    r = c.post("/projects", data={"name": "DupeTest"})
    assert r.status_code == 200
    projects = list_projects(settings=cfg)
    assert len(projects) == 1


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------


def test_static_htmx_served(client):
    r = client.get("/static/htmx.min.js")
    assert r.status_code == 200
    assert "htmx" in r.text


def test_dashboard_references_local_htmx(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/static/htmx.min.js" in r.text
    assert "unpkg.com" not in r.text


# ---------------------------------------------------------------------------
# DB helpers: projects and voice_profiles
# ---------------------------------------------------------------------------


def test_db_list_projects_empty(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    assert list_projects(settings=cfg) == []


def test_db_create_and_delete_project(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    p = create_project("Gamma", settings=cfg)
    assert p["name"] == "Gamma"
    assert p["auto_publish"] == 0
    projects = list_projects(settings=cfg)
    assert len(projects) == 1

    from lan_app.db import delete_project
    assert delete_project(p["id"], settings=cfg) is True
    assert list_projects(settings=cfg) == []


def test_db_list_voice_profiles_empty(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    assert list_voice_profiles(settings=cfg) == []


def test_db_create_and_delete_voice_profile(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    vp = create_voice_profile("Carol", notes="engineer", settings=cfg)
    assert vp["display_name"] == "Carol"
    assert vp["notes"] == "engineer"
    profiles = list_voice_profiles(settings=cfg)
    assert len(profiles) == 1

    from lan_app.db import delete_voice_profile
    assert delete_voice_profile(vp["id"], settings=cfg) is True
    assert list_voice_profiles(settings=cfg) == []
