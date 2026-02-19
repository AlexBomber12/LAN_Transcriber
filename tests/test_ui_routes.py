"""Tests for the server-rendered HTML UI routes (PR-UI-SHELL-01)."""

from __future__ import annotations

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


def test_recording_detail_placeholder_tabs(seeded_client):
    for tab in ("calendar", "project", "speakers", "language", "metrics"):
        r = seeded_client.get(f"/recordings/rec-ui-1?tab={tab}")
        assert r.status_code == 200
        assert "placeholder" in r.text.lower() or "available after" in r.text.lower()


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
