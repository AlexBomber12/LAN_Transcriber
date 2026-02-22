from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from lan_app import api, onenote, ui_routes
from lan_app.config import AppSettings
from lan_app.constants import (
    RECORDING_STATUS_PUBLISHED,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_READY,
)
from lan_app.db import (
    create_project,
    create_recording,
    get_project,
    get_recording,
    init_db,
    set_recording_publish_result,
    update_project_onenote_mapping,
)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_db_project_mapping_and_publish_result(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    project = create_project("Ops", settings=cfg)
    updated = update_project_onenote_mapping(
        project["id"],
        onenote_notebook_id="nb-100",
        onenote_section_id="sec-100",
        settings=cfg,
    )
    assert updated is not None
    assert updated["onenote_notebook_id"] == "nb-100"
    assert updated["onenote_section_id"] == "sec-100"
    check = get_project(project["id"], settings=cfg)
    assert check is not None
    assert check["onenote_section_id"] == "sec-100"

    create_recording(
        "rec-published-1",
        source="drive",
        source_filename="published.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    assert set_recording_publish_result(
        "rec-published-1",
        onenote_page_id="page-100",
        onenote_page_url="https://onenote.local/page-100",
        settings=cfg,
    )
    rec = get_recording("rec-published-1", settings=cfg)
    assert rec is not None
    assert rec["status"] == RECORDING_STATUS_PUBLISHED
    assert rec["onenote_page_id"] == "page-100"
    assert rec["onenote_page_url"] == "https://onenote.local/page-100"


def test_publish_recording_to_onenote_success(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    project = create_project("Finance", settings=cfg)
    update_project_onenote_mapping(
        project["id"],
        onenote_notebook_id="nb-1",
        onenote_section_id="sec-1",
        settings=cfg,
    )
    create_recording(
        "rec-pub-1",
        source="drive",
        source_filename="finance.mp3",
        status=RECORDING_STATUS_READY,
        project_id=project["id"],
        drive_file_id="drive-123",
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-pub-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Budget Review",
                "summary_bullets": ["Discussed Q2 targets"],
                "decisions": ["Approve spending freeze"],
                "action_items": [
                    {"task": "Draft budget", "owner": "Alex", "deadline": "2026-03-01"}
                ],
            }
        ),
        encoding="utf-8",
    )

    class _FakeGraphClient:
        def __init__(self, settings=None):
            self.settings = settings

        def graph_post_html(self, path: str, html: str):
            assert path.endswith("/me/onenote/sections/sec-1/pages")
            assert "<h2>Summary</h2>" in html
            return {
                "id": "page-xyz",
                "links": {"oneNoteWebUrl": {"href": "https://onenote.local/page-xyz"}},
            }

    monkeypatch.setattr(onenote, "MicrosoftGraphClient", _FakeGraphClient)

    published = onenote.publish_recording_to_onenote("rec-pub-1", settings=cfg)
    assert published["onenote_page_id"] == "page-xyz"
    assert published["onenote_page_url"] == "https://onenote.local/page-xyz"

    rec = get_recording("rec-pub-1", settings=cfg)
    assert rec is not None
    assert rec["status"] == RECORDING_STATUS_PUBLISHED
    assert rec["onenote_page_id"] == "page-xyz"
    assert rec["onenote_page_url"] == "https://onenote.local/page-xyz"


def test_extract_page_url_ignores_graph_resource_links():
    payload = {
        "location": "https://graph.microsoft.com/v1.0/me/onenote/pages/page-123",
        "content_location": "https://graph.microsoft.com/v1.0/me/onenote/pages/page-123/content",
        "webUrl": "https://graph.microsoft.com/v1.0/me/onenote/pages/page-123",
        "contentUrl": "https://graph.microsoft.com/v1.0/me/onenote/pages/page-123/content",
    }
    assert onenote._extract_page_url(payload) is None


def test_load_metrics_context_does_not_duplicate_fallback_participants(
    tmp_path: Path, monkeypatch
):
    cfg = _cfg(tmp_path)
    recording_id = "rec-metrics-no-dup-1"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "metrics.json").write_text(
        json.dumps(
            {
                "meeting": {"total_questions": 3},
                "participants": [
                    {
                        "speaker": "Speaker From Artifact",
                        "airtime_seconds": 20,
                        "airtime_share": 0.2,
                        "turns": 2,
                        "interruptions_done": 0,
                        "interruptions_received": 0,
                        "questions_count": 1,
                        "role_hint": "observer",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(onenote, "get_meeting_metrics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        onenote,
        "list_participant_metrics",
        lambda *_args, **_kwargs: [
            {
                "diar_speaker_label": "Speaker From DB",
                "json": {
                    "speaker": "Speaker From DB",
                    "airtime_seconds": 120,
                    "airtime_share": 0.8,
                    "turns": 8,
                    "interruptions_done": 1,
                    "interruptions_received": 0,
                    "questions_count": 2,
                    "role_hint": "host",
                },
            }
        ],
    )

    metrics = onenote._load_metrics_context(recording_id, settings=cfg)
    speakers = [row["speaker"] for row in metrics["participants"]]
    assert speakers == ["Speaker From DB"]
    assert metrics["meeting"]["total_questions"] == 3


def test_publish_recording_to_onenote_requires_ready_or_needs_review(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    project = create_project("Sales", settings=cfg)
    update_project_onenote_mapping(
        project["id"],
        onenote_notebook_id="nb-1",
        onenote_section_id="sec-1",
        settings=cfg,
    )
    create_recording(
        "rec-not-ready-1",
        source="drive",
        source_filename="not-ready.mp3",
        status=RECORDING_STATUS_QUEUED,
        project_id=project["id"],
        settings=cfg,
    )
    with pytest.raises(onenote.PublishPreconditionError):
        onenote.publish_recording_to_onenote("rec-not-ready-1", settings=cfg)


def test_api_publish_endpoint(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-api-publish-1",
        source="drive",
        source_filename="publish.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    monkeypatch.setattr(
        api,
        "publish_recording_to_onenote",
        lambda recording_id, *, settings=None: {
            "recording_id": recording_id,
            "onenote_page_id": "page-api-1",
        },
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.post("/api/recordings/rec-api-publish-1/publish")
    assert resp.status_code == 200
    assert resp.json()["onenote_page_id"] == "page-api-1"


def test_api_publish_endpoint_returns_422_on_precondition(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-api-publish-2",
        source="drive",
        source_filename="publish-2.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    def _fail(_recording_id: str, *, settings=None):
        raise onenote.PublishPreconditionError("project mapping missing")

    monkeypatch.setattr(api, "publish_recording_to_onenote", _fail)
    client = TestClient(api.app, follow_redirects=True)
    resp = client.post("/api/recordings/rec-api-publish-2/publish")
    assert resp.status_code == 422
    assert "project mapping missing" in resp.text


def test_projects_page_browses_and_updates_mapping(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    project = create_project("BrowseProject", settings=cfg)

    monkeypatch.setattr(
        ui_routes,
        "list_onenote_notebooks",
        lambda *, settings=None: [
            {"id": "nb-1", "display_name": "Notebook A", "web_url": "https://onenote/notebook"}
        ],
    )
    monkeypatch.setattr(
        ui_routes,
        "list_onenote_sections",
        lambda notebook_id, *, settings=None: [
            {"id": "sec-1", "display_name": "Section A", "web_url": "https://onenote/section"}
        ],
    )

    client = TestClient(api.app, follow_redirects=True)
    page = client.get(f"/projects?browse_project_id={project['id']}&browse_notebook_id=nb-1")
    assert page.status_code == 200
    assert "OneNote Browser for BrowseProject" in page.text
    assert "Notebook A" in page.text
    assert "Section A" in page.text

    updated = client.post(
        f"/projects/{project['id']}/onenote",
        data={"onenote_notebook_id": "nb-1", "onenote_section_id": "sec-1"},
    )
    assert updated.status_code == 200
    project_after = get_project(project["id"], settings=cfg)
    assert project_after is not None
    assert project_after["onenote_notebook_id"] == "nb-1"
    assert project_after["onenote_section_id"] == "sec-1"


def test_recording_overview_publish_controls(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    project = create_project("UI Publish", settings=cfg)
    create_recording(
        "rec-ui-publish-1",
        source="drive",
        source_filename="ui-publish.mp3",
        status=RECORDING_STATUS_READY,
        project_id=project["id"],
        settings=cfg,
    )

    client = TestClient(api.app, follow_redirects=True)
    before = client.get("/recordings/rec-ui-publish-1?tab=overview")
    assert before.status_code == 200
    assert "Publish to OneNote" in before.text

    set_recording_publish_result(
        "rec-ui-publish-1",
        onenote_page_id="page-ui-1",
        onenote_page_url="https://onenote.local/page-ui-1",
        settings=cfg,
    )
    after = client.get("/recordings/rec-ui-publish-1?tab=overview")
    assert after.status_code == 200
    assert "Open OneNote page" in after.text


def test_recording_overview_hides_publish_without_project_assignment(
    tmp_path: Path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-no-project-1",
        source="drive",
        source_filename="ui-no-project.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    client = TestClient(api.app, follow_redirects=True)
    page = client.get("/recordings/rec-ui-no-project-1?tab=overview")
    assert page.status_code == 200
    assert "Publish to OneNote" not in page.text
    assert "Assign this recording to a project before publishing to OneNote." in page.text


def test_recording_overview_does_not_link_graph_api_when_page_url_missing(
    tmp_path: Path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-publish-no-url-1",
        source="drive",
        source_filename="ui-publish-no-url.mp3",
        status=RECORDING_STATUS_PUBLISHED,
        onenote_page_id="page-id-only",
        settings=cfg,
    )

    client = TestClient(api.app, follow_redirects=True)
    page = client.get("/recordings/rec-ui-publish-no-url-1?tab=overview")
    assert page.status_code == 200
    assert "https://graph.microsoft.com/v1.0/me/onenote/pages/" not in page.text
    assert "Open OneNote page" not in page.text
    assert "page-id-only" in page.text


def test_ui_publish_action_redirects_on_success(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-publish-2",
        source="drive",
        source_filename="ui-publish-2.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    monkeypatch.setattr(
        ui_routes,
        "publish_recording_to_onenote",
        lambda recording_id, *, settings=None: {"recording_id": recording_id},
    )
    client = TestClient(api.app, follow_redirects=False)
    resp = client.post("/ui/recordings/rec-ui-publish-2/publish")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/recordings/rec-ui-publish-2?tab=overview"
