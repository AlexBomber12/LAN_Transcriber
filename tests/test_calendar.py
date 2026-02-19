from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from lan_app import api, calendar, ui_routes
from lan_app.config import AppSettings
from lan_app.db import (
    create_recording,
    get_calendar_match,
    init_db,
    upsert_calendar_match,
)
from lan_app.ms_graph import GraphNotConfiguredError


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _fake_events() -> list[dict[str, object]]:
    return [
        {
            "id": "evt-1",
            "subject": "Weekly sync meeting",
            "start": {"dateTime": "2026-02-19T09:50:00Z"},
            "end": {"dateTime": "2026-02-19T10:35:00Z"},
            "organizer": {
                "emailAddress": {"name": "Alex", "address": "alex@example.com"}
            },
            "attendees": [
                {"emailAddress": {"name": "Priya", "address": "priya@example.com"}},
                {"emailAddress": {"name": "Lee", "address": "lee@example.com"}},
            ],
            "location": {"displayName": "Room 3"},
        },
        {
            "id": "evt-2",
            "subject": "Unrelated lunch",
            "start": {"dateTime": "2026-02-19T13:00:00Z"},
            "end": {"dateTime": "2026-02-19T14:00:00Z"},
            "organizer": {
                "emailAddress": {"name": "Taylor", "address": "taylor@example.com"}
            },
            "attendees": [],
            "location": {"displayName": "Cafeteria"},
        },
    ]


def test_refresh_calendar_context_auto_selects_best_candidate(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-cal-1",
        source="drive",
        source_filename="weekly_sync.mp3",
        captured_at="2026-02-19T10:00:00Z",
        duration_sec=1800,
        settings=cfg,
    )

    seen_path: dict[str, str] = {}

    class _FakeClient:
        def __init__(self, settings=None):
            self.settings = settings

        def graph_get(self, path: str):
            seen_path["value"] = path
            return {"value": _fake_events()}

    monkeypatch.setattr(calendar, "MicrosoftGraphClient", _FakeClient)

    context = calendar.refresh_calendar_context("rec-cal-1", settings=cfg)
    assert context["selected_event_id"] == "evt-1"
    assert context["selected_event"]["subject"] == "Weekly sync meeting"
    assert context["candidate_total"] == 2
    assert context["signals"]["organizer"] == "Alex"
    assert "calendarView" in seen_path["value"]

    stored = get_calendar_match("rec-cal-1", settings=cfg)
    assert stored is not None
    assert stored["selected_event_id"] == "evt-1"
    assert len(stored["candidates_json"]) == 2
    assert stored["candidates_json"][0]["rationale"]


def test_manual_no_event_override_persists_after_refresh(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-cal-2",
        source="drive",
        source_filename="weekly_sync.mp3",
        captured_at="2026-02-19T10:00:00Z",
        duration_sec=1800,
        settings=cfg,
    )

    class _FakeClient:
        def __init__(self, settings=None):
            self.settings = settings

        def graph_get(self, _path: str):
            return {"value": _fake_events()}

    monkeypatch.setattr(calendar, "MicrosoftGraphClient", _FakeClient)

    first = calendar.refresh_calendar_context("rec-cal-2", settings=cfg)
    assert first["selected_event_id"] == "evt-1"

    manual = calendar.select_calendar_event("rec-cal-2", None, settings=cfg)
    assert manual["manual_no_event"] is True
    assert manual["selected_event_id"] is None

    refreshed = calendar.refresh_calendar_context("rec-cal-2", settings=cfg)
    assert refreshed["selected_event_id"] is None
    assert refreshed["manual_no_event"] is True


def test_api_calendar_get_falls_back_to_cached_context(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-cal-api-1",
        source="drive",
        source_filename="meeting.mp3",
        settings=cfg,
    )

    def _refresh(_recording_id: str, *, settings=None):
        raise GraphNotConfiguredError("missing env")

    monkeypatch.setattr(api, "refresh_calendar_context", _refresh)
    monkeypatch.setattr(
        api,
        "load_calendar_context",
        lambda recording_id, *, settings=None: {
            "recording_id": recording_id,
            "selected_event_id": None,
            "selected_confidence": None,
            "selected_event": None,
            "signals": {"title_tokens": [], "attendees": [], "organizer": None},
            "candidates": [],
            "candidate_total": 0,
            "manual_no_event": False,
        },
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/api/recordings/rec-cal-api-1/calendar")
    assert resp.status_code == 200
    body = resp.json()
    assert body["recording_id"] == "rec-cal-api-1"
    assert body["fetch_error"] == "missing env"


def test_api_calendar_select_validation_error(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-cal-api-2",
        source="drive",
        source_filename="meeting.mp3",
        settings=cfg,
    )

    def _invalid(_recording_id: str, _event_id: str | None, *, settings=None):
        raise ValueError("Unknown event_id for this recording")

    monkeypatch.setattr(api, "select_calendar_event", _invalid)

    client = TestClient(api.app, follow_redirects=True)
    resp = client.post(
        "/api/recordings/rec-cal-api-2/calendar/select",
        json={"event_id": "missing"},
    )
    assert resp.status_code == 422
    assert "Unknown event_id" in resp.text


def test_ui_calendar_tab_and_save_selection(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-cal-ui-1",
        source="drive",
        source_filename="meeting.mp3",
        settings=cfg,
    )

    monkeypatch.setattr(
        ui_routes,
        "refresh_calendar_context",
        lambda recording_id, *, settings=None: {
            "recording_id": recording_id,
            "selected_event_id": "evt-1",
            "selected_confidence": 0.92,
            "selected_event": {
                "event_id": "evt-1",
                "subject": "Weekly sync meeting",
                "start": "2026-02-19T09:50:00Z",
                "end": "2026-02-19T10:35:00Z",
                "organizer": "Alex",
                "attendees": ["Priya", "Lee"],
                "location": "Room 3",
                "score": 0.92,
                "rationale": "time_overlap=1.00; proximity=0.00; subject_match=0.20",
            },
            "signals": {
                "title_tokens": ["meeting", "sync", "weekly"],
                "attendees": ["Priya", "Lee"],
                "organizer": "Alex",
            },
            "candidates": [
                {
                    "event_id": "evt-1",
                    "subject": "Weekly sync meeting",
                    "start": "2026-02-19T09:50:00Z",
                    "end": "2026-02-19T10:35:00Z",
                    "organizer": "Alex",
                    "attendees": ["Priya", "Lee"],
                    "location": "Room 3",
                    "score": 0.92,
                    "rationale": "time_overlap=1.00; proximity=0.00; subject_match=0.20",
                }
            ],
            "candidate_total": 1,
            "manual_no_event": False,
        },
    )

    selected: dict[str, str | None] = {}

    def _select(recording_id: str, event_id: str | None, *, settings=None):
        selected["recording_id"] = recording_id
        selected["event_id"] = event_id
        return {
            "recording_id": recording_id,
            "selected_event_id": event_id,
            "selected_confidence": 0.0,
            "selected_event": None,
            "signals": {"title_tokens": [], "attendees": [], "organizer": None},
            "candidates": [],
            "candidate_total": 0,
            "manual_no_event": event_id is None,
        }

    monkeypatch.setattr(ui_routes, "select_calendar_event", _select)

    browser = TestClient(api.app, follow_redirects=True)
    page = browser.get("/recordings/rec-cal-ui-1?tab=calendar")
    assert page.status_code == 200
    assert "Weekly sync meeting" in page.text
    assert "Save selection" in page.text

    post = browser.post(
        "/ui/recordings/rec-cal-ui-1/calendar/select",
        data={"event_id": "evt-1"},
        follow_redirects=False,
    )
    assert post.status_code == 303
    assert post.headers["location"] == "/recordings/rec-cal-ui-1?tab=calendar"
    assert selected["recording_id"] == "rec-cal-ui-1"
    assert selected["event_id"] == "evt-1"


def test_parse_event_datetime_accepts_7_digit_graph_precision():
    parsed = calendar._parse_event_datetime(
        {
            "dateTime": "2026-02-19T09:50:00.0000000Z",
            "timeZone": "UTC",
        }
    )
    assert parsed == datetime(2026, 2, 19, 9, 50, tzinfo=timezone.utc)


def test_parse_event_datetime_uses_declared_timezone_for_naive_values():
    parsed = calendar._parse_event_datetime(
        {
            "dateTime": "2026-02-19T09:50:00.0000000",
            "timeZone": "Pacific Standard Time",
        }
    )
    assert parsed == datetime(2026, 2, 19, 17, 50, tzinfo=timezone.utc)


def test_parse_event_datetime_unknown_timezone_does_not_fallback_to_utc():
    parsed = calendar._parse_event_datetime(
        {
            "dateTime": "2026-02-19T09:50:00.0000000",
            "timeZone": "Unknown/Timezone",
        }
    )
    assert parsed is None


def test_context_keeps_selected_event_when_not_in_top_five(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-cal-top5-1",
        source="drive",
        source_filename="meeting.mp3",
        captured_at="2026-02-19T10:00:00Z",
        settings=cfg,
    )

    candidates = []
    for idx in range(6):
        candidates.append(
            {
                "event_id": f"evt-{idx + 1}",
                "subject": f"Event {idx + 1}",
                "start": "2026-02-19T09:00:00Z",
                "end": "2026-02-19T09:30:00Z",
                "organizer": "Alex",
                "attendees": [],
                "location": None,
                "title_tokens": [f"event{idx + 1}"],
                "score": round(0.9 - idx * 0.1, 4),
                "rationale": "test",
            }
        )
    upsert_calendar_match(
        recording_id="rec-cal-top5-1",
        candidates=candidates,
        selected_event_id="evt-6",
        selected_confidence=0.4,
        settings=cfg,
    )

    context = calendar.load_calendar_context("rec-cal-top5-1", settings=cfg)
    rendered_ids = [item["event_id"] for item in context["candidates"]]
    assert context["selected_event_id"] == "evt-6"
    assert "evt-6" in rendered_ids
    assert rendered_ids.count("evt-6") == 1
    assert context["candidate_total"] == 6


def test_proximity_component_point_recording_uses_nearest_event_edge():
    recording_point = datetime(2026, 2, 19, 10, 0, tzinfo=timezone.utc)
    event_start = datetime(2026, 2, 19, 9, 0, tzinfo=timezone.utc)
    event_end = datetime(2026, 2, 19, 9, 59, tzinfo=timezone.utc)
    proximity = calendar._proximity_component(
        recording_start=recording_point,
        recording_end=recording_point,
        event_start=event_start,
        event_end=event_end,
        window_seconds=3600,
    )
    expected = 1.0 - (60.0 / 3600.0)
    assert abs(proximity - expected) < 1e-9
