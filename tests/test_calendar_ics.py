from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from lan_app import api
from lan_app.calendar import ics as ics_module
from lan_app.calendar import service as calendar_service
from lan_app.calendar.ics import parse_ics_events
from lan_app.config import AppSettings
from lan_app.db import (
    create_calendar_source,
    get_calendar_match,
    init_db,
    replace_calendar_events_for_window,
)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_parse_ics_timezone_fixture_expands_to_utc():
    payload = (Path(__file__).parent / "fixtures" / "calendar_timezone.ics").read_text(
        encoding="utf-8"
    )
    events = parse_ics_events(
        payload,
        window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 2, 20, tzinfo=timezone.utc),
    )
    assert len(events) == 1
    assert events[0]["uid"] == "tz-event-1"
    assert events[0]["starts_at"] == "2026-02-10T08:00:00Z"
    assert events[0]["ends_at"] == "2026-02-10T09:00:00Z"
    assert events[0]["organizer"] == "Alice Example"
    assert events[0]["organizer_name"] == "Alice Example"
    assert events[0]["organizer_email"] == "alice@example.com"
    assert events[0]["attendees"] == []


def test_parse_ics_recurrence_fixture_expands_weekly_instances():
    payload = (Path(__file__).parent / "fixtures" / "calendar_recurrence.ics").read_text(
        encoding="utf-8"
    )
    events = parse_ics_events(
        payload,
        window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    starts = [row["starts_at"] for row in events]
    assert len(events) == 4
    assert starts == [
        "2026-02-01T12:00:00Z",
        "2026-02-08T12:00:00Z",
        "2026-02-15T12:00:00Z",
        "2026-02-22T12:00:00Z",
    ]


def test_parse_ics_recurring_event_retains_attendees_and_organizer():
    payload = "\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//LAN Transcriber//Calendar Recurrence Attendees Test//EN",
            "BEGIN:VEVENT",
            "UID:weekly-attendees-1",
            "DTSTAMP:20260201T120000Z",
            "DTSTART:20260201T120000Z",
            "DTEND:20260201T130000Z",
            "RRULE:FREQ=WEEKLY;COUNT=3",
            "SUMMARY:Weekly planning",
            "ORGANIZER;CN=Alex Example:mailto:alex@example.com",
            "ATTENDEE;CN=Priya Kapoor:mailto:priya@example.com",
            "ATTENDEE;CN=Marco Rossi:mailto:marco@example.com",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
    )
    events = parse_ics_events(
        payload,
        window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
    )

    assert len(events) == 3
    for event in events:
        assert event["organizer"] == "Alex Example"
        assert event["organizer_email"] == "alex@example.com"
        assert event["attendees"] == [
            {
                "label": "Priya Kapoor",
                "name": "Priya Kapoor",
                "email": "priya@example.com",
            },
            {
                "label": "Marco Rossi",
                "name": "Marco Rossi",
                "email": "marco@example.com",
            },
        ]


def test_parse_ics_floating_datetimes_use_calendar_timezone():
    payload = "\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//LAN Transcriber//Calendar Floating Time Test//EN",
            "X-WR-TIMEZONE:Europe/Berlin",
            "BEGIN:VEVENT",
            "UID:floating-berlin-1",
            "DTSTAMP:20260201T120000Z",
            "DTSTART:20260210T090000",
            "DTEND:20260210T100000",
            "SUMMARY:Floating Time Event",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
    )
    events = parse_ics_events(
        payload,
        window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 2, 20, tzinfo=timezone.utc),
    )
    assert len(events) == 1
    assert events[0]["starts_at"] == "2026-02-10T08:00:00Z"
    assert events[0]["ends_at"] == "2026-02-10T09:00:00Z"


def test_ics_private_participant_helpers_cover_empty_and_duplicate_inputs():
    assert ics_module._participant_email("") is None  # noqa: SLF001
    assert ics_module._participant_email("mailto:alex@example.com") == "alex@example.com"  # noqa: SLF001

    class _BlankParticipant:
        params: dict[str, str] = {}

        def __str__(self) -> str:
            return ""

    assert ics_module._participant_payload(_BlankParticipant()) is None  # noqa: SLF001
    assert ics_module._participant_payload(  # noqa: SLF001
        type("Participant", (), {"params": {"CN": "Alex"}, "__str__": lambda self: ""})()
    ) == {
        "label": "Alex",
        "name": "Alex",
    }

    class _Component:
        def get(self, key: str):  # noqa: ANN001
            if key != "ATTENDEE":
                return None
            return [
                type("Participant", (), {"params": {"CN": "Alex"}, "__str__": lambda self: "mailto:alex@example.com"})(),
                type("Participant", (), {"params": {"CN": "Alex"}, "__str__": lambda self: "mailto:alex@example.com"})(),
                type("Participant", (), {"params": {}, "__str__": lambda self: ""})(),
            ]

    assert ics_module._attendees(_Component()) == [  # noqa: SLF001
        {
            "label": "Alex",
            "name": "Alex",
            "email": "alex@example.com",
        }
    ]


@pytest.fixture()
def calendar_client(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    return TestClient(api.app, follow_redirects=True)


def test_calendar_source_create_sync_and_events(calendar_client: TestClient, monkeypatch):
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    starts_at = now + timedelta(days=1)
    ends_at = starts_at + timedelta(hours=1)
    payload = "\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//LAN Transcriber//Calendar API Test//EN",
            "BEGIN:VEVENT",
            "UID:api-event-1",
            "DTSTAMP:20260201T120000Z",
            f"DTSTART:{starts_at.strftime('%Y%m%dT%H%M%SZ')}",
            f"DTEND:{ends_at.strftime('%Y%m%dT%H%M%SZ')}",
            "SUMMARY:API Calendar Event",
            "LOCATION:Room A",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
    ).encode("utf-8")

    def _fake_fetch(*_args, **_kwargs) -> bytes:
        return payload

    monkeypatch.setattr(calendar_service, "fetch_ics_url", _fake_fetch)

    create = calendar_client.post(
        "/api/calendar/sources",
        json={
            "name": "Team Calendar",
            "kind": "url",
            "url": "https://calendar.example.com/team.ics",
        },
    )
    assert create.status_code == 200
    source = create.json()
    assert "url" not in source
    assert source["name"] == "Team Calendar"
    assert source["url_configured"] is True
    source_id = int(source["id"])

    sync = calendar_client.post(f"/api/calendar/sources/{source_id}/sync")
    assert sync.status_code == 200
    assert sync.json()["events_count"] == 1

    # Re-sync should remain idempotent due bounded-window replace.
    resync = calendar_client.post(f"/api/calendar/sources/{source_id}/sync")
    assert resync.status_code == 200
    assert resync.json()["events_count"] == 1

    sources = calendar_client.get("/api/calendar/sources")
    assert sources.status_code == 200
    listed = sources.json()["items"]
    assert listed and listed[0]["id"] == source_id
    assert "url" not in listed[0]

    events = calendar_client.get(
        "/api/calendar/events",
        params={
            "from": now.isoformat().replace("+00:00", "Z"),
            "to": (now + timedelta(days=3)).isoformat().replace("+00:00", "Z"),
            "source_id": source_id,
        },
    )
    assert events.status_code == 200
    body = events.json()
    assert body["total"] == 1
    assert body["items"][0]["summary"] == "API Calendar Event"


def test_upload_runtime_populates_calendar_matches(calendar_client: TestClient, monkeypatch):
    cfg = api._settings
    source = create_calendar_source(
        name="Team Calendar",
        kind="file",
        file_ics="BEGIN:VCALENDAR\nEND:VCALENDAR",
        settings=cfg,
    )
    replace_calendar_events_for_window(
        source_id=int(source["id"]),
        window_start="2026-03-01T00:00:00Z",
        window_end="2026-03-02T00:00:00Z",
        events=[
            {
                "uid": "evt-upload-1",
                "starts_at": "2026-03-01T10:00:00Z",
                "ends_at": "2026-03-01T11:00:00Z",
                "summary": "Roadmap Review",
                "organizer": "Alex",
                "attendees": [{"label": "Priya Kapoor"}],
                "updated_at": "2026-03-01T00:00:00Z",
            }
        ],
        settings=cfg,
    )
    monkeypatch.setattr(api, "infer_captured_at", lambda *_a, **_k: "2026-03-01T10:05:00Z")
    monkeypatch.setattr(
        api,
        "enqueue_recording_job",
        lambda *_a, **_k: type("Job", (), {"job_id": "job-upload-1"})(),
    )

    response = calendar_client.post(
        "/api/uploads",
        files={"file": ("roadmap-review.mp3", b"fake-mp3", "audio/mpeg")},
    )
    assert response.status_code == 200
    recording_id = response.json()["recording_id"]
    row = get_calendar_match(recording_id, settings=cfg)
    assert row is not None
    assert row["selected_event_id"] == f"{source['id']}:evt-upload-1:2026-03-01T10:00:00Z"
    assert row["candidates_json"][0]["subject"] == "Roadmap Review"


def test_upload_runtime_ignores_calendar_matching_failures(
    calendar_client: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(api, "refresh_recording_calendar_match", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(api, "infer_captured_at", lambda *_a, **_k: "2026-03-01T10:05:00Z")
    monkeypatch.setattr(
        api,
        "enqueue_recording_job",
        lambda *_a, **_k: type("Job", (), {"job_id": "job-upload-fail"})(),
    )

    response = calendar_client.post(
        "/api/uploads",
        files={"file": ("calendar-fail.mp3", b"fake-mp3", "audio/mpeg")},
    )
    assert response.status_code == 200


def test_calendar_source_rejects_non_http_url(calendar_client: TestClient):
    response = calendar_client.post(
        "/api/calendar/sources",
        json={
            "name": "Bad Calendar",
            "kind": "url",
            "url": "file:///tmp/test.ics",
        },
    )
    assert response.status_code == 422
    assert "http or https" in response.json()["detail"]


def test_calendar_source_rejects_mixed_loopback_dns_answers(calendar_client: TestClient, monkeypatch):
    def _fake_getaddrinfo(*_args, **_kwargs):
        return [
            (0, 0, 0, "", ("127.0.0.1", 443)),
            (0, 0, 0, "", ("203.0.113.10", 443)),
        ]

    monkeypatch.setattr("lan_app.calendar.ics.socket.getaddrinfo", _fake_getaddrinfo)
    response = calendar_client.post(
        "/api/calendar/sources",
        json={
            "name": "Mixed DNS Calendar",
            "kind": "url",
            "url": "https://calendar.example.com/team.ics",
        },
    )
    assert response.status_code == 422
    assert "localhost/loopback" in response.json()["detail"]
