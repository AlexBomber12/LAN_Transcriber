from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lan_app.calendar import matching
from lan_app.config import AppSettings
from lan_app.db import (
    create_calendar_source,
    create_recording,
    get_calendar_match,
    init_db,
    replace_calendar_events_for_window,
    upsert_calendar_match,
)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_matching_private_helpers_cover_edge_cases() -> None:
    assert matching._parse_datetime(None) is None  # noqa: SLF001
    assert matching._parse_datetime("2026-03-01T10:00:00Z") == datetime(  # noqa: SLF001
        2026,
        3,
        1,
        10,
        0,
        tzinfo=timezone.utc,
    )
    assert matching._parse_datetime("2026-03-01T10:00:00") == datetime(  # noqa: SLF001
        2026,
        3,
        1,
        10,
        0,
        tzinfo=timezone.utc,
    )
    assert matching._parse_datetime("bad") is None  # noqa: SLF001
    assert matching._safe_duration_seconds("12.5") == 12.5  # noqa: SLF001
    assert matching._safe_duration_seconds(0) is None  # noqa: SLF001
    assert matching._tokenize("The Roadmap sync audio") == {"roadmap"}  # noqa: SLF001
    assert matching._recording_tokens({"source_filename": "Roadmap Review.wav"}) == {  # noqa: SLF001
        "review",
        "roadmap",
    }
    assert matching._candidate_rows("{bad") == []  # noqa: SLF001
    assert matching._candidate_rows("{}") == []  # noqa: SLF001
    assert matching._candidate_rows([{"event_id": "evt-1"}, "skip"]) == [  # noqa: SLF001
        {"event_id": "evt-1"}
    ]
    details = matching._candidate_attendee_details(  # noqa: SLF001
        [
            {"name": "Alex", "email": "alex@example.com"},
            {"label": "Priya"},
            "Priya",
            "",
        ]
    )
    assert details == [
        {
            "label": "Alex",
            "name": "Alex",
            "email": "alex@example.com",
        },
        {"label": "Priya"},
    ]
    assert matching._candidate_attendees({"attendees": [" Alex ", " "]}) == ["Alex"]  # noqa: SLF001
    assert matching._candidate_attendees(  # noqa: SLF001
        {"attendees": [" ", ""], "attendee_details": details}
    ) == ["Alex", "Priya"]
    assert matching._candidate_attendees({"attendee_details": details}) == [  # noqa: SLF001
        "Alex",
        "Priya",
    ]
    assert matching._score_event(  # noqa: SLF001
        event={"starts_at": "2026-03-01T11:00:00Z", "ends_at": "2026-03-01T10:00:00Z"},
        captured_at=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        duration_sec=None,
        recording_tokens=set(),
    ) is None
    no_token_score = matching._score_event(  # noqa: SLF001
        event={
            "source_id": 1,
            "uid": "evt-no-token",
            "starts_at": "2026-03-01T10:00:00Z",
            "ends_at": "2026-03-01T11:00:00Z",
            "summary": "Budget review",
            "attendees_json": [],
        },
        captured_at=datetime(2026, 3, 1, 10, 5, tzinfo=timezone.utc),
        duration_sec=None,
        recording_tokens={"roadmap"},
    )
    assert no_token_score is not None
    assert all("Shared filename/title tokens" not in row for row in no_token_score["rationale"])
    assert matching._auto_selection([{"event_id": "evt-low", "score": 0.5}]) == (None, None)  # noqa: SLF001


def test_match_recording_to_calendar_scores_and_auto_selects(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
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
                "uid": "evt-roadmap",
                "starts_at": "2026-03-01T10:00:00Z",
                "ends_at": "2026-03-01T11:00:00Z",
                "summary": "Roadmap review",
                "organizer": "Alex",
                "organizer_name": "Alex",
                "organizer_email": "alex@example.com",
                "attendees": [
                    {"name": "Priya Kapoor", "email": "priya@example.com", "label": "Priya Kapoor"}
                ],
                "updated_at": "2026-03-01T00:00:00Z",
            },
            {
                "uid": "evt-budget",
                "starts_at": "2026-03-01T13:00:00Z",
                "ends_at": "2026-03-01T14:00:00Z",
                "summary": "Budget review",
                "organizer": "Jordan",
                "attendees": [],
                "updated_at": "2026-03-01T00:00:00Z",
            },
        ],
        settings=cfg,
    )

    result = matching.match_recording_to_calendar(
        {
            "id": "rec-match-1",
            "captured_at": "2026-03-01T10:05:00Z",
            "duration_sec": 3300,
            "source_filename": "Roadmap review.wav",
        },
        settings=cfg,
    )

    assert result["selected_event_id"] == f"{source['id']}:evt-roadmap:2026-03-01T10:00:00Z"
    assert result["selected_confidence"] == result["candidates"][0]["score"]
    assert result["candidates"][0]["attendees"] == ["Priya Kapoor"]
    assert result["candidates"][0]["attendee_details"][0]["email"] == "priya@example.com"
    assert result["candidates"][0]["source_name"] == "Team Calendar"
    assert "Shared filename/title tokens" in " ".join(result["candidates"][0]["rationale"])
    assert result["candidates"][0]["score"] > result["candidates"][1]["score"]


def test_match_recording_to_calendar_keeps_ambiguous_candidates_unselected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
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
                "uid": "evt-a",
                "starts_at": "2026-03-01T10:00:00Z",
                "ends_at": "2026-03-01T11:00:00Z",
                "summary": "Planning session",
                "updated_at": "2026-03-01T00:00:00Z",
            },
            {
                "uid": "evt-b",
                "starts_at": "2026-03-01T10:03:00Z",
                "ends_at": "2026-03-01T11:03:00Z",
                "summary": "Planning session",
                "updated_at": "2026-03-01T00:00:00Z",
            },
        ],
        settings=cfg,
    )

    result = matching.match_recording_to_calendar(
        {
            "id": "rec-match-ambiguous",
            "captured_at": "2026-03-01T10:05:00Z",
            "source_filename": "planning.wav",
        },
        settings=cfg,
    )

    assert len(result["candidates"]) == 2
    assert result["selected_event_id"] is None
    assert result["selected_confidence"] is None
    assert matching.match_recording_to_calendar(  # noqa: SLF001
        {"id": "rec-no-capture", "captured_at": None},
        settings=cfg,
    ) == {
        "recording_id": "rec-no-capture",
        "candidates": [],
        "selected_event_id": None,
        "selected_confidence": None,
    }


def test_selected_calendar_helpers_and_refresh_cover_existing_and_missing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-match-refresh-1",
        source="upload",
        source_filename="meeting.wav",
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-match-refresh-1",
        candidates=[
            {
                "event_id": "evt-kept",
                "subject": "Kept meeting",
                "attendee_details": [{"label": "Alex Example"}],
            }
        ],
        selected_event_id="evt-kept",
        selected_confidence=0.8,
        settings=cfg,
    )

    monkeypatch.setattr(
        matching,
        "match_recording_to_calendar",
        lambda *_a, **_k: {
            "recording_id": "rec-match-refresh-1",
            "candidates": [],
            "selected_event_id": None,
            "selected_confidence": None,
        },
    )
    refreshed = matching.refresh_recording_calendar_match(
        "rec-match-refresh-1",
        settings=cfg,
    )
    assert refreshed["selected_event_id"] == "evt-kept"
    assert matching.calendar_match_candidates("rec-match-refresh-1", settings=cfg) == [
        {
            "event_id": "evt-kept",
            "subject": "Kept meeting",
            "attendee_details": [{"label": "Alex Example"}],
        }
    ]
    assert matching.selected_calendar_candidate("rec-match-refresh-1", settings=cfg) == {
        "event_id": "evt-kept",
        "subject": "Kept meeting",
        "attendee_details": [{"label": "Alex Example"}],
        "attendees": ["Alex Example"],
    }
    assert matching.calendar_summary_context("rec-match-refresh-1", settings=cfg) == (
        "Kept meeting",
        ["Alex Example"],
    )

    upsert_calendar_match(
        recording_id="rec-match-refresh-1",
        candidates=[],
        selected_event_id=None,
        selected_confidence=None,
        settings=cfg,
    )
    row = get_calendar_match("rec-match-refresh-1", settings=cfg)
    assert row is not None
    assert row["candidates_json"] == []

    upsert_calendar_match(
        recording_id="rec-match-refresh-1",
        candidates=[
            {"event_id": "evt-other", "subject": "Other"},
            {"event_id": "evt-selected", "summary": "Selected later", "attendees": ["Priya"]},
        ],
        selected_event_id="evt-selected",
        selected_confidence=0.7,
        settings=cfg,
    )
    assert matching.selected_calendar_candidate("rec-match-refresh-1", settings=cfg) == {
        "event_id": "evt-selected",
        "summary": "Selected later",
        "attendees": ["Priya"],
    }

    monkeypatch.setattr(
        matching,
        "get_calendar_match",
        lambda *_a, **_k: {"selected_event_id": "evt-1", "candidates_json": "{bad"},
    )
    assert matching.calendar_match_candidates("rec-bad", settings=cfg) == []
    assert matching.selected_calendar_candidate("rec-bad", settings=cfg) == {}
    assert matching.calendar_summary_context("rec-bad", settings=cfg) == (None, [])

    with pytest.raises(KeyError):
        matching.refresh_recording_calendar_match("missing-recording", settings=cfg)
