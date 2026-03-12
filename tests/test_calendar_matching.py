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
    assert matching._coerce_bool(True) is True  # noqa: SLF001
    assert matching._coerce_bool("1") is True  # noqa: SLF001
    assert matching._coerce_bool("no") is False  # noqa: SLF001
    assert matching._normalize_token_text("Riunione Caf\u00e8") == "riunione cafe"  # noqa: SLF001
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

    corrected = matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_source": "2026-03-01T10:05:00",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 1,
        }
    )
    assert corrected["status"] == "corrected_local"
    assert corrected["warning"] is None

    weak = matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 0,
        }
    )
    assert weak["status"] == "weak_upload_time"
    assert "receipt time" in weak["warning"]

    suspicious = matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_source": "2026-03-01T10:05:00",
            "captured_at_timezone": "UTC",
            "captured_at_inferred_from_filename": 1,
        }
    )
    assert suspicious["status"] == "suspicious"
    assert suspicious["auto_select_mode"] == "blocked"

    assert matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "calendar",
            "captured_at": "2026-03-01T09:05:00Z",
        }
    )["status"] == "stored_utc"

    assert matching._score_event(  # noqa: SLF001
        event={"starts_at": "2026-03-01T11:00:00Z", "ends_at": "2026-03-01T10:00:00Z"},
        captured_at=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        duration_sec=None,
        recording_tokens=set(),
        capture_time_assessment={"status": "stored_utc", "penalty": 0.0},
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
        capture_time_assessment={"status": "stored_utc", "penalty": 0.0},
    )
    assert no_token_score is not None
    assert all("Shared subject tokens" not in row for row in no_token_score["rationale"])
    assert no_token_score["score_details"]["shared_tokens"]["subject"] == []

    no_candidates = matching._auto_selection_decision([], capture_time_assessment=weak)  # noqa: SLF001
    assert no_candidates["reason_code"] == "no_candidates"
    assert matching._auto_selection(  # noqa: SLF001
        [{"event_id": "evt-low", "score": 0.5}],
        capture_time_assessment=weak,
    ) == (None, None)
    warnings = matching.calendar_match_warnings(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 0,
        },
        [{"event_id": "evt-low", "score": 0.5}],
        selected_event_id=None,
    )
    assert len(warnings) == 2


def test_matching_private_helper_branch_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)

    missing_provenance = matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_inferred_from_filename": 1,
        }
    )
    assert missing_provenance["status"] == "suspicious"

    invalid_provenance = matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_source": "not-a-date",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 1,
        }
    )
    assert invalid_provenance["status"] == "suspicious"

    aware_source = matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_source": "2026-03-01T10:05:00+01:00",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 1,
        }
    )
    assert aware_source["status"] == "corrected_local"

    upload_with_source = matching._capture_time_assessment(  # noqa: SLF001
        {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_source": "2026-03-01T10:05:00",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 0,
        }
    )
    assert upload_with_source["status"] == "suspicious"

    assert matching._candidate_score_value({"score": "bad"}) == 0.0  # noqa: SLF001

    exact_start = matching._score_event(  # noqa: SLF001
        event={
            "source_id": 1,
            "uid": "evt-exact",
            "starts_at": "2026-03-01T10:00:00Z",
            "ends_at": "2026-03-01T11:00:00Z",
            "summary": "Exact start",
            "attendees_json": [],
        },
        captured_at=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        duration_sec=1800,
        recording_tokens=set(),
        capture_time_assessment={"status": "stored_utc", "penalty": 0.0},
    )
    assert exact_start is not None
    assert exact_start["rationale"][0] == "Capture starts exactly at the event start."

    single_candidate = matching._auto_selection_decision(  # noqa: SLF001
        [{"event_id": "evt-only", "score": 0.9}],
        capture_time_assessment={"auto_select_mode": "normal", "penalty": 0.0},
    )
    assert single_candidate["selected_event_id"] == "evt-only"

    matching._annotate_candidates([], decision=single_candidate)  # noqa: SLF001
    candidate = {"event_id": "evt-edge", "rationale": "not-a-list"}
    matching._annotate_candidates(  # noqa: SLF001
        [candidate],
        decision={
            "reason_code": "selected",
            "reason": None,
            "min_score": 0.1,
            "min_margin": 0.1,
        },
    )
    assert candidate["auto_select_note"] is None

    no_close_warning = matching.calendar_match_warnings(  # noqa: SLF001
        {
            "source": "calendar",
            "captured_at": "2026-03-01T09:05:00Z",
        },
        [{"event_id": "evt-a", "score": 0.91}, {"event_id": "evt-b", "score": 0.4}],
        selected_event_id=None,
    )
    assert all("too closely" not in row for row in no_close_warning)

    duplicate_warning = "No calendar candidate is strong enough for auto-selection yet. Review the candidates on this tab."
    monkeypatch.setattr(
        matching,
        "_capture_time_assessment",
        lambda *_a, **_k: {
            "warning": duplicate_warning,
            "auto_select_mode": "normal",
            "penalty": 0.0,
        },
    )
    deduped = matching.calendar_match_warnings(  # noqa: SLF001
        {"captured_at": "2026-03-01T09:05:00Z"},
        [{"event_id": "evt-low", "score": 0.2}],
        selected_event_id=None,
    )
    assert deduped == [duplicate_warning]

    create_recording(
        "rec-refresh-upsert",
        source="upload",
        source_filename="refresh.wav",
        settings=cfg,
    )
    monkeypatch.setattr(
        matching,
        "match_recording_to_calendar",
        lambda *_a, **_k: {
            "recording_id": "rec-refresh-upsert",
            "candidates": [{"event_id": "evt-new", "subject": "Fresh"}],
            "selected_event_id": "evt-new",
            "selected_confidence": 0.9,
            "warnings": [],
        },
    )
    refreshed = matching.refresh_recording_calendar_match("rec-refresh-upsert", settings=cfg)
    assert refreshed["selected_event_id"] == "evt-new"


def test_match_recording_to_calendar_uses_corrected_upload_capture_time(tmp_path: Path) -> None:
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
                "starts_at": "2026-03-01T09:00:00Z",
                "ends_at": "2026-03-01T10:00:00Z",
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
                "starts_at": "2026-03-01T12:00:00Z",
                "ends_at": "2026-03-01T13:00:00Z",
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
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_source": "2026-03-01T10:05:00",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 1,
            "duration_sec": 3300,
            "source_filename": "Roadmap review.wav",
        },
        settings=cfg,
    )

    assert result["selected_event_id"] == f"{source['id']}:evt-roadmap:2026-03-01T09:00:00Z"
    assert result["selected_confidence"] == result["candidates"][0]["score"]
    assert result["warnings"] == []
    assert result["candidates"][0]["attendees"] == ["Priya Kapoor"]
    assert result["candidates"][0]["attendee_details"][0]["email"] == "priya@example.com"
    assert result["candidates"][0]["source_name"] == "Team Calendar"
    assert result["candidates"][0]["capture_time_status"] == "corrected_local"
    assert "corrected filename timestamp" in " ".join(result["candidates"][0]["rationale"])
    assert result["candidates"][0]["score_details"]["shared_tokens"]["subject"] == [
        "review",
        "roadmap",
    ]
    assert result["candidates"][0]["score"] > result["candidates"][1]["score"]


def test_match_recording_to_calendar_scores_prestart_overlap_well(tmp_path: Path) -> None:
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
                "uid": "evt-target",
                "starts_at": "2026-03-01T10:00:00Z",
                "ends_at": "2026-03-01T11:00:00Z",
                "summary": "Planning session",
                "updated_at": "2026-03-01T00:00:00Z",
            },
            {
                "uid": "evt-earlier",
                "starts_at": "2026-03-01T09:00:00Z",
                "ends_at": "2026-03-01T09:45:00Z",
                "summary": "Earlier session",
                "updated_at": "2026-03-01T00:00:00Z",
            },
        ],
        settings=cfg,
    )

    result = matching.match_recording_to_calendar(
        {
            "id": "rec-prestart",
            "source": "upload",
            "captured_at": "2026-03-01T09:55:00Z",
            "captured_at_source": "2026-03-01T10:55:00",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 1,
            "duration_sec": 4200,
            "source_filename": "session.wav",
        },
        settings=cfg,
    )

    assert result["selected_event_id"] == f"{source['id']}:evt-target:2026-03-01T10:00:00Z"
    assert result["candidates"][0]["score"] > 0.79
    assert "shortly before the event begins" in " ".join(result["candidates"][0]["rationale"])
    assert result["candidates"][0]["score_details"]["event_overlap_ratio"] == 1.0


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
            "source": "calendar",
            "captured_at": "2026-03-01T10:05:00Z",
            "source_filename": "planning.wav",
        },
        settings=cfg,
    )

    assert len(result["candidates"]) == 2
    assert result["selected_event_id"] is None
    assert result["selected_confidence"] is None
    assert result["candidates"][0]["auto_select_reason"] == "margin_too_small"
    assert result["candidates"][0]["margin_to_next"] < result["candidates"][0]["auto_select_min_margin"]
    assert any("Multiple nearby calendar candidates" in warning for warning in result["warnings"])
    assert matching.match_recording_to_calendar(  # noqa: SLF001
        {"id": "rec-no-capture", "captured_at": None},
        settings=cfg,
    ) == {
        "recording_id": "rec-no-capture",
        "candidates": [],
        "selected_event_id": None,
        "selected_confidence": None,
        "warnings": [
            "This recording has no usable capture timestamp, so calendar matching needs manual review."
        ],
    }


def test_strong_overlap_can_outrank_slightly_closer_start_time(tmp_path: Path) -> None:
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
                "uid": "evt-close-short",
                "starts_at": "2026-03-01T10:02:00Z",
                "ends_at": "2026-03-01T10:20:00Z",
                "summary": "Quick check-in",
                "updated_at": "2026-03-01T00:00:00Z",
            },
            {
                "uid": "evt-roadmap",
                "starts_at": "2026-03-01T10:08:00Z",
                "ends_at": "2026-03-01T11:08:00Z",
                "summary": "Roadmap workshop",
                "updated_at": "2026-03-01T00:00:00Z",
            },
        ],
        settings=cfg,
    )

    result = matching.match_recording_to_calendar(
        {
            "id": "rec-overlap-wins",
            "source": "calendar",
            "captured_at": "2026-03-01T10:00:00Z",
            "duration_sec": 3600,
            "source_filename": "roadmap.wav",
        },
        settings=cfg,
    )

    assert result["selected_event_id"] == f"{source['id']}:evt-roadmap:2026-03-01T10:08:00Z"
    assert result["candidates"][0]["uid"] == "evt-roadmap"
    assert result["candidates"][0]["score"] > result["candidates"][1]["score"]
    assert (
        result["candidates"][0]["score_details"]["recording_overlap_ratio"]
        > result["candidates"][1]["score_details"]["recording_overlap_ratio"]
    )


def test_weak_and_suspicious_capture_time_trigger_conservative_behavior(tmp_path: Path) -> None:
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
                "starts_at": "2026-03-01T09:00:00Z",
                "ends_at": "2026-03-01T10:00:00Z",
                "summary": "Roadmap review",
                "updated_at": "2026-03-01T00:00:00Z",
            }
        ],
        settings=cfg,
    )

    weak = matching.match_recording_to_calendar(
        {
            "id": "rec-weak",
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 0,
            "source_filename": "meeting.wav",
        },
        settings=cfg,
    )
    assert weak["selected_event_id"] is None
    assert weak["candidates"][0]["capture_time_status"] == "weak_upload_time"
    assert any("receipt time" in warning for warning in weak["warnings"])
    assert any("strong enough for auto-selection" in warning for warning in weak["warnings"])

    suspicious = matching.match_recording_to_calendar(
        {
            "id": "rec-suspicious",
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_source": "2026-03-01T10:05:00",
            "captured_at_timezone": "UTC",
            "captured_at_inferred_from_filename": 1,
            "duration_sec": 3300,
            "source_filename": "Roadmap review.wav",
        },
        settings=cfg,
    )
    assert suspicious["selected_event_id"] is None
    assert suspicious["candidates"][0]["capture_time_status"] == "suspicious"
    assert suspicious["candidates"][0]["auto_select_reason"] == "blocked_suspicious_time"
    assert any("intentionally conservative" in warning for warning in suspicious["warnings"])


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
            "warnings": [],
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
