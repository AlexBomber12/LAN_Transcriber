from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from lan_app import db as db_module
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FINISHED,
    JOB_STATUS_QUEUED,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_PUBLISHED,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_QUARANTINE,
)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_db_internal_helpers_and_validation_paths():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT '{' AS json, '{' AS payload_json, 1 AS number").fetchone()
    parsed = db_module._as_dict(row)  # noqa: SLF001
    conn.close()

    assert parsed == {"json": "{", "payload_json": "{", "number": 1}
    assert db_module._as_dict(None) is None  # noqa: SLF001

    with pytest.raises(ValueError, match="Unsupported recording status"):
        db_module._validate_recording_status("bad-status")  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported job status"):
        db_module._validate_job_status("bad-status")  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported job type"):
        db_module._validate_job_type("bad-type")  # noqa: SLF001

    assert db_module._normalise_project_assignment_source("   ") is None  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported project assignment source"):
        db_module._normalise_project_assignment_source("semi-auto")  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported calendar source kind"):
        db_module._normalise_calendar_source_kind("ical")  # noqa: SLF001


def test_with_db_retry_error_paths_and_unreachable(monkeypatch: pytest.MonkeyPatch):
    def _raise_non_lock() -> None:
        raise sqlite3.OperationalError("syntax error")

    with pytest.raises(sqlite3.OperationalError, match="syntax error"):
        db_module.with_db_retry(_raise_non_lock, retries=2, base_sleep_ms=0)

    def _raise_lock() -> None:
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        db_module.with_db_retry(_raise_lock, retries=0, base_sleep_ms=0)

    monkeypatch.setattr(  # noqa: A003
        db_module,
        "range",
        lambda *_args, **_kwargs: (),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="unreachable"):
        db_module.with_db_retry(lambda: "ok", retries=1, base_sleep_ms=0)


def test_migration_file_discovery_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing_dir = tmp_path / "missing-migrations"
    monkeypatch.setattr(db_module, "_MIGRATIONS_DIR", missing_dir)  # noqa: SLF001
    with pytest.raises(FileNotFoundError, match="Migrations directory not found"):
        db_module._migration_files()  # noqa: SLF001

    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    (migrations_dir / "notes.sql").write_text("-- ignored", encoding="utf-8")
    (migrations_dir / "001_init.sql").write_text("SELECT 1;", encoding="utf-8")

    monkeypatch.setattr(db_module, "_MIGRATIONS_DIR", migrations_dir)  # noqa: SLF001
    files = db_module._migration_files()  # noqa: SLF001
    assert files == [(1, migrations_dir / "001_init.sql")]

    (migrations_dir / "001_duplicate.sql").write_text("SELECT 2;", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate migration version"):
        db_module._migration_files()  # noqa: SLF001


def test_executescript_allowing_duplicate_columns_error_paths():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY)")

    with pytest.raises(sqlite3.OperationalError):
        db_module._executescript_allowing_duplicate_columns(conn, "THIS IS NOT SQL;")  # noqa: SLF001

    with pytest.raises(sqlite3.OperationalError):
        db_module._executescript_allowing_duplicate_columns(  # noqa: SLF001
            conn,
            """
            ALTER TABLE sample ADD COLUMN marker TEXT;
            ALTER TABLE sample ADD COLUMN marker TEXT;
            THIS IS BAD SQL;
            """,
        )
    conn.close()


def test_init_db_skips_already_applied_live_migration_and_db_path_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _cfg(tmp_path)
    migration_path = tmp_path / "001_test.sql"
    migration_path.write_text("CREATE TABLE IF NOT EXISTS skip_me (id INTEGER);", encoding="utf-8")

    monkeypatch.setattr(db_module, "_migration_files", lambda: [(1, migration_path)])  # noqa: SLF001
    versions = iter([0, 1])
    monkeypatch.setattr(db_module, "_read_user_version", lambda _conn: next(versions))  # noqa: SLF001
    applied = {"called": False}

    def _fake_executescript(_conn: sqlite3.Connection, _migration_sql: str) -> None:
        applied["called"] = True

    monkeypatch.setattr(  # noqa: SLF001
        db_module,
        "_executescript_allowing_duplicate_columns",
        _fake_executescript,
    )

    assert db_module.init_db(cfg) == cfg.db_path
    assert applied["called"] is False

    env_db = tmp_path / "env-default.db"
    monkeypatch.setenv("LAN_DB_PATH", str(env_db))
    assert db_module.db_path() == env_db


def test_recording_status_helpers_language_settings_and_publish_edges(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    db_module.create_recording(
        "rec-db-cov-1",
        source="upload",
        source_filename="input.wav",
        settings=cfg,
    )

    with pytest.raises(ValueError, match="stage is required"):
        db_module.set_recording_progress(
            "rec-db-cov-1",
            stage="   ",
            progress=0.2,
            settings=cfg,
        )

    assert (
        db_module.set_recording_status_if_current_in(
            "rec-db-cov-1",
            RECORDING_STATUS_PROCESSING,
            current_statuses=[],
            settings=cfg,
        )
        is False
    )
    with pytest.raises(ValueError, match="Unsupported recording status"):
        db_module.set_recording_status_if_current_in(
            "rec-db-cov-1",
            RECORDING_STATUS_PROCESSING,
            current_statuses=["not-a-status"],
            settings=cfg,
        )

    assert db_module.set_recording_status_if_current_in(
        "rec-db-cov-1",
        RECORDING_STATUS_QUARANTINE,
        current_statuses=[RECORDING_STATUS_QUEUED],
        quarantine_reason="manual-check",
        settings=cfg,
    )
    assert (
        db_module.set_recording_status_if_current_in(
            "rec-db-cov-1",
            RECORDING_STATUS_PROCESSING,
            current_statuses=[RECORDING_STATUS_QUEUED],
            settings=cfg,
        )
        is False
    )
    rec = db_module.get_recording("rec-db-cov-1", settings=cfg) or {}
    assert rec["status"] == RECORDING_STATUS_QUARANTINE
    assert rec["quarantine_reason"] == "manual-check"

    assert (
        db_module.set_recording_status_if_current_in_and_no_started_job(
            "rec-db-cov-1",
            RECORDING_STATUS_PROCESSING,
            current_statuses=[],
            settings=cfg,
        )
        is False
    )
    assert (
        db_module.set_recording_status_if_current_in_and_job_started(
            "rec-db-cov-1",
            RECORDING_STATUS_PROCESSING,
            job_id="missing-job",
            current_statuses=[],
            settings=cfg,
        )
        is False
    )

    assert db_module.set_recording_routing_suggestion(
        "rec-db-cov-1",
        suggested_project_id=None,
        routing_confidence=None,
        routing_rationale=[" first ", " ", "second"],
        settings=cfg,
    )
    rec = db_module.get_recording("rec-db-cov-1", settings=cfg) or {}
    assert rec["routing_confidence"] is None
    assert rec["routing_rationale_json"] == ["first", "second"]

    assert db_module.set_recording_language_settings("rec-db-cov-1", settings=cfg) is False
    assert db_module.set_recording_language_settings(
        "rec-db-cov-1",
        language_auto="en",
        settings=cfg,
    )
    assert db_module.set_recording_language_settings(
        "rec-db-cov-1",
        target_summary_language="de",
        settings=cfg,
    )

    with pytest.raises(ValueError, match="onenote_page_id is required"):
        db_module.set_recording_publish_result(
            "rec-db-cov-1",
            onenote_page_id="  ",
            settings=cfg,
        )
    assert db_module.set_recording_publish_result(
        "rec-db-cov-1",
        onenote_page_id="page-1",
        onenote_page_url="   ",
        settings=cfg,
    )
    rec = db_module.get_recording("rec-db-cov-1", settings=cfg) or {}
    assert rec["status"] == RECORDING_STATUS_PUBLISHED
    assert rec["onenote_page_id"] == "page-1"
    assert rec["onenote_page_url"] is None


def test_job_helpers_has_find_requeue_finish_and_invalid_terminal_status(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    db_module.create_recording(
        "rec-db-cov-job-1",
        source="upload",
        source_filename="input.wav",
        settings=cfg,
    )
    db_module.create_job(
        "job-db-cov-1",
        "rec-db-cov-job-1",
        JOB_TYPE_PRECHECK,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
    )

    assert db_module.has_started_job_for_recording("rec-db-cov-job-1", settings=cfg) is False
    active = db_module.find_active_job_for_recording(
        "rec-db-cov-job-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )
    assert active is not None
    assert active["id"] == "job-db-cov-1"
    with pytest.raises(ValueError, match="Unsupported job type"):
        db_module.find_active_job_for_recording(
            "rec-db-cov-job-1",
            job_type="legacy",
            settings=cfg,
        )

    assert db_module.requeue_job("missing-job", settings=cfg) is False
    assert db_module.start_job("job-db-cov-1", settings=cfg)
    assert db_module.has_started_job_for_recording("rec-db-cov-job-1", settings=cfg)
    assert db_module.requeue_job("job-db-cov-1", error="retry once", settings=cfg)
    assert db_module.finish_job("job-db-cov-1", error="done", settings=cfg)

    job = db_module.get_job("job-db-cov-1", settings=cfg) or {}
    assert job["status"] == JOB_STATUS_FINISHED
    assert job["error"] == "done"

    with pytest.raises(ValueError, match="Unsupported terminal state"):
        db_module._set_job_terminal_state(  # noqa: SLF001
            job_id="job-db-cov-1",
            status="not-terminal",
            error=None,
            settings=cfg,
        )


def test_project_keyword_weights_and_voice_profile_crud_edges(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    project = db_module.create_project("Coverage Project", settings=cfg)
    project_id = int(project["id"])

    assert db_module.update_project_onenote_mapping(9999, settings=cfg) is None
    mapped = db_module.update_project_onenote_mapping(
        project_id,
        onenote_notebook_id=" notebook-1 ",
        onenote_section_id="   ",
        settings=cfg,
    )
    assert mapped is not None
    assert mapped["onenote_notebook_id"] == "notebook-1"
    assert mapped["onenote_section_id"] is None

    assert db_module.increment_project_keyword_weights(
        project_id=project_id,
        keyword_deltas={
            "  ": 2.0,
            "alpha": 0.0,
            "beta signal": 1.5,
        },
        settings=cfg,
    ) == 1

    weights_for_project = db_module.list_project_keyword_weights(
        project_id=project_id,
        settings=cfg,
    )
    all_weights = db_module.list_project_keyword_weights(settings=cfg)
    assert len(weights_for_project) == 1
    assert weights_for_project[0]["keyword"] == "beta signal"
    assert any(item["keyword"] == "beta signal" for item in all_weights)

    assert db_module.list_voice_profiles(settings=cfg) == []
    profile = db_module.create_voice_profile(
        "Speaker One",
        notes="notes",
        settings=cfg,
    )
    assert db_module.delete_voice_profile(int(profile["id"]), settings=cfg)
    assert db_module.delete_voice_profile(int(profile["id"]), settings=cfg) is False


def test_speaker_assignments_and_voice_samples_paths(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    db_module.create_recording(
        "rec-db-cov-voice-1",
        source="upload",
        source_filename="voice.wav",
        settings=cfg,
    )
    profile = db_module.create_voice_profile("Voice A", settings=cfg)
    profile_id = int(profile["id"])

    with pytest.raises(ValueError, match="diar_speaker_label is required"):
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="   ",
            voice_profile_id=profile_id,
            settings=cfg,
        )

    assigned = db_module.set_speaker_assignment(
        recording_id="rec-db-cov-voice-1",
        diar_speaker_label="S1",
        voice_profile_id=profile_id,
        confidence=2.0,
        settings=cfg,
    )
    assert assigned is not None
    assert assigned["confidence"] == 1.0
    assert (
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="S1",
            voice_profile_id=None,
            settings=cfg,
        )
        is None
    )

    with pytest.raises(ValueError, match="snippet_path is required"):
        db_module.create_voice_sample(
            voice_profile_id=profile_id,
            snippet_path=" ",
            settings=cfg,
        )

    sample = db_module.create_voice_sample(
        voice_profile_id=profile_id,
        recording_id=" rec-db-cov-voice-1 ",
        diar_speaker_label=" S1 ",
        snippet_path="snippet.wav",
        settings=cfg,
    )
    assert sample["recording_id"] == "rec-db-cov-voice-1"
    assert sample["diar_speaker_label"] == "S1"

    assert db_module.list_voice_samples(voice_profile_id=profile_id, settings=cfg)
    by_recording = db_module.list_voice_samples(
        voice_profile_id=profile_id,
        recording_id="rec-db-cov-voice-1",
        settings=cfg,
    )
    assert len(by_recording) == 1

    loaded = db_module.get_voice_sample(int(sample["id"]), settings=cfg)
    assert loaded is not None
    assert loaded["id"] == sample["id"]


def test_calendar_sources_events_matches_and_metrics_edges(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    db_module.create_recording(
        "rec-db-cov-calendar-1",
        source="upload",
        source_filename="calendar.wav",
        settings=cfg,
    )

    with pytest.raises(ValueError, match="name is required"):
        db_module.create_calendar_source(
            "  ",
            kind="url",
            url="https://example.com/calendar.ics",
            settings=cfg,
        )
    with pytest.raises(ValueError, match="url is required"):
        db_module.create_calendar_source(
            "Team Calendar",
            kind="url",
            url="  ",
            settings=cfg,
        )
    with pytest.raises(ValueError, match="file_ics is required"):
        db_module.create_calendar_source(
            "Team Calendar",
            kind="file",
            file_ics=" ",
            settings=cfg,
        )

    source = db_module.create_calendar_source(
        "Team Calendar",
        kind="url",
        url="https://example.com/calendar.ics",
        file_ics="ignored.ics",
        settings=cfg,
    )
    source_id = int(source["id"])
    assert source["file_ics"] is None

    assert db_module.get_calendar_source(source_id, settings=cfg) is not None
    assert any(
        int(row["id"]) == source_id
        for row in db_module.list_calendar_sources(settings=cfg)
    )

    with pytest.raises(ValueError, match="at least one field"):
        db_module.update_calendar_source_sync_state(source_id, settings=cfg)
    assert (
        db_module.update_calendar_source_sync_state(
            9999,
            last_error="sync failed",
            settings=cfg,
        )
        is None
    )
    updated = db_module.update_calendar_source_sync_state(
        source_id,
        last_synced_at="2026-03-01T10:00:00Z",
        settings=cfg,
    )
    assert updated is not None
    assert updated["last_synced_at"] == "2026-03-01T10:00:00Z"
    updated = db_module.update_calendar_source_sync_state(
        source_id,
        last_error="temporary",
        settings=cfg,
    )
    assert updated is not None
    assert updated["last_error"] == "temporary"

    with pytest.raises(ValueError, match="window_start and window_end are required"):
        db_module.replace_calendar_events_for_window(
            source_id=source_id,
            window_start=" ",
            window_end="2026-03-02T00:00:00Z",
            events=[],
            settings=cfg,
        )

    inserted = db_module.replace_calendar_events_for_window(
        source_id=source_id,
        window_start="2026-03-01T00:00:00Z",
        window_end="2026-03-03T00:00:00Z",
        events=[
            {
                "uid": "",
                "starts_at": "2026-03-01T10:00:00Z",
                "ends_at": "2026-03-01T11:00:00Z",
            },
            {
                "uid": "event-1",
                "starts_at": "2026-03-01T10:00:00Z",
                "ends_at": "2026-03-01T11:00:00Z",
                "summary": " Standup ",
                "description": " notes ",
                "location": " room ",
                "organizer": " owner@example.com ",
                "all_day": True,
            },
        ],
        settings=cfg,
    )
    assert inserted == 1

    with pytest.raises(ValueError, match="starts_from and ends_to are required"):
        db_module.list_calendar_events(
            starts_from=" ",
            ends_to="2026-03-03T00:00:00Z",
            settings=cfg,
        )

    rows = db_module.list_calendar_events(
        starts_from="2026-03-01T00:00:00Z",
        ends_to="2026-03-03T00:00:00Z",
        source_id=source_id,
        settings=cfg,
    )
    assert len(rows) == 1
    assert rows[0]["uid"] == "event-1"

    inserted_match = db_module.set_calendar_match_selection(
        recording_id="rec-db-cov-calendar-1",
        event_id="event-1",
        selected_confidence=0.9,
        settings=cfg,
    )
    assert inserted_match["selected_event_id"] == "event-1"
    updated_match = db_module.set_calendar_match_selection(
        recording_id="rec-db-cov-calendar-1",
        event_id=None,
        selected_confidence=None,
        settings=cfg,
    )
    assert updated_match["selected_event_id"] is None

    assert db_module.get_meeting_metrics("rec-db-cov-calendar-1", settings=cfg) is None
    db_module.upsert_meeting_metrics(
        recording_id="rec-db-cov-calendar-1",
        payload={"score": 1},
        settings=cfg,
    )
    metrics = db_module.get_meeting_metrics("rec-db-cov-calendar-1", settings=cfg)
    assert metrics is not None
    assert metrics["json"] == {"score": 1}

    replaced = db_module.replace_participant_metrics(
        recording_id="rec-db-cov-calendar-1",
        rows=[
            {
                "diar_speaker_label": " ",
                "voice_profile_id": 7,
                "payload": {"skip": True},
            },
            {
                "diar_speaker_label": "S1",
                "voice_profile_id": "not-int",
                "payload": "bad-payload",
            },
            {
                "diar_speaker_label": "S2",
                "voice_profile_id": None,
                "payload": {"words": 42},
            },
        ],
        settings=cfg,
    )
    assert len(replaced) == 2
    listed = db_module.list_participant_metrics("rec-db-cov-calendar-1", settings=cfg)
    assert len(listed) == 2
