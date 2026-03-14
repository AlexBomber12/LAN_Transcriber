from __future__ import annotations

import json
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
    RECORDING_STATUS_READY,
    RECORDING_STATUS_STOPPING,
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
    assert db_module._clamp_score("bad", default=0.3) == 0.3  # noqa: SLF001
    assert db_module._normalise_candidate_matches(  # noqa: SLF001
        [
            "skip",
            {"voice_profile_id": "bad"},
            {"voice_profile_id": "7", "score": "bad"},
            {"voice_profile_id": 7, "score": 0.5, "display_name": "Voice 7"},
        ]
    ) == [{"voice_profile_id": 7, "score": 0.5, "display_name": "Voice 7"}]
    assert db_module._rewrite_voice_profile_ids_for_merge(  # noqa: SLF001
        "{bad-json",
        source_profile_id=1,
        target_profile_id=2,
    ) == []
    assert db_module._rewrite_voice_profile_ids_for_merge(  # noqa: SLF001
        123,
        source_profile_id=1,
        target_profile_id=2,
    ) == []
    assert db_module._rewrite_voice_profile_ids_for_merge(  # noqa: SLF001
        [1, "bad", 3],
        source_profile_id=1,
        target_profile_id=2,
    ) == [2, 3]
    assert db_module._rewrite_candidate_matches_for_merge(  # noqa: SLF001
        "{bad-json",
        source_profile_id=1,
        target_profile_id=2,
    ) == []
    assert db_module._rewrite_candidate_matches_for_merge(  # noqa: SLF001
        123,
        source_profile_id=1,
        target_profile_id=2,
    ) == []
    assert db_module._rewrite_candidate_matches_for_merge(  # noqa: SLF001
        ["skip", {"voice_profile_id": 1, "score": 0.9}, {"voice_profile_id": "bad"}],
        source_profile_id=1,
        target_profile_id=2,
    ) == [{"voice_profile_id": 2, "score": 0.9}]
    assert db_module._sqlite_like_query(" A_B% ") == r"%A\_B\%%"  # noqa: SLF001
    assert db_module._sqlite_like_query(" Straße ") == "%Straße%"  # noqa: SLF001
    assert db_module._sqlite_casefold(None) == ""  # noqa: SLF001
    assert db_module._sqlite_casefold("Ärger") == "ärger"  # noqa: SLF001

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


def test_recording_cancel_and_finish_if_queued_helpers_cover_false_paths(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    db_module.create_recording(
        "rec-cancel-db-1",
        source="test",
        source_filename="cancel.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    db_module.create_job(
        "job-cancel-db-1",
        recording_id="rec-cancel-db-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_FINISHED,
    )

    assert (
        db_module.acknowledge_recording_cancel_request(
            "rec-cancel-db-1",
            reason_text="Cancelled by user",
            settings=cfg,
        )
        is False
    )
    assert db_module.finish_job_if_queued("job-cancel-db-1", settings=cfg) is False

    assert db_module.set_recording_cancel_request(
        "rec-cancel-db-1",
        requested_by="user",
        reason_code="user_stop",
        reason_text="Stop requested by user",
        settings=cfg,
    )
    assert db_module.set_recording_status(
        "rec-cancel-db-1",
        RECORDING_STATUS_STOPPING,
        settings=cfg,
    )
    assert (
        db_module.acknowledge_recording_cancel_request(
            "rec-cancel-db-1",
            reason_text="Cancelled by user",
            settings=cfg,
        )
        is True
    )


def test_list_recordings_q_search_is_conservative_and_escaped(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    db_module.create_recording(
        "rec-db-search-1",
        source="upload",
        source_filename="Budget_100%.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    db_module.create_recording(
        "rec-db-search-2",
        source="upload",
        source_filename="notes.wav",
        status=RECORDING_STATUS_PUBLISHED,
        settings=cfg,
    )
    db_module.create_recording(
        "rec-db-search-3",
        source="upload",
        source_filename="Straße.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    db_module.create_recording(
        "rec-db-search-4",
        source="upload",
        source_filename="Ärger.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    by_filename, total_by_filename = db_module.list_recordings(
        settings=cfg,
        q="Budget_100%",
    )
    assert total_by_filename == 1
    assert [row["id"] for row in by_filename] == ["rec-db-search-1"]

    by_id, total_by_id = db_module.list_recordings(
        settings=cfg,
        q="search-2",
    )
    assert total_by_id == 1
    assert [row["source_filename"] for row in by_id] == ["notes.wav"]

    by_source, total_by_source = db_module.list_recordings(
        settings=cfg,
        q="upload",
    )
    assert total_by_source == 0
    assert by_source == []

    by_unicode_filename, total_by_unicode_filename = db_module.list_recordings(
        settings=cfg,
        q="straße",
    )
    assert total_by_unicode_filename == 1
    assert [row["id"] for row in by_unicode_filename] == ["rec-db-search-3"]

    by_unicode_fragment, total_by_unicode_fragment = db_module.list_recordings(
        settings=cfg,
        q="ß",
    )
    assert total_by_unicode_fragment == 1
    assert [row["source_filename"] for row in by_unicode_fragment] == ["Straße.wav"]

    by_upper_unicode, total_by_upper_unicode = db_module.list_recordings(
        settings=cfg,
        q="Ä",
    )
    assert total_by_upper_unicode == 1
    assert [row["id"] for row in by_upper_unicode] == ["rec-db-search-4"]

    by_casefold_unicode, total_by_casefold_unicode = db_module.list_recordings(
        settings=cfg,
        q="är",
    )
    assert total_by_casefold_unicode == 1
    assert [row["source_filename"] for row in by_casefold_unicode] == ["Ärger.wav"]


def test_glossary_entry_helpers_cover_crud_and_validation_paths(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)

    created = db_module.create_glossary_entry(
        "  Sander  ",
        aliases=["Sandia", " sandia ", "Sander"],
        term_kind="person",
        source="correction",
        enabled=True,
        notes="  common mis-hearing  ",
        metadata={"recording_id": "rec-glossary-1"},
        settings=cfg,
    )
    assert created["canonical_text"] == "Sander"
    assert created["aliases_json"] == ["Sandia"]
    assert created["kind"] == "person"
    assert created["source"] == "correction"
    assert created["metadata_json"] == {"recording_id": "rec-glossary-1"}

    listed = db_module.list_glossary_entries(settings=cfg)
    assert [row["canonical_text"] for row in listed] == ["Sander"]
    assert db_module.list_glossary_entries(source="correction", settings=cfg)
    assert db_module.list_glossary_entries(enabled=False, settings=cfg) == []

    updated = db_module.update_glossary_entry(
        int(created["id"]),
        canonical_text="Sander Van Doorn",
        aliases="Sandia\nSandoor\nSander Van Doorn",
        term_kind="person",
        source="manual",
        enabled=False,
        notes="updated",
        metadata={"recording_id": "rec-glossary-2"},
        settings=cfg,
    )
    assert updated is not None
    assert updated["canonical_text"] == "Sander Van Doorn"
    assert updated["aliases_json"] == ["Sandia", "Sandoor"]
    assert updated["source"] == "manual"
    assert updated["enabled"] == 0
    assert updated["notes"] == "updated"
    assert updated["metadata_json"] == {"recording_id": "rec-glossary-2"}

    fetched = db_module.get_glossary_entry(int(created["id"]), settings=cfg)
    assert fetched == updated
    assert db_module.list_glossary_entries(enabled=True, settings=cfg) == []
    assert db_module.list_glossary_entries(enabled=False, settings=cfg)
    assert db_module.update_glossary_entry(999, settings=cfg) is None
    assert db_module.delete_glossary_entry(int(created["id"]), settings=cfg) is True
    assert db_module.get_glossary_entry(int(created["id"]), settings=cfg) is None
    assert db_module.delete_glossary_entry(int(created["id"]), settings=cfg) is False

    assert db_module._normalise_glossary_aliases(  # noqa: SLF001
        123,
        canonical_text="Sander",
    ) == []
    assert db_module._normalise_glossary_aliases(  # noqa: SLF001
        ["", "Sandia"],
        canonical_text="Sander",
    ) == ["Sandia"]
    assert db_module.create_glossary_entry(
        "fallback-meta",
        aliases=None,
        metadata="bad",  # type: ignore[arg-type]
        settings=cfg,
    )["metadata_json"] == {}

    with pytest.raises(ValueError, match="canonical_text is required"):
        db_module.create_glossary_entry("   ", settings=cfg)
    created_for_error = db_module.create_glossary_entry("Editable", settings=cfg)
    with pytest.raises(ValueError, match="canonical_text is required"):
        db_module.update_glossary_entry(
            int(created_for_error["id"]),
            canonical_text="   ",
            settings=cfg,
        )
    with pytest.raises(ValueError, match="Unsupported glossary kind"):
        db_module.create_glossary_entry("Bad Kind", term_kind="alias", settings=cfg)
    with pytest.raises(ValueError, match="Unsupported glossary source"):
        db_module.list_glossary_entries(source="seeded", settings=cfg)


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


def test_llm_chunk_state_helpers_and_pipeline_clear_reset(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    db_module.create_recording(
        "rec-db-cov-chunks-1",
        source="upload",
        source_filename="input.wav",
        settings=cfg,
    )

    planned = db_module.upsert_recording_llm_chunk_state(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        chunk_index="1",
        chunk_total=2,
        status="planned",
        metadata={"order_path": [1], "text": "chunk one", "base_text": "chunk one"},
        settings=cfg,
    )
    assert planned is not None
    started = db_module.mark_recording_llm_chunk_started(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        chunk_index="1",
        chunk_total=2,
        metadata={"order_path": [1], "text": "chunk one", "base_text": "chunk one"},
        settings=cfg,
    )
    assert started is not None
    assert started["attempt"] == 1
    completed = db_module.mark_recording_llm_chunk_completed(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        chunk_index="1",
        chunk_total=2,
        metadata={"order_path": [1], "text": "chunk one", "base_text": "chunk one"},
        settings=cfg,
    )
    assert completed is not None
    assert completed["status"] == "completed"

    failed = db_module.mark_recording_llm_chunk_failed(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        chunk_index="2",
        chunk_total=2,
        error_code="llm_chunk_timeout",
        error_text="timed out",
        parent_chunk_index="1",
        metadata={"order_path": [2], "text": "chunk two", "base_text": "chunk two"},
        settings=cfg,
    )
    assert failed is not None
    assert failed["status"] == "failed"

    split = db_module.mark_recording_llm_chunk_split(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        chunk_index="3",
        chunk_total=3,
        error_code="llm_chunk_timeout",
        error_text="request timed out",
        metadata={
            "order_path": [3],
            "text": "chunk three",
            "base_text": "chunk three",
            "split_child_chunk_indexes": ["3a", "3b"],
        },
        settings=cfg,
    )
    assert split is not None
    assert split["status"] == "split"
    cancelled = db_module.mark_recording_llm_chunk_cancelled(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        chunk_index="4",
        chunk_total=4,
        metadata={"order_path": [4], "text": "chunk four", "base_text": "chunk four"},
        settings=cfg,
    )
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"

    rows = db_module.list_recording_llm_chunk_states(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        settings=cfg,
    )
    assert [row["chunk_index"] for row in rows] == ["1", "2", "3", "4"]
    assert rows[2]["metadata_json"]["split_child_chunk_indexes"] == ["3a", "3b"]

    assert db_module.clear_recording_pipeline_stages(
        "rec-db-cov-chunks-1",
        from_stage="metrics",
        settings=cfg,
    ) == 0
    assert len(
        db_module.list_recording_llm_chunk_states(
            "rec-db-cov-chunks-1",
            chunk_group="extract",
            settings=cfg,
        )
    ) == 4

    assert db_module.clear_recording_pipeline_stages(
        "rec-db-cov-chunks-1",
        from_stage="llm_extract",
        settings=cfg,
    ) == 0
    assert (
        db_module.list_recording_llm_chunk_states(
            "rec-db-cov-chunks-1",
            chunk_group="extract",
            settings=cfg,
        )
        == []
    )

    with pytest.raises(ValueError, match="chunk_group is required"):
        db_module._normalise_llm_chunk_group("  ")  # noqa: SLF001
    with pytest.raises(ValueError, match="chunk_index is required"):
        db_module._normalise_llm_chunk_index(" ")  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported LLM chunk status"):
        db_module._validate_llm_chunk_status("bad")  # noqa: SLF001
    with pytest.raises(ValueError, match="Terminal chunk status must not be running"):
        db_module._mark_recording_llm_chunk_terminal(  # noqa: SLF001
            "rec-db-cov-chunks-1",
            chunk_group="extract",
            chunk_index="5",
            chunk_total=1,
            status="running",
            settings=cfg,
        )

    upsert_recording = db_module.upsert_recording_llm_chunk_state(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        chunk_index="6",
        chunk_total=1,
        status="planned",
        metadata={"order_path": [6], "text": "chunk six", "base_text": "chunk six"},
        settings=cfg,
    )
    assert upsert_recording is not None
    assert db_module.clear_recording_llm_chunk_states(
        "rec-db-cov-chunks-1",
        chunk_group="extract",
        settings=cfg,
    ) == 1


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
    with pytest.raises(ValueError, match="Unsupported speaker review_state"):
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="S0",
            voice_profile_id=profile_id,
            review_state="bad",
            settings=cfg,
        )
    with pytest.raises(ValueError, match="confirmed_canonical requires voice_profile_id"):
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="S0",
            voice_profile_id=None,
            review_state="confirmed_canonical",
            settings=cfg,
        )
    with pytest.raises(ValueError, match="local_label requires local_display_name"):
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="S0",
            voice_profile_id=None,
            review_state="local_label",
            settings=cfg,
        )
    with pytest.raises(
        ValueError,
        match="local_display_name cannot be combined with voice_profile_id",
    ):
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="S0",
            voice_profile_id=profile_id,
            local_display_name="Guest",
            settings=cfg,
        )
    with pytest.raises(ValueError, match="local_label cannot set voice_profile_id"):
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="S0",
            voice_profile_id=profile_id,
            review_state="local_label",
            local_display_name="Guest",
            settings=cfg,
        )
    with pytest.raises(ValueError, match="kept_unknown cannot set voice_profile_id"):
        db_module.set_speaker_assignment(
            recording_id="rec-db-cov-voice-1",
            diar_speaker_label="S0",
            voice_profile_id=profile_id,
            review_state="kept_unknown",
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
    assert assigned["review_state"] == "confirmed_canonical"
    assert assigned["local_display_name"] is None
    unmatched = db_module.set_speaker_assignment(
        recording_id="rec-db-cov-voice-1",
        diar_speaker_label="S2",
        voice_profile_id=None,
        confidence=0.42,
        candidate_matches=[
            {"voice_profile_id": profile_id, "score": 0.42, "display_name": "Voice A"},
            {"voice_profile_id": "bad"},
            {"voice_profile_id": profile_id, "score": 0.4},
        ],
        low_confidence=True,
        keep_unmatched=True,
        settings=cfg,
    )
    assert unmatched is not None
    assert unmatched["voice_profile_id"] is None
    assert unmatched["low_confidence"] == 0
    assert unmatched["review_state"] == "kept_unknown"
    assert unmatched["candidate_matches_json"] == [
        {"voice_profile_id": profile_id, "score": 0.42, "display_name": "Voice A"}
    ]
    local_only = db_module.set_speaker_assignment(
        recording_id="rec-db-cov-voice-1",
        diar_speaker_label="S3",
        voice_profile_id=None,
        local_display_name="Meeting Guest",
        settings=cfg,
    )
    assert local_only is not None
    assert local_only["review_state"] == "local_label"
    assert local_only["local_display_name"] == "Meeting Guest"
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
    review_sample = db_module.create_voice_sample(
        voice_profile_id=None,
        recording_id="rec-db-cov-voice-1",
        diar_speaker_label="S2",
        snippet_path="snippet-review.wav",
        sample_source="auto",
        sample_start_sec=1.25,
        sample_end_sec=2.5,
        embedding=[0.1, "bad", 0.2],
        candidate_matches=[{"voice_profile_id": profile_id, "score": 0.41}],
        needs_review=True,
        confidence=0.41,
        settings=cfg,
    )
    assert review_sample["voice_profile_id"] is None
    assert review_sample["sample_source"] == "auto"
    assert review_sample["embedding_json"] == [0.1, 0.2]
    assert review_sample["candidate_matches_json"] == [
        {"voice_profile_id": profile_id, "score": 0.41}
    ]
    assert review_sample["needs_review"] == 1

    assert db_module.list_voice_samples(voice_profile_id=profile_id, settings=cfg)
    by_recording = db_module.list_voice_samples(
        voice_profile_id=profile_id,
        recording_id="rec-db-cov-voice-1",
        settings=cfg,
    )
    assert len(by_recording) == 1
    all_recording_samples = db_module.list_voice_samples(
        recording_id="rec-db-cov-voice-1",
        settings=cfg,
    )
    assert len(all_recording_samples) == 2

    loaded = db_module.get_voice_sample(int(sample["id"]), settings=cfg)
    assert loaded is not None
    assert loaded["id"] == sample["id"]


def test_canonical_speaker_migration_from_legacy_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _cfg(tmp_path)
    all_migrations = db_module._migration_files()  # noqa: SLF001
    legacy_migrations = [item for item in all_migrations if item[0] < 15]

    monkeypatch.setattr(db_module, "_migration_files", lambda: legacy_migrations)  # noqa: SLF001
    db_module.init_db(cfg)
    with db_module.connect(cfg) as conn:
        conn.execute(
            """
            INSERT INTO recordings (
                id,
                source,
                source_filename,
                captured_at,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rec-db-cov-legacy-1",
                "upload",
                "legacy.wav",
                "2026-03-01T10:00:00Z",
                "Queued",
                "2026-03-01T10:00:00Z",
                "2026-03-01T10:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO voice_profiles (id, display_name, notes) VALUES (?, ?, ?)",
            (1, "Legacy Voice", "notes"),
        )
        conn.execute(
            """
            INSERT INTO speaker_assignments (
                recording_id,
                diar_speaker_label,
                voice_profile_id,
                confidence
            )
            VALUES (?, ?, ?, ?)
            """,
            ("rec-db-cov-legacy-1", "S1", 1, 0.93),
        )
        conn.execute(
            """
            INSERT INTO voice_samples (
                id,
                voice_profile_id,
                recording_id,
                diar_speaker_label,
                snippet_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                1,
                "rec-db-cov-legacy-1",
                "S1",
                "recordings/rec-db-cov-legacy-1/derived/snippets/S1/1.wav",
                "2026-03-01T10:00:00Z",
            ),
        )
        conn.commit()

    monkeypatch.setattr(db_module, "_migration_files", lambda: all_migrations)  # noqa: SLF001
    db_module.init_db(cfg)

    profile = db_module.get_voice_profile(1, include_merged=True, settings=cfg)
    assert profile is not None
    assert profile["display_name"] == "Legacy Voice"
    assert profile["created_at"] == "2026-03-01T10:00:00Z"
    assert profile["updated_at"] == "2026-03-01T10:00:00Z"
    assignment = db_module.list_speaker_assignments(
        "rec-db-cov-legacy-1",
        settings=cfg,
    )
    assert len(assignment) == 1
    assert assignment[0]["recording_id"] == "rec-db-cov-legacy-1"
    assert assignment[0]["diar_speaker_label"] == "S1"
    assert assignment[0]["voice_profile_id"] == 1
    assert assignment[0]["confidence"] == 0.93
    assert assignment[0]["candidate_matches_json"] == []
    assert assignment[0]["low_confidence"] == 0
    assert assignment[0]["review_state"] == "confirmed_canonical"
    assert assignment[0]["local_display_name"] is None
    assert assignment[0]["voice_profile_name"] == "Legacy Voice"
    assert assignment[0]["updated_at"]
    sample = db_module.get_voice_sample(1, settings=cfg)
    assert sample is not None
    assert sample["sample_source"] == "manual"
    assert sample["candidate_matches_json"] == []
    assert sample["needs_review"] == 0
    assert sample["confidence"] == 1.0


def test_speaker_review_state_migration_from_pre_023_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _cfg(tmp_path)
    all_migrations = db_module._migration_files()  # noqa: SLF001
    legacy_migrations = [item for item in all_migrations if item[0] < 23]

    monkeypatch.setattr(db_module, "_migration_files", lambda: legacy_migrations)  # noqa: SLF001
    db_module.init_db(cfg)
    with db_module.connect(cfg) as conn:
        conn.execute(
            """
            INSERT INTO recordings (
                id,
                source,
                source_filename,
                captured_at,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rec-db-cov-review-migration-1",
                "upload",
                "review.wav",
                "2026-03-01T10:00:00Z",
                "Queued",
                "2026-03-01T10:00:00Z",
                "2026-03-01T10:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO voice_profiles (id, display_name, notes) VALUES (?, ?, ?)",
            (1, "Legacy Canonical", "notes"),
        )
        conn.execute(
            """
            INSERT INTO speaker_assignments (
                recording_id,
                diar_speaker_label,
                voice_profile_id,
                confidence,
                candidate_matches_json,
                low_confidence,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rec-db-cov-review-migration-1",
                "S1",
                1,
                0.95,
                "[]",
                0,
                "2026-03-01T10:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO speaker_assignments (
                recording_id,
                diar_speaker_label,
                voice_profile_id,
                confidence,
                candidate_matches_json,
                low_confidence,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rec-db-cov-review-migration-1",
                "S2",
                None,
                0.44,
                json.dumps([{"voice_profile_id": 1, "score": 0.44}]),
                1,
                "2026-03-01T10:00:00Z",
            ),
        )
        conn.commit()

    monkeypatch.setattr(db_module, "_migration_files", lambda: all_migrations)  # noqa: SLF001
    db_module.init_db(cfg)

    assignments = db_module.list_speaker_assignments(
        "rec-db-cov-review-migration-1",
        settings=cfg,
    )
    assert assignments == [
        {
            "recording_id": "rec-db-cov-review-migration-1",
            "diar_speaker_label": "S1",
            "voice_profile_id": 1,
            "confidence": 0.95,
            "candidate_matches_json": [],
            "low_confidence": 0,
            "review_state": "confirmed_canonical",
            "local_display_name": None,
            "updated_at": "2026-03-01T10:00:00Z",
            "voice_profile_name": "Legacy Canonical",
        },
        {
            "recording_id": "rec-db-cov-review-migration-1",
            "diar_speaker_label": "S2",
            "voice_profile_id": None,
            "confidence": 0.44,
            "candidate_matches_json": [{"voice_profile_id": 1, "score": 0.44}],
            "low_confidence": 1,
            "review_state": "system_suggested",
            "local_display_name": None,
            "updated_at": "2026-03-01T10:00:00Z",
            "voice_profile_name": None,
        },
    ]


def test_merge_voice_profiles_moves_references_and_rewrites_candidates(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    project = db_module.create_project("Merge Project", settings=cfg)
    db_module.create_recording(
        "rec-db-cov-merge-1",
        source="upload",
        source_filename="merge.wav",
        settings=cfg,
    )
    source = db_module.create_voice_profile("Source Voice", settings=cfg)
    target = db_module.create_voice_profile("Target Voice", settings=cfg)

    db_module.set_speaker_assignment(
        recording_id="rec-db-cov-merge-1",
        diar_speaker_label="S1",
        voice_profile_id=source["id"],
        confidence=0.98,
        candidate_matches=[
            {"voice_profile_id": source["id"], "score": 0.98},
            {"voice_profile_id": target["id"], "score": 0.72},
        ],
        keep_unmatched=True,
        settings=cfg,
    )
    db_module.create_voice_sample(
        voice_profile_id=source["id"],
        recording_id="rec-db-cov-merge-1",
        diar_speaker_label="S1",
        snippet_path="recordings/rec-db-cov-merge-1/derived/snippets/S1/1.wav",
        candidate_matches=[
            {"voice_profile_id": source["id"], "score": 0.98},
            {"voice_profile_id": target["id"], "score": 0.72},
        ],
        settings=cfg,
    )
    db_module.create_voice_sample(
        voice_profile_id=None,
        recording_id="rec-db-cov-merge-1",
        diar_speaker_label="S2",
        snippet_path="recordings/rec-db-cov-merge-1/derived/snippets/S2/1.wav",
        candidate_matches=[
            {"voice_profile_id": source["id"], "score": 0.44},
            {"voice_profile_id": target["id"], "score": 0.41},
        ],
        needs_review=True,
        confidence=0.44,
        settings=cfg,
    )
    db_module.replace_participant_metrics(
        recording_id="rec-db-cov-merge-1",
        rows=[
            {
                "diar_speaker_label": "S1",
                "voice_profile_id": source["id"],
                "payload": {"words": 5},
            }
        ],
        settings=cfg,
    )
    db_module.create_routing_training_example(
        recording_id="rec-db-cov-merge-1",
        project_id=project["id"],
        calendar_subject_tokens=["merge"],
        tags=["speaker"],
        voice_profile_ids=[source["id"], source["id"]],
        settings=cfg,
    )
    db_module.increment_project_keyword_weights(
        project_id=project["id"],
        keyword_deltas={
            f"voice:{source['id']}": 1.2,
            f"voice:{target['id']}": 0.3,
        },
        settings=cfg,
    )

    merged = db_module.merge_voice_profiles(
        source["id"],
        target["id"],
        settings=cfg,
    )
    assert merged["samples_moved"] == 1
    assert merged["assignments_moved"] == 1
    assert merged["participant_metrics_moved"] == 1
    assert merged["assignment_candidate_updates"] == 1
    assert merged["sample_candidate_updates"] == 2
    assert merged["routing_example_updates"] == 1
    assert merged["routing_keyword_updates"] == 1

    assert [row["id"] for row in db_module.list_voice_profiles(settings=cfg)] == [
        target["id"]
    ]
    all_profiles = db_module.list_voice_profiles(include_merged=True, settings=cfg)
    assert {row["id"] for row in all_profiles} == {source["id"], target["id"]}
    assert db_module.get_voice_profile(source["id"], settings=cfg) is None
    merged_profile = db_module.get_voice_profile(
        source["id"],
        include_merged=True,
        settings=cfg,
    )
    assert merged_profile is not None
    assert merged_profile["merged_into_voice_profile_id"] == target["id"]

    assignments = db_module.list_speaker_assignments(
        "rec-db-cov-merge-1",
        settings=cfg,
    )
    assert assignments[0]["voice_profile_id"] == target["id"]
    assert assignments[0]["candidate_matches_json"] == [
        {"voice_profile_id": target["id"], "score": 0.98}
    ]

    samples = db_module.list_voice_samples(settings=cfg)
    assigned_sample = next(
        row for row in samples if row["voice_profile_id"] == target["id"]
    )
    review_sample = next(row for row in samples if row["voice_profile_id"] is None)
    assert assigned_sample["candidate_matches_json"] == [
        {"voice_profile_id": target["id"], "score": 0.98}
    ]
    assert review_sample["candidate_matches_json"] == [
        {"voice_profile_id": target["id"], "score": 0.44}
    ]
    participant_rows = db_module.list_participant_metrics(
        "rec-db-cov-merge-1",
        settings=cfg,
    )
    assert participant_rows[0]["voice_profile_id"] == target["id"]

    with db_module.connect(cfg) as conn:
        training_row = conn.execute(
            "SELECT voice_profile_ids_json FROM routing_training_examples"
        ).fetchone()
        keyword_rows = conn.execute(
            """
            SELECT keyword, weight
            FROM routing_project_keyword_weights
            WHERE project_id = ?
            ORDER BY keyword
            """,
            (project["id"],),
        ).fetchall()
    assert json.loads(training_row[0]) == [target["id"]]
    assert [(row[0], row[1]) for row in keyword_rows] == [
        (f"voice:{target['id']}", 1.5)
    ]

    with pytest.raises(ValueError, match="must differ"):
        db_module.merge_voice_profiles(target["id"], target["id"], settings=cfg)
    with pytest.raises(ValueError, match="source_profile_id was not found"):
        db_module.merge_voice_profiles(9999, target["id"], settings=cfg)
    with pytest.raises(ValueError, match="target_profile_id was not found"):
        db_module.merge_voice_profiles(target["id"], 9999, settings=cfg)
    with pytest.raises(ValueError, match="source_profile_id is already merged"):
        db_module.merge_voice_profiles(source["id"], target["id"], settings=cfg)

    third = db_module.create_voice_profile("Third Voice", settings=cfg)
    db_module.merge_voice_profiles(third["id"], target["id"], settings=cfg)
    with pytest.raises(ValueError, match="target_profile_id is already merged"):
        db_module.merge_voice_profiles(target["id"], third["id"], settings=cfg)


def test_delete_voice_profile_clears_merged_pointer_via_trigger(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db_module.init_db(cfg)
    source = db_module.create_voice_profile("Delete Source", settings=cfg)
    target = db_module.create_voice_profile("Delete Target", settings=cfg)

    db_module.merge_voice_profiles(source["id"], target["id"], settings=cfg)
    assert db_module.get_voice_profile(source["id"], settings=cfg) is None

    assert db_module.delete_voice_profile(target["id"], settings=cfg) is True

    reactivated = db_module.get_voice_profile(
        source["id"],
        include_merged=True,
        settings=cfg,
    )
    assert reactivated is not None
    assert reactivated["merged_into_voice_profile_id"] is None
    assert reactivated["merged_at"] is None
    assert [row["id"] for row in db_module.list_voice_profiles(settings=cfg)] == [
        source["id"]
    ]


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
                "starts_at": "2026-03-01T23:00:00Z",
                "ends_at": "2026-03-02T01:00:00Z",
                "summary": " Standup ",
                "description": " notes ",
                "location": " room ",
                "organizer": " owner@example.com ",
                "organizer_name": " Owner Example ",
                "organizer_email": " owner@example.com ",
                "attendees": [{"name": "Alex", "email": "alex@example.com", "label": "Alex"}],
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
    assert rows[0]["organizer_name"] == "Owner Example"
    assert rows[0]["organizer_email"] == "owner@example.com"
    assert rows[0]["attendees_json"] == [
        {
            "name": "Alex",
            "email": "alex@example.com",
            "label": "Alex",
        }
    ]

    replaced = db_module.replace_calendar_events_for_window(
        source_id=source_id,
        window_start="2026-03-02T00:00:00Z",
        window_end="2026-03-04T00:00:00Z",
        events=[
            {
                "uid": "event-2",
                "starts_at": "2026-03-02T09:00:00Z",
                "ends_at": "2026-03-02T10:00:00Z",
                "summary": "Boundary replacement",
                "updated_at": "2026-03-01T00:00:00Z",
            }
        ],
        settings=cfg,
    )
    assert replaced == 1
    boundary_rows = db_module.list_calendar_events(
        starts_from="2026-03-01T00:00:00Z",
        ends_to="2026-03-04T00:00:00Z",
        source_id=source_id,
        settings=cfg,
    )
    assert [row["uid"] for row in boundary_rows] == ["event-2"]

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
