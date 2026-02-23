from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_STARTED,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUS_NEEDS_REVIEW,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_READY,
)
from lan_app import db as db_module
from lan_app.db import (
    connect,
    create_job,
    create_recording,
    get_job,
    get_recording,
    init_db,
    set_recording_status,
)
from lan_app.reaper import run_stuck_job_reaper_once


def _test_settings(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    cfg.stuck_job_seconds = 60
    return cfg


def test_reaper_recovers_stale_started_job(tmp_path: Path):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-stale-1",
        source="test",
        source_filename="stale.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-stale-1",
        recording_id="rec-reaper-stale-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_STARTED,
        settings=cfg,
        attempt=1,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET started_at = ?, updated_at = ?
            WHERE id = ?
            """,
            ("2026-02-22T00:00:00Z", "2026-02-22T00:00:00Z", "job-reaper-stale-1"),
        )
        conn.commit()

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc),
    )

    job = get_job("job-reaper-stale-1", settings=cfg)
    recording = get_recording("rec-reaper-stale-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["stale_started_jobs"] == 1
    assert summary["recovered_jobs"] == 1
    assert job["status"] == JOB_STATUS_FAILED
    assert job["error"] == "stuck job recovered"
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW

    step_log = cfg.recordings_root / "rec-reaper-stale-1" / "logs" / "step-precheck.log"
    assert step_log.exists()
    assert "stuck job recovery applied" in step_log.read_text(encoding="utf-8")


def test_reaper_skips_stale_started_job_when_it_already_transitioned(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-race-1",
        source="test",
        source_filename="race.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-race-1",
        recording_id="rec-reaper-race-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_STARTED,
        settings=cfg,
        attempt=1,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET started_at = ?, updated_at = ?
            WHERE id = ?
            """,
            ("2026-02-22T00:00:00Z", "2026-02-22T00:00:00Z", "job-reaper-race-1"),
        )
        conn.commit()

    monkeypatch.setattr("lan_app.reaper.fail_job_if_started", lambda *_a, **_k: False)

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc),
    )

    job = get_job("job-reaper-race-1", settings=cfg)
    recording = get_recording("rec-reaper-race-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["stale_started_jobs"] == 1
    assert summary["recovered_jobs"] == 0
    assert summary["recovered_recordings"] == 0
    assert job["status"] == JOB_STATUS_STARTED
    assert recording["status"] == RECORDING_STATUS_PROCESSING


def test_reaper_recovers_stale_started_job_when_recording_status_is_queued(
    tmp_path: Path,
):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-stale-queued-1",
        source="test",
        source_filename="stale-queued.wav",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-reaper-stale-queued-1",
        recording_id="rec-reaper-stale-queued-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_STARTED,
        settings=cfg,
        attempt=1,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET started_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                "2026-02-22T00:00:00Z",
                "2026-02-22T00:00:00Z",
                "job-reaper-stale-queued-1",
            ),
        )
        conn.commit()

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc),
    )

    job = get_job("job-reaper-stale-queued-1", settings=cfg)
    recording = get_recording("rec-reaper-stale-queued-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["stale_started_jobs"] == 1
    assert summary["recovered_jobs"] == 1
    assert summary["recovered_recordings"] == 1
    assert job["status"] == JOB_STATUS_FAILED
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW


def test_reaper_does_not_downgrade_ready_recording_when_clearing_stale_started_job(
    tmp_path: Path,
):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-stale-ready-1",
        source="test",
        source_filename="stale-ready.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    create_job(
        "job-reaper-stale-ready-1",
        recording_id="rec-reaper-stale-ready-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_STARTED,
        settings=cfg,
        attempt=1,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET started_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                "2026-02-22T00:00:00Z",
                "2026-02-22T00:00:00Z",
                "job-reaper-stale-ready-1",
            ),
        )
        conn.commit()

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc),
    )

    job = get_job("job-reaper-stale-ready-1", settings=cfg)
    recording = get_recording("rec-reaper-stale-ready-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["stale_started_jobs"] == 1
    assert summary["recovered_jobs"] == 1
    assert summary["recovered_recordings"] == 0
    assert job["status"] == JOB_STATUS_FAILED
    assert recording["status"] == RECORDING_STATUS_READY


def test_reaper_skips_stale_downgrade_if_status_changes_after_selection(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-stale-race-1",
        source="test",
        source_filename="stale-race.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-stale-race-1",
        recording_id="rec-reaper-stale-race-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_STARTED,
        settings=cfg,
        attempt=1,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET started_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                "2026-02-22T00:00:00Z",
                "2026-02-22T00:00:00Z",
                "job-reaper-stale-race-1",
            ),
        )
        conn.commit()

    original = db_module.set_recording_status_if_current_in

    def _racy_set_status(*args, **kwargs):
        set_recording_status(
            "rec-reaper-stale-race-1",
            RECORDING_STATUS_READY,
            settings=cfg,
        )
        return original(*args, **kwargs)

    monkeypatch.setattr("lan_app.reaper.set_recording_status_if_current_in", _racy_set_status)

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc),
    )

    job = get_job("job-reaper-stale-race-1", settings=cfg)
    recording = get_recording("rec-reaper-stale-race-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["stale_started_jobs"] == 1
    assert summary["recovered_jobs"] == 1
    assert summary["recovered_recordings"] == 0
    assert job["status"] == JOB_STATUS_FAILED
    assert recording["status"] == RECORDING_STATUS_READY


def test_reaper_recovers_processing_recording_without_started_job(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-orphan-1",
        source="test",
        source_filename="orphan.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-orphan-1",
        recording_id="rec-reaper-orphan-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE recordings
            SET updated_at = ?
            WHERE id = ?
            """,
            ("2026-02-22T00:00:00Z", "rec-reaper-orphan-1"),
        )
        conn.commit()

    cancelled_job_ids: list[str] = []

    def _cancel_pending(job_id: str, *, settings=None) -> bool:
        cancelled_job_ids.append(job_id)
        return True

    monkeypatch.setattr("lan_app.reaper.cancel_pending_queue_job", _cancel_pending)

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=1),
    )

    job = get_job("job-reaper-orphan-1", settings=cfg)
    recording = get_recording("rec-reaper-orphan-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["processing_without_started"] == 1
    assert summary["recovered_jobs"] == 1
    assert job["status"] == JOB_STATUS_FAILED
    assert job["error"] == "stuck job recovered"
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW
    assert cancelled_job_ids == ["job-reaper-orphan-1"]

    step_log = cfg.recordings_root / "rec-reaper-orphan-1" / "logs" / "step-precheck.log"
    assert step_log.exists()
    assert "stuck job recovery applied" in step_log.read_text(encoding="utf-8")


def test_reaper_does_not_recover_recent_processing_without_started_job(tmp_path: Path):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-recent-1",
        source="test",
        source_filename="recent.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-recent-1",
        recording_id="rec-reaper-recent-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
    )

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime.now(tz=timezone.utc),
    )

    job = get_job("job-reaper-recent-1", settings=cfg)
    recording = get_recording("rec-reaper-recent-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["processing_without_started"] == 0
    assert summary["recovered_jobs"] == 0
    assert job["status"] == JOB_STATUS_QUEUED
    assert recording["status"] == RECORDING_STATUS_PROCESSING


def test_reaper_skips_orphan_recovery_when_active_job_is_no_longer_queued(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-race-queued-1",
        source="test",
        source_filename="race-queued.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-race-queued-1",
        recording_id="rec-reaper-race-queued-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE recordings
            SET updated_at = ?
            WHERE id = ?
            """,
            ("2026-02-22T00:00:00Z", "rec-reaper-race-queued-1"),
        )
        conn.commit()

    monkeypatch.setattr("lan_app.reaper.fail_job_if_queued", lambda *_a, **_k: False)

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 1, tzinfo=timezone.utc),
    )

    job = get_job("job-reaper-race-queued-1", settings=cfg)
    recording = get_recording("rec-reaper-race-queued-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["processing_without_started"] == 1
    assert summary["recovered_jobs"] == 0
    assert summary["recovered_recordings"] == 0
    assert job["status"] == JOB_STATUS_QUEUED
    assert recording["status"] == RECORDING_STATUS_PROCESSING


def test_reaper_skips_orphan_downgrade_if_status_changes_after_selection(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-reaper-race-downgrade-1",
        source="test",
        source_filename="race-downgrade.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-race-downgrade-1",
        recording_id="rec-reaper-race-downgrade-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
    )
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE recordings
            SET updated_at = ?
            WHERE id = ?
            """,
            ("2026-02-22T00:00:00Z", "rec-reaper-race-downgrade-1"),
        )
        conn.commit()

    monkeypatch.setattr(
        "lan_app.reaper.cancel_pending_queue_job",
        lambda *_a, **_k: True,
    )
    original = db_module.set_recording_status_if_current_in

    def _racy_set_status(*args, **kwargs):
        set_recording_status(
            "rec-reaper-race-downgrade-1",
            RECORDING_STATUS_READY,
            settings=cfg,
        )
        return original(*args, **kwargs)

    monkeypatch.setattr("lan_app.reaper.set_recording_status_if_current_in", _racy_set_status)

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 1, tzinfo=timezone.utc),
    )

    job = get_job("job-reaper-race-downgrade-1", settings=cfg)
    recording = get_recording("rec-reaper-race-downgrade-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert summary["processing_without_started"] == 1
    assert summary["recovered_jobs"] == 1
    assert summary["recovered_recordings"] == 0
    assert job["status"] == JOB_STATUS_FAILED
    assert recording["status"] == RECORDING_STATUS_READY


def test_reaper_continues_when_step_log_append_fails(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)

    create_recording(
        "rec-reaper-logfail-stale-1",
        source="test",
        source_filename="logfail-stale-1.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-logfail-stale-1",
        recording_id="rec-reaper-logfail-stale-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_STARTED,
        settings=cfg,
        attempt=1,
    )
    create_recording(
        "rec-reaper-logfail-stale-2",
        source="test",
        source_filename="logfail-stale-2.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-logfail-stale-2",
        recording_id="rec-reaper-logfail-stale-2",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_STARTED,
        settings=cfg,
        attempt=1,
    )
    create_recording(
        "rec-reaper-logfail-orphan-1",
        source="test",
        source_filename="logfail-orphan.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-reaper-logfail-orphan-1",
        recording_id="rec-reaper-logfail-orphan-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
        attempt=1,
    )

    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET started_at = ?, updated_at = ?
            WHERE id IN (?, ?)
            """,
            (
                "2026-02-22T00:00:00Z",
                "2026-02-22T00:00:00Z",
                "job-reaper-logfail-stale-1",
                "job-reaper-logfail-stale-2",
            ),
        )
        conn.execute(
            """
            UPDATE recordings
            SET updated_at = ?
            WHERE id = ?
            """,
            ("2026-02-22T00:00:00Z", "rec-reaper-logfail-orphan-1"),
        )
        conn.commit()

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("lan_app.reaper._append_step_log", _boom)
    monkeypatch.setattr("lan_app.reaper.cancel_pending_queue_job", lambda *_a, **_k: True)

    summary = run_stuck_job_reaper_once(
        settings=cfg,
        now=datetime(2026, 2, 23, 0, 0, 1, tzinfo=timezone.utc),
    )

    stale_job_1 = get_job("job-reaper-logfail-stale-1", settings=cfg)
    stale_job_2 = get_job("job-reaper-logfail-stale-2", settings=cfg)
    orphan_job = get_job("job-reaper-logfail-orphan-1", settings=cfg)
    stale_rec_1 = get_recording("rec-reaper-logfail-stale-1", settings=cfg)
    stale_rec_2 = get_recording("rec-reaper-logfail-stale-2", settings=cfg)
    orphan_rec = get_recording("rec-reaper-logfail-orphan-1", settings=cfg)

    assert stale_job_1 is not None
    assert stale_job_2 is not None
    assert orphan_job is not None
    assert stale_rec_1 is not None
    assert stale_rec_2 is not None
    assert orphan_rec is not None
    assert summary["stale_started_jobs"] == 2
    assert summary["processing_without_started"] == 1
    assert summary["recovered_jobs"] == 3
    assert summary["recovered_recordings"] == 3
    assert stale_job_1["status"] == JOB_STATUS_FAILED
    assert stale_job_2["status"] == JOB_STATUS_FAILED
    assert orphan_job["status"] == JOB_STATUS_FAILED
    assert stale_rec_1["status"] == RECORDING_STATUS_NEEDS_REVIEW
    assert stale_rec_2["status"] == RECORDING_STATUS_NEEDS_REVIEW
    assert orphan_rec["status"] == RECORDING_STATUS_NEEDS_REVIEW
