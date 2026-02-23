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
)
from lan_app.db import connect, create_job, create_recording, get_job, get_recording, init_db
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


def test_reaper_recovers_processing_recording_without_started_job(tmp_path: Path):
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

    step_log = cfg.recordings_root / "rec-reaper-orphan-1" / "logs" / "step-precheck.log"
    assert step_log.exists()
    assert "stuck job recovery applied" in step_log.read_text(encoding="utf-8")
