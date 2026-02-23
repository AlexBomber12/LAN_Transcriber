from __future__ import annotations

import asyncio
import builtins
from pathlib import Path
import sys
from types import ModuleType

from fastapi.testclient import TestClient
import pytest

from lan_app import api
from lan_app import db as db_module
from lan_app import worker_tasks
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FAILED,
    JOB_STATUS_FINISHED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_STARTED,
    JOB_TYPE_ALIGN,
    JOB_TYPE_DIARIZE,
    JOB_TYPE_LANGUAGE,
    JOB_TYPE_LLM,
    JOB_TYPE_METRICS,
    JOB_TYPE_PRECHECK,
    JOB_TYPE_STT,
    RECORDING_STATUS_FAILED,
    RECORDING_STATUS_NEEDS_REVIEW,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_READY,
)
from lan_app.db import (
    connect,
    create_job,
    create_project,
    create_recording,
    fail_job_if_started,
    get_job,
    get_recording,
    init_db,
    list_jobs,
    set_recording_status,
    start_job,
    upsert_calendar_match,
)
from lan_app.jobs import RecordingJob, enqueue_recording_job, purge_pending_recording_jobs
from lan_app.worker_tasks import process_job
from lan_transcriber import aliases
from lan_transcriber.llm_client import LLMClient
from lan_transcriber.pipeline import PrecheckResult


def _test_settings(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_init_db_creates_mvp_tables(tmp_path: Path):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    with connect(cfg) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    expected = {
        "recordings",
        "jobs",
        "projects",
        "voice_profiles",
        "speaker_assignments",
        "voice_samples",
        "calendar_matches",
        "meeting_metrics",
        "participant_metrics",
        "routing_training_examples",
        "routing_project_keyword_weights",
    }
    assert expected.issubset(names)


def test_placeholder_cleanup_migration_only_removes_legacy_placeholders(tmp_path: Path):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-mig-placeholder-1",
        source="test",
        source_filename="placeholder.mp3",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_recording(
        "rec-mig-real-queued-1",
        source="test",
        source_filename="queued.mp3",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    placeholder_types = (
        JOB_TYPE_STT,
        JOB_TYPE_DIARIZE,
        JOB_TYPE_ALIGN,
        JOB_TYPE_LANGUAGE,
        JOB_TYPE_LLM,
        JOB_TYPE_METRICS,
    )
    for job_type in placeholder_types:
        create_job(
            f"job-mig-placeholder-{job_type}-1",
            recording_id="rec-mig-placeholder-1",
            job_type=job_type,
            settings=cfg,
            status=JOB_STATUS_QUEUED,
        )

    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE recordings
            SET created_at = ?, updated_at = ?
            WHERE id = ?
            """,
            ("2020-01-02T00:00:00Z", "2020-01-02T00:00:00Z", "rec-mig-placeholder-1"),
        )
        conn.execute(
            """
            UPDATE jobs
            SET created_at = ?, updated_at = ?
            WHERE recording_id = ?
            """,
            ("2020-01-02T00:00:08Z", "2020-01-02T00:00:08Z", "rec-mig-placeholder-1"),
        )
        conn.commit()

    # Simulate a recording where all six legacy jobs were intentionally enqueued later.
    with connect(cfg) as conn:
        conn.execute(
            """
            UPDATE recordings
            SET created_at = ?, updated_at = ?
            WHERE id = ?
            """,
            ("2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z", "rec-mig-real-queued-1"),
        )
        conn.commit()

    for job_type in placeholder_types:
        create_job(
            f"job-mig-real-{job_type}-1",
            recording_id="rec-mig-real-queued-1",
            job_type=job_type,
            settings=cfg,
            status=JOB_STATUS_QUEUED,
        )

    with connect(cfg) as conn:
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        conn.execute(f"PRAGMA user_version = {current_version - 1}")
        conn.commit()

    init_db(cfg)

    for job_type in placeholder_types:
        assert get_job(f"job-mig-placeholder-{job_type}-1", settings=cfg) is None

    for job_type in placeholder_types:
        queued_job = get_job(f"job-mig-real-{job_type}-1", settings=cfg)
        assert queued_job is not None
        assert queued_job["status"] == JOB_STATUS_QUEUED


def test_start_job_only_transitions_queued_jobs(tmp_path: Path):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-start-job-1",
        source="test",
        source_filename="start-job.wav",
        settings=cfg,
    )
    create_job(
        "job-start-job-queued-1",
        recording_id="rec-start-job-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_QUEUED,
    )
    create_job(
        "job-start-job-failed-1",
        recording_id="rec-start-job-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_FAILED,
    )

    assert start_job("job-start-job-queued-1", settings=cfg) is True
    assert start_job("job-start-job-failed-1", settings=cfg) is False

    queued_job = get_job("job-start-job-queued-1", settings=cfg)
    failed_job = get_job("job-start-job-failed-1", settings=cfg)
    assert queued_job is not None
    assert failed_job is not None
    assert queued_job["status"] == JOB_STATUS_STARTED
    assert int(queued_job["attempt"]) == 1
    assert failed_job["status"] == JOB_STATUS_FAILED
    assert int(failed_job["attempt"]) == 0


def test_worker_ignores_stale_execution_for_non_queued_job(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-stale-exec-1",
        source="test",
        source_filename="stale-exec.wav",
        status=RECORDING_STATUS_NEEDS_REVIEW,
        settings=cfg,
    )
    create_job(
        "job-worker-stale-exec-1",
        recording_id="rec-worker-stale-exec-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_FAILED,
        settings=cfg,
    )

    result = process_job(
        "job-worker-stale-exec-1",
        "rec-worker-stale-exec-1",
        JOB_TYPE_PRECHECK,
    )

    job = get_job("job-worker-stale-exec-1", settings=cfg)
    recording = get_recording("rec-worker-stale-exec-1", settings=cfg)
    assert result["status"] == "ignored"
    assert job is not None
    assert recording is not None
    assert job["status"] == JOB_STATUS_FAILED
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW
    step_log = (
        cfg.recordings_root
        / "rec-worker-stale-exec-1"
        / "logs"
        / "step-precheck.log"
    )
    assert step_log.exists()
    assert "ignored stale queue execution" in step_log.read_text(encoding="utf-8")


def test_worker_ignores_stale_inflight_execution_for_recovered_started_job(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-stale-inflight-1",
        source="test",
        source_filename="stale-inflight.wav",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-worker-stale-inflight-1",
        recording_id="rec-worker-stale-inflight-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    def _simulate_reaper(*_args, **_kwargs):
        assert (
            fail_job_if_started(
                "job-worker-stale-inflight-1",
                "stuck job recovered",
                settings=cfg,
            )
            is True
        )
        assert (
            set_recording_status(
                "rec-worker-stale-inflight-1",
                RECORDING_STATUS_NEEDS_REVIEW,
                settings=cfg,
            )
            is True
        )
        return RECORDING_STATUS_READY, None

    monkeypatch.setattr("lan_app.worker_tasks._run_precheck_pipeline", _simulate_reaper)

    result = process_job(
        "job-worker-stale-inflight-1",
        "rec-worker-stale-inflight-1",
        JOB_TYPE_PRECHECK,
    )

    job = get_job("job-worker-stale-inflight-1", settings=cfg)
    recording = get_recording("rec-worker-stale-inflight-1", settings=cfg)
    assert result["status"] == "ignored"
    assert job is not None
    assert recording is not None
    assert job["status"] == JOB_STATUS_FAILED
    assert job["error"] == "stuck job recovered"
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW
    step_log = (
        cfg.recordings_root
        / "rec-worker-stale-inflight-1"
        / "logs"
        / "step-precheck.log"
    )
    assert step_log.exists()
    assert "ignored stale in-flight execution" in step_log.read_text(encoding="utf-8")


def test_worker_finalizes_started_job_when_recording_leaves_processing_mid_run(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-recording-race-1",
        source="test",
        source_filename="recording-race.wav",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-worker-recording-race-1",
        recording_id="rec-worker-recording-race-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    def _simulate_manual_status_change(*_args, **_kwargs):
        assert (
            set_recording_status(
                "rec-worker-recording-race-1",
                RECORDING_STATUS_QUARANTINE,
                settings=cfg,
                quarantine_reason="manual_override",
            )
            is True
        )
        return RECORDING_STATUS_READY, None

    monkeypatch.setattr(
        "lan_app.worker_tasks._run_precheck_pipeline",
        _simulate_manual_status_change,
    )

    result = process_job(
        "job-worker-recording-race-1",
        "rec-worker-recording-race-1",
        JOB_TYPE_PRECHECK,
    )

    job = get_job("job-worker-recording-race-1", settings=cfg)
    recording = get_recording("rec-worker-recording-race-1", settings=cfg)
    assert result["status"] == "ignored"
    assert job is not None
    assert recording is not None
    assert job["status"] == JOB_STATUS_FAILED
    assert "stale in-flight execution ignored" in str(job["error"])
    assert recording["status"] == RECORDING_STATUS_QUARANTINE
    assert recording["quarantine_reason"] == "manual_override"
    step_log = (
        cfg.recordings_root
        / "rec-worker-recording-race-1"
        / "logs"
        / "step-precheck.log"
    )
    assert step_log.exists()
    assert "ignored stale in-flight execution" in step_log.read_text(encoding="utf-8")


def test_worker_noop_updates_job_and_recording_state(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-1",
        source="test",
        source_filename="sample.mp3",
        settings=cfg,
    )
    create_job(
        "job-worker-1",
        recording_id="rec-worker-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    result = process_job("job-worker-1", "rec-worker-1", JOB_TYPE_PRECHECK)
    recording = get_recording("rec-worker-1", settings=cfg)
    job = get_job("job-worker-1", settings=cfg)

    assert result["status"] == "ok"
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_QUARANTINE
    assert recording["quarantine_reason"] == "raw_audio_missing"
    assert job is not None
    assert job["status"] == JOB_STATUS_FINISHED

    step_log = cfg.recordings_root / "rec-worker-1" / "logs" / "step-precheck.log"
    assert step_log.exists()
    assert "finished job=job-worker-1" in step_log.read_text(encoding="utf-8")


def test_worker_legacy_job_restores_status_from_precheck_log(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-legacy-1",
        source="test",
        source_filename="legacy.mp3",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-worker-legacy-1",
        recording_id="rec-worker-legacy-1",
        job_type=JOB_TYPE_STT,
        settings=cfg,
    )

    precheck_log = cfg.recordings_root / "rec-worker-legacy-1" / "logs" / "step-precheck.log"
    precheck_log.parent.mkdir(parents=True, exist_ok=True)
    precheck_log.write_text(
        "[2026-02-22T00:00:00Z] finished job=job-precheck-1 type=precheck recording_status=Ready\n",
        encoding="utf-8",
    )

    result = process_job("job-worker-legacy-1", "rec-worker-legacy-1", JOB_TYPE_STT)
    recording = get_recording("rec-worker-legacy-1", settings=cfg)
    job = get_job("job-worker-legacy-1", settings=cfg)

    assert result["status"] == "ignored"
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_READY
    assert job is not None
    assert job["status"] == JOB_STATUS_FAILED
    assert "unsupported legacy job type under single-job pipeline" in str(job["error"])


def test_worker_legacy_job_keeps_queued_when_precheck_pending(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-legacy-pending-1",
        source="test",
        source_filename="legacy-pending.mp3",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-worker-legacy-pending-precheck-1",
        recording_id="rec-worker-legacy-pending-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )
    create_job(
        "job-worker-legacy-pending-stt-1",
        recording_id="rec-worker-legacy-pending-1",
        job_type=JOB_TYPE_STT,
        settings=cfg,
    )

    precheck_log = (
        cfg.recordings_root
        / "rec-worker-legacy-pending-1"
        / "logs"
        / "step-precheck.log"
    )
    precheck_log.parent.mkdir(parents=True, exist_ok=True)
    precheck_log.write_text(
        "[2026-02-22T00:00:00Z] finished job=job-precheck-old type=precheck recording_status=Ready\n",
        encoding="utf-8",
    )

    result = process_job(
        "job-worker-legacy-pending-stt-1",
        "rec-worker-legacy-pending-1",
        JOB_TYPE_STT,
    )
    recording = get_recording("rec-worker-legacy-pending-1", settings=cfg)

    assert result["status"] == "ignored"
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_QUEUED


def test_worker_legacy_job_restores_quarantine_reason(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-legacy-q-1",
        source="test",
        source_filename="legacy-q.mp3",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-worker-legacy-q-1",
        recording_id="rec-worker-legacy-q-1",
        job_type=JOB_TYPE_STT,
        settings=cfg,
    )

    precheck_log = cfg.recordings_root / "rec-worker-legacy-q-1" / "logs" / "step-precheck.log"
    precheck_log.parent.mkdir(parents=True, exist_ok=True)
    precheck_log.write_text(
        (
            "[2026-02-22T00:00:00Z] quarantined reason=duration_lt_20s\n"
            "[2026-02-22T00:00:01Z] finished job=job-precheck-q-1 "
            "type=precheck recording_status=Quarantine\n"
        ),
        encoding="utf-8",
    )

    result = process_job("job-worker-legacy-q-1", "rec-worker-legacy-q-1", JOB_TYPE_STT)
    recording = get_recording("rec-worker-legacy-q-1", settings=cfg)

    assert result["status"] == "ignored"
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_QUARANTINE
    assert recording["quarantine_reason"] == "duration_lt_20s"


def test_worker_legacy_job_falls_back_to_failed_without_precheck_log(
    tmp_path: Path, monkeypatch
):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-legacy-fallback-1",
        source="test",
        source_filename="legacy-fallback.mp3",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-worker-legacy-fallback-1",
        recording_id="rec-worker-legacy-fallback-1",
        job_type=JOB_TYPE_STT,
        settings=cfg,
    )

    result = process_job(
        "job-worker-legacy-fallback-1",
        "rec-worker-legacy-fallback-1",
        JOB_TYPE_STT,
    )
    recording = get_recording("rec-worker-legacy-fallback-1", settings=cfg)

    assert result["status"] == "ignored"
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_FAILED


def test_load_calendar_summary_context_uses_preparsed_candidates(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setattr(
        worker_tasks,
        "get_calendar_match",
        lambda *_a, **_k: {
            "selected_event_id": "evt-1",
            "candidates_json": [
                {
                    "event_id": "evt-1",
                    "subject": "Roadmap Review",
                    "attendees": ["Alex", " Priya ", ""],
                }
            ],
        },
    )

    title, attendees = worker_tasks._load_calendar_summary_context("rec-cal-ctx-1", settings=cfg)
    assert title == "Roadmap Review"
    assert attendees == ["Alex", "Priya"]


def test_recordings_and_jobs_api_actions(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-api-1",
        source="test",
        source_filename="api.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = JOB_TYPE_PRECHECK,
        settings: AppSettings | None = None,
    ) -> RecordingJob:
        effective = settings or cfg
        create_job(
            "job-api-1",
            recording_id=recording_id,
            job_type=job_type,
            settings=effective,
        )
        set_recording_status(
            recording_id,
            RECORDING_STATUS_QUEUED,
            settings=effective,
        )
        return RecordingJob(
            job_id="job-api-1",
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)
    purged = {"recording_id": None}

    def _fake_purge(recording_id: str, *, settings: AppSettings | None = None) -> int:
        purged["recording_id"] = recording_id
        return 1

    monkeypatch.setattr(api, "purge_pending_recording_jobs", _fake_purge)

    client = TestClient(api.app)

    listed = client.get("/api/recordings")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    detail = client.get("/api/recordings/rec-api-1")
    assert detail.status_code == 200
    assert detail.json()["id"] == "rec-api-1"

    requeue = client.post(
        "/api/recordings/rec-api-1/actions/requeue",
        json={"job_type": JOB_TYPE_PRECHECK},
    )
    assert requeue.status_code == 200
    assert requeue.json()["job_id"] == "job-api-1"
    after_requeue = client.get("/api/recordings/rec-api-1")
    assert after_requeue.status_code == 200
    assert after_requeue.json()["status"] == RECORDING_STATUS_QUEUED

    jobs = client.get("/api/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["total"] == 1

    quarantined = client.post(
        "/api/recordings/rec-api-1/actions/quarantine",
        json={"reason": "manual_review"},
    )
    assert quarantined.status_code == 200
    assert quarantined.json()["status"] == RECORDING_STATUS_QUARANTINE
    assert quarantined.json()["quarantine_reason"] == "manual_review"

    deleted = client.post("/api/recordings/rec-api-1/actions/delete")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert purged["recording_id"] == "rec-api-1"

    missing = client.get("/api/recordings/rec-api-1")
    assert missing.status_code == 404


def test_api_requeue_rejects_non_precheck_job_type(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-api-rq-legacy-1",
        source="test",
        source_filename="legacy.mp3",
        settings=cfg,
    )

    client = TestClient(api.app)
    response = client.post(
        "/api/recordings/rec-api-rq-legacy-1/actions/requeue",
        json={"job_type": JOB_TYPE_STT},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Only precheck is supported in single-job pipeline mode"
    }


def test_api_requeue_dedupes_active_precheck_job(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-api-rq-dedupe-1",
        source="test",
        source_filename="dedupe.mp3",
        settings=cfg,
    )

    class _FakeQueue:
        def enqueue(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("lan_app.jobs.get_queue", lambda _cfg: _FakeQueue())

    client = TestClient(api.app)
    first = client.post(
        "/api/recordings/rec-api-rq-dedupe-1/actions/requeue",
        json={"job_type": JOB_TYPE_PRECHECK},
    )
    assert first.status_code == 200
    first_job_id = first.json()["job_id"]

    second = client.post(
        "/api/recordings/rec-api-rq-dedupe-1/actions/requeue",
        json={"job_type": JOB_TYPE_PRECHECK},
    )
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["existing_job_id"] == first_job_id
    assert "already queued or started" in detail["message"].lower()


def test_api_alias_requires_auth_when_token_enabled(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    cfg.api_bearer_token = "alias-secret"
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)

    alias_path = tmp_path / "db" / "speaker_bank.yaml"
    aliases.save_aliases({}, alias_path)
    monkeypatch.setattr(aliases, "ALIAS_PATH", alias_path)

    client = TestClient(api.app)
    blocked = client.post("/alias/S1", json={"alias": "Alice"})
    assert blocked.status_code == 401

    allowed = client.post(
        "/alias/S1",
        json={"alias": "Alice"},
        headers={"Authorization": "Bearer alias-secret"},
    )
    assert allowed.status_code == 200
    assert aliases.load_aliases(alias_path).get("S1") == "Alice"


def test_enqueue_marks_job_failed_when_redis_enqueue_fails(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-enqueue-fail-1",
        source="test",
        source_filename="enqueue.mp3",
        settings=cfg,
    )

    class _BrokenQueue:
        def enqueue(self, *_args, **_kwargs):
            raise RuntimeError("redis down")

    monkeypatch.setattr("lan_app.jobs.get_queue", lambda _cfg: _BrokenQueue())

    with pytest.raises(RuntimeError) as exc_info:
        enqueue_recording_job(
            "rec-enqueue-fail-1",
            job_type=JOB_TYPE_PRECHECK,
            settings=cfg,
        )
    assert "redis down" in str(exc_info.value)

    jobs, total = list_jobs(settings=cfg, recording_id="rec-enqueue-fail-1")
    assert total == 1
    assert jobs[0]["status"] == JOB_STATUS_FAILED
    assert "queue enqueue failed" in jobs[0]["error"]


def test_worker_setup_failure_marks_job_and_recording_failed(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-fail-1",
        source="test",
        source_filename="setup-fail.mp3",
        settings=cfg,
    )
    create_job(
        "job-worker-fail-1",
        recording_id="rec-worker-fail-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("lan_app.worker_tasks._append_step_log", _boom)

    with pytest.raises(OSError) as exc_info:
        process_job("job-worker-fail-1", "rec-worker-fail-1", JOB_TYPE_PRECHECK)
    assert "disk full" in str(exc_info.value)

    job = get_job("job-worker-fail-1", settings=cfg)
    recording = get_recording("rec-worker-fail-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert job["status"] == JOB_STATUS_FAILED
    assert recording["status"] == RECORDING_STATUS_FAILED


def test_worker_retryable_failure_retries_before_marking_failed(
    tmp_path: Path, monkeypatch
):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-retry-1",
        source="test",
        source_filename="retryable.wav",
        settings=cfg,
    )
    create_job(
        "job-worker-retry-1",
        recording_id="rec-worker-retry-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-worker-retry-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"\x00")
    monkeypatch.setattr("lan_app.worker_tasks._resolve_raw_audio_path", lambda *_a, **_k: raw_audio)

    attempts = {"count": 0}

    def _retryable_failure(*_args, **_kwargs):
        attempts["count"] += 1
        raise RuntimeError("transient failure")

    monkeypatch.setattr("lan_app.worker_tasks.run_precheck", _retryable_failure)
    sleeps: list[int] = []
    retry_job_statuses: list[str] = []

    def _capture_sleep(seconds: int) -> None:
        sleeps.append(seconds)
        in_retry = get_job("job-worker-retry-1", settings=cfg)
        assert in_retry is not None
        retry_job_statuses.append(str(in_retry["status"]))

    monkeypatch.setattr("lan_app.worker_tasks.time.sleep", _capture_sleep)

    with pytest.raises(RuntimeError, match="transient failure"):
        process_job("job-worker-retry-1", "rec-worker-retry-1", JOB_TYPE_PRECHECK)

    job = get_job("job-worker-retry-1", settings=cfg)
    recording = get_recording("rec-worker-retry-1", settings=cfg)
    assert attempts["count"] == 3
    assert sleeps == [1, 2]
    assert retry_job_statuses == [JOB_STATUS_QUEUED, JOB_STATUS_QUEUED]
    assert job is not None
    assert recording is not None
    assert job["attempt"] == 3
    assert job["status"] == JOB_STATUS_FAILED
    assert job["error"] == "max attempts exceeded"
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW


def test_worker_retry_does_not_revive_recovered_started_job(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-worker-retry-race-1",
        source="test",
        source_filename="retry-race.wav",
        settings=cfg,
    )
    create_job(
        "job-worker-retry-race-1",
        recording_id="rec-worker-retry-race-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-worker-retry-race-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"\x00")
    monkeypatch.setattr("lan_app.worker_tasks._resolve_raw_audio_path", lambda *_a, **_k: raw_audio)
    monkeypatch.setattr(
        "lan_app.worker_tasks.run_precheck",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("retry failure")),
    )

    def _requeue_after_recovery(job_id: str, *, error: str | None = None, settings=None) -> bool:
        assert fail_job_if_started(job_id, "stuck job recovered", settings=cfg) is True
        assert (
            set_recording_status(
                "rec-worker-retry-race-1",
                RECORDING_STATUS_NEEDS_REVIEW,
                settings=cfg,
            )
            is True
        )
        return db_module.requeue_job_if_started(
            job_id,
            error=error,
            settings=settings or cfg,
        )

    monkeypatch.setattr(
        "lan_app.worker_tasks.requeue_job_if_started",
        _requeue_after_recovery,
    )
    monkeypatch.setattr("lan_app.worker_tasks.time.sleep", lambda _seconds: None)

    result = process_job(
        "job-worker-retry-race-1",
        "rec-worker-retry-race-1",
        JOB_TYPE_PRECHECK,
    )

    job = get_job("job-worker-retry-race-1", settings=cfg)
    recording = get_recording("rec-worker-retry-race-1", settings=cfg)
    assert result["status"] == "ignored"
    assert job is not None
    assert recording is not None
    assert job["status"] == JOB_STATUS_FAILED
    assert job["error"] == "stuck job recovered"
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW
    step_log = (
        cfg.recordings_root
        / "rec-worker-retry-race-1"
        / "logs"
        / "step-precheck.log"
    )
    assert step_log.exists()
    assert "ignored stale in-flight execution" in step_log.read_text(encoding="utf-8")


def test_worker_retry_terminal_uses_effective_max_attempts_cap(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))
    monkeypatch.setenv("LAN_MAX_JOB_ATTEMPTS", "5")

    init_db(cfg)
    create_recording(
        "rec-worker-retry-cap-1",
        source="test",
        source_filename="retry-cap.wav",
        settings=cfg,
    )
    create_job(
        "job-worker-retry-cap-1",
        recording_id="rec-worker-retry-cap-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-worker-retry-cap-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"\x00")
    monkeypatch.setattr("lan_app.worker_tasks._resolve_raw_audio_path", lambda *_a, **_k: raw_audio)
    monkeypatch.setattr(
        "lan_app.worker_tasks.run_precheck",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("retry failure")),
    )
    monkeypatch.setattr("lan_app.worker_tasks.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="retry failure"):
        process_job("job-worker-retry-cap-1", "rec-worker-retry-cap-1", JOB_TYPE_PRECHECK)

    job = get_job("job-worker-retry-cap-1", settings=cfg)
    recording = get_recording("rec-worker-retry-cap-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert int(job["attempt"]) == 3
    assert job["status"] == JOB_STATUS_FAILED
    assert job["error"] == "max attempts exceeded"
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW


def test_worker_max_attempts_exceeded_before_processing_sets_terminal_state(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))
    monkeypatch.setenv("LAN_MAX_JOB_ATTEMPTS", "3")

    init_db(cfg)
    create_recording(
        "rec-worker-max-attempts-1",
        source="test",
        source_filename="max-attempts.wav",
        settings=cfg,
    )
    create_job(
        "job-worker-max-attempts-1",
        recording_id="rec-worker-max-attempts-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        attempt=3,
    )

    def _should_not_run_pipeline(*_args, **_kwargs):
        raise AssertionError("pipeline execution should not run after max attempts")

    monkeypatch.setattr("lan_app.worker_tasks._run_precheck_pipeline", _should_not_run_pipeline)

    with pytest.raises(RuntimeError, match="max attempts exceeded"):
        process_job(
            "job-worker-max-attempts-1",
            "rec-worker-max-attempts-1",
            JOB_TYPE_PRECHECK,
        )

    job = get_job("job-worker-max-attempts-1", settings=cfg)
    recording = get_recording("rec-worker-max-attempts-1", settings=cfg)
    assert job is not None
    assert recording is not None
    assert job["attempt"] == 4
    assert job["status"] == JOB_STATUS_FAILED
    assert job["error"] == "max attempts exceeded"
    assert recording["status"] == RECORDING_STATUS_NEEDS_REVIEW


def test_enqueue_sets_recording_status_to_queued_on_success(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-requeue-status-1",
        source="test",
        source_filename="retry.mp3",
        status=RECORDING_STATUS_FAILED,
        settings=cfg,
    )

    class _QueueOK:
        def enqueue(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("lan_app.jobs.get_queue", lambda _cfg: _QueueOK())

    enqueue_recording_job(
        "rec-requeue-status-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )
    recording = get_recording("rec-requeue-status-1", settings=cfg)
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_QUEUED


def test_enqueue_uses_configured_rq_job_timeout(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    cfg.rq_job_timeout_seconds = 321
    init_db(cfg)
    create_recording(
        "rec-rq-timeout-1",
        source="test",
        source_filename="timeout.mp3",
        settings=cfg,
    )

    captured: dict[str, object] = {}

    class _QueueCapture:
        def enqueue(self, *_args, **kwargs):
            captured.update(kwargs)
            return None

    monkeypatch.setattr("lan_app.jobs.get_queue", lambda _cfg: _QueueCapture())

    enqueue_recording_job(
        "rec-rq-timeout-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    assert captured["job_timeout"] == 321


def test_purge_pending_recording_jobs_deletes_only_pending(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-purge-1",
        source="test",
        source_filename="purge.mp3",
        settings=cfg,
    )
    create_job(
        "job-purge-queued",
        recording_id="rec-purge-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )
    create_job(
        "job-purge-finished",
        recording_id="rec-purge-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_FINISHED,
    )

    class _FakeRQJob:
        def __init__(self, status: str):
            self._status = status
            self.deleted = False

        def get_status(self, refresh: bool = False):
            class _Status:
                def __init__(self, value: str):
                    self.value = value

            return _Status(self._status)

        def delete(self, remove_from_queue: bool = True, delete_dependents: bool = False):
            self.deleted = True

    class _FakeQueue:
        def __init__(self):
            self.removed: list[str] = []
            self.jobs = {
                "job-purge-queued": _FakeRQJob("queued"),
                "job-purge-finished": _FakeRQJob("finished"),
            }

        def fetch_job(self, job_id: str):
            return self.jobs.get(job_id)

        def remove(self, job_id: str):
            self.removed.append(job_id)

    fake_queue = _FakeQueue()
    monkeypatch.setattr("lan_app.jobs.get_queue", lambda _cfg: fake_queue)

    removed = purge_pending_recording_jobs("rec-purge-1", settings=cfg)
    assert removed == 1
    assert fake_queue.removed == ["job-purge-queued"]
    assert fake_queue.jobs["job-purge-queued"].deleted is True
    assert fake_queue.jobs["job-purge-finished"].deleted is False


def test_worker_precheck_quarantines_and_writes_artifacts(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-precheck-q-1",
        source="test",
        source_filename="short.wav",
        settings=cfg,
    )
    create_job(
        "job-precheck-q-1",
        recording_id="rec-precheck-q-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-precheck-q-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"\x00")

    monkeypatch.setattr("lan_app.worker_tasks._resolve_raw_audio_path", lambda *_a, **_k: raw_audio)
    monkeypatch.setattr(
        "lan_app.worker_tasks.run_precheck",
        lambda *_a, **_k: PrecheckResult(
            duration_sec=5.0,
            speech_ratio=0.5,
            quarantine_reason="duration_lt_20s",
        ),
    )
    build_called = {"value": False}

    def _should_not_build_diariser(*_args, **_kwargs):
        build_called["value"] = True
        raise AssertionError("_build_diariser should be skipped for quarantined precheck")

    monkeypatch.setattr(
        "lan_app.worker_tasks._build_diariser",
        _should_not_build_diariser,
    )

    called = {"value": False}

    async def _fake_run_pipeline(*_args, **_kwargs):
        called["value"] = True
        return None

    monkeypatch.setattr("lan_app.worker_tasks.run_pipeline", _fake_run_pipeline)

    result = process_job("job-precheck-q-1", "rec-precheck-q-1", JOB_TYPE_PRECHECK)
    assert result["status"] == "ok"
    assert called["value"] is True
    assert build_called["value"] is False

    recording = get_recording("rec-precheck-q-1", settings=cfg)
    job = get_job("job-precheck-q-1", settings=cfg)
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_QUARANTINE
    assert recording["quarantine_reason"] == "duration_lt_20s"
    assert job is not None
    assert job["status"] == JOB_STATUS_FINISHED


def test_worker_precheck_missing_audio_quarantines_recording(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-precheck-missing-audio-1",
        source="test",
        source_filename="missing.wav",
        settings=cfg,
    )
    create_job(
        "job-precheck-missing-audio-1",
        recording_id="rec-precheck-missing-audio-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    monkeypatch.setattr(
        "lan_app.worker_tasks._resolve_raw_audio_path",
        lambda *_a, **_k: None,
    )

    def _should_not_precheck(*_args, **_kwargs):
        raise AssertionError("run_precheck should be skipped when audio is missing")

    monkeypatch.setattr("lan_app.worker_tasks.run_precheck", _should_not_precheck)

    result = process_job(
        "job-precheck-missing-audio-1",
        "rec-precheck-missing-audio-1",
        JOB_TYPE_PRECHECK,
    )
    assert result["status"] == "ok"

    recording = get_recording("rec-precheck-missing-audio-1", settings=cfg)
    job = get_job("job-precheck-missing-audio-1", settings=cfg)
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_QUARANTINE
    assert recording["quarantine_reason"] == "raw_audio_missing"
    assert job is not None
    assert job["status"] == JOB_STATUS_FINISHED


def test_worker_precheck_runs_pipeline_when_safe(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))
    monkeypatch.setenv("ROUTING_AUTO_SELECT_THRESHOLD", "0.1")

    init_db(cfg)
    create_project("Normal", settings=cfg)
    create_recording(
        "rec-precheck-ok-1",
        source="test",
        source_filename="normal.wav",
        settings=cfg,
    )
    create_job(
        "job-precheck-ok-1",
        recording_id="rec-precheck-ok-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-precheck-ok-1",
        candidates=[
            {
                "event_id": "evt-normal",
                "subject": "Normal sync",
                "organizer": "Alex",
                "attendees": ["Priya"],
                "score": 0.9,
                "rationale": "test",
            }
        ],
        selected_event_id="evt-normal",
        selected_confidence=0.9,
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-precheck-ok-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"\x00")

    monkeypatch.setattr("lan_app.worker_tasks._resolve_raw_audio_path", lambda *_a, **_k: raw_audio)
    monkeypatch.setattr(
        "lan_app.worker_tasks.run_precheck",
        lambda *_a, **_k: PrecheckResult(
            duration_sec=35.0,
            speech_ratio=0.7,
            quarantine_reason=None,
        ),
    )
    called = {"value": False}
    observed_llm: dict[str, object] = {}

    async def _fake_run_pipeline(*_args, **kwargs):
        called["value"] = True
        observed_llm["value"] = kwargs.get("llm")
        return None

    monkeypatch.setattr("lan_app.worker_tasks.run_pipeline", _fake_run_pipeline)

    result = process_job("job-precheck-ok-1", "rec-precheck-ok-1", JOB_TYPE_PRECHECK)
    assert result["status"] == "ok"
    assert called["value"] is True
    assert isinstance(observed_llm["value"], LLMClient)

    recording = get_recording("rec-precheck-ok-1", settings=cfg)
    job = get_job("job-precheck-ok-1", settings=cfg)
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_READY
    assert job is not None
    assert job["status"] == JOB_STATUS_FINISHED


def test_worker_precheck_keeps_auto_summary_target_unset(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setenv("LAN_DATA_ROOT", str(cfg.data_root))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(cfg.recordings_root))
    monkeypatch.setenv("LAN_DB_PATH", str(cfg.db_path))
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(cfg.metrics_snapshot_path))

    init_db(cfg)
    create_recording(
        "rec-precheck-auto-target-1",
        source="test",
        source_filename="normal.wav",
        settings=cfg,
    )
    create_job(
        "job-precheck-auto-target-1",
        recording_id="rec-precheck-auto-target-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-precheck-auto-target-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"\x00")

    monkeypatch.setattr("lan_app.worker_tasks._resolve_raw_audio_path", lambda *_a, **_k: raw_audio)
    monkeypatch.setattr(
        "lan_app.worker_tasks.run_precheck",
        lambda *_a, **_k: PrecheckResult(
            duration_sec=40.0,
            speech_ratio=0.8,
            quarantine_reason=None,
        ),
    )

    async def _fake_run_pipeline(*_args, **_kwargs):
        return None

    monkeypatch.setattr("lan_app.worker_tasks.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        "lan_app.worker_tasks._load_transcript_language_payload",
        lambda *_a, **_k: ("es", "es"),
    )

    observed_updates: dict[str, object] = {}

    def _capture_language_settings(recording_id: str, *, settings=None, **kwargs):
        observed_updates["recording_id"] = recording_id
        observed_updates["kwargs"] = kwargs
        return True

    monkeypatch.setattr(
        "lan_app.worker_tasks.set_recording_language_settings",
        _capture_language_settings,
    )

    result = process_job(
        "job-precheck-auto-target-1",
        "rec-precheck-auto-target-1",
        JOB_TYPE_PRECHECK,
    )
    assert result["status"] == "ok"
    assert observed_updates["recording_id"] == "rec-precheck-auto-target-1"
    assert observed_updates["kwargs"] == {"language_auto": "es"}


def test_build_diariser_wraps_sync_pyannote_pipeline(monkeypatch):
    from lan_app import worker_tasks

    class _FakeModel:
        def __init__(self):
            self.calls: list[object] = []

        def __call__(self, input_payload: object):
            self.calls.append(input_payload)
            return {"ok": True}

    fake_model = _FakeModel()

    class _FakePipeline:
        @staticmethod
        def from_pretrained(_name: str):
            return fake_model

    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = _FakePipeline  # type: ignore[attr-defined]
    pyannote_pkg = ModuleType("pyannote")
    pyannote_pkg.audio = pyannote_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    diariser = worker_tasks._build_diariser(duration_sec=30.0)
    result = asyncio.run(diariser(Path("/tmp/fake.wav")))

    assert result == {"ok": True}
    assert fake_model.calls == ["/tmp/fake.wav"]


def test_build_diariser_surfaces_pyannote_model_load_errors(monkeypatch):
    from lan_app import worker_tasks

    class _BrokenPipeline:
        @staticmethod
        def from_pretrained(_name: str):
            raise RuntimeError("auth failed")

    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = _BrokenPipeline  # type: ignore[attr-defined]
    pyannote_pkg = ModuleType("pyannote")
    pyannote_pkg.audio = pyannote_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    with pytest.raises(RuntimeError, match="auth failed"):
        worker_tasks._build_diariser(duration_sec=30.0)


def test_build_diariser_surfaces_non_pyannote_import_errors(monkeypatch):
    from lan_app import worker_tasks

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyannote.audio":
            err = ModuleNotFoundError("No module named 'torch'")
            err.name = "torch"
            raise err
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ModuleNotFoundError, match="torch"):
        worker_tasks._build_diariser(duration_sec=30.0)
