from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lan_app import api
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FINISHED,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_READY,
)
from lan_app.db import (
    connect,
    create_job,
    create_recording,
    get_job,
    get_recording,
    init_db,
)
from lan_app.jobs import RecordingJob
from lan_app.worker_tasks import process_job


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
        "calendar_matches",
        "meeting_metrics",
        "participant_metrics",
    }
    assert expected.issubset(names)


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
    assert recording["status"] == RECORDING_STATUS_READY
    assert job is not None
    assert job["status"] == JOB_STATUS_FINISHED

    step_log = cfg.recordings_root / "rec-worker-1" / "logs" / "step-precheck.log"
    assert step_log.exists()
    assert "finished job=job-worker-1" in step_log.read_text(encoding="utf-8")


def test_recordings_and_jobs_api_actions(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-api-1",
        source="test",
        source_filename="api.mp3",
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
        return RecordingJob(
            job_id="job-api-1",
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)

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

    missing = client.get("/api/recordings/rec-api-1")
    assert missing.status_code == 404
