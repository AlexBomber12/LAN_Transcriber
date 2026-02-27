from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from lan_app import api
from lan_app.config import AppSettings
from lan_app.constants import JOB_STATUS_QUEUED, JOB_TYPE_PRECHECK, RECORDING_STATUS_QUEUED
from lan_app.db import (
    create_job,
    get_recording,
    init_db,
    list_jobs,
    list_recordings,
    set_recording_status,
)
from lan_app.jobs import RecordingJob


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _stub_enqueue(monkeypatch, cfg: AppSettings) -> None:
    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = JOB_TYPE_PRECHECK,
        settings: AppSettings | None = None,
    ) -> RecordingJob:
        effective = settings or cfg
        job_id = f"job-upload-{uuid4().hex[:8]}"
        create_job(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
            status=JOB_STATUS_QUEUED,
            settings=effective,
        )
        set_recording_status(
            recording_id,
            RECORDING_STATUS_QUEUED,
            settings=effective,
        )
        return RecordingJob(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)


def test_upload_success_creates_recording_and_job(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    client = TestClient(api.app)
    response = client.post(
        "/api/uploads",
        files={"file": ("2026-02-18 16_01_43.mp3", b"abc", "audio/mpeg")},
    )
    assert response.status_code == 200
    payload = response.json()

    recording_id = payload["recording_id"]
    assert recording_id.startswith("trs_")
    assert payload["captured_at"] == "2026-02-18T16:01:43Z"
    assert payload["bytes_written"] == 3

    raw_file = cfg.recordings_root / recording_id / "raw" / "audio.mp3"
    assert raw_file.exists()
    assert raw_file.read_bytes() == b"abc"

    recording = get_recording(recording_id, settings=cfg)
    assert recording is not None
    assert recording["source"] == "upload"
    assert ".mp3" in str(recording["source_filename"])
    assert recording["captured_at"] == "2026-02-18T16:01:43Z"

    jobs, total = list_jobs(settings=cfg, recording_id=recording_id)
    assert total == 1
    assert jobs[0]["status"] == JOB_STATUS_QUEUED
    assert jobs[0]["type"] == JOB_TYPE_PRECHECK


def test_upload_unsupported_extension_returns_422(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    client = TestClient(api.app)
    response = client.post(
        "/api/uploads",
        files={"file": ("bad.exe", b"abc", "application/octet-stream")},
    )
    assert response.status_code == 422
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_max_bytes_returns_413(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.upload_max_bytes = 2
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    client = TestClient(api.app)
    response = client.post(
        "/api/uploads",
        files={"file": ("tiny.mp3", b"abc", "audio/mpeg")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "max upload size exceeded"

    items, total = list_recordings(settings=cfg)
    assert total == 0
    assert items == []
