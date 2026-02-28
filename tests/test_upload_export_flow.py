from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4
import zipfile

import pytest
from fastapi.testclient import TestClient

from lan_app import api, ui_routes
from lan_app.config import AppSettings
from lan_app.constants import JOB_STATUS_QUEUED, JOB_TYPE_PRECHECK
from lan_app.db import create_job, init_db
from lan_app.jobs import RecordingJob


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


@pytest.fixture()
def cfg(tmp_path: Path, monkeypatch):
    app_settings = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", app_settings)
    monkeypatch.setattr(ui_routes, "_settings", app_settings)
    init_db(app_settings)

    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = JOB_TYPE_PRECHECK,
        settings: AppSettings | None = None,
    ) -> RecordingJob:
        effective = settings or app_settings
        job_id = f"job-upload-{uuid4().hex[:8]}"
        create_job(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
            status=JOB_STATUS_QUEUED,
            settings=effective,
        )
        return RecordingJob(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)
    return app_settings


@pytest.fixture()
def client(cfg: AppSettings):
    return TestClient(api.app, follow_redirects=True)


def _upload_recording(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/uploads",
        files={"file": ("meeting-flow.mp3", b"fake-mp3", "audio/mpeg")},
    )
    assert response.status_code == 200
    return response.json()


def test_upload_creates_recording_and_file(client: TestClient, cfg: AppSettings):
    payload = _upload_recording(client)
    recording_id = str(payload["recording_id"])

    assert recording_id
    raw_audio = cfg.recordings_root / recording_id / "raw" / "audio.mp3"
    assert raw_audio.exists()
    assert raw_audio.read_bytes() == b"fake-mp3"


def test_ui_pages_render_for_new_recording(client: TestClient):
    payload = _upload_recording(client)
    recording_id = str(payload["recording_id"])

    upload_page = client.get("/upload")
    assert upload_page.status_code == 200
    assert 'id="file-input"' in upload_page.text

    recordings_page = client.get("/recordings")
    assert recordings_page.status_code == 200
    assert recording_id in recordings_page.text or "meeting-flow.mp3" in recordings_page.text

    detail_page = client.get(f"/recordings/{recording_id}")
    assert detail_page.status_code == 200


def test_export_zip_downloads_required_files(client: TestClient):
    payload = _upload_recording(client)
    recording_id = str(payload["recording_id"])

    response = client.get(f"/ui/recordings/{recording_id}/export.zip")
    assert response.status_code == 200

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    assert "onenote.md" in names
    assert "manifest.json" in names
