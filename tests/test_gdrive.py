from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lan_app import api
from lan_app.config import AppSettings
from lan_app.db import get_recording, init_db, list_jobs, list_recordings
from lan_app.gdrive import (
    _PIPELINE_STEPS,
    _known_drive_file_ids,
    _suffix_from_name,
    download_file,
    ingest_once,
    list_inbox_files,
    parse_plaud_captured_at,
)
from lan_app.jobs import RecordingJob


def _test_settings(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    cfg.gdrive_sa_json_path = tmp_path / "sa.json"
    cfg.gdrive_inbox_folder_id = "folder-abc"
    return cfg


def _test_settings_no_gdrive(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _stub_enqueue(monkeypatch: Any, cfg: AppSettings) -> None:
    """Replace enqueue_recording_job with a DB-only stub (no Redis)."""
    from lan_app.db import create_job, set_recording_status
    from lan_app.constants import JOB_STATUS_QUEUED, RECORDING_STATUS_QUEUED
    from uuid import uuid4

    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = "precheck",
        settings: AppSettings | None = None,
    ) -> RecordingJob:
        effective = settings or cfg
        job_id = uuid4().hex
        create_job(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
            status=JOB_STATUS_QUEUED,
            settings=effective,
        )
        set_recording_status(
            recording_id, RECORDING_STATUS_QUEUED, settings=effective
        )
        return RecordingJob(
            job_id=job_id, recording_id=recording_id, job_type=job_type
        )

    monkeypatch.setattr("lan_app.jobs.enqueue_recording_job", _fake_enqueue)


# --------------------------------------------------------------------------
# parse_plaud_captured_at
# --------------------------------------------------------------------------


class TestParsePlaudCapturedAt:
    def test_underscore_separator(self):
        result = parse_plaud_captured_at("2026-02-18 16_01_43.mp3")
        assert result == "2026-02-18T16:01:43Z"

    def test_dash_separator(self):
        result = parse_plaud_captured_at("2026-02-18 16-01-43.mp3")
        assert result == "2026-02-18T16:01:43Z"

    def test_mixed_separator(self):
        result = parse_plaud_captured_at("2026-02-18 16_01-43.mp3")
        assert result == "2026-02-18T16:01:43Z"

    def test_no_match_returns_none(self):
        assert parse_plaud_captured_at("random_file.mp3") is None

    def test_invalid_date_returns_none(self):
        assert parse_plaud_captured_at("2026-13-40 99_99_99.mp3") is None

    def test_embedded_in_longer_name(self):
        result = parse_plaud_captured_at("Plaud_2026-02-18 16_01_43_notes.mp3")
        assert result == "2026-02-18T16:01:43Z"


# --------------------------------------------------------------------------
# _suffix_from_name
# --------------------------------------------------------------------------


class TestSuffixFromName:
    def test_mp3(self):
        assert _suffix_from_name("file.mp3") == ".mp3"

    def test_wav(self):
        assert _suffix_from_name("recording.wav") == ".wav"

    def test_no_extension_defaults_mp3(self):
        assert _suffix_from_name("noext") == ".mp3"

    def test_dotfile_defaults_mp3(self):
        assert _suffix_from_name(".hidden") == ".mp3"


# --------------------------------------------------------------------------
# list_inbox_files
# --------------------------------------------------------------------------


class _FakeFilesResource:
    """Fake Drive files().list() chain."""

    def __init__(self, pages: list[dict[str, Any]]):
        self._pages = pages
        self._call_idx = 0

    def list(self, **kwargs: Any) -> "_FakeFilesResource":
        return self

    def execute(self) -> dict[str, Any]:
        page = self._pages[self._call_idx]
        self._call_idx += 1
        return page


class _FakeService:
    def __init__(self, pages: list[dict[str, Any]]):
        self._files = _FakeFilesResource(pages)

    def files(self) -> _FakeFilesResource:
        return self._files


def test_list_inbox_files_single_page():
    files = [{"id": "f1", "name": "a.mp3"}]
    svc = _FakeService([{"files": files}])
    result = list_inbox_files(svc, "folder-abc")
    assert result == files


def test_list_inbox_files_pagination():
    svc = _FakeService([
        {"files": [{"id": "f1"}], "nextPageToken": "tok"},
        {"files": [{"id": "f2"}]},
    ])
    result = list_inbox_files(svc, "folder-abc")
    assert len(result) == 2
    assert result[0]["id"] == "f1"
    assert result[1]["id"] == "f2"


# --------------------------------------------------------------------------
# _known_drive_file_ids
# --------------------------------------------------------------------------


def test_known_drive_file_ids(tmp_path: Path):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    from lan_app.db import create_recording

    create_recording(
        "rec-1",
        source="gdrive",
        source_filename="a.mp3",
        drive_file_id="drive-1",
        settings=cfg,
    )
    create_recording(
        "rec-2",
        source="test",
        source_filename="b.mp3",
        settings=cfg,
    )
    ids = _known_drive_file_ids(cfg)
    assert ids == {"drive-1"}


# --------------------------------------------------------------------------
# download_file
# --------------------------------------------------------------------------


class _FakeMediaDownload:
    """Simulates MediaIoBaseDownload by writing bytes to the file handle."""

    def __init__(self, fh: Any, request: Any):
        self._fh = fh
        self._data = request._data
        self._done = False

    def next_chunk(self) -> tuple[None, bool]:
        if not self._done:
            self._fh.write(self._data)
            self._done = True
        return None, True


class _FakeGetMediaRequest:
    def __init__(self, data: bytes):
        self._data = data


class _FakeDownloadService:
    def __init__(self, data: bytes):
        self._data = data

    def files(self) -> "_FakeDownloadService":
        return self

    def get_media(self, fileId: str) -> _FakeGetMediaRequest:
        return _FakeGetMediaRequest(self._data)


def test_download_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "lan_app.gdrive.MediaIoBaseDownload", _FakeMediaDownload
    )
    svc = _FakeDownloadService(b"hello audio")
    dest = tmp_path / "raw" / "audio.mp3"
    result = download_file(svc, "file-1", dest)
    assert result == dest
    assert dest.read_bytes() == b"hello audio"


# --------------------------------------------------------------------------
# ingest_once
# --------------------------------------------------------------------------


def test_ingest_once_raises_when_no_sa_path(tmp_path: Path):
    cfg = _test_settings_no_gdrive(tmp_path)
    cfg.gdrive_inbox_folder_id = "folder-abc"
    with pytest.raises(ValueError, match="GDRIVE_SA_JSON_PATH"):
        ingest_once(cfg)


def test_ingest_once_raises_when_no_folder_id(tmp_path: Path):
    cfg = _test_settings_no_gdrive(tmp_path)
    cfg.gdrive_sa_json_path = tmp_path / "sa.json"
    with pytest.raises(ValueError, match="GDRIVE_INBOX_FOLDER_ID"):
        ingest_once(cfg)


def test_ingest_once_raises_when_empty_folder_id(tmp_path: Path):
    cfg = _test_settings_no_gdrive(tmp_path)
    cfg.gdrive_sa_json_path = tmp_path / "sa.json"
    cfg.gdrive_inbox_folder_id = ""
    with pytest.raises(ValueError, match="GDRIVE_INBOX_FOLDER_ID"):
        ingest_once(cfg)


def test_ingest_once_raises_when_whitespace_folder_id(tmp_path: Path):
    cfg = _test_settings_no_gdrive(tmp_path)
    cfg.gdrive_sa_json_path = tmp_path / "sa.json"
    cfg.gdrive_inbox_folder_id = "   "
    with pytest.raises(ValueError, match="GDRIVE_INBOX_FOLDER_ID"):
        ingest_once(cfg)


def test_ingest_once_skips_known_files(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    from lan_app.db import create_recording

    create_recording(
        "rec-existing",
        source="gdrive",
        source_filename="old.mp3",
        drive_file_id="drive-known",
        settings=cfg,
    )

    inbox_files = [
        {
            "id": "drive-known",
            "name": "old.mp3",
            "md5Checksum": "abc",
            "createdTime": "2026-01-01T00:00:00Z",
        },
    ]
    svc = _FakeService([{"files": inbox_files}])

    results = ingest_once(cfg, service=svc)
    assert results == []


def test_ingest_once_downloads_and_creates_recording(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    inbox_files = [
        {
            "id": "drive-new-1",
            "name": "2026-02-18 16_01_43.mp3",
            "md5Checksum": "md5abc",
            "createdTime": "2026-02-18T16:01:43Z",
        },
    ]
    svc = _FakeService([{"files": inbox_files}])

    monkeypatch.setattr(
        "lan_app.gdrive.download_file",
        lambda svc, fid, dest: _write_fake_download(dest),
    )

    results = ingest_once(cfg, service=svc)
    assert len(results) == 1
    result = results[0]
    assert result["drive_file_id"] == "drive-new-1"
    assert result["captured_at"] == "2026-02-18T16:01:43Z"
    assert result["jobs_created"] == len(_PIPELINE_STEPS)
    assert "warning" not in result

    # Verify DB recording
    rec = get_recording(result["recording_id"], settings=cfg)
    assert rec is not None
    assert rec["source"] == "gdrive"
    assert rec["drive_file_id"] == "drive-new-1"
    assert rec["drive_md5"] == "md5abc"
    assert rec["captured_at"] == "2026-02-18T16:01:43Z"

    # Verify DB jobs
    jobs, total = list_jobs(settings=cfg, recording_id=result["recording_id"])
    assert total == len(_PIPELINE_STEPS)


def test_ingest_once_fallback_captured_at_from_drive(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    inbox_files = [
        {
            "id": "drive-noparse",
            "name": "meeting_notes.mp3",
            "md5Checksum": "md5xyz",
            "createdTime": "2026-03-01T10:00:00Z",
        },
    ]
    svc = _FakeService([{"files": inbox_files}])
    monkeypatch.setattr(
        "lan_app.gdrive.download_file",
        lambda svc, fid, dest: _write_fake_download(dest),
    )

    results = ingest_once(cfg, service=svc)
    assert len(results) == 1
    assert results[0]["captured_at"] == "2026-03-01T10:00:00Z"
    assert "warning" in results[0]


def test_ingest_once_continues_on_download_failure(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    inbox_files = [
        {
            "id": "drive-fail",
            "name": "2026-02-18 10_00_00.mp3",
            "createdTime": "2026-02-18T10:00:00Z",
        },
        {
            "id": "drive-ok",
            "name": "2026-02-18 11_00_00.mp3",
            "createdTime": "2026-02-18T11:00:00Z",
        },
    ]
    svc = _FakeService([{"files": inbox_files}])

    def _download_maybe_fail(svc: Any, fid: str, dest: Path) -> Path:
        if fid == "drive-fail":
            # Simulate a partial write before failure
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"partial")
            raise OSError("network error")
        return _write_fake_download(dest)

    monkeypatch.setattr("lan_app.gdrive.download_file", _download_maybe_fail)

    results = ingest_once(cfg, service=svc)
    assert len(results) == 1
    assert results[0]["drive_file_id"] == "drive-ok"


def test_ingest_once_cleans_up_partial_files_on_download_failure(
    tmp_path: Path, monkeypatch
):
    """Failed downloads should not leave orphaned recording directories."""
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    inbox_files = [
        {
            "id": "drive-partial",
            "name": "2026-02-18 10_00_00.mp3",
            "createdTime": "2026-02-18T10:00:00Z",
        },
    ]
    svc = _FakeService([{"files": inbox_files}])

    created_dirs: list[Path] = []

    def _download_fail(svc: Any, fid: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial data")
        created_dirs.append(dest.parent.parent)  # recordings/<id>
        raise OSError("network error")

    monkeypatch.setattr("lan_app.gdrive.download_file", _download_fail)

    results = ingest_once(cfg, service=svc)
    assert results == []
    assert len(created_dirs) == 1
    # The orphaned directory should have been cleaned up
    assert not created_dirs[0].exists()


def test_ingest_idempotent(tmp_path: Path, monkeypatch):
    """Running ingest twice with the same files should not duplicate."""
    cfg = _test_settings(tmp_path)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    inbox_files = [
        {
            "id": "drive-idem",
            "name": "2026-02-18 16_01_43.mp3",
            "md5Checksum": "md5idem",
            "createdTime": "2026-02-18T16:01:43Z",
        },
    ]
    svc = _FakeService([{"files": inbox_files}])
    monkeypatch.setattr(
        "lan_app.gdrive.download_file",
        lambda svc, fid, dest: _write_fake_download(dest),
    )

    first = ingest_once(cfg, service=svc)
    assert len(first) == 1

    # Second call: same file already known
    svc2 = _FakeService([{"files": inbox_files}])
    second = ingest_once(cfg, service=svc2)
    assert len(second) == 0

    # Only 1 recording in DB
    recs, total = list_recordings(settings=cfg)
    assert total == 1


# --------------------------------------------------------------------------
# API endpoint
# --------------------------------------------------------------------------


def test_api_ingest_endpoint_returns_422_when_not_configured(
    tmp_path: Path, monkeypatch
):
    cfg = _test_settings_no_gdrive(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    client = TestClient(api.app)
    resp = client.post("/api/actions/ingest")
    assert resp.status_code == 422


def test_api_ingest_endpoint_success(tmp_path: Path, monkeypatch):
    cfg = _test_settings(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)

    def _fake_ingest(settings: Any = None, *, service: Any = None) -> list[dict]:
        return [{"recording_id": "trs_test1", "drive_file_id": "f1"}]

    monkeypatch.setattr("lan_app.gdrive.ingest_once", _fake_ingest)

    client = TestClient(api.app)
    resp = client.post("/api/actions/ingest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["ingested"][0]["recording_id"] == "trs_test1"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _write_fake_download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"fake audio data")
    return dest
