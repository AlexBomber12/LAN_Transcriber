from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from lan_app import api, uploads
from lan_app.config import AppSettings
from lan_app.constants import JOB_STATUS_QUEUED, JOB_TYPE_PRECHECK, RECORDING_STATUS_QUEUED
from lan_app.db import (
    create_recording,
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
        force_reprocess: bool = False,
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
    assert payload["captured_at"] == "2026-02-18T15:01:43Z"
    assert payload["bytes_written"] == 3

    raw_file = cfg.recordings_root / recording_id / "raw" / "audio.mp3"
    assert raw_file.exists()
    assert raw_file.read_bytes() == b"abc"

    recording = get_recording(recording_id, settings=cfg)
    assert recording is not None
    assert recording["source"] == "upload"
    assert ".mp3" in str(recording["source_filename"])
    assert recording["captured_at"] == "2026-02-18T15:01:43Z"
    assert recording["captured_at_source"] == "2026-02-18T16:01:43"
    assert recording["captured_at_timezone"] == "Europe/Rome"
    assert recording["captured_at_inferred_from_filename"] == 1

    jobs, total = list_jobs(settings=cfg, recording_id=recording_id)
    assert total == 1
    assert jobs[0]["status"] == JOB_STATUS_QUEUED
    assert jobs[0]["type"] == JOB_TYPE_PRECHECK


def test_upload_without_embedded_timestamp_falls_back_to_current_utc(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    _stub_enqueue(monkeypatch, cfg)

    fixed_now = datetime(2026, 2, 3, 4, 5, 6, 789123, tzinfo=timezone.utc)

    class _FixedDateTime:
        @staticmethod
        def now(*, tz):
            assert tz == timezone.utc
            return fixed_now

    monkeypatch.setattr(uploads, "datetime", _FixedDateTime)

    client = TestClient(api.app)
    response = client.post(
        "/api/uploads",
        files={"file": ("plain-name.mp3", b"abc", "audio/mpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["captured_at"] == "2026-02-03T04:05:06Z"

    recording = get_recording(str(payload["recording_id"]), settings=cfg)
    assert recording is not None
    assert recording["captured_at"] == "2026-02-03T04:05:06Z"
    assert recording["captured_at_source"] is None
    assert recording["captured_at_timezone"] == "Europe/Rome"
    assert recording["captured_at_inferred_from_filename"] == 0


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


def test_upload_queue_failure_rolls_back_recording(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)

    def _fail_enqueue(*_args, **_kwargs):
        raise RuntimeError("redis down")

    monkeypatch.setattr(api, "enqueue_recording_job", _fail_enqueue)

    client = TestClient(api.app)
    response = client.post(
        "/api/uploads",
        files={"file": ("2026-02-18 16_01_43.mp3", b"abc", "audio/mpeg")},
    )
    assert response.status_code == 503
    assert "Queue unavailable" in response.json()["detail"]

    items, total = list_recordings(settings=cfg)
    assert total == 0
    assert items == []
    assert list(cfg.recordings_root.glob("trs_*")) == []


def test_find_matching_upload_recording_returns_existing_id(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-missing-raw", source="upload", source_filename="missing.mp3", settings=cfg)
    create_recording("rec-match", source="upload", source_filename="match.mp3", settings=cfg)
    raw_dir = cfg.recordings_root / "rec-match" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "audio.mp3").write_bytes(b"same-audio")
    upload_path = tmp_path / "incoming.mp3"
    upload_path.write_bytes(b"same-audio")

    assert uploads.find_matching_upload_recording(upload_path, settings=cfg) == "rec-match"


def test_find_matching_upload_recording_returns_none_for_missing_or_different_upload(
    tmp_path: Path,
):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    upload_path = tmp_path / "incoming.mp3"
    upload_path.write_bytes(b"same-size")
    assert uploads.find_matching_upload_recording(upload_path, settings=cfg) is None

    create_recording("rec-other", source="upload", source_filename="other.mp3", settings=cfg)
    raw_dir = cfg.recordings_root / "rec-other" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "audio.mp3").write_bytes(b"different")

    missing_upload = tmp_path / "missing.mp3"
    assert uploads.find_matching_upload_recording(missing_upload, settings=cfg) is None

    assert uploads.find_matching_upload_recording(upload_path, settings=cfg) is None


def test_find_matching_upload_recording_returns_none_when_upload_stat_fails(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    upload_path = tmp_path / "incoming.mp3"
    upload_path.write_bytes(b"abc")

    real_stat = Path.stat

    def _broken_stat(self: Path, *args, **kwargs):
        if self == upload_path:
            raise OSError("bad upload stat")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _broken_stat)

    assert uploads.find_matching_upload_recording(upload_path, settings=cfg) is None


def test_find_matching_upload_recording_returns_none_for_directory_path(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    upload_dir = tmp_path / "incoming-dir"
    upload_dir.mkdir()

    assert uploads.find_matching_upload_recording(upload_dir, settings=cfg) is None


def test_find_matching_upload_recording_skips_unreadable_existing_audio(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-bad", source="upload", source_filename="bad.mp3", settings=cfg)
    raw_audio = cfg.recordings_root / "rec-bad" / "raw" / "audio.mp3"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"abc")
    upload_path = tmp_path / "incoming.mp3"
    upload_path.write_bytes(b"abc")

    real_sha256 = uploads._sha256_file

    def _broken_sha256(path: Path) -> str:
        if path == raw_audio:
            raise OSError("bad raw audio")
        return real_sha256(path)

    monkeypatch.setattr(uploads, "_sha256_file", _broken_sha256)

    assert uploads.find_matching_upload_recording(upload_path, settings=cfg) is None


def test_find_matching_upload_recording_skips_size_mismatch(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-size", source="upload", source_filename="size.mp3", settings=cfg)
    raw_audio = cfg.recordings_root / "rec-size" / "raw" / "audio.mp3"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"abcd")
    upload_path = tmp_path / "incoming.mp3"
    upload_path.write_bytes(b"abc")

    assert uploads.find_matching_upload_recording(upload_path, settings=cfg) is None


def test_reupload_reuses_existing_recording_and_enqueues_force_reprocess(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)
    caplog.set_level("INFO")

    observed: list[dict[str, object]] = []

    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = JOB_TYPE_PRECHECK,
        force_reprocess: bool = False,
        settings: AppSettings | None = None,
    ) -> RecordingJob:
        observed.append(
            {
                "recording_id": recording_id,
                "job_type": job_type,
                "force_reprocess": force_reprocess,
            }
        )
        return RecordingJob(
            job_id=f"job-{len(observed)}",
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)

    client = TestClient(api.app)
    first = client.post(
        "/api/uploads",
        files={"file": ("meeting.mp3", b"abc", "audio/mpeg")},
    )
    assert first.status_code == 200
    first_payload = first.json()

    caplog.clear()
    second = client.post(
        "/api/uploads",
        files={"file": ("meeting-copy.mp3", b"abc", "audio/mpeg")},
    )
    assert second.status_code == 200
    second_payload = second.json()

    assert second_payload["recording_id"] == first_payload["recording_id"]
    assert second_payload["captured_at"] == first_payload["captured_at"]
    items, total = list_recordings(settings=cfg)
    assert total == 1
    assert items[0]["id"] == first_payload["recording_id"]
    assert observed == [
        {
            "recording_id": first_payload["recording_id"],
            "job_type": JOB_TYPE_PRECHECK,
            "force_reprocess": False,
        },
        {
            "recording_id": first_payload["recording_id"],
            "job_type": JOB_TYPE_PRECHECK,
            "force_reprocess": True,
        },
    ]
    assert f"re-upload detected for {first_payload['recording_id']}" in caplog.text
    upload_dirs = sorted(cfg.recordings_root.glob("trs_*"))
    assert len(upload_dirs) == 1


def test_reupload_active_job_conflict_returns_409(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)

    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = JOB_TYPE_PRECHECK,
        force_reprocess: bool = False,
        settings: AppSettings | None = None,
    ) -> RecordingJob:
        if force_reprocess:
            from lan_app.jobs import DuplicateRecordingJobError

            raise DuplicateRecordingJobError(recording_id=recording_id, job_id="existing-job")
        return RecordingJob(
            job_id="job-first",
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)

    client = TestClient(api.app)
    first = client.post(
        "/api/uploads",
        files={"file": ("meeting.mp3", b"abc", "audio/mpeg")},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/uploads",
        files={"file": ("meeting-again.mp3", b"abc", "audio/mpeg")},
    )
    assert second.status_code == 409
    assert second.json()["detail"]["existing_job_id"] == "existing-job"
    items, total = list_recordings(settings=cfg)
    assert total == 1
    assert len(sorted(cfg.recordings_root.glob("trs_*"))) == 1


def test_reupload_queue_failure_returns_503(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_db(cfg)

    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = JOB_TYPE_PRECHECK,
        force_reprocess: bool = False,
        settings: AppSettings | None = None,
    ) -> RecordingJob:
        if force_reprocess:
            raise RuntimeError("redis down")
        return RecordingJob(
            job_id="job-first",
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)

    client = TestClient(api.app)
    first = client.post(
        "/api/uploads",
        files={"file": ("meeting.mp3", b"abc", "audio/mpeg")},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/uploads",
        files={"file": ("meeting-again.mp3", b"abc", "audio/mpeg")},
    )
    assert second.status_code == 503
    assert second.json()["detail"] == "Queue unavailable: redis down"
    items, total = list_recordings(settings=cfg)
    assert total == 1
    assert len(sorted(cfg.recordings_root.glob("trs_*"))) == 1
