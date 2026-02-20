from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import ModuleType

from fastapi.testclient import TestClient
import pytest

from lan_app import api
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FAILED,
    JOB_STATUS_FINISHED,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUS_FAILED,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_READY,
)
from lan_app.db import (
    connect,
    create_job,
    create_recording,
    get_job,
    get_recording,
    init_db,
    list_jobs,
    set_recording_status,
)
from lan_app.jobs import RecordingJob, enqueue_recording_job, purge_pending_recording_jobs
from lan_app.worker_tasks import process_job
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
    assert recording["status"] == RECORDING_STATUS_QUARANTINE
    assert recording["quarantine_reason"] == "raw_audio_missing"
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


def test_worker_precheck_quarantines_and_skips_pipeline(tmp_path: Path, monkeypatch):
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

    async def _should_not_run(*_args, **_kwargs):
        raise AssertionError("run_pipeline should be skipped for quarantined recordings")

    monkeypatch.setattr("lan_app.worker_tasks.run_pipeline", _should_not_run)

    result = process_job("job-precheck-q-1", "rec-precheck-q-1", JOB_TYPE_PRECHECK)
    assert result["status"] == "ok"

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

    init_db(cfg)
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

    async def _fake_run_pipeline(*_args, **_kwargs):
        called["value"] = True
        return None

    monkeypatch.setattr("lan_app.worker_tasks.run_pipeline", _fake_run_pipeline)

    result = process_job("job-precheck-ok-1", "rec-precheck-ok-1", JOB_TYPE_PRECHECK)
    assert result["status"] == "ok"
    assert called["value"] is True

    recording = get_recording("rec-precheck-ok-1", settings=cfg)
    job = get_job("job-precheck-ok-1", settings=cfg)
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_READY
    assert job is not None
    assert job["status"] == JOB_STATUS_FINISHED


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
