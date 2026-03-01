from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from rq import SimpleWorker

from lan_app import worker_tasks
from lan_app.config import AppSettings
from lan_app.constants import JOB_STATUS_FINISHED, RECORDING_STATUS_READY
from lan_app.db import create_recording, get_job, get_recording, init_db
from lan_app.jobs import enqueue_recording_job, get_queue

_REDIS_DB15_URL = "redis://127.0.0.1:6379/15"


def _is_github_actions() -> bool:
    return os.getenv("GITHUB_ACTIONS", "").strip().lower() == "true"


def _set_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LAN_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LAN_RECORDINGS_ROOT", str(tmp_path / "recordings"))
    monkeypatch.setenv("LAN_DB_PATH", str(tmp_path / "db" / "app.db"))
    monkeypatch.setenv("LAN_REDIS_URL", _REDIS_DB15_URL)


@pytest.fixture
def redis_db_15() -> Iterator[Redis]:
    client = Redis.from_url(_REDIS_DB15_URL)
    try:
        client.ping()
    except (RedisConnectionError, RedisTimeoutError) as exc:
        if _is_github_actions():
            raise
        pytest.skip(f"Redis is unavailable at {_REDIS_DB15_URL}: {exc}")
    client.flushdb()
    try:
        yield client
    finally:
        try:
            client.flushdb()
        except (RedisConnectionError, RedisTimeoutError):
            pass
        client.close()


def test_enqueue_job_runs_via_rq_simple_worker_and_updates_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    redis_db_15: Redis,
) -> None:
    assert redis_db_15 is not None
    _set_runtime_env(monkeypatch, tmp_path)
    settings = AppSettings()
    init_db(settings)

    recording_id = "rec-rq-integration-1"
    create_recording(
        recording_id,
        source="upload",
        source_filename="integration.wav",
        settings=settings,
    )

    def _fast_precheck_stub(
        *,
        recording_id: str,
        settings: AppSettings,
        log_path: Path,
    ) -> tuple[str, str | None]:
        assert recording_id == "rec-rq-integration-1"
        assert settings.db_path == tmp_path / "db" / "app.db"
        assert log_path.name == "step-precheck.log"
        return RECORDING_STATUS_READY, None

    monkeypatch.setattr(worker_tasks, "_run_precheck_pipeline", _fast_precheck_stub)

    queued_job = enqueue_recording_job(recording_id, settings=settings)
    queue = get_queue(settings)
    worker = SimpleWorker([queue], connection=queue.connection)
    worker.work(burst=True, with_scheduler=False)

    db_job = get_job(queued_job.job_id, settings=settings)
    assert db_job is not None
    assert db_job["status"] == JOB_STATUS_FINISHED
    assert db_job["error"] is None
    assert db_job["finished_at"] is not None

    recording = get_recording(recording_id, settings=settings)
    assert recording is not None
    assert recording["status"] == RECORDING_STATUS_READY
    assert recording["quarantine_reason"] is None
