from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from redis import Redis
from rq import Queue

from .config import AppSettings
from .constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    JOB_STATUS_QUEUED,
    JOB_TYPES,
)
from .db import create_job, get_recording, init_db


class RecordingNotFoundError(ValueError):
    """Raised when trying to enqueue work for an unknown recording."""


def _validate_job_type(job_type: str) -> None:
    if job_type not in JOB_TYPES:
        raise ValueError(f"Unsupported job type: {job_type}")


@dataclass(frozen=True)
class RecordingJob:
    """Queue envelope persisted in DB and mirrored into Redis RQ."""

    job_id: str
    recording_id: str
    job_type: str
    audio_path: Path | None = None


def get_queue(settings: AppSettings | None = None) -> Queue:
    cfg = settings or AppSettings()
    connection = Redis.from_url(cfg.redis_url)
    return Queue(name=cfg.rq_queue_name, connection=connection)


def enqueue_recording_job(
    recording_id: str,
    *,
    job_type: str = DEFAULT_REQUEUE_JOB_TYPE,
    settings: AppSettings | None = None,
) -> RecordingJob:
    cfg = settings or AppSettings()
    init_db(cfg)
    _validate_job_type(job_type)
    if get_recording(recording_id, settings=cfg) is None:
        raise RecordingNotFoundError(f"Recording not found: {recording_id}")

    job_id = uuid4().hex
    create_job(
        job_id=job_id,
        recording_id=recording_id,
        job_type=job_type,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
    )

    from .worker_tasks import process_job

    queue = get_queue(cfg)
    queue.enqueue(
        process_job,
        job_id,
        recording_id,
        job_type,
        job_id=job_id,
    )
    return RecordingJob(job_id=job_id, recording_id=recording_id, job_type=job_type)


__all__ = [
    "RecordingJob",
    "RecordingNotFoundError",
    "enqueue_recording_job",
    "get_queue",
]
