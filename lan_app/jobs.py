from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from redis import Redis
from rq import Queue
from rq.registry import DeferredJobRegistry, ScheduledJobRegistry

from .config import AppSettings
from .constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    JOB_STATUS_QUEUED,
    JOB_TYPES,
    RECORDING_STATUS_QUEUED,
)
from .db import (
    create_job_if_no_active_for_recording,
    create_job,
    fail_job,
    get_recording,
    init_db,
    list_jobs,
    set_recording_status,
)


class RecordingNotFoundError(ValueError):
    """Raised when trying to enqueue work for an unknown recording."""


class DuplicateRecordingJobError(ValueError):
    """Raised when an active precheck job already exists for a recording."""

    def __init__(self, *, recording_id: str, job_id: str):
        self.recording_id = recording_id
        self.job_id = job_id
        super().__init__(
            f"recording {recording_id} already has queued/started job {job_id}"
        )


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


def _status_value(status: object | None) -> str | None:
    if status is None:
        return None
    return getattr(status, "value", str(status))


def _purge_pending_queue_job(queue: Queue, job_id: str) -> bool:
    job = queue.fetch_job(job_id)
    if job is None:
        return False

    status = _status_value(job.get_status(refresh=True))
    if status not in {"queued", "deferred", "scheduled"}:
        return False

    if status == "queued":
        queue.remove(job_id)
    elif status == "deferred":
        DeferredJobRegistry(queue=queue).remove(job_id, delete_job=False)
    elif status == "scheduled":
        ScheduledJobRegistry(queue=queue).remove(job_id, delete_job=False)

    job.delete(remove_from_queue=True, delete_dependents=True)
    return True


def purge_pending_recording_jobs(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> int:
    cfg = settings or AppSettings()
    init_db(cfg)
    queue = get_queue(cfg)

    removed = 0
    offset = 0
    while True:
        rows, total = list_jobs(
            settings=cfg,
            status=JOB_STATUS_QUEUED,
            recording_id=recording_id,
            limit=500,
            offset=offset,
        )
        if not rows:
            break
        for row in rows:
            job_id = str(row["id"])
            if _purge_pending_queue_job(queue, job_id):
                removed += 1
        offset += len(rows)
        if offset >= total:
            break
    return removed


def cancel_pending_queue_job(
    job_id: str,
    *,
    settings: AppSettings | None = None,
) -> bool:
    cfg = settings or AppSettings()
    try:
        queue = get_queue(cfg)
    except Exception:
        return False
    try:
        return _purge_pending_queue_job(queue, job_id)
    except Exception:
        return False


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
    if job_type == DEFAULT_REQUEUE_JOB_TYPE:
        _created, existing = create_job_if_no_active_for_recording(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
            settings=cfg,
            status=JOB_STATUS_QUEUED,
        )
        if existing is not None:
            existing_job_id = str(existing.get("id") or "").strip()
            if existing_job_id:
                raise DuplicateRecordingJobError(
                    recording_id=recording_id,
                    job_id=existing_job_id,
                )
    else:
        create_job(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
            status=JOB_STATUS_QUEUED,
            settings=cfg,
        )

    from .worker_tasks import process_job

    queue = get_queue(cfg)
    try:
        queue.enqueue(
            process_job,
            job_id,
            recording_id,
            job_type,
            job_id=job_id,
            job_timeout=cfg.rq_job_timeout_seconds,
        )
    except Exception as exc:
        # Keep DB queue state terminal when Redis/RQ enqueue fails.
        try:
            fail_job(
                job_id,
                error=f"queue enqueue failed: {exc}",
                settings=cfg,
            )
        except Exception:
            pass
        raise

    set_recording_status(
        recording_id,
        RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    return RecordingJob(job_id=job_id, recording_id=recording_id, job_type=job_type)


__all__ = [
    "cancel_pending_queue_job",
    "DuplicateRecordingJobError",
    "RecordingJob",
    "RecordingNotFoundError",
    "enqueue_recording_job",
    "get_queue",
    "purge_pending_recording_jobs",
]
