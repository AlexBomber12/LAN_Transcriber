from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import AppSettings
from .constants import (
    JOB_TYPE_CLEANUP,
    JOB_TYPE_PUBLISH,
    JOB_TYPES,
    RECORDING_STATUS_FAILED,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_PUBLISHED,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_READY,
)
from .db import fail_job, finish_job, init_db, set_recording_status, start_job


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _step_log_path(recording_id: str, job_type: str, settings: AppSettings) -> Path:
    return settings.recordings_root / recording_id / "logs" / f"step-{job_type}.log"


def _append_step_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{_utc_now()}] {message}\n")


def _success_status(job_type: str) -> str:
    if job_type == JOB_TYPE_PUBLISH:
        return RECORDING_STATUS_PUBLISHED
    if job_type == JOB_TYPE_CLEANUP:
        return RECORDING_STATUS_QUARANTINE
    return RECORDING_STATUS_READY


def _record_failure(
    *,
    job_id: str,
    job_type: str,
    recording_id: str,
    settings: AppSettings,
    log_path: Path,
    exc: Exception,
) -> None:
    error = str(exc)
    try:
        fail_job(job_id, error, settings=settings)
    except Exception:
        pass
    try:
        set_recording_status(recording_id, RECORDING_STATUS_FAILED, settings=settings)
    except Exception:
        pass
    try:
        _append_step_log(log_path, f"failed job={job_id} type={job_type}: {error}")
    except Exception:
        pass


def process_job(job_id: str, recording_id: str, job_type: str) -> dict[str, str]:
    """
    Execute a queue job.

    MVP behavior intentionally runs a no-op body and only records lifecycle state.
    """

    if job_type not in JOB_TYPES:
        raise ValueError(f"Unsupported job type: {job_type}")

    settings = AppSettings()
    init_db(settings)
    log_path = _step_log_path(recording_id, job_type, settings)

    try:
        if not start_job(job_id, settings=settings):
            raise ValueError(f"Job not found: {job_id}")
        if not set_recording_status(
            recording_id,
            RECORDING_STATUS_PROCESSING,
            settings=settings,
        ):
            raise ValueError(f"Recording not found: {recording_id}")
        _append_step_log(log_path, f"started job={job_id} type={job_type}")

        final_status = _success_status(job_type)
        if not set_recording_status(recording_id, final_status, settings=settings):
            raise ValueError(f"Recording not found: {recording_id}")
        if not finish_job(job_id, settings=settings):
            raise ValueError(f"Job not found: {job_id}")
        _append_step_log(
            log_path,
            f"finished job={job_id} type={job_type} recording_status={final_status}",
        )
    except Exception as exc:
        _record_failure(
            job_id=job_id,
            job_type=job_type,
            recording_id=recording_id,
            settings=settings,
            log_path=log_path,
            exc=exc,
        )
        raise

    return {
        "job_id": job_id,
        "recording_id": recording_id,
        "job_type": job_type,
        "status": "ok",
    }


__all__ = ["process_job"]
