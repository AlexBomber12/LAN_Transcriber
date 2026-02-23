from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any

from .config import AppSettings
from .constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    RECORDING_STATUS_NEEDS_REVIEW,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_QUEUED,
)
from .db import (
    fail_job_if_queued,
    fail_job_if_started,
    list_processing_recordings_without_started_job,
    list_stale_started_jobs,
    set_recording_status_if_current_in_and_no_started_job,
)
from .jobs import cancel_pending_queue_job

_LOG = logging.getLogger(__name__)
_RECOVERY_ERROR = "stuck job recovered"
_STALE_DOWNGRADE_STATUSES = (
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_NEEDS_REVIEW,
)
_ORPHAN_DOWNGRADE_STATUSES = (RECORDING_STATUS_PROCESSING,)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(microsecond=0)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _step_log_path(recording_id: str, job_type: str, settings: AppSettings) -> Path:
    return settings.recordings_root / recording_id / "logs" / f"step-{job_type}.log"


def _append_step_log(path: Path, message: str, *, now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{_iso_z(now)}] {message}\n")


def _append_step_log_best_effort(path: Path, message: str, *, now: datetime) -> None:
    try:
        _append_step_log(path, message, now=now)
    except OSError:
        _LOG.warning(
            "stuck job recovery step-log append failed path=%s",
            path,
            exc_info=True,
        )


def run_stuck_job_reaper_once(
    *,
    settings: AppSettings | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    current_time = now or _utc_now()
    threshold = max(int(cfg.stuck_job_seconds), 1)
    stale_before = _iso_z(current_time - timedelta(seconds=threshold))

    recovered_job_ids: list[str] = []
    recovered_recording_ids: set[str] = set()
    stale_rows = list_stale_started_jobs(
        before_started_at=stale_before,
        settings=cfg,
    )
    for row in stale_rows:
        job_id = str(row.get("id") or "").strip()
        recording_id = str(row.get("recording_id") or "").strip()
        job_type = str(row.get("type") or "").strip() or DEFAULT_REQUEUE_JOB_TYPE
        if not job_id or not recording_id:
            continue
        if not fail_job_if_started(job_id, _RECOVERY_ERROR, settings=cfg):
            # Job completed or moved to another state after selection.
            continue
        if set_recording_status_if_current_in_and_no_started_job(
            recording_id,
            RECORDING_STATUS_NEEDS_REVIEW,
            current_statuses=_STALE_DOWNGRADE_STATUSES,
            settings=cfg,
        ):
            recovered_recording_ids.add(recording_id)
        _append_step_log_best_effort(
            _step_log_path(recording_id, job_type, cfg),
            f"stuck job recovery applied job={job_id}",
            now=current_time,
        )
        recovered_job_ids.append(job_id)

    processing_rows = list_processing_recordings_without_started_job(
        settings=cfg,
        before_updated_at=stale_before,
    )
    for row in processing_rows:
        recording_id = str(row.get("id") or "").strip()
        if not recording_id:
            continue
        active_job_id = str(row.get("active_job_id") or "").strip()
        active_job_type = (
            str(row.get("active_job_type") or "").strip() or DEFAULT_REQUEUE_JOB_TYPE
        )
        if active_job_id:
            if not fail_job_if_queued(active_job_id, _RECOVERY_ERROR, settings=cfg):
                # Job transitioned after selection; skip this recovery pass.
                continue
            # Best-effort dequeue to avoid executing a recovered queued job later.
            cancel_pending_queue_job(active_job_id, settings=cfg)
            recovered_job_ids.append(active_job_id)
        if set_recording_status_if_current_in_and_no_started_job(
            recording_id,
            RECORDING_STATUS_NEEDS_REVIEW,
            current_statuses=_ORPHAN_DOWNGRADE_STATUSES,
            settings=cfg,
        ):
            recovered_recording_ids.add(recording_id)
        _append_step_log_best_effort(
            _step_log_path(recording_id, active_job_type, cfg),
            (
                "stuck job recovery applied "
                f"recording={recording_id} "
                f"job={active_job_id or 'none'}"
            ),
            now=current_time,
        )

    return {
        "stale_started_jobs": len(stale_rows),
        "processing_without_started": len(processing_rows),
        "recovered_jobs": len(recovered_job_ids),
        "recovered_recordings": len(recovered_recording_ids),
        "job_ids": recovered_job_ids,
        "recording_ids": sorted(recovered_recording_ids),
    }


__all__ = ["run_stuck_job_reaper_once"]
