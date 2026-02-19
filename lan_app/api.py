from __future__ import annotations

import asyncio
import shutil
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from lan_transcriber import aliases
from lan_transcriber.metrics import write_metrics_snapshot
from lan_transcriber.models import TranscriptResult
from lan_transcriber.pipeline import refresh_aliases

from .config import AppSettings
from .constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    JOB_STATUSES,
    JOB_TYPES,
    RECORDING_STATUSES,
    RECORDING_STATUS_QUARANTINE,
)
from .db import (
    delete_recording,
    get_recording,
    init_db,
    list_jobs,
    list_recordings,
    set_recording_status,
)
from .jobs import RecordingNotFoundError, enqueue_recording_job

app = FastAPI()
ALIAS_PATH = aliases.ALIAS_PATH
_subscribers: List[asyncio.Queue[str]] = []
_current_result: TranscriptResult | None = None
_settings = AppSettings()


class AliasUpdate(BaseModel):
    alias: str


class RequeueAction(BaseModel):
    job_type: str = DEFAULT_REQUEUE_JOB_TYPE


class QuarantineAction(BaseModel):
    reason: str | None = None


def _validate_recording_status(status: str | None) -> str | None:
    if status is None:
        return None
    if status not in RECORDING_STATUSES:
        raise HTTPException(status_code=422, detail=f"Unsupported recording status: {status}")
    return status


def _validate_job_status(status: str | None) -> str | None:
    if status is None:
        return None
    if status not in JOB_STATUSES:
        raise HTTPException(status_code=422, detail=f"Unsupported job status: {status}")
    return status


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Simple health check used by monitoring."""
    return {"status": "ok"}


@app.on_event("startup")
async def _start_metrics() -> None:
    init_db(_settings)
    _settings.metrics_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(write_metrics_snapshot(_settings.metrics_snapshot_path))


@app.post("/alias/{speaker_id}")
async def update_alias(speaker_id: str, upd: AliasUpdate):
    path = aliases.ALIAS_PATH
    known = aliases.load_aliases(path)
    known[speaker_id] = upd.alias
    aliases.save_aliases(known, path)
    if _current_result is not None:
        refresh_aliases(_current_result, path)
    for queue in list(_subscribers):
        queue.put_nowait("updated")
    return {"speaker": speaker_id, "alias": upd.alias}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/events")
async def events():
    queue: asyncio.Queue[str] = asyncio.Queue()
    _subscribers.append(queue)

    async def gen():
        try:
            while True:
                await queue.get()
                yield "event: speaker_alias_updated\ndata: updated\n\n"
        finally:
            _subscribers.remove(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/recordings")
async def api_list_recordings(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    valid_status = _validate_recording_status(status)
    items, total = list_recordings(
        settings=_settings,
        status=valid_status,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/recordings/{recording_id}")
async def api_get_recording(recording_id: str) -> dict[str, object]:
    item = get_recording(recording_id, settings=_settings)
    if item is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return item


@app.post("/api/recordings/{recording_id}/actions/requeue")
async def api_requeue_recording(
    recording_id: str,
    action: RequeueAction | None = None,
) -> dict[str, str]:
    payload = action or RequeueAction()
    if payload.job_type not in JOB_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported job type: {payload.job_type}")

    if get_recording(recording_id, settings=_settings) is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    try:
        job = enqueue_recording_job(
            recording_id,
            job_type=payload.job_type,
            settings=_settings,
        )
    except RecordingNotFoundError:
        raise HTTPException(status_code=404, detail="Recording not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}")

    return {
        "recording_id": recording_id,
        "job_id": job.job_id,
        "job_type": job.job_type,
    }


@app.post("/api/recordings/{recording_id}/actions/quarantine")
async def api_quarantine_recording(
    recording_id: str,
    action: QuarantineAction | None = None,
) -> dict[str, object]:
    if get_recording(recording_id, settings=_settings) is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    payload = action or QuarantineAction()
    set_recording_status(
        recording_id,
        RECORDING_STATUS_QUARANTINE,
        settings=_settings,
        quarantine_reason=payload.reason,
    )
    item = get_recording(recording_id, settings=_settings)
    if item is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return item


@app.post("/api/recordings/{recording_id}/actions/delete")
async def api_delete_recording(recording_id: str) -> dict[str, object]:
    deleted = delete_recording(recording_id, settings=_settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="Recording not found")

    recording_path = _settings.recordings_root / recording_id
    shutil.rmtree(recording_path, ignore_errors=True)
    return {"recording_id": recording_id, "deleted": True}


@app.get("/api/jobs")
async def api_list_jobs(
    status: str | None = Query(default=None),
    recording_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    valid_status = _validate_job_status(status)
    items, total = list_jobs(
        settings=_settings,
        status=valid_status,
        recording_id=recording_id,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def set_current_result(result: TranscriptResult | None) -> None:
    global _current_result
    _current_result = result


__all__ = ["ALIAS_PATH", "app", "set_current_result", "healthz"]
