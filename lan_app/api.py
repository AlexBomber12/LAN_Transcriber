from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import shutil
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from lan_transcriber import aliases
from lan_transcriber.metrics import write_metrics_snapshot
from lan_transcriber.models import TranscriptResult
from lan_transcriber.pipeline import refresh_aliases

from .calendar import (
    load_calendar_context,
    refresh_calendar_context,
    select_calendar_event,
)
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
from .jobs import purge_pending_recording_jobs
from .healthchecks import (
    check_app_health,
    check_db_health,
    check_redis_health,
    check_worker_health,
    collect_health_checks,
)
from .ms_graph import (
    GraphAuthError,
    GraphDeviceFlowLimitError,
    GraphNotConfiguredError,
    get_device_flow_session,
    ms_connection_state,
    start_device_flow_session,
)
from .onenote import PublishPreconditionError, publish_recording_to_onenote
from .ops import run_retention_cleanup
from .ui_routes import _STATIC_DIR, ui_router

app = FastAPI()
app.include_router(ui_router)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
ALIAS_PATH = aliases.ALIAS_PATH
_subscribers: List[asyncio.Queue[str]] = []
_current_result: TranscriptResult | None = None
_settings = AppSettings()
_cleanup_task: asyncio.Task[None] | None = None
_CLEANUP_INTERVAL_SECONDS = 3600
_logger = logging.getLogger(__name__)


class AliasUpdate(BaseModel):
    alias: str


class RequeueAction(BaseModel):
    job_type: str = DEFAULT_REQUEUE_JOB_TYPE


class QuarantineAction(BaseModel):
    reason: str | None = None


class CalendarSelectAction(BaseModel):
    event_id: str | None = None


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
async def healthz() -> dict[str, object]:
    checks = await run_in_threadpool(collect_health_checks, _settings)
    return {
        "status": "ok" if all(item["ok"] for item in checks.values()) else "degraded",
        "checks": checks,
    }


def _health_check_by_component(component: str, settings: AppSettings) -> dict[str, object]:
    if component == "app":
        return check_app_health()
    if component == "db":
        return check_db_health(settings)
    if component == "redis":
        return check_redis_health(settings)
    if component == "worker":
        return check_worker_health(settings)
    raise KeyError(component)


@app.get("/healthz/{component}")
async def healthz_component(component: str) -> dict[str, object]:
    try:
        payload = await run_in_threadpool(_health_check_by_component, component, _settings)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown health component")
    if not payload["ok"]:
        raise HTTPException(status_code=503, detail=str(payload["detail"]))
    return payload


async def _retention_cleanup_loop() -> None:
    while True:
        try:
            await run_in_threadpool(run_retention_cleanup, settings=_settings)
        except Exception:
            _logger.exception("Retention cleanup job failed")
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)


@app.get("/api/connections/ms/verify")
async def api_verify_ms_connection() -> dict[str, object]:
    """Validate Microsoft Graph auth by calling /me via cached delegated token."""
    state = await run_in_threadpool(ms_connection_state, _settings)
    if state["status"] != "connected":
        return {
            "ok": False,
            "error": state["status"],
            "detail": state.get("error"),
            "account_display_name": state.get("account_display_name"),
            "tenant_id": state.get("tenant_id"),
            "granted_scopes": state.get("granted_scopes", []),
        }
    return {
        "ok": True,
        "account_display_name": state.get("account_display_name"),
        "tenant_id": state.get("tenant_id"),
        "granted_scopes": state.get("granted_scopes", []),
    }


@app.post("/api/connections/ms/connect")
async def api_start_ms_connection(
    reconnect: bool = Query(default=False),
) -> dict[str, object]:
    """Start device-code flow and return code/URL details for UI polling."""
    try:
        return await run_in_threadpool(
            start_device_flow_session,
            _settings,
            reconnect=reconnect,
        )
    except GraphNotConfiguredError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except GraphDeviceFlowLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except GraphAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/connections/ms/connect/{session_id}")
async def api_get_ms_connection_status(session_id: str) -> dict[str, object]:
    try:
        return get_device_flow_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown device-flow session")


@app.on_event("startup")
async def _start_metrics() -> None:
    global _cleanup_task
    init_db(_settings)
    _settings.metrics_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(write_metrics_snapshot(_settings.metrics_snapshot_path))
    _cleanup_task = asyncio.create_task(_retention_cleanup_loop())


@app.on_event("shutdown")
async def _stop_background_tasks() -> None:
    global _cleanup_task
    if _cleanup_task is None:
        return
    _cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await _cleanup_task
    _cleanup_task = None


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


@app.get("/api/recordings/{recording_id}/calendar")
async def api_get_recording_calendar(recording_id: str) -> dict[str, object]:
    if get_recording(recording_id, settings=_settings) is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    try:
        return await run_in_threadpool(
            refresh_calendar_context,
            recording_id,
            settings=_settings,
        )
    except GraphAuthError as exc:
        fallback = await run_in_threadpool(
            load_calendar_context,
            recording_id,
            settings=_settings,
        )
        fallback["fetch_error"] = str(exc)
        return fallback


@app.post("/api/recordings/{recording_id}/calendar/select")
async def api_select_recording_calendar(
    recording_id: str,
    action: CalendarSelectAction | None = None,
) -> dict[str, object]:
    if get_recording(recording_id, settings=_settings) is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    payload = action or CalendarSelectAction()
    try:
        return await run_in_threadpool(
            select_calendar_event,
            recording_id,
            payload.event_id,
            settings=_settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/recordings/{recording_id}/publish")
async def api_publish_recording(recording_id: str) -> dict[str, object]:
    if get_recording(recording_id, settings=_settings) is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    try:
        return await run_in_threadpool(
            publish_recording_to_onenote,
            recording_id,
            settings=_settings,
        )
    except PublishPreconditionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except GraphNotConfiguredError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except GraphAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


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
    if get_recording(recording_id, settings=_settings) is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    try:
        purge_pending_recording_jobs(recording_id, settings=_settings)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Queue unavailable: {exc}",
        )

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


@app.post("/api/actions/ingest")
async def api_ingest_once() -> dict[str, object]:
    """Trigger a single Google Drive ingest cycle."""
    from .gdrive import ingest_once

    try:
        results = ingest_once(settings=_settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ingest failed: {exc}")
    return {"ingested": results, "count": len(results)}


def set_current_result(result: TranscriptResult | None) -> None:
    global _current_result
    _current_result = result


__all__ = ["ALIAS_PATH", "app", "set_current_result", "healthz"]
