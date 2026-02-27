from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import shutil
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from lan_transcriber import aliases
from lan_transcriber.metrics import write_metrics_snapshot
from lan_transcriber.models import TranscriptResult
from lan_transcriber.pipeline import refresh_aliases

from .auth import auth_enabled, request_is_authenticated, request_requires_auth
from .config import AppSettings
from .constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    JOB_STATUSES,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUSES,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_QUARANTINE,
)
from .db import (
    create_recording,
    delete_recording,
    get_recording,
    init_db,
    list_jobs,
    list_recordings,
    set_recording_status,
)
from .jobs import (
    DuplicateRecordingJobError,
    RecordingNotFoundError,
    enqueue_recording_job,
)
from .jobs import purge_pending_recording_jobs
from .healthchecks import (
    check_app_health,
    check_db_health,
    check_redis_health,
    check_worker_health,
    collect_health_checks,
)
from .locks import release_ingest_lock, try_acquire_ingest_lock
from .ops import run_retention_cleanup
from .reaper import run_stuck_job_reaper_once
from .uploads import (
    ALLOWED_UPLOAD_EXTENSIONS,
    infer_captured_at,
    safe_filename,
    suffix_from_name,
    write_upload_to_path,
)
from .ui_routes import _STATIC_DIR, ui_router

ALIAS_PATH = aliases.ALIAS_PATH
_subscribers: List[asyncio.Queue[str]] = []
_current_result: TranscriptResult | None = None
_settings = AppSettings()
_CLEANUP_INTERVAL_SECONDS = 3600
_logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    init_db(_settings)
    _settings.metrics_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_task = asyncio.create_task(write_metrics_snapshot(_settings.metrics_snapshot_path))
    cleanup_task = asyncio.create_task(_retention_cleanup_loop())
    reaper_task = asyncio.create_task(_stuck_job_reaper_loop())

    try:
        yield
    finally:
        for task in (reaper_task, cleanup_task, metrics_task):
            task.cancel()
        for task in (reaper_task, cleanup_task, metrics_task):
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(lifespan=_lifespan)
app.include_router(ui_router)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


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


def _is_public_auth_exempt(request: Request) -> bool:
    if request.method.upper() != "GET":
        return False
    path = request.url.path
    if path == "/healthz" or path.startswith("/healthz/"):
        return True
    if path in {"/metrics", "/openapi.json"}:
        return True
    return False


@app.middleware("http")
async def _enforce_optional_bearer_auth(request: Request, call_next):
    if not auth_enabled(_settings):
        return await call_next(request)
    if _is_public_auth_exempt(request):
        return await call_next(request)
    if not request_requires_auth(request):
        return await call_next(request)
    if request_is_authenticated(request, _settings):
        return await call_next(request)
    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})


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


async def _stuck_job_reaper_loop() -> None:
    while True:
        try:
            summary = await run_in_threadpool(
                run_stuck_job_reaper_once,
                settings=_settings,
            )
            recovered_jobs = int(summary.get("recovered_jobs") or 0)
            if recovered_jobs > 0:
                _logger.warning(
                    "Recovered %s stuck jobs for %s recordings",
                    recovered_jobs,
                    int(summary.get("recovered_recordings") or 0),
                )
        except Exception:
            _logger.exception("Stuck-job reaper failed")
        await asyncio.sleep(_settings.reaper_interval_seconds)


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
) -> dict[str, object]:
    payload = action or RequeueAction()
    if payload.job_type != DEFAULT_REQUEUE_JOB_TYPE:
        raise HTTPException(
            status_code=422,
            detail="Only precheck is supported in single-job pipeline mode",
        )

    if get_recording(recording_id, settings=_settings) is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    try:
        job = enqueue_recording_job(
            recording_id,
            job_type=DEFAULT_REQUEUE_JOB_TYPE,
            settings=_settings,
        )
    except DuplicateRecordingJobError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "A precheck job is already queued or started for this recording.",
                "existing_job_id": exc.job_id,
            },
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


@app.post("/api/uploads")
async def api_upload_file(file: UploadFile = File(...)) -> dict[str, object]:
    recording_id = f"trs_{uuid4().hex[:8]}"
    ext = suffix_from_name(file.filename or "")
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported file extension: {ext}")

    recording_dir = _settings.recordings_root / recording_id
    raw_dir = recording_dir / "raw"
    dest = raw_dir / f"audio{ext}"
    completed = False
    recording_created = False

    try:
        bytes_written = write_upload_to_path(
            file,
            dest,
            max_bytes=_settings.upload_max_bytes,
        )
        captured_at = infer_captured_at(file.filename or "")
        create_recording(
            recording_id,
            source="upload",
            source_filename=safe_filename(file.filename or ""),
            captured_at=captured_at,
            status=RECORDING_STATUS_QUEUED,
            settings=_settings,
        )
        recording_created = True
        try:
            job = enqueue_recording_job(
                recording_id,
                job_type=JOB_TYPE_PRECHECK,
                settings=_settings,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}")

        completed = True
        return {
            "recording_id": recording_id,
            "job_id": job.job_id,
            "captured_at": captured_at,
            "bytes_written": bytes_written,
        }
    except ValueError as exc:
        message = str(exc)
        if message == "max upload size exceeded":
            raise HTTPException(status_code=413, detail=message)
        raise HTTPException(status_code=422, detail=message)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Upload failed: {exc}")
    finally:
        await file.close()
        if not completed and recording_dir.exists():
            shutil.rmtree(recording_dir, ignore_errors=True)
        if not completed and recording_created:
            with suppress(Exception):
                delete_recording(recording_id, settings=_settings)


@app.post("/api/actions/ingest")
async def api_ingest_once() -> dict[str, object]:
    """Trigger a single Google Drive ingest cycle."""
    from .gdrive import ingest_once

    try:
        acquired, retry_after, lock_token = try_acquire_ingest_lock(_settings)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ingest lock unavailable: {exc}")

    if not acquired:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Ingest is already running.",
                "retry_after_seconds": retry_after,
            },
        )

    try:
        results = ingest_once(settings=_settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ingest failed: {exc}")
    finally:
        if lock_token is not None:
            with suppress(Exception):
                release_ingest_lock(_settings, token=lock_token)
    return {"ingested": results, "count": len(results)}


def set_current_result(result: TranscriptResult | None) -> None:
    global _current_result
    _current_result = result


__all__ = ["ALIAS_PATH", "app", "set_current_result", "healthz"]
