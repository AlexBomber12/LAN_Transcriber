"""Server-rendered HTML pages for the LAN Transcriber control panel.

Uses Jinja2 templates + HTMX (bundled locally) for a minimal, DB-window-style UI.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .config import AppSettings
from .constants import (
    JOB_STATUSES,
    RECORDING_STATUSES,
    RECORDING_STATUS_QUARANTINE,
)
from .db import (
    create_project,
    create_voice_profile,
    delete_project,
    delete_voice_profile,
    get_recording,
    list_jobs,
    list_projects,
    list_recordings,
    list_voice_profiles,
    set_recording_status,
)
from .jobs import enqueue_recording_job, purge_pending_recording_jobs

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

ui_router = APIRouter()
_settings = AppSettings()


def _status_counts(settings: AppSettings) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in RECORDING_STATUSES:
        _, total = list_recordings(settings=settings, status=status, limit=1, offset=0)
        counts[status] = total
    return counts


def _job_counts(settings: AppSettings) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in JOB_STATUSES:
        _, total = list_jobs(settings=settings, status=status, limit=1, offset=0)
        counts[status] = total
    return counts


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@ui_router.get("/", response_class=HTMLResponse)
async def ui_dashboard(request: Request) -> Any:
    rec_counts = _status_counts(_settings)
    job_counts = _job_counts(_settings)
    recent, _ = list_recordings(settings=_settings, limit=10)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active": "dashboard",
            "rec_counts": rec_counts,
            "job_counts": job_counts,
            "recent": recent,
        },
    )


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------


@ui_router.get("/recordings", response_class=HTMLResponse)
async def ui_recordings(
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Any:
    valid_status = status if status in RECORDING_STATUSES else None
    items, total = list_recordings(
        settings=_settings,
        status=valid_status,
        limit=limit,
        offset=offset,
    )
    return templates.TemplateResponse(
        request,
        "recordings.html",
        {
            "active": "recordings",
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "status_filter": valid_status or "",
            "statuses": RECORDING_STATUSES,
            "prev_offset": max(offset - limit, 0),
            "next_offset": offset + limit,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
        },
    )


@ui_router.get("/recordings/{recording_id}", response_class=HTMLResponse)
async def ui_recording_detail(
    request: Request,
    recording_id: str,
    tab: str = Query(default="overview"),
) -> Any:
    rec = get_recording(recording_id, settings=_settings)
    if rec is None:
        return HTMLResponse("<h1>404 – Recording not found</h1>", status_code=404)
    jobs, _ = list_jobs(settings=_settings, recording_id=recording_id, limit=100)
    tabs = ["overview", "calendar", "project", "speakers", "language", "metrics", "log"]
    return templates.TemplateResponse(
        request,
        "recording_detail.html",
        {
            "active": "recordings",
            "rec": rec,
            "jobs": jobs,
            "tabs": tabs,
            "current_tab": tab if tab in tabs else "overview",
        },
    )


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@ui_router.get("/projects", response_class=HTMLResponse)
async def ui_projects(request: Request) -> Any:
    items = list_projects(settings=_settings)
    return templates.TemplateResponse(
        request,
        "projects.html",
        {
            "active": "projects",
            "items": items,
        },
    )


@ui_router.post("/projects", response_class=HTMLResponse)
async def ui_create_project(
    name: str = Form(...),
) -> Any:
    name = name.strip()
    if name:
        try:
            create_project(name, settings=_settings)
        except sqlite3.IntegrityError:
            pass  # duplicate name — silently redirect back
    return RedirectResponse("/projects", status_code=303)


@ui_router.post("/projects/{project_id}/delete", response_class=HTMLResponse)
async def ui_delete_project(project_id: int) -> Any:
    delete_project(project_id, settings=_settings)
    return RedirectResponse("/projects", status_code=303)


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


@ui_router.get("/voices", response_class=HTMLResponse)
async def ui_voices(request: Request) -> Any:
    items = list_voice_profiles(settings=_settings)
    return templates.TemplateResponse(
        request,
        "voices.html",
        {
            "active": "voices",
            "items": items,
        },
    )


@ui_router.post("/voices", response_class=HTMLResponse)
async def ui_create_voice(
    display_name: str = Form(...),
    notes: str = Form(default=""),
) -> Any:
    display_name = display_name.strip()
    if display_name:
        create_voice_profile(display_name, notes.strip() or None, settings=_settings)
    return RedirectResponse("/voices", status_code=303)


@ui_router.post("/voices/{profile_id}/delete", response_class=HTMLResponse)
async def ui_delete_voice(profile_id: int) -> Any:
    delete_voice_profile(profile_id, settings=_settings)
    return RedirectResponse("/voices", status_code=303)


# ---------------------------------------------------------------------------
# Queue / Jobs
# ---------------------------------------------------------------------------


@ui_router.get("/queue", response_class=HTMLResponse)
async def ui_queue(
    request: Request,
    status: str | None = Query(default=None),
    recording_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Any:
    valid_status = status if status in JOB_STATUSES else None
    items, total = list_jobs(
        settings=_settings,
        status=valid_status,
        recording_id=recording_id or None,
        limit=limit,
        offset=offset,
    )
    return templates.TemplateResponse(
        request,
        "queue.html",
        {
            "active": "queue",
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "status_filter": valid_status or "",
            "recording_id_filter": recording_id or "",
            "statuses": JOB_STATUSES,
            "prev_offset": max(offset - limit, 0),
            "next_offset": offset + limit,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
        },
    )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


@ui_router.get("/connections", response_class=HTMLResponse)
async def ui_connections(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "connections.html",
        {
            "active": "connections",
        },
    )


# ---------------------------------------------------------------------------
# Inline recording actions (HTMX targets returning HX-Redirect)
# ---------------------------------------------------------------------------


@ui_router.post("/ui/recordings/{recording_id}/requeue")
async def ui_action_requeue(recording_id: str) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    try:
        enqueue_recording_job(recording_id, settings=_settings)
    except Exception as exc:
        return HTMLResponse(f"Requeue failed: {exc}", status_code=503)
    resp = HTMLResponse("")
    resp.headers["HX-Redirect"] = f"/recordings/{recording_id}"
    return resp


@ui_router.post("/ui/recordings/{recording_id}/quarantine")
async def ui_action_quarantine(recording_id: str) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    set_recording_status(
        recording_id,
        RECORDING_STATUS_QUARANTINE,
        settings=_settings,
    )
    resp = HTMLResponse("")
    resp.headers["HX-Redirect"] = f"/recordings/{recording_id}"
    return resp


@ui_router.post("/ui/recordings/{recording_id}/delete")
async def ui_action_delete(recording_id: str) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    try:
        purge_pending_recording_jobs(recording_id, settings=_settings)
    except Exception as exc:
        return HTMLResponse(f"Delete failed (queue unavailable): {exc}", status_code=503)
    from .db import delete_recording

    delete_recording(recording_id, settings=_settings)
    recording_path = _settings.recordings_root / recording_id
    shutil.rmtree(recording_path, ignore_errors=True)
    resp = HTMLResponse("")
    resp.headers["HX-Redirect"] = "/recordings"
    return resp
