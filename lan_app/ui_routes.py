"""Server-rendered HTML pages for the LAN Transcriber control panel.

Uses Jinja2 templates + HTMX (bundled locally) for a minimal, DB-window-style UI.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .calendar import (
    load_calendar_context,
    refresh_calendar_context,
    select_calendar_event,
)
from .config import AppSettings
from .constants import (
    JOB_STATUSES,
    JOB_TYPE_PRECHECK,
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
    set_recording_language_settings,
    set_recording_status,
)
from .jobs import enqueue_recording_job, purge_pending_recording_jobs
from .ms_graph import GraphAuthError, ms_connection_state
from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.llm_client import LLMClient
from lan_transcriber.pipeline import Settings as PipelineSettings
from lan_transcriber.pipeline import build_structured_summary_prompts, build_summary_payload

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

ui_router = APIRouter()
_settings = AppSettings()

_LANGUAGE_NAME_MAP: dict[str, str] = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
}

_LANGUAGE_CODE_MAP: dict[str, str] = {
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "fre": "fr",
    "deu": "de",
    "ger": "de",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "ukr": "uk",
    "jpn": "ja",
    "kor": "ko",
    "zho": "zh",
    "chi": "zh",
}

_COMMON_LANGUAGE_CODES = ("en", "es", "fr", "de", "pt", "it", "zh", "ja", "ko", "ru")


def _normalise_language_code(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    token = raw.replace("_", "-").split("-", 1)[0]
    if len(token) == 2 and token.isalpha():
        return token
    if len(token) == 3 and token.isalpha():
        return _LANGUAGE_CODE_MAP.get(token, None)
    return None


def _language_display_name(code: str | None) -> str:
    if not code:
        return "—"
    if code == "unknown":
        return "Unknown"
    return f"{_LANGUAGE_NAME_MAP.get(code, code.upper())} ({code})"


def _parse_language_form_value(value: str, *, field_name: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    parsed = _normalise_language_code(stripped)
    if parsed is None:
        raise ValueError(f"{field_name} must be a language code such as en or es")
    return parsed


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _load_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return payload


def _normalise_text_items(value: Any, *, max_items: int) -> list[str]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, str):
        rows = [line.strip() for line in value.splitlines() if line.strip()]
    else:
        return []

    out: list[str] = []
    for row in rows:
        if len(out) >= max_items:
            break
        text = str(row).strip()
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        if text:
            out.append(text)
    return out


def _summary_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    _transcript_path, summary_path = _recording_derived_paths(recording_id, settings)
    payload = _load_json_dict(summary_path)
    summary_text = str(payload.get("summary") or "").strip()
    summary_bullets = _normalise_text_items(payload.get("summary_bullets"), max_items=12)
    if not summary_bullets:
        summary_bullets = _normalise_text_items(summary_text, max_items=12)

    decisions = _normalise_text_items(payload.get("decisions"), max_items=20)
    raw_action_items = payload.get("action_items")
    action_items: list[dict[str, Any]] = []
    if isinstance(raw_action_items, list):
        for row in raw_action_items[:30]:
            if not isinstance(row, dict):
                continue
            task = str(row.get("task") or "").strip()
            if not task:
                continue
            owner = str(row.get("owner") or "").strip()
            deadline = str(row.get("deadline") or "").strip()
            try:
                confidence = float(row.get("confidence"))
            except (TypeError, ValueError):
                confidence = 0.5
            action_items.append(
                {
                    "task": task,
                    "owner": owner or None,
                    "deadline": deadline or None,
                    "confidence": max(0.0, min(confidence, 1.0)),
                }
            )

    question_types = {
        "open": 0,
        "yes_no": 0,
        "clarification": 0,
        "status": 0,
        "decision_seeking": 0,
    }
    questions_total = 0
    extracted_questions: list[str] = []
    questions_payload = payload.get("questions")
    if isinstance(questions_payload, dict):
        types_payload = questions_payload.get("types")
        if isinstance(types_payload, dict):
            for key in question_types:
                try:
                    question_types[key] = max(0, int(types_payload.get(key, 0)))
                except (TypeError, ValueError):
                    question_types[key] = 0
        try:
            questions_total = max(0, int(questions_payload.get("total_count", 0)))
        except (TypeError, ValueError):
            questions_total = 0
        extracted_questions = _normalise_text_items(
            questions_payload.get("extracted"),
            max_items=20,
        )
    if questions_total == 0:
        questions_total = max(sum(question_types.values()), len(extracted_questions))

    topic = str(payload.get("topic") or "").strip()
    emotional_summary = str(payload.get("emotional_summary") or "").strip()
    return {
        "topic": topic or "—",
        "summary_bullets": summary_bullets,
        "summary_text": summary_text,
        "decisions": decisions,
        "action_items": action_items,
        "emotional_summary": emotional_summary or "—",
        "questions": {
            "total_count": questions_total,
            "types": question_types,
            "extracted": extracted_questions,
        },
    }


def _chunk_text_for_turns(text: str, *, chunk_size: int = 450) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    words = normalized.split(" ")
    current: list[str] = []
    current_len = 0
    for word in words:
        word_len = len(word)
        sep = 1 if current else 0
        if current and current_len + sep + word_len > chunk_size:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
            continue
        if not current and word_len > chunk_size:
            start = 0
            while start < word_len:
                end = min(start + chunk_size, word_len)
                chunks.append(word[start:end])
                start = end
            current = []
            current_len = 0
            continue
        if sep:
            current_len += 1
        current.append(word)
        current_len += word_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def _fallback_speaker_turns_from_transcript(transcript_payload: dict[str, Any]) -> list[dict[str, Any]]:
    segments_payload = transcript_payload.get("segments")
    if isinstance(segments_payload, list):
        segment_turns: list[dict[str, Any]] = []
        for row in segments_payload:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(row.get("start") or 0.0)
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(row.get("end") or row.get("start") or start)
            except (TypeError, ValueError):
                end = start
            segment_turns.append(
                {
                    "start": start,
                    "end": max(end, start),
                    "speaker": "S1",
                    "text": text,
                    "language": row.get("language"),
                }
            )
        if segment_turns:
            return segment_turns

    transcript_text = str(transcript_payload.get("text") or "").strip()
    chunks = _chunk_text_for_turns(transcript_text)
    turns: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        turns.append(
            {
                "start": float(idx),
                "end": float(idx + 1),
                "speaker": "S1",
                "text": chunk,
            }
        )
    return turns


def _recording_derived_paths(recording_id: str, settings: AppSettings) -> tuple[Path, Path]:
    derived = settings.recordings_root / recording_id / "derived"
    return derived / "transcript.json", derived / "summary.json"


def _sync_transcript_language_settings(
    recording_id: str,
    *,
    settings: AppSettings,
    target_summary_language: str | None,
    transcript_language_override: str | None,
) -> None:
    transcript_path, _summary_path = _recording_derived_paths(recording_id, settings)
    payload = _load_json_dict(transcript_path)
    if not payload:
        return
    payload["target_summary_language"] = target_summary_language
    payload["transcript_language_override"] = transcript_language_override
    atomic_write_json(transcript_path, payload)


def _language_options(
    *,
    distribution_codes: list[str],
    target_summary_language: str | None,
    transcript_language_override: str | None,
) -> list[dict[str, str]]:
    ordered: list[str] = []
    seen: set[str] = set()
    for code in distribution_codes:
        if code == "unknown":
            continue
        if code not in seen:
            ordered.append(code)
            seen.add(code)
    for code in (target_summary_language, transcript_language_override):
        if code and code not in seen:
            ordered.append(code)
            seen.add(code)
    for code in _COMMON_LANGUAGE_CODES:
        if code not in seen:
            ordered.append(code)
            seen.add(code)
    return [
        {"code": code, "label": _language_display_name(code)}
        for code in ordered
    ]


def _language_tab_context(recording_id: str, rec: dict[str, Any], settings: AppSettings) -> dict[str, Any]:
    transcript_path, summary_path = _recording_derived_paths(recording_id, settings)
    transcript_payload = _load_json_dict(transcript_path)
    summary_payload = _load_json_dict(summary_path)

    language_payload = transcript_payload.get("language")
    language_obj = language_payload if isinstance(language_payload, dict) else {}
    detected = _normalise_language_code(language_obj.get("detected")) or _normalise_language_code(
        rec.get("language_auto")
    )
    dominant = _normalise_language_code(transcript_payload.get("dominant_language")) or detected

    distribution_payload = transcript_payload.get("language_distribution")
    distribution_obj = distribution_payload if isinstance(distribution_payload, dict) else {}
    distribution_rows: list[dict[str, Any]] = []
    for code_raw, pct_raw in distribution_obj.items():
        code = _normalise_language_code(code_raw) or str(code_raw)
        try:
            percent = float(pct_raw)
        except (TypeError, ValueError):
            continue
        distribution_rows.append(
            {
                "code": code,
                "label": _language_display_name(code),
                "percent": round(percent, 2),
            }
        )
    distribution_rows.sort(key=lambda row: (-row["percent"], row["code"]))

    spans_payload = transcript_payload.get("language_spans")
    spans_raw = spans_payload if isinstance(spans_payload, list) else []
    span_rows: list[dict[str, Any]] = []
    for row in spans_raw:
        if not isinstance(row, dict):
            continue
        try:
            start = float(row.get("start", 0.0))
            end = float(row.get("end", start))
        except (TypeError, ValueError):
            continue
        code = _normalise_language_code(row.get("lang")) or str(row.get("lang") or "unknown")
        span_rows.append(
            {
                "start": round(start, 3),
                "end": round(max(end, start), 3),
                "code": code,
                "label": _language_display_name(code),
            }
        )
    span_rows.sort(key=lambda row: (row["start"], row["end"]))

    target_summary_language = _normalise_language_code(rec.get("target_summary_language"))
    transcript_target_summary_language = _normalise_language_code(
        transcript_payload.get("target_summary_language")
    )
    resolved_target_summary_language = (
        target_summary_language or transcript_target_summary_language or dominant
    )
    transcript_language_override = _normalise_language_code(
        rec.get("language_override")
    ) or _normalise_language_code(
        transcript_payload.get("transcript_language_override")
    )

    distribution_codes = [str(row["code"]) for row in distribution_rows]
    non_unknown_distribution = [code for code in distribution_codes if code != "unknown"]

    return {
        "detected": detected,
        "detected_label": _language_display_name(detected),
        "dominant": dominant,
        "dominant_label": _language_display_name(dominant),
        "distribution": distribution_rows,
        "spans": span_rows,
        "is_mixed": len(set(non_unknown_distribution)) >= 2,
        "target_summary_language": target_summary_language or "",
        "target_summary_language_label": _language_display_name(resolved_target_summary_language),
        "transcript_language_override": transcript_language_override or "",
        "transcript_language_override_label": _language_display_name(transcript_language_override),
        "summary_preview": str(summary_payload.get("summary") or "").strip(),
        "summary_model": str(summary_payload.get("model") or ""),
        "options": _language_options(
            distribution_codes=distribution_codes,
            target_summary_language=resolved_target_summary_language,
            transcript_language_override=transcript_language_override,
        ),
    }


def _resummarize_recording(
    recording_id: str,
    *,
    settings: AppSettings,
    target_summary_language: str | None,
) -> None:
    transcript_path, summary_path = _recording_derived_paths(recording_id, settings)
    speaker_turns_path = transcript_path.parent / "speaker_turns.json"
    transcript_payload = _load_json_dict(transcript_path)
    if not transcript_payload:
        raise ValueError("No transcript.json found for this recording")

    transcript_text = str(transcript_payload.get("text") or "").strip()
    if not transcript_text:
        raise ValueError("Transcript text is empty; re-transcribe first")

    language_payload = transcript_payload.get("language")
    language_obj = language_payload if isinstance(language_payload, dict) else {}
    resolved_target = (
        target_summary_language
        or _normalise_language_code(transcript_payload.get("target_summary_language"))
        or _normalise_language_code(transcript_payload.get("dominant_language"))
        or _normalise_language_code(language_obj.get("detected"))
        or "en"
    )
    pipeline_settings = PipelineSettings(
        recordings_root=settings.recordings_root,
        voices_dir=settings.data_root / "voices",
        unknown_dir=settings.recordings_root / "unknown",
        tmp_root=settings.data_root / "tmp",
    )
    speaker_turns_raw = _load_json_list(speaker_turns_path)
    speaker_turns = [row for row in speaker_turns_raw if isinstance(row, dict)]
    if not speaker_turns:
        speaker_turns = _fallback_speaker_turns_from_transcript(transcript_payload)
    if not speaker_turns:
        speaker_turns = [{"start": 0.0, "end": 0.0, "speaker": "S1", "text": transcript_text}]

    calendar_title = str(transcript_payload.get("calendar_title") or "").strip() or None
    attendees_payload = transcript_payload.get("calendar_attendees")
    calendar_attendees: list[str] = []
    if isinstance(attendees_payload, list):
        calendar_attendees = [
            str(attendee).strip()
            for attendee in attendees_payload
            if str(attendee).strip()
        ]

    system_prompt, user_prompt = build_structured_summary_prompts(
        speaker_turns,
        resolved_target,
        calendar_title=calendar_title,
        calendar_attendees=calendar_attendees,
    )
    message = asyncio.run(
        LLMClient().generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=pipeline_settings.llm_model,
            response_format={"type": "json_object"},
        )
    )
    raw_summary = message.get("content", "") if isinstance(message, dict) else str(message)

    summary_payload = _load_json_dict(summary_path)
    friendly = summary_payload.get("friendly")
    if not isinstance(friendly, int):
        friendly = 0
    structured_payload = build_summary_payload(
        raw_llm_content=raw_summary,
        model=pipeline_settings.llm_model,
        target_summary_language=resolved_target,
        friendly=friendly,
        default_topic=calendar_title or "Meeting summary",
    )
    summary_payload.update(structured_payload)
    atomic_write_json(summary_path, summary_payload)

    transcript_payload["target_summary_language"] = resolved_target
    atomic_write_json(transcript_path, transcript_payload)


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
    current_tab = tab if tab in tabs else "overview"
    calendar: dict[str, Any] | None = None
    language: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    if current_tab == "calendar":
        try:
            calendar = await run_in_threadpool(
                refresh_calendar_context,
                recording_id,
                settings=_settings,
            )
        except GraphAuthError as exc:
            calendar = await run_in_threadpool(
                load_calendar_context,
                recording_id,
                settings=_settings,
            )
            calendar["fetch_error"] = str(exc)
    if current_tab == "language":
        language = _language_tab_context(recording_id, rec, _settings)
    if current_tab in {"overview", "metrics"}:
        summary = _summary_context(recording_id, _settings)

    return templates.TemplateResponse(
        request,
        "recording_detail.html",
        {
            "active": "recordings",
            "rec": rec,
            "jobs": jobs,
            "tabs": tabs,
            "current_tab": current_tab,
            "calendar": calendar,
            "language": language,
            "summary": summary,
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
    ms_state = await run_in_threadpool(ms_connection_state, _settings)
    return templates.TemplateResponse(
        request,
        "connections.html",
        {
            "active": "connections",
            "ms": ms_state,
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


def _save_language_settings(
    recording_id: str,
    *,
    target_summary_language: str,
    transcript_language_override: str,
) -> tuple[str | None, str | None]:
    target = _parse_language_form_value(
        target_summary_language,
        field_name="target_summary_language",
    )
    transcript_override = _parse_language_form_value(
        transcript_language_override,
        field_name="transcript_language_override",
    )
    set_recording_language_settings(
        recording_id,
        settings=_settings,
        target_summary_language=target,
        transcript_language_override=transcript_override,
    )
    _sync_transcript_language_settings(
        recording_id,
        settings=_settings,
        target_summary_language=target,
        transcript_language_override=transcript_override,
    )
    return target, transcript_override


@ui_router.post("/ui/recordings/{recording_id}/language/settings")
async def ui_save_language_settings(
    recording_id: str,
    target_summary_language: str = Form(default=""),
    transcript_language_override: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    try:
        _save_language_settings(
            recording_id,
            target_summary_language=target_summary_language,
            transcript_language_override=transcript_language_override,
        )
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    return RedirectResponse(f"/recordings/{recording_id}?tab=language", status_code=303)


@ui_router.post("/ui/recordings/{recording_id}/language/resummarize")
async def ui_resummarize_language(
    recording_id: str,
    target_summary_language: str = Form(default=""),
    transcript_language_override: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    try:
        target, _transcript_override = _save_language_settings(
            recording_id,
            target_summary_language=target_summary_language,
            transcript_language_override=transcript_language_override,
        )
        await run_in_threadpool(
            _resummarize_recording,
            recording_id,
            settings=_settings,
            target_summary_language=target,
        )
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    except Exception as exc:
        return HTMLResponse(f"Re-summarize failed: {exc}", status_code=503)
    return RedirectResponse(f"/recordings/{recording_id}?tab=language", status_code=303)


@ui_router.post("/ui/recordings/{recording_id}/language/retranscribe")
async def ui_retranscribe_language(
    recording_id: str,
    target_summary_language: str = Form(default=""),
    transcript_language_override: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    try:
        _save_language_settings(
            recording_id,
            target_summary_language=target_summary_language,
            transcript_language_override=transcript_language_override,
        )
        enqueue_recording_job(
            recording_id,
            job_type=JOB_TYPE_PRECHECK,
            settings=_settings,
        )
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    except Exception as exc:
        return HTMLResponse(f"Re-transcribe failed: {exc}", status_code=503)
    return RedirectResponse(f"/recordings/{recording_id}?tab=log", status_code=303)


@ui_router.post("/ui/recordings/{recording_id}/calendar/select")
async def ui_select_calendar(recording_id: str, event_id: str = Form(default="")) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    selected_event_id = event_id.strip() or None
    try:
        await run_in_threadpool(
            select_calendar_event,
            recording_id,
            selected_event_id,
            settings=_settings,
        )
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    return RedirectResponse(f"/recordings/{recording_id}?tab=calendar", status_code=303)
