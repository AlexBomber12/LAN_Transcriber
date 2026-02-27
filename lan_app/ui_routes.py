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
from urllib.parse import quote

from fastapi import APIRouter, Form, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .auth import (
    auth_enabled,
    cookie_secure_flag,
    clear_auth_cookie,
    expected_bearer_token,
    request_is_authenticated,
    safe_next_path,
    set_auth_cookie,
)
from .config import AppSettings
from .conversation_metrics import refresh_recording_metrics
from .constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    JOB_STATUS_FAILED,
    JOB_STATUSES,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUSES,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_QUARANTINE,
)
from .db import (
    count_routing_training_examples,
    create_voice_sample,
    create_project,
    create_voice_profile,
    delete_project,
    delete_voice_profile,
    get_meeting_metrics,
    get_job,
    get_recording,
    get_voice_sample,
    list_participant_metrics,
    list_jobs,
    list_projects,
    list_recordings,
    list_speaker_assignments,
    list_voice_samples,
    list_voice_profiles,
    set_recording_project,
    set_speaker_assignment,
    set_recording_language_settings,
    set_recording_status,
)
from .exporter import build_export_zip_bytes, build_onenote_markdown
from .gdrive import build_drive_service
from .jobs import (
    DuplicateRecordingJobError,
    enqueue_recording_job,
    purge_pending_recording_jobs,
)
from .routing import refresh_recording_routing, train_routing_from_manual_selection
from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.llm_client import LLMClient
from lan_transcriber.pipeline import Settings as PipelineSettings
from lan_transcriber.pipeline import build_structured_summary_prompts, build_summary_payload
from lan_transcriber.utils import normalise_language_code as _normalise_language_code_shared

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

_COMMON_LANGUAGE_CODES = ("en", "es", "fr", "de", "pt", "it", "zh", "ja", "ko", "ru")
_STUCK_JOB_RECOVERY_ERROR = "stuck job recovered"


def _normalise_language_code(value: object | None) -> str | None:
    return _normalise_language_code_shared(value)


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


def _recording_recovery_warning(jobs: list[dict[str, Any]]) -> str | None:
    for job in jobs:
        error = str(job.get("error") or "").strip().lower()
        if error != _STUCK_JOB_RECOVERY_ERROR:
            continue
        finished_at = str(job.get("finished_at") or "").strip()
        if finished_at:
            return (
                "Warning: this recording was recovered from a stuck job at "
                f"{finished_at[:19].replace('T', ' ')} UTC."
            )
        return "Warning: this recording was recovered from a stuck job."
    return None


def _safe_pipeline_progress(value: object) -> float:
    try:
        progress = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(progress, 1.0))


def _pipeline_stage_label(stage: object) -> str:
    text = str(stage or "").strip()
    if not text:
        return "Waiting"
    return text.replace("_", " ").title()


def _gdrive_connection_state(settings: AppSettings) -> dict[str, Any]:
    sa_path_value = str(settings.gdrive_sa_json_path or "").strip()
    folder_id_value = str(settings.gdrive_inbox_folder_id or "").strip()
    return {
        "configured": bool(sa_path_value and folder_id_value),
        "sa_path": sa_path_value,
        "folder_id": folder_id_value,
    }


def _test_gdrive_connection(settings: AppSettings) -> dict[str, Any]:
    state = _gdrive_connection_state(settings)
    if not state["configured"]:
        raise ValueError("Google Drive is not configured.")
    sa_path = Path(str(state["sa_path"]))
    folder_id = str(state["folder_id"])
    service = build_drive_service(sa_path)
    query = f"'{folder_id}' in parents and trashed=false"
    response = (
        service.files()
        .list(
            q=query,
            fields="files(id,name,createdTime)",
            pageSize=1,
        )
        .execute()
    )
    rows = response.get("files", []) or []
    if rows:
        first = rows[0]
        name = str(first.get("name") or "").strip() or "(unnamed)"
        file_id = str(first.get("id") or "").strip() or "unknown-id"
        return {
            "ok": True,
            "message": f"Connected. Sample file: {name} ({file_id}).",
        }
    return {
        "ok": True,
        "message": "Connected. Inbox is reachable, but no files were found.",
    }


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


def _metrics_tab_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    def _to_int(value: Any, *, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _to_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    meeting_payload: dict[str, Any] = {}
    participants_payload: list[dict[str, Any]] = []

    meeting_row = get_meeting_metrics(recording_id, settings=settings) or {}
    meeting_json = meeting_row.get("json")
    if isinstance(meeting_json, dict):
        meeting_payload = dict(meeting_json)

    participant_rows = list_participant_metrics(recording_id, settings=settings)
    for row in participant_rows:
        payload = row.get("json")
        participant = payload if isinstance(payload, dict) else {}
        if not participant:
            continue
        normalized = dict(participant)
        speaker = str(normalized.get("speaker") or row.get("diar_speaker_label") or "").strip()
        if not speaker:
            continue
        normalized["speaker"] = speaker
        participants_payload.append(normalized)

    metrics_payload: dict[str, Any] = {}
    if not meeting_payload or not participants_payload:
        derived = settings.recordings_root / recording_id / "derived"
        metrics_payload = _load_json_dict(derived / "metrics.json")
        meeting_raw = metrics_payload.get("meeting")
        participants_raw = metrics_payload.get("participants")
        if isinstance(meeting_raw, dict):
            if not meeting_payload:
                meeting_payload = dict(meeting_raw)
            else:
                for key, value in meeting_raw.items():
                    meeting_payload.setdefault(key, value)
        if isinstance(participants_raw, list):
            artifact_participants = [row for row in participants_raw if isinstance(row, dict)]
            if not participants_payload:
                participants_payload = artifact_participants
            else:
                by_speaker: dict[str, dict[str, Any]] = {}
                for row in participants_payload:
                    speaker = str(row.get("speaker") or "").strip()
                    if not speaker:
                        continue
                    by_speaker[speaker] = row
                for row in artifact_participants:
                    speaker = str(row.get("speaker") or "").strip()
                    if not speaker:
                        continue
                    existing = by_speaker.get(speaker)
                    if existing is None:
                        participants_payload.append(row)
                        by_speaker[speaker] = row
                        continue
                    for key, value in row.items():
                        existing.setdefault(key, value)

    meeting = {
        "total_interruptions": _to_int(meeting_payload.get("total_interruptions")),
        "total_questions": _to_int(meeting_payload.get("total_questions")),
        "decisions_count": _to_int(meeting_payload.get("decisions_count")),
        "action_items_count": _to_int(meeting_payload.get("action_items_count")),
        "actionability_ratio": _to_float(meeting_payload.get("actionability_ratio")),
        "emotional_summary": str(meeting_payload.get("emotional_summary") or "—"),
        "total_speech_time_seconds": _to_float(meeting_payload.get("total_speech_time_seconds")),
    }

    participants: list[dict[str, Any]] = []
    for row in participants_payload:
        speaker = str(row.get("speaker") or "").strip()
        if not speaker:
            continue
        participants.append(
            {
                "speaker": speaker,
                "airtime_seconds": round(_to_float(row.get("airtime_seconds")), 3),
                "airtime_share": round(_to_float(row.get("airtime_share")), 4),
                "turns": _to_int(row.get("turns")),
                "interruptions_done": _to_int(row.get("interruptions_done")),
                "interruptions_received": _to_int(row.get("interruptions_received")),
                "questions_count": _to_int(row.get("questions_count")),
                "role_hint": str(row.get("role_hint") or "—"),
            }
        )
    participants.sort(key=lambda item: (-item["airtime_seconds"], item["speaker"]))

    return {
        "meeting": meeting,
        "participants": participants,
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


def _speaker_slug(label: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    slug = slug.strip("_")
    return slug or "speaker"


def _safe_path(candidate: Path, *, root: Path) -> Path | None:
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return None
    return resolved


def _safe_audio_path(candidate: Path, *, root: Path) -> Path | None:
    resolved = _safe_path(candidate, root=root)
    if resolved is None:
        return None
    if resolved.suffix.lower() != ".wav":
        return None
    return resolved


def _speaker_snippet_files(
    recording_id: str,
    speaker_label: str,
    *,
    settings: AppSettings,
) -> list[Path]:
    snippets_root = settings.recordings_root / recording_id / "derived" / "snippets"
    speaker_dir = snippets_root / _speaker_slug(speaker_label)
    safe_dir = _safe_path(speaker_dir, root=snippets_root)
    if safe_dir is None or not safe_dir.exists() or not safe_dir.is_dir():
        return []

    out: list[Path] = []
    for child in safe_dir.iterdir():
        if not child.is_file():
            continue
        safe_audio = _safe_audio_path(child, root=snippets_root)
        if safe_audio is None:
            continue
        out.append(safe_audio)
    out.sort(key=lambda path: path.name)
    return out


def _as_data_relative_path(path: Path, *, settings: AppSettings) -> str | None:
    safe = _safe_path(path, root=settings.data_root)
    if safe is None:
        return None
    root_resolved = settings.data_root.resolve()
    return safe.relative_to(root_resolved).as_posix()


def _speakers_tab_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    transcript_path, _summary_path = _recording_derived_paths(recording_id, settings)
    speaker_turns_path = transcript_path.parent / "speaker_turns.json"
    transcript_payload = _load_json_dict(transcript_path)

    speaker_turns_raw = _load_json_list(speaker_turns_path)
    speaker_turns = [row for row in speaker_turns_raw if isinstance(row, dict)]
    if not speaker_turns:
        speaker_turns = _fallback_speaker_turns_from_transcript(transcript_payload)

    per_speaker: dict[str, dict[str, Any]] = {}
    for row in speaker_turns:
        speaker = str(row.get("speaker") or "S1").strip() or "S1"
        slot = per_speaker.setdefault(
            speaker,
            {
                "speaker": speaker,
                "turn_count": 0,
                "duration_sec": 0.0,
                "preview_text": "",
            },
        )
        slot["turn_count"] += 1
        text = str(row.get("text") or "").strip()
        if text and not slot["preview_text"]:
            slot["preview_text"] = text
        try:
            start = float(row.get("start") or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(row.get("end") or start)
        except (TypeError, ValueError):
            end = start
        slot["duration_sec"] += max(0.0, end - start)

    assignments = list_speaker_assignments(recording_id, settings=settings)
    assignment_by_speaker = {
        str(row.get("diar_speaker_label") or ""): row
        for row in assignments
    }
    voice_profiles = list_voice_profiles(settings=settings)

    speaker_rows: list[dict[str, Any]] = []
    recording_token = quote(recording_id, safe="")
    for speaker in sorted(per_speaker):
        row = per_speaker[speaker]
        snippets = _speaker_snippet_files(recording_id, speaker, settings=settings)
        speaker_token = quote(_speaker_slug(speaker), safe="")
        snippet_urls = [
            (
                f"/ui/recordings/{recording_token}/snippets/"
                f"{speaker_token}/{quote(path.name, safe='')}"
            )
            for path in snippets
        ]
        assignment = assignment_by_speaker.get(speaker, {})
        profile_id_raw = assignment.get("voice_profile_id")
        try:
            profile_id = int(profile_id_raw) if profile_id_raw is not None else None
        except (TypeError, ValueError):
            profile_id = None
        try:
            confidence = float(assignment.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        speaker_rows.append(
            {
                "speaker": speaker,
                "turn_count": int(row["turn_count"]),
                "duration_sec": round(float(row["duration_sec"]), 3),
                "preview_text": str(row["preview_text"]),
                "snippet_urls": snippet_urls,
                "voice_profile_id": profile_id,
                "voice_profile_name": str(assignment.get("voice_profile_name") or ""),
                "confidence": max(0.0, min(confidence, 1.0)),
            }
        )

    return {
        "speaker_rows": speaker_rows,
        "voice_profiles": voice_profiles,
    }


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _project_tab_context(recording_id: str, rec: dict[str, Any], settings: AppSettings) -> dict[str, Any]:
    decision = refresh_recording_routing(
        recording_id,
        settings=settings,
        apply_workflow=False,
    )
    refreshed = get_recording(recording_id, settings=settings) or rec
    projects = list_projects(settings=settings)
    selected_project_id = _as_int(refreshed.get("project_id"))
    selected_project = None
    for project in projects:
        if _as_int(project.get("id")) == selected_project_id:
            selected_project = project
            break

    suggested_project_id = _as_int(decision.get("suggested_project_id"))
    if suggested_project_id is None:
        suggested_project_id = _as_int(refreshed.get("suggested_project_id"))
    suggested_project_name = str(decision.get("suggested_project_name") or "").strip()
    if not suggested_project_name:
        suggested_project_name = str(refreshed.get("suggested_project_name") or "").strip()
    confidence_raw = decision.get("confidence", refreshed.get("routing_confidence"))
    try:
        confidence = max(0.0, min(float(confidence_raw), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = float(decision.get("threshold") or settings.routing_auto_select_threshold)

    rationale_payload = decision.get("rationale")
    if not isinstance(rationale_payload, list):
        rationale_payload = refreshed.get("routing_rationale_json")
    rationale = [str(item).strip() for item in (rationale_payload or []) if str(item).strip()]

    selected_training_examples = (
        count_routing_training_examples(project_id=selected_project_id, settings=settings)
        if selected_project_id is not None
        else 0
    )
    return {
        "projects": projects,
        "selected_project_id": selected_project_id,
        "selected_project_name": str((selected_project or {}).get("name") or "").strip(),
        "suggested_project_id": suggested_project_id,
        "suggested_project_name": suggested_project_name,
        "confidence": confidence,
        "threshold": threshold,
        "rationale": rationale,
        "training_examples_total": count_routing_training_examples(settings=settings),
        "training_examples_selected_project": selected_training_examples,
    }


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
    refresh_recording_metrics(recording_id, settings=settings)


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
# UI auth shell
# ---------------------------------------------------------------------------


@ui_router.get("/ui")
async def ui_root(request: Request) -> Any:
    if auth_enabled(_settings) and not request_is_authenticated(request, _settings):
        return RedirectResponse("/ui/login?next=%2Fui", status_code=303)
    return RedirectResponse("/", status_code=303)


@ui_router.get("/ui/login", response_class=HTMLResponse)
async def ui_login(
    request: Request,
    next: str = Query(default="/ui"),
) -> Any:
    if not auth_enabled(_settings):
        return RedirectResponse("/", status_code=303)
    target = safe_next_path(next, default="/ui")
    if request_is_authenticated(request, _settings):
        return RedirectResponse(target, status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "active": "",
            "next": target,
            "error": "",
        },
    )


@ui_router.post("/ui/login", response_class=HTMLResponse)
async def ui_login_submit(
    request: Request,
    token: str = Form(default=""),
    next: str = Form(default="/ui"),
) -> Any:
    if not auth_enabled(_settings):
        return RedirectResponse("/", status_code=303)

    expected = expected_bearer_token(_settings) or ""
    submitted = token.strip()
    target = safe_next_path(next, default="/ui")
    if not submitted or submitted != expected:
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "active": "",
                "next": target,
                "error": "Invalid token.",
            },
            status_code=401,
        )

    response = RedirectResponse(target, status_code=303)
    set_auth_cookie(
        response,
        submitted,
        secure=cookie_secure_flag(request),
    )
    return response


@ui_router.get("/ui/logout")
async def ui_logout() -> Any:
    target = "/ui/login" if auth_enabled(_settings) else "/"
    response = RedirectResponse(target, status_code=303)
    clear_auth_cookie(response)
    return response


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
    for item in items:
        progress_ratio = _safe_pipeline_progress(item.get("pipeline_progress"))
        item["progress_percent"] = int(round(progress_ratio * 100))
        item["progress_stage_label"] = _pipeline_stage_label(item.get("pipeline_stage"))
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
    recovery_warning = _recording_recovery_warning(jobs)
    tabs = ["overview", "project", "speakers", "language", "metrics", "log"]
    current_tab = tab if tab in tabs else "overview"
    language: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    speakers: dict[str, Any] | None = None
    project: dict[str, Any] | None = None
    export_text = ""
    if current_tab == "language":
        language = _language_tab_context(recording_id, rec, _settings)
    if current_tab == "speakers":
        speakers = _speakers_tab_context(recording_id, _settings)
    if current_tab == "project":
        project = _project_tab_context(recording_id, rec, _settings)
        rec = get_recording(recording_id, settings=_settings) or rec
    if current_tab in {"overview", "metrics"}:
        summary = _summary_context(recording_id, _settings)
    if current_tab == "overview":
        export_text = build_onenote_markdown(rec, settings=_settings)
    if current_tab == "metrics":
        metrics = _metrics_tab_context(recording_id, _settings)

    return templates.TemplateResponse(
        request,
        "recording_detail.html",
        {
            "active": "recordings",
            "rec": rec,
            "jobs": jobs,
            "recovery_warning": recovery_warning,
            "tabs": tabs,
            "current_tab": current_tab,
            "language": language,
            "summary": summary,
            "metrics": metrics,
            "speakers": speakers,
            "project": project,
            "export_text": export_text,
        },
    )


@ui_router.get("/ui/recordings/{recording_id}/progress", response_class=HTMLResponse)
async def ui_recording_progress(request: Request, recording_id: str) -> Any:
    rec = get_recording(recording_id, settings=_settings)
    if rec is None:
        return HTMLResponse("Not found", status_code=404)
    progress_ratio = _safe_pipeline_progress(rec.get("pipeline_progress"))
    progress_percent = int(round(progress_ratio * 100))
    return templates.TemplateResponse(
        request,
        "partials/recording_progress.html",
        {
            "rec": rec,
            "progress_ratio": progress_ratio,
            "progress_percent": progress_percent,
            "stage_code": str(rec.get("pipeline_stage") or "").strip() or "waiting",
            "stage_label": _pipeline_stage_label(rec.get("pipeline_stage")),
            "updated_at": str(rec.get("pipeline_updated_at") or "").strip(),
            "warning": str(rec.get("last_warning") or "").strip(),
            "is_processing": str(rec.get("status") or "") == RECORDING_STATUS_PROCESSING,
        },
    )


@ui_router.get("/ui/recordings/{recording_id}/export.zip")
async def ui_recording_export_zip(
    recording_id: str,
    include_snippets: int = Query(default=0),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    zip_bytes = build_export_zip_bytes(
        recording_id,
        settings=_settings,
        include_snippets=include_snippets == 1,
    )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="export_{recording_id}.zip"',
        },
    )


# ---------------------------------------------------------------------------
# Recording speaker assignment + snippet audio
# ---------------------------------------------------------------------------


@ui_router.get("/ui/recordings/{recording_id}/snippets/{speaker_slug}/{filename}")
async def ui_recording_snippet_audio(
    recording_id: str,
    speaker_slug: str,
    filename: str,
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    snippets_root = _settings.recordings_root / recording_id / "derived" / "snippets"
    safe_file = _safe_audio_path(
        snippets_root / speaker_slug / filename,
        root=snippets_root,
    )
    if safe_file is None or not safe_file.exists() or not safe_file.is_file():
        return HTMLResponse("Snippet not found", status_code=404)
    return FileResponse(path=str(safe_file), media_type="audio/wav", filename=safe_file.name)


@ui_router.post("/ui/recordings/{recording_id}/speakers/assign")
async def ui_assign_speaker(
    recording_id: str,
    diar_speaker_label: str = Form(...),
    voice_profile_id: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    diar_label = diar_speaker_label.strip()
    if not diar_label:
        return HTMLResponse("diar_speaker_label is required", status_code=422)

    profile_token = voice_profile_id.strip()
    profile_id: int | None = None
    if profile_token:
        try:
            profile_id = int(profile_token)
        except ValueError:
            return HTMLResponse("voice_profile_id must be an integer", status_code=422)
    try:
        set_speaker_assignment(
            recording_id=recording_id,
            diar_speaker_label=diar_label,
            voice_profile_id=profile_id,
            confidence=1.0,
            settings=_settings,
        )
    except sqlite3.IntegrityError:
        return HTMLResponse("Voice profile not found", status_code=404)
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    return RedirectResponse(f"/recordings/{recording_id}?tab=speakers", status_code=303)


@ui_router.post("/ui/recordings/{recording_id}/speakers/create-and-assign")
async def ui_create_and_assign_speaker(
    recording_id: str,
    diar_speaker_label: str = Form(...),
    display_name: str = Form(...),
    notes: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    diar_label = diar_speaker_label.strip()
    if not diar_label:
        return HTMLResponse("diar_speaker_label is required", status_code=422)
    clean_name = display_name.strip()
    if not clean_name:
        return HTMLResponse("display_name is required", status_code=422)

    profile = create_voice_profile(clean_name, notes.strip() or None, settings=_settings)
    try:
        profile_id = int(profile.get("id"))
    except (TypeError, ValueError):
        return HTMLResponse("Voice profile create failed", status_code=503)
    set_speaker_assignment(
        recording_id=recording_id,
        diar_speaker_label=diar_label,
        voice_profile_id=profile_id,
        confidence=1.0,
        settings=_settings,
    )
    return RedirectResponse(f"/recordings/{recording_id}?tab=speakers", status_code=303)


@ui_router.post("/ui/recordings/{recording_id}/speakers/add-sample")
async def ui_add_speaker_sample(
    recording_id: str,
    diar_speaker_label: str = Form(...),
    voice_profile_id: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    diar_label = diar_speaker_label.strip()
    if not diar_label:
        return HTMLResponse("diar_speaker_label is required", status_code=422)
    profile_token = voice_profile_id.strip()
    if not profile_token:
        return HTMLResponse("voice_profile_id is required", status_code=422)
    try:
        profile_id = int(profile_token)
    except ValueError:
        return HTMLResponse("voice_profile_id must be an integer", status_code=422)

    snippet_files = _speaker_snippet_files(
        recording_id,
        diar_label,
        settings=_settings,
    )
    if not snippet_files:
        return HTMLResponse("No snippets available for this speaker", status_code=422)

    rel_path = _as_data_relative_path(snippet_files[0], settings=_settings)
    if rel_path is None:
        return HTMLResponse("Snippet path is outside runtime data root", status_code=422)
    try:
        create_voice_sample(
            voice_profile_id=profile_id,
            snippet_path=rel_path,
            recording_id=recording_id,
            diar_speaker_label=diar_label,
            settings=_settings,
        )
    except sqlite3.IntegrityError:
        return HTMLResponse("Voice profile not found", status_code=404)
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    return RedirectResponse(f"/recordings/{recording_id}?tab=speakers", status_code=303)


@ui_router.post("/ui/recordings/{recording_id}/project")
async def ui_set_recording_project(
    recording_id: str,
    project_id: str = Form(default=""),
    train_routing: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)

    token = project_id.strip()
    resolved_project_id: int | None = None
    if token:
        try:
            resolved_project_id = int(token)
        except ValueError:
            return HTMLResponse("project_id must be an integer", status_code=422)

    try:
        updated = set_recording_project(
            recording_id,
            resolved_project_id,
            settings=_settings,
        )
    except sqlite3.IntegrityError:
        return HTMLResponse("Project not found", status_code=404)
    if not updated:
        return HTMLResponse("Not found", status_code=404)

    should_train = train_routing.strip().lower() in {"1", "true", "on", "yes"}
    if should_train and resolved_project_id is not None:
        try:
            train_routing_from_manual_selection(
                recording_id,
                resolved_project_id,
                settings=_settings,
            )
        except KeyError:
            return HTMLResponse("Project not found", status_code=404)
    refresh_recording_routing(
        recording_id,
        settings=_settings,
        apply_workflow=False,
    )
    return RedirectResponse(f"/recordings/{recording_id}?tab=project", status_code=303)


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
    samples = list_voice_samples(settings=_settings)
    for sample in samples:
        sample_id = sample.get("id")
        sample["audio_url"] = f"/ui/voice-samples/{sample_id}/audio"
    samples_by_profile: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        try:
            profile_id = int(sample.get("voice_profile_id"))
        except (TypeError, ValueError):
            continue
        samples_by_profile.setdefault(profile_id, []).append(sample)
    for item in items:
        try:
            profile_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        item["sample_count"] = len(samples_by_profile.get(profile_id, []))
    return templates.TemplateResponse(
        request,
        "voices.html",
        {
            "active": "voices",
            "items": items,
            "samples": samples,
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


@ui_router.get("/ui/voice-samples/{sample_id}/audio")
async def ui_voice_sample_audio(sample_id: int) -> Any:
    sample = get_voice_sample(sample_id, settings=_settings)
    if sample is None:
        return HTMLResponse("Voice sample not found", status_code=404)
    snippet_raw = str(sample.get("snippet_path") or "").strip()
    if not snippet_raw:
        return HTMLResponse("Voice sample has no snippet path", status_code=404)
    source_path = Path(snippet_raw)
    if not source_path.is_absolute():
        source_path = _settings.data_root / source_path
    safe_path = _safe_audio_path(source_path, root=_settings.data_root)
    if safe_path is None or not safe_path.exists() or not safe_path.is_file():
        return HTMLResponse("Snippet not found", status_code=404)
    return FileResponse(path=str(safe_path), media_type="audio/wav", filename=safe_path.name)


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
# Upload
# ---------------------------------------------------------------------------


@ui_router.get("/upload", response_class=HTMLResponse)
async def ui_upload(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "active": "upload",
        },
    )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


@ui_router.get("/connections", response_class=HTMLResponse)
async def ui_connections(request: Request) -> Any:
    gdrive_state = _gdrive_connection_state(_settings)
    return templates.TemplateResponse(
        request,
        "connections.html",
        {
            "active": "connections",
            "gdrive": gdrive_state,
        },
    )


@ui_router.post("/ui/connections/gdrive/test", response_class=HTMLResponse)
async def ui_test_gdrive_connection() -> Any:
    try:
        result = await run_in_threadpool(_test_gdrive_connection, _settings)
    except ValueError as exc:
        return HTMLResponse(f"<span style='color:#92400e'>{exc}</span>")
    except Exception as exc:
        return HTMLResponse(f"<span style='color:#b42318'>Google Drive test failed: {exc}</span>")
    return HTMLResponse(f"<span style='color:#14532d'>{result['message']}</span>")


# ---------------------------------------------------------------------------
# Inline recording actions (HTMX targets returning HX-Redirect)
# ---------------------------------------------------------------------------


@ui_router.post("/ui/recordings/{recording_id}/requeue")
async def ui_action_requeue(recording_id: str) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    try:
        enqueue_recording_job(recording_id, settings=_settings)
    except DuplicateRecordingJobError as exc:
        return HTMLResponse(
            f"Requeue skipped: precheck job already active ({exc.job_id}).",
            status_code=409,
        )
    except Exception as exc:
        return HTMLResponse(f"Requeue failed: {exc}", status_code=503)
    resp = HTMLResponse("")
    resp.headers["HX-Redirect"] = f"/recordings/{recording_id}"
    return resp


@ui_router.post("/ui/recordings/{recording_id}/jobs/{job_id}/retry")
async def ui_action_retry_failed_step(recording_id: str, job_id: str) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    job = get_job(job_id, settings=_settings)
    if job is None or str(job.get("recording_id") or "") != recording_id:
        return HTMLResponse("Job not found", status_code=404)
    if str(job.get("status") or "") != JOB_STATUS_FAILED:
        return HTMLResponse("Only failed jobs can be retried", status_code=422)
    try:
        enqueue_recording_job(
            recording_id,
            job_type=DEFAULT_REQUEUE_JOB_TYPE,
            settings=_settings,
        )
    except DuplicateRecordingJobError as exc:
        return HTMLResponse(
            f"Retry skipped: precheck job already active ({exc.job_id}).",
            status_code=409,
        )
    except Exception as exc:
        return HTMLResponse(f"Retry failed: {exc}", status_code=503)
    return RedirectResponse(f"/recordings/{recording_id}?tab=log", status_code=303)


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
async def ui_action_delete(
    recording_id: str,
    confirm_delete: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    if confirm_delete.strip().upper() != "DELETE":
        return HTMLResponse('Type "DELETE" to confirm deletion.', status_code=422)
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

