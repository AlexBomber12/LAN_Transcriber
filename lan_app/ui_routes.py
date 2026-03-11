"""Server-rendered HTML pages for the LAN Transcriber control panel.

Uses Jinja2 templates + HTMX (bundled locally) for a minimal, DB-window-style UI.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
from .calendar.ics import validate_ics_url
from .calendar.matching import (
    calendar_match_candidates,
    calendar_summary_context,
    selected_calendar_candidate,
)
from .calendar.service import CalendarSyncError, redacted_calendar_source, sync_calendar_source
from .conversation_metrics import refresh_recording_metrics
from .constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    JOB_STATUS_FAILED,
    JOB_STATUSES,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUSES,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_QUARANTINE,
)
from .db import (
    create_calendar_source,
    create_glossary_entry,
    count_routing_training_examples,
    create_voice_sample,
    create_project,
    create_voice_profile,
    delete_glossary_entry,
    get_calendar_source,
    get_glossary_entry,
    delete_project,
    delete_voice_profile,
    get_calendar_match,
    get_meeting_metrics,
    get_job,
    get_recording,
    get_voice_sample,
    list_calendar_events,
    list_calendar_sources,
    list_glossary_entries,
    list_participant_metrics,
    list_jobs,
    list_recording_pipeline_stages,
    list_projects,
    list_recordings,
    list_speaker_assignments,
    list_voice_samples,
    list_voice_profiles,
    set_calendar_match_selection,
    set_recording_duration,
    set_recording_project,
    set_speaker_assignment,
    set_recording_language_settings,
    set_recording_status,
    update_glossary_entry,
)
from .exporter import build_export_zip_bytes, build_onenote_markdown
from .jobs import (
    DuplicateRecordingJobError,
    enqueue_recording_job,
    purge_pending_recording_jobs,
)
from .ops import RecordingDeleteError, delete_recording_with_artifacts
from .routing import refresh_recording_routing, train_routing_from_manual_selection
from .speaker_bank import DEFAULT_ASSIGNMENT_THRESHOLD, merge_canonical_speakers
from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.llm_client import LLMClient
from lan_transcriber.pipeline import Settings as PipelineSettings
from lan_transcriber.pipeline import build_structured_summary_prompts, build_summary_payload
from lan_transcriber.pipeline_steps.precheck import (
    _audio_duration_from_ffprobe,
    _audio_duration_from_wave,
)
from lan_transcriber.utils import normalise_language_code as _normalise_language_code_shared

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
_LOG = logging.getLogger(__name__)

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
_DISPLAY_TIMEZONE = "Europe/Rome"
_GLOSSARY_KIND_OPTIONS = ("person", "company", "product", "project", "term")
_GLOSSARY_SOURCE_OPTIONS = (
    "manual",
    "correction",
    "system",
    "speaker_bank",
    "calendar",
    "project",
)
_TERMINAL_RECORDING_STATUSES = frozenset(RECORDING_STATUSES) - {
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_QUEUED,
}
_PIPELINE_STAGE_LABELS = {
    "sanitize_audio": "Sanitize Audio",
    "precheck": "Sanitize & Precheck",
    "calendar_refresh": "Calendar Refresh",
    "asr": "ASR / VAD",
    "stt": "ASR / VAD",
    "diarization": "Diarization",
    "diarize": "Diarization",
    "align": "Word Alignment",
    "speaker_turns": "Speaker Turns",
    "language": "Language Analysis",
    "language_analysis": "Language Analysis",
    "llm_extract": "LLM Summary",
    "llm": "LLM Summary",
    "llm_merge": "LLM Merge",
    "export_artifacts": "Export Artifacts",
    "metrics": "Post-process & Metrics",
    "routing": "Routing",
    "done": "Done",
}


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
                f"{_format_local_timestamp(finished_at)}."
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
    if text in _PIPELINE_STAGE_LABELS:
        return _PIPELINE_STAGE_LABELS[text]
    if text.startswith("llm_chunk_"):
        parts = text.split("_")
        if len(parts) == 5 and parts[3] == "of":
            return f"LLM Chunk {parts[2]} of {parts[4]}"
    return text.replace("_", " ").title()


def _pipeline_stage_rows_for_display(recording_id: str) -> list[dict[str, Any]]:
    rows = list_recording_pipeline_stages(recording_id, settings=_settings)
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        stage_name = str(row.get("stage_name") or "").strip()
        status = str(row.get("status") or "").strip()
        display_rows.append(
            {
                "stage_name": stage_name,
                "stage_label": _pipeline_stage_label(stage_name),
                "status": status,
                "status_label": status.replace("_", " ").title() if status else "Unknown",
                "attempt": int(row.get("attempt") or 0),
                "updated_at_display": _format_local_timestamp(row.get("updated_at")),
                "started_at_display": _format_local_timestamp(row.get("started_at")),
                "finished_at_display": _format_local_timestamp(row.get("finished_at")),
                "duration_seconds": (
                    round(float(row.get("duration_ms") or 0) / 1000.0, 3)
                    if row.get("duration_ms") is not None
                    else None
                ),
                "error_text": str(row.get("error_text") or "").strip(),
            }
        )
    return display_rows


def _safe_duration_seconds(value: object) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return round(duration, 3)


def _format_duration_seconds(value: object) -> str:
    duration = _safe_duration_seconds(value)
    if duration is None:
        return "—"
    total_seconds = int(duration)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _display_timezone() -> ZoneInfo | timezone:
    try:
        return ZoneInfo(_DISPLAY_TIMEZONE)
    except ZoneInfoNotFoundError:
        return timezone.utc


def _format_local_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "—"
    try:
        parsed = _parse_iso_datetime(value, field_name="timestamp")
    except ValueError:
        return str(value)
    return parsed.astimezone(_display_timezone()).strftime("%Y-%m-%d %H:%M:%S %Z")


def _recording_audio_candidates(recording_id: str, settings: AppSettings) -> list[Path]:
    candidates: list[Path] = []
    sanitized = settings.recordings_root / recording_id / "derived" / "audio_sanitized.wav"
    if sanitized.exists():
        candidates.append(sanitized)
    raw_dir = settings.recordings_root / recording_id / "raw"
    for raw_path in sorted(raw_dir.glob("audio.*")):
        candidates.append(raw_path)
    return candidates


def _probe_duration_seconds(audio_path: Path) -> float | None:
    return _safe_duration_seconds(
        _audio_duration_from_wave(audio_path) or _audio_duration_from_ffprobe(audio_path)
    )


def _prepare_recording_for_display(
    recording: dict[str, Any],
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    item = dict(recording)
    recording_id = str(item.get("id") or "").strip()
    duration_sec = _safe_duration_seconds(item.get("duration_sec"))
    if duration_sec is None and recording_id:
        for candidate in _recording_audio_candidates(recording_id, settings):
            duration_sec = _probe_duration_seconds(candidate)
            if duration_sec is None:
                continue
            item["duration_sec"] = duration_sec
            try:
                set_recording_duration(
                    recording_id,
                    duration_sec,
                    settings=settings,
                    touch_updated_at=False,
                )
            except sqlite3.Error:
                _LOG.warning(
                    "Failed to backfill display duration for recording %s",
                    recording_id,
                    exc_info=True,
                )
            break

    item["duration_display"] = _format_duration_seconds(item.get("duration_sec"))
    item["captured_at_display"] = _format_local_timestamp(item.get("captured_at"))
    item["created_at_display"] = _format_local_timestamp(item.get("created_at"))
    item["updated_at_display"] = _format_local_timestamp(item.get("updated_at"))
    item["pipeline_updated_at_display"] = _format_local_timestamp(
        item.get("pipeline_updated_at")
    )
    item["review_reason_text_display"] = str(
        item.get("review_reason_text") or ""
    ).strip()
    return item


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
    speaker_name_map = _recording_speaker_name_map(recording_id, settings=settings)

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
        diar_label = str(row.get("speaker") or "").strip()
        if not diar_label:
            continue
        participants.append(
            {
                "speaker": _speaker_display_label(
                    diar_label,
                    speaker_name_map=speaker_name_map,
                ),
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


def _asr_glossary_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    derived = settings.recordings_root / recording_id / "derived"
    payload = _load_json_dict(derived / "asr_glossary.json")
    if not payload:
        return {
            "available": False,
            "entries": [],
            "entry_count": 0,
            "term_count": 0,
            "truncated": False,
        }

    entry_rows: list[dict[str, Any]] = []
    raw_entries = payload.get("entries")
    if isinstance(raw_entries, list):
        for row in raw_entries:
            if not isinstance(row, dict):
                continue
            canonical_text = " ".join(str(row.get("canonical_text") or "").split())
            if not canonical_text:
                continue
            aliases_raw = row.get("aliases")
            aliases = (
                [
                    " ".join(str(alias).split())
                    for alias in aliases_raw
                    if " ".join(str(alias).split())
                ]
                if isinstance(aliases_raw, list)
                else []
            )
            sources_raw = row.get("sources")
            sources = (
                [
                    " ".join(str(source).split())
                    for source in sources_raw
                    if " ".join(str(source).split())
                ]
                if isinstance(sources_raw, list)
                else []
            )
            entry_rows.append(
                {
                    "canonical_text": canonical_text,
                    "aliases": aliases,
                    "kind": " ".join(str(row.get("kind") or "term").split()) or "term",
                    "sources": sources,
                    "sources_label": ", ".join(
                        source.replace("_", " ") for source in sources
                    )
                    or "—",
                }
            )

    def _safe_count(value: Any, *, default: int) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    return {
        "available": True,
        "entries": entry_rows,
        "entry_count": _safe_count(payload.get("entry_count"), default=len(entry_rows)),
        "term_count": _safe_count(
            payload.get("term_count"),
            default=sum(1 + len(row["aliases"]) for row in entry_rows),
        ),
        "truncated": bool(payload.get("truncated")),
    }


def _glossary_form_payload(
    *,
    canonical_text: str,
    aliases_text: str,
    kind: str,
    source: str,
    enabled: str,
    notes: str,
    recording_id: str,
    existing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
    clean_recording_id = " ".join(recording_id.strip().split())
    if clean_recording_id:
        metadata["recording_id"] = clean_recording_id
    else:
        metadata.pop("recording_id", None)
    return {
        "canonical_text": canonical_text.strip(),
        "aliases": aliases_text,
        "term_kind": kind,
        "source": source,
        "enabled": enabled.strip().lower() in {"1", "true", "on", "yes"},
        "notes": notes,
        "metadata": metadata,
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


def _snippets_manifest_path(recording_id: str, *, settings: AppSettings) -> Path:
    return settings.recordings_root / recording_id / "derived" / "snippets_manifest.json"


def _speaker_snippet_manifest_entries(
    recording_id: str,
    speaker_label: str,
    *,
    settings: AppSettings,
) -> list[dict[str, Any]]:
    payload = _load_json_dict(_snippets_manifest_path(recording_id, settings=settings))
    speakers_payload = payload.get("speakers")
    if not isinstance(speakers_payload, dict):
        return []
    rows = speakers_payload.get(speaker_label)
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("speaker") or speaker_label).strip() != speaker_label:
            continue
        out.append(dict(row))
    out.sort(
        key=lambda item: (
            int(item.get("ranking_position") or 0),
            str(item.get("snippet_id") or ""),
        )
    )
    return out


def _snippet_audio_url(
    recording_id: str,
    relative_path: str,
) -> str | None:
    normalized = Path(relative_path)
    if len(normalized.parts) != 2:
        return None
    speaker_slug, filename = normalized.parts
    return (
        f"/ui/recordings/{quote(recording_id, safe='')}/snippets/"
        f"{quote(speaker_slug, safe='')}/{quote(filename, safe='')}"
    )


def _snippet_choice_label(entry: dict[str, Any]) -> str:
    prefix = "Recommended" if bool(entry.get("recommended")) else "Clean"
    clip_start = float(entry.get("clip_start") or 0.0)
    clip_end = float(entry.get("clip_end") or clip_start)
    purity = max(0.0, min(float(entry.get("purity_score") or 0.0), 1.0))
    return (
        f"{prefix}: {clip_start:.2f}s-{clip_end:.2f}s "
        f"(purity {purity * 100:.0f}%)"
    )


def _snippet_warning_messages(entries: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("status") or "").strip()
        if not status or status == "accepted":
            continue
        counts[status] = counts.get(status, 0) + 1

    messages: list[str] = []
    if counts.get("rejected_degraded"):
        messages.append(
            "Diarization ran in degraded mode, so snippet samples from this speaker were blocked."
        )
    if counts.get("rejected_overlap"):
        count = counts["rejected_overlap"]
        noun = "candidate was" if count == 1 else "candidates were"
        messages.append(f"{count} snippet {noun} rejected because it overlaps another speaker.")
    if counts.get("rejected_failed_extract"):
        count = counts["rejected_failed_extract"]
        noun = "candidate could" if count == 1 else "candidates could"
        messages.append(
            f"{count} snippet {noun} not be extracted from the sanitized WAV."
        )
    if counts.get("rejected_short"):
        count = counts["rejected_short"]
        noun = "candidate was" if count == 1 else "candidates were"
        messages.append(f"{count} snippet {noun} too short to trust as a voice sample.")
    return messages


def _no_clean_snippet_message(entries: list[dict[str, Any]]) -> str:
    statuses = {str(entry.get("status") or "").strip() for entry in entries}
    if "rejected_degraded" in statuses:
        return "No clean snippets are available because diarization ran in degraded mode."
    if "rejected_overlap" in statuses:
        return "No clean snippets are available because every candidate overlaps another speaker."
    if "rejected_failed_extract" in statuses:
        return "No clean snippets are available because extraction failed for the clean candidates."
    if "rejected_short" in statuses:
        return "No clean snippets are available because every candidate was too short."
    if entries:
        return "No accepted clean snippets are available for this speaker."
    return "No snippet quality data is available for this speaker yet."


def _speaker_snippet_context(
    recording_id: str,
    speaker_label: str,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    snippets_root = settings.recordings_root / recording_id / "derived" / "snippets"
    expected_prefix = f"{_speaker_slug(speaker_label)}/"
    clean_snippets: list[dict[str, Any]] = []
    for entry in _speaker_snippet_manifest_entries(
        recording_id,
        speaker_label,
        settings=settings,
    ):
        if str(entry.get("status") or "").strip() != "accepted":
            continue
        relative_path = str(entry.get("relative_path") or "").strip()
        if not relative_path.startswith(expected_prefix):
            continue
        safe_file = _safe_audio_path(snippets_root / relative_path, root=snippets_root)
        if safe_file is None or not safe_file.exists() or not safe_file.is_file():
            continue
        safe_relative = safe_file.relative_to(snippets_root.resolve()).as_posix()
        audio_url = _snippet_audio_url(recording_id, safe_relative)
        if audio_url is None:
            continue
        clean_snippets.append(
            {
                "relative_path": safe_relative,
                "audio_url": audio_url,
                "clip_label": (
                    f"{float(entry.get('clip_start') or 0.0):.2f}s-"
                    f"{float(entry.get('clip_end') or 0.0):.2f}s"
                ),
                "source_label": (
                    f"{str(entry.get('source_kind') or 'segment').title()} "
                    f"{float(entry.get('source_start') or 0.0):.2f}s-"
                    f"{float(entry.get('source_end') or 0.0):.2f}s"
                ),
                "purity_label": f"{max(0.0, min(float(entry.get('purity_score') or 0.0), 1.0)) * 100:.0f}%",
                "recommended": bool(entry.get("recommended")),
                "choice_label": _snippet_choice_label(entry),
            }
        )

    entries = _speaker_snippet_manifest_entries(
        recording_id,
        speaker_label,
        settings=settings,
    )
    return {
        "clean_snippets": clean_snippets,
        "snippet_warnings": _snippet_warning_messages(entries),
        "no_clean_snippet_message": (
            None if clean_snippets else _no_clean_snippet_message(entries)
        ),
    }


def _selected_clean_snippet(
    recording_id: str,
    speaker_label: str,
    snippet_path: str,
    *,
    settings: AppSettings,
) -> Path:
    chosen_path = snippet_path.strip()
    if not chosen_path:
        raise ValueError("snippet_path is required")
    accepted = next(
        (
            entry
            for entry in _speaker_snippet_manifest_entries(
                recording_id,
                speaker_label,
                settings=settings,
            )
            if str(entry.get("status") or "").strip() == "accepted"
            and str(entry.get("relative_path") or "").strip() == chosen_path
        ),
        None,
    )
    if accepted is None:
        raise ValueError("Selected snippet is not a clean snippet for this speaker")
    snippets_root = settings.recordings_root / recording_id / "derived" / "snippets"
    safe_file = _safe_audio_path(snippets_root / chosen_path, root=snippets_root)
    if safe_file is None:
        raise ValueError("Selected snippet path is invalid")
    safe_relative = safe_file.relative_to(snippets_root.resolve()).as_posix()
    if not safe_relative.startswith(f"{_speaker_slug(speaker_label)}/"):
        raise ValueError("Selected snippet does not belong to this speaker")
    if not safe_file.exists() or not safe_file.is_file():
        raise ValueError("Selected snippet file does not exist")
    return safe_file


def _candidate_match_rows(
    candidate_matches: Any,
    *,
    voice_profiles_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = candidate_matches if isinstance(candidate_matches, list) else []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile_id = _as_int(row.get("voice_profile_id"))
        if profile_id is None:
            continue
        try:
            score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        profile = voice_profiles_by_id.get(profile_id, {})
        display_name = str(
            row.get("display_name")
            or profile.get("display_name")
            or f"#{profile_id}"
        ).strip() or f"#{profile_id}"
        normalized.append(
            {
                "voice_profile_id": profile_id,
                "display_name": display_name,
                "score": max(0.0, min(score, 1.0)),
            }
        )
    normalized.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["display_name"]),
            int(item["voice_profile_id"]),
        )
    )
    return normalized


def _recording_speaker_name_map(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, str]:
    name_map: dict[str, str] = {}
    for row in list_speaker_assignments(recording_id, settings=settings):
        diar_label = str(row.get("diar_speaker_label") or "").strip()
        display_name = str(row.get("voice_profile_name") or "").strip()
        if diar_label and display_name:
            name_map[diar_label] = display_name
    return name_map


def _speaker_display_label(
    speaker_label: str,
    *,
    speaker_name_map: dict[str, str],
) -> str:
    diar_label = str(speaker_label or "").strip() or "S1"
    mapped_name = str(speaker_name_map.get(diar_label) or "").strip()
    if not mapped_name:
        return diar_label
    return f"{mapped_name} ({diar_label})"


def _voice_duplicate_candidates(
    *,
    voice_samples: list[dict[str, Any]],
    voice_profiles_by_id: dict[int, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for sample in voice_samples:
        source_profile_id = _as_int(sample.get("voice_profile_id"))
        if source_profile_id is None or source_profile_id not in voice_profiles_by_id:
            continue
        seen_targets: set[int] = set()
        for candidate in _candidate_match_rows(
            sample.get("candidate_matches_json"),
            voice_profiles_by_id=voice_profiles_by_id,
        ):
            target_profile_id = int(candidate["voice_profile_id"])
            if target_profile_id == source_profile_id or target_profile_id in seen_targets:
                continue
            seen_targets.add(target_profile_id)
            slot = evidence.setdefault(
                (source_profile_id, target_profile_id),
                {
                    "voice_profile_id": target_profile_id,
                    "display_name": str(candidate["display_name"]),
                    "best_score": 0.0,
                    "match_count": 0,
                },
            )
            slot["match_count"] += 1
            slot["best_score"] = max(slot["best_score"], float(candidate["score"]))

    grouped: dict[int, list[dict[str, Any]]] = {}
    for (source_profile_id, _target_profile_id), item in evidence.items():
        grouped.setdefault(source_profile_id, []).append(item)
    for rows in grouped.values():
        rows.sort(
            key=lambda item: (
                -int(item["match_count"]),
                -float(item["best_score"]),
                str(item["display_name"]),
                int(item["voice_profile_id"]),
            )
        )
    return grouped


def _speaker_review_notices(
    recording_id: str,
    *,
    low_confidence_count: int,
    settings: AppSettings,
) -> list[str]:
    derived_root = settings.recordings_root / recording_id / "derived"
    status_payload = _load_json_dict(derived_root / "diarization_status.json")
    metadata_payload = _load_json_dict(derived_root / "diarization_metadata.json")

    notices: list[str] = []
    degraded = bool(status_payload.get("degraded")) or bool(metadata_payload.get("degraded"))
    if degraded:
        mode = str(status_payload.get("mode") or metadata_payload.get("mode") or "").strip()
        reason = str(status_payload.get("reason") or metadata_payload.get("reason") or "").strip()
        message = "Diarization ran in degraded fallback mode"
        if mode and mode != "pyannote":
            message += f" ({mode})"
        if reason:
            message += f": {reason}."
        else:
            message += "; speaker results may need manual review."
        notices.append(message)
    if low_confidence_count == 1:
        notices.append("1 speaker match is low confidence and needs manual review.")
    elif low_confidence_count > 1:
        notices.append(
            f"{low_confidence_count} speaker matches are low confidence and need manual review."
        )
    return notices


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
    voice_profiles_by_id = {
        int(profile["id"]): profile
        for profile in voice_profiles
        if _as_int(profile.get("id")) is not None
    }

    speaker_rows: list[dict[str, Any]] = []
    low_confidence_count = 0
    for speaker in sorted(per_speaker):
        row = per_speaker[speaker]
        snippet_context = _speaker_snippet_context(recording_id, speaker, settings=settings)
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
        candidate_matches = _candidate_match_rows(
            assignment.get("candidate_matches_json"),
            voice_profiles_by_id=voice_profiles_by_id,
        )
        low_confidence = bool(assignment.get("low_confidence"))
        if low_confidence:
            low_confidence_count += 1
        voice_profile_name = str(assignment.get("voice_profile_name") or "")
        speaker_rows.append(
            {
                "speaker": speaker,
                "turn_count": int(row["turn_count"]),
                "duration_sec": round(float(row["duration_sec"]), 3),
                "preview_text": str(row["preview_text"]),
                "clean_snippets": snippet_context["clean_snippets"],
                "snippet_warnings": snippet_context["snippet_warnings"],
                "no_clean_snippet_message": snippet_context["no_clean_snippet_message"],
                "voice_profile_id": profile_id,
                "voice_profile_name": voice_profile_name,
                "confidence": max(0.0, min(confidence, 1.0)),
                "candidate_matches": candidate_matches,
                "low_confidence": low_confidence,
                "display_label": _speaker_display_label(
                    speaker,
                    speaker_name_map={speaker: voice_profile_name.strip()},
                ),
                "needs_review": low_confidence or (
                    profile_id is None and bool(candidate_matches)
                ),
                "assignment_threshold": DEFAULT_ASSIGNMENT_THRESHOLD,
            }
        )

    return {
        "speaker_rows": speaker_rows,
        "voice_profiles": voice_profiles,
        "review_notices": _speaker_review_notices(
            recording_id,
            low_confidence_count=low_confidence_count,
            settings=settings,
        ),
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
        llm_model=settings.llm_model,
    )
    speaker_turns_raw = _load_json_list(speaker_turns_path)
    speaker_turns = [row for row in speaker_turns_raw if isinstance(row, dict)]
    if not speaker_turns:
        speaker_turns = _fallback_speaker_turns_from_transcript(transcript_payload)
    if not speaker_turns:
        speaker_turns = [{"start": 0.0, "end": 0.0, "speaker": "S1", "text": transcript_text}]

    calendar_title, calendar_attendees = calendar_summary_context(
        recording_id,
        settings=settings,
    )
    if calendar_title is None and not calendar_attendees:
        calendar_title = str(transcript_payload.get("calendar_title") or "").strip() or None
        attendees_payload = transcript_payload.get("calendar_attendees")
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


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_iso_datetime(value: str, *, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_ymd_date(value: str, *, field_name: str) -> date:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD") from exc


def _calendar_date_range_defaults() -> tuple[str, str]:
    today = datetime.now(tz=timezone.utc).date()
    end = today + timedelta(days=7)
    return today.isoformat(), end.isoformat()


def _calendar_page_data(
    *,
    date_from: str,
    date_to: str,
    source_id: int | None,
    settings: AppSettings,
) -> dict[str, Any]:
    start_date = _parse_ymd_date(date_from, field_name="from")
    end_date = _parse_ymd_date(date_to, field_name="to")
    if end_date < start_date:
        raise ValueError("to must be after from")

    start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    # `to` is user-facing inclusive date, while DB query upper bound is exclusive.
    end_dt = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)

    sources_raw = list_calendar_sources(settings=settings)
    sources = [redacted_calendar_source(row) for row in sources_raw]
    for source in sources:
        source["last_synced_at_display"] = _format_local_timestamp(source.get("last_synced_at"))
    events = list_calendar_events(
        starts_from=_utc_iso(start_dt),
        ends_to=_utc_iso(end_dt),
        source_id=source_id,
        settings=settings,
    )
    for event in events:
        event["starts_at_display"] = _format_local_timestamp(event.get("starts_at"))
        event["ends_at_display"] = _format_local_timestamp(event.get("ends_at"))
        attendees = _calendar_attendee_labels(event.get("attendees_json"))
        event["attendees_preview"] = attendees[:4]
        event["attendees_label"] = ", ".join(attendees) if attendees else "—"
    return {
        "sources": sources,
        "events": events,
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "selected_source_id": source_id,
    }


def _calendar_attendee_labels(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for row in value:
        if isinstance(row, dict):
            label = str(row.get("label") or row.get("name") or row.get("email") or "").strip()
        else:
            label = str(row or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def _calendar_rationale_rows(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(row).strip() for row in value if str(row).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _calendar_candidate_context(candidate: dict[str, Any], *, selected_event_id: str) -> dict[str, Any]:
    attendees = _calendar_attendee_labels(candidate.get("attendees"))
    if not attendees:
        attendees = _calendar_attendee_labels(candidate.get("attendee_details"))
    confidence_raw = candidate.get("confidence", candidate.get("score"))
    try:
        confidence_value = float(confidence_raw)
    except (TypeError, ValueError):
        confidence_value = None
    organizer_display = str(
        candidate.get("organizer")
        or candidate.get("organizer_name")
        or candidate.get("organizer_email")
        or ""
    ).strip() or None
    event_id = str(candidate.get("event_id") or "").strip()
    return {
        **candidate,
        "event_id": event_id,
        "selected": event_id == selected_event_id and bool(event_id),
        "subject_display": str(candidate.get("subject") or candidate.get("summary") or "").strip()
        or "Untitled event",
        "starts_at_display": _format_local_timestamp(candidate.get("starts_at")),
        "ends_at_display": _format_local_timestamp(candidate.get("ends_at")),
        "organizer_display": organizer_display or "—",
        "attendees": attendees,
        "attendees_preview": attendees[:4],
        "attendees_label": ", ".join(attendees) if attendees else "—",
        "confidence_value": confidence_value,
        "confidence_display": (
            f"{confidence_value:.2f}" if confidence_value is not None else "—"
        ),
        "rationale_rows": _calendar_rationale_rows(candidate.get("rationale")),
        "source_label": str(
            candidate.get("source_name") or candidate.get("source_kind") or ""
        ).strip()
        or "—",
    }


def _calendar_tab_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    match_row = get_calendar_match(recording_id, settings=settings) or {}
    selected_event_id = str(match_row.get("selected_event_id") or "").strip()
    selected_candidate = selected_calendar_candidate(recording_id, settings=settings)
    selected = None
    if selected_candidate:
        selected = _calendar_candidate_context(
            selected_candidate,
            selected_event_id=selected_event_id,
        )
    candidates = [
        _calendar_candidate_context(candidate, selected_event_id=selected_event_id)
        for candidate in calendar_match_candidates(recording_id, settings=settings)
        if str(candidate.get("event_id") or "").strip()
    ]
    return {
        "selected": selected,
        "selected_event_id": selected_event_id or None,
        "selected_confidence": match_row.get("selected_confidence"),
        "candidates": candidates,
        "error_message": "",
    }


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
    for idx, item in enumerate(items):
        item = _prepare_recording_for_display(item, settings=_settings)
        progress_ratio = _safe_pipeline_progress(item.get("pipeline_progress"))
        item["progress_percent"] = int(round(progress_ratio * 100))
        item["progress_stage_label"] = _pipeline_stage_label(item.get("pipeline_stage"))
        items[idx] = item
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
    calendar_error: str = Query(default=""),
) -> Any:
    rec = get_recording(recording_id, settings=_settings)
    if rec is None:
        return HTMLResponse("<h1>404 – Recording not found</h1>", status_code=404)
    rec = _prepare_recording_for_display(rec, settings=_settings)
    jobs, _ = list_jobs(settings=_settings, recording_id=recording_id, limit=100)
    recovery_warning = _recording_recovery_warning(jobs)
    tabs = ["overview", "calendar", "project", "speakers", "language", "metrics", "log"]
    current_tab = tab if tab in tabs else "overview"
    calendar: dict[str, Any] | None = None
    language: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    speakers: dict[str, Any] | None = None
    project: dict[str, Any] | None = None
    glossary: dict[str, Any] | None = None
    export_text = ""
    pipeline_stages = _pipeline_stage_rows_for_display(recording_id)
    if current_tab == "calendar":
        calendar = _calendar_tab_context(recording_id, _settings)
        if calendar_error.strip():
            calendar["error_message"] = calendar_error.strip()
    if current_tab == "language":
        language = _language_tab_context(recording_id, rec, _settings)
    if current_tab == "speakers":
        speakers = _speakers_tab_context(recording_id, _settings)
    if current_tab == "project":
        project = _project_tab_context(recording_id, rec, _settings)
        rec = _prepare_recording_for_display(
            get_recording(recording_id, settings=_settings) or rec,
            settings=_settings,
        )
    if current_tab in {"overview", "metrics"}:
        summary = _summary_context(recording_id, _settings)
    if current_tab == "overview":
        export_text = build_onenote_markdown(rec, settings=_settings)
        glossary = _asr_glossary_context(recording_id, _settings)
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
            "calendar": calendar,
            "language": language,
            "summary": summary,
            "metrics": metrics,
            "speakers": speakers,
            "project": project,
            "glossary": glossary,
            "export_text": export_text,
            "pipeline_stages": pipeline_stages,
        },
    )


@ui_router.get("/ui/recordings/{recording_id}/progress", response_class=HTMLResponse)
async def ui_recording_progress(
    request: Request,
    recording_id: str,
    tab: str = Query(default="overview"),
) -> Any:
    rec = get_recording(recording_id, settings=_settings)
    if rec is None:
        return HTMLResponse("Not found", status_code=404)
    rec = _prepare_recording_for_display(rec, settings=_settings)
    progress_ratio = _safe_pipeline_progress(rec.get("pipeline_progress"))
    progress_percent = int(round(progress_ratio * 100))
    response = templates.TemplateResponse(
        request,
        "partials/recording_progress.html",
        {
            "rec": rec,
            "progress_ratio": progress_ratio,
            "progress_percent": progress_percent,
            "stage_code": str(rec.get("pipeline_stage") or "").strip() or "waiting",
            "stage_label": _pipeline_stage_label(rec.get("pipeline_stage")),
            "updated_at": str(rec.get("pipeline_updated_at") or "").strip(),
            "updated_at_display": rec.get("pipeline_updated_at_display"),
            "warning": str(rec.get("last_warning") or "").strip(),
            "is_processing": str(rec.get("status") or "") == RECORDING_STATUS_PROCESSING,
        },
    )
    if (
        request.headers.get("HX-Request") == "true"
        and str(rec.get("status") or "") in _TERMINAL_RECORDING_STATUSES
    ):
        response.headers["HX-Redirect"] = f"/recordings/{recording_id}?tab={quote(tab)}"
    return response


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
# Recording calendar selection
# ---------------------------------------------------------------------------


@ui_router.post("/ui/recordings/{recording_id}/calendar/select")
async def ui_select_calendar_match(
    recording_id: str,
    event_id: str = Form(default=""),
) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)

    clean_event_id = event_id.strip()
    selected_confidence: float | None = None
    if clean_event_id:
        selected_candidate = None
        for candidate in calendar_match_candidates(recording_id, settings=_settings):
            if str(candidate.get("event_id") or "").strip() == clean_event_id:
                selected_candidate = candidate
                break
        if selected_candidate is None:
            return HTMLResponse("Calendar candidate not found", status_code=422)
        try:
            selected_confidence = float(
                selected_candidate.get("confidence", selected_candidate.get("score"))
            )
        except (TypeError, ValueError):
            selected_confidence = None

    set_calendar_match_selection(
        recording_id=recording_id,
        event_id=clean_event_id or None,
        selected_confidence=selected_confidence,
        settings=_settings,
    )
    refresh_recording_routing(
        recording_id,
        settings=_settings,
        apply_workflow=False,
    )
    return RedirectResponse(f"/recordings/{recording_id}?tab=calendar", status_code=303)


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
    existing_assignment = next(
        (
            row
            for row in list_speaker_assignments(recording_id, settings=_settings)
            if str(row.get("diar_speaker_label") or "").strip() == diar_label
        ),
        {},
    )
    try:
        set_speaker_assignment(
            recording_id=recording_id,
            diar_speaker_label=diar_label,
            voice_profile_id=profile_id,
            confidence=(
                1.0
                if profile_id is not None
                else float(existing_assignment.get("confidence") or 0.0)
            ),
            candidate_matches=existing_assignment.get("candidate_matches_json"),
            low_confidence=(
                False if profile_id is not None else existing_assignment.get("low_confidence")
            ),
            keep_unmatched=(
                profile_id is None and bool(existing_assignment.get("candidate_matches_json"))
            ),
            settings=_settings,
        )
    except sqlite3.IntegrityError:
        return HTMLResponse("Voice profile not found", status_code=404)
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    _LOG.info(
        "speaker remap saved recording_id=%s diar_speaker_label=%s from_voice_profile_id=%s to_voice_profile_id=%s",
        recording_id,
        diar_label,
        existing_assignment.get("voice_profile_id"),
        profile_id,
    )
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
    _LOG.info(
        "speaker profile created and assigned recording_id=%s diar_speaker_label=%s voice_profile_id=%s",
        recording_id,
        diar_label,
        profile_id,
    )
    return RedirectResponse(f"/recordings/{recording_id}?tab=speakers", status_code=303)


@ui_router.post("/ui/recordings/{recording_id}/speakers/add-sample")
async def ui_add_speaker_sample(
    recording_id: str,
    diar_speaker_label: str = Form(...),
    voice_profile_id: str = Form(default=""),
    snippet_path: str = Form(default=""),
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

    try:
        selected_snippet = _selected_clean_snippet(
            recording_id,
            diar_label,
            snippet_path,
            settings=_settings,
        )
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)

    rel_path = _as_data_relative_path(selected_snippet, settings=_settings)
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
    _LOG.info(
        "speaker sample linked recording_id=%s diar_speaker_label=%s voice_profile_id=%s snippet_path=%s",
        recording_id,
        diar_label,
        profile_id,
        rel_path,
    )
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
# Glossary
# ---------------------------------------------------------------------------


@ui_router.get("/glossary", response_class=HTMLResponse)
async def ui_glossary(
    request: Request,
    edit_id: int | None = Query(default=None),
) -> Any:
    items = list_glossary_entries(settings=_settings)
    for item in items:
        aliases = item.get("aliases_json")
        item["aliases"] = aliases if isinstance(aliases, list) else []
        item["aliases_text"] = "\n".join(item["aliases"])
        metadata = item.get("metadata_json")
        item["metadata"] = metadata if isinstance(metadata, dict) else {}
        item["recording_id"] = str(item["metadata"].get("recording_id") or "").strip()
        item["source_label"] = str(item.get("source") or "").replace("_", " ")
    editing_entry = None if edit_id is None else get_glossary_entry(edit_id, settings=_settings)
    if isinstance(editing_entry, dict):
        aliases = editing_entry.get("aliases_json")
        editing_entry["aliases"] = aliases if isinstance(aliases, list) else []
        editing_entry["aliases_text"] = "\n".join(editing_entry["aliases"])
        metadata = editing_entry.get("metadata_json")
        editing_entry["metadata"] = metadata if isinstance(metadata, dict) else {}
        editing_entry["recording_id"] = str(
            editing_entry["metadata"].get("recording_id") or ""
        ).strip()
    return templates.TemplateResponse(
        request,
        "glossary.html",
        {
            "active": "glossary",
            "items": items,
            "editing_entry": editing_entry,
            "kind_options": _GLOSSARY_KIND_OPTIONS,
            "source_options": _GLOSSARY_SOURCE_OPTIONS,
        },
    )


@ui_router.post("/glossary", response_class=HTMLResponse)
async def ui_create_glossary(
    canonical_text: str = Form(...),
    aliases_text: str = Form(default=""),
    kind: str = Form(default="term"),
    source: str = Form(default="manual"),
    enabled: str = Form(default=""),
    notes: str = Form(default=""),
    recording_id: str = Form(default=""),
) -> Any:
    payload = _glossary_form_payload(
        canonical_text=canonical_text,
        aliases_text=aliases_text,
        kind=kind,
        source=source,
        enabled=enabled,
        notes=notes,
        recording_id=recording_id,
    )
    try:
        create_glossary_entry(settings=_settings, **payload)
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    return RedirectResponse("/glossary", status_code=303)


@ui_router.post("/glossary/{entry_id}", response_class=HTMLResponse)
async def ui_update_glossary(
    entry_id: int,
    canonical_text: str = Form(...),
    aliases_text: str = Form(default=""),
    kind: str = Form(default="term"),
    source: str = Form(default="manual"),
    enabled: str = Form(default=""),
    notes: str = Form(default=""),
    recording_id: str = Form(default=""),
) -> Any:
    existing_entry = get_glossary_entry(entry_id, settings=_settings)
    if existing_entry is None:
        return HTMLResponse("Glossary entry not found", status_code=404)
    payload = _glossary_form_payload(
        canonical_text=canonical_text,
        aliases_text=aliases_text,
        kind=kind,
        source=source,
        enabled=enabled,
        notes=notes,
        recording_id=recording_id,
        existing_metadata=(
            existing_entry.get("metadata_json")
            if isinstance(existing_entry.get("metadata_json"), dict)
            else None
        ),
    )
    try:
        update_glossary_entry(entry_id, settings=_settings, **payload)
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=422)
    return RedirectResponse("/glossary", status_code=303)


@ui_router.post("/glossary/{entry_id}/delete", response_class=HTMLResponse)
async def ui_delete_glossary(entry_id: int) -> Any:
    delete_glossary_entry(entry_id, settings=_settings)
    return RedirectResponse("/glossary", status_code=303)


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
    voice_profiles_by_id = {
        int(item["id"]): item
        for item in items
        if _as_int(item.get("id")) is not None
    }
    samples_by_profile: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        try:
            profile_id = int(sample.get("voice_profile_id"))
        except (TypeError, ValueError):
            continue
        samples_by_profile.setdefault(profile_id, []).append(sample)
    duplicate_candidates_by_profile = _voice_duplicate_candidates(
        voice_samples=samples,
        voice_profiles_by_id=voice_profiles_by_id,
    )
    for item in items:
        try:
            profile_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        item_samples = samples_by_profile.get(profile_id, [])
        item["sample_count"] = len(item_samples)
        item["samples_preview"] = item_samples[:3]
        item["duplicate_candidates"] = duplicate_candidates_by_profile.get(profile_id, [])
        item["merge_targets"] = [
            target
            for target in items
            if _as_int(target.get("id")) is not None and int(target["id"]) != profile_id
        ]
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


@ui_router.post("/voices/{profile_id}/merge", response_class=HTMLResponse)
async def ui_merge_voice(
    profile_id: int,
    target_profile_id: str = Form(default=""),
) -> Any:
    target_token = target_profile_id.strip()
    if not target_token:
        return HTMLResponse("target_profile_id is required", status_code=422)
    try:
        target_id = int(target_token)
    except ValueError:
        return HTMLResponse("target_profile_id must be an integer", status_code=422)
    try:
        merged = merge_canonical_speakers(
            profile_id,
            target_id,
            settings=_settings,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message else 422
        return HTMLResponse(message, status_code=status_code)
    _LOG.info(
        "canonical speaker merged source_profile_id=%s target_profile_id=%s merged=%s",
        profile_id,
        target_id,
        merged,
    )
    return RedirectResponse(f"/voices#voice-{target_id}", status_code=303)


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
# Calendars
# ---------------------------------------------------------------------------


@ui_router.get("/calendars", response_class=HTMLResponse)
async def ui_calendars(
    request: Request,
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    source_id: int | None = Query(default=None),
    error: str = Query(default=""),
) -> Any:
    default_from, default_to = _calendar_date_range_defaults()
    requested_from = from_ or default_from
    requested_to = to or default_to
    error_message = error.strip()
    try:
        context = _calendar_page_data(
            date_from=requested_from,
            date_to=requested_to,
            source_id=source_id,
            settings=_settings,
        )
    except ValueError as exc:
        error_message = str(exc)
        context = _calendar_page_data(
            date_from=default_from,
            date_to=default_to,
            source_id=None,
            settings=_settings,
        )
    context.update(
        {
            "active": "calendars",
            "error_message": error_message,
        }
    )
    return templates.TemplateResponse(request, "calendars.html", context)


@ui_router.post("/calendars/sources", response_class=HTMLResponse)
async def ui_create_calendar_source(
    name: str = Form(...),
    kind: str = Form(default="url"),
    url: str = Form(default=""),
    file: str = Form(default=""),
) -> Any:
    try:
        normalized_url: str | None = None
        if kind.strip().lower() == "url":
            normalized_url = validate_ics_url(url.strip())
        source = create_calendar_source(
            name=name.strip(),
            kind=kind,
            url=normalized_url,
            file_ics=file,
            settings=_settings,
        )
        return RedirectResponse(
            f"/calendars?source_id={int(source.get('id'))}",
            status_code=303,
        )
    except ValueError as exc:
        return RedirectResponse(
            f"/calendars?error={quote(str(exc), safe='')}",
            status_code=303,
        )


@ui_router.post("/calendars/sources/{source_id}/sync", response_class=HTMLResponse)
async def ui_sync_calendar_source(source_id: int) -> Any:
    if get_calendar_source(source_id, settings=_settings) is None:
        return HTMLResponse("Calendar source not found", status_code=404)
    try:
        await run_in_threadpool(
            sync_calendar_source,
            source_id,
            settings=_settings,
        )
    except CalendarSyncError as exc:
        return RedirectResponse(
            f"/calendars?source_id={source_id}&error={quote(str(exc), safe='')}",
            status_code=303,
        )
    return RedirectResponse(f"/calendars?source_id={source_id}", status_code=303)


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
async def ui_action_delete(recording_id: str) -> Any:
    if get_recording(recording_id, settings=_settings) is None:
        return HTMLResponse("Not found", status_code=404)
    try:
        purge_pending_recording_jobs(recording_id, settings=_settings)
    except Exception as exc:
        return HTMLResponse(f"Delete failed (queue unavailable): {exc}", status_code=503)
    try:
        deleted = delete_recording_with_artifacts(recording_id, settings=_settings)
    except RecordingDeleteError as exc:
        return HTMLResponse(str(exc), status_code=500)
    if not deleted:
        return HTMLResponse("Not found", status_code=404)
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
