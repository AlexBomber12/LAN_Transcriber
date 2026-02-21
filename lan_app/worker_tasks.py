from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lan_transcriber.llm_client import LLMClient
from lan_transcriber.pipeline import Settings as PipelineSettings
from lan_transcriber.pipeline import run_pipeline, run_precheck

from .config import AppSettings
from .constants import (
    JOB_TYPE_CLEANUP,
    JOB_TYPE_PRECHECK,
    JOB_TYPE_PUBLISH,
    JOB_TYPES,
    RECORDING_STATUS_FAILED,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_PUBLISHED,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_READY,
)
from .db import (
    fail_job,
    finish_job,
    get_calendar_match,
    get_recording,
    init_db,
    set_recording_language_settings,
    set_recording_status,
    start_job,
)


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


def _resolve_raw_audio_path(recording_id: str, settings: AppSettings) -> Path | None:
    raw_dir = settings.recordings_root / recording_id / "raw"
    candidates = sorted(raw_dir.glob("audio.*"))
    if not candidates:
        return None
    return candidates[0]


def _clean_language_value(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _load_transcript_language_payload(
    recording_id: str,
    settings: AppSettings,
) -> tuple[str | None, str | None]:
    transcript_path = settings.recordings_root / recording_id / "derived" / "transcript.json"
    if not transcript_path.exists():
        return None, None
    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    dominant = _clean_language_value(payload.get("dominant_language"))
    target = _clean_language_value(payload.get("target_summary_language"))
    return dominant, target


def _load_calendar_summary_context(
    recording_id: str,
    settings: AppSettings,
) -> tuple[str | None, list[str]]:
    row = get_calendar_match(recording_id, settings=settings) or {}
    selected_event_id = str(row.get("selected_event_id") or "").strip()
    if not selected_event_id:
        return None, []
    try:
        candidates = json.loads(str(row.get("candidates_json") or "[]"))
    except ValueError:
        return None, []
    if not isinstance(candidates, list):
        return None, []

    selected: dict[str, Any] | None = None
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if str(item.get("event_id") or "").strip() == selected_event_id:
            selected = item
            break
    if selected is None:
        return None, []

    title = str(selected.get("subject") or "").strip() or None
    attendees_raw = selected.get("attendees")
    attendees = []
    if isinstance(attendees_raw, list):
        attendees = [
            str(attendee).strip()
            for attendee in attendees_raw
            if str(attendee).strip()
        ]
    return title, attendees


class _FallbackDiariser:
    def __init__(self, duration_sec: float | None) -> None:
        self._duration_sec = max(duration_sec or 0.1, 0.1)

    async def __call__(self, _audio_path: Path):
        duration = self._duration_sec

        class _Annotation:
            def itertracks(self, yield_label: bool = False):
                if yield_label:
                    yield SimpleNamespace(start=0.0, end=duration), "S1"
                else:  # pragma: no cover - legacy branch
                    yield (SimpleNamespace(start=0.0, end=duration),)

        return _Annotation()


class _PyannoteDiariser:
    def __init__(self, pipeline_model: Any) -> None:
        self._pipeline_model = pipeline_model

    async def __call__(self, audio_path: Path):
        def _run_sync():
            try:
                return self._pipeline_model(str(audio_path))
            except Exception:
                return self._pipeline_model({"audio": str(audio_path)})

        return await asyncio.to_thread(_run_sync)


def _build_pipeline_settings(settings: AppSettings) -> PipelineSettings:
    return PipelineSettings(
        recordings_root=settings.recordings_root,
        voices_dir=settings.data_root / "voices",
        unknown_dir=settings.recordings_root / "unknown",
        tmp_root=settings.data_root / "tmp",
    )


def _build_diariser(duration_sec: float | None):
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing == "pyannote":
            return _FallbackDiariser(duration_sec)
        raise
    model = Pipeline.from_pretrained("pyannote/speaker-diarization@3.2")
    return _PyannoteDiariser(model)


def _run_precheck_pipeline(
    *,
    recording_id: str,
    settings: AppSettings,
    log_path: Path,
) -> tuple[str, str | None]:
    recording = get_recording(recording_id, settings=settings) or {}
    transcript_language_override = _clean_language_value(recording.get("language_override"))
    target_summary_language = _clean_language_value(recording.get("target_summary_language"))
    has_explicit_summary_target = target_summary_language is not None

    audio_path = _resolve_raw_audio_path(recording_id, settings)
    if audio_path is None:
        _append_step_log(log_path, "precheck skipped: raw audio not found")
        return RECORDING_STATUS_QUARANTINE, "raw_audio_missing"

    pipeline_settings = _build_pipeline_settings(settings)
    precheck = run_precheck(audio_path, pipeline_settings)
    _append_step_log(
        log_path,
        (
            "precheck "
            f"duration_sec={precheck.duration_sec} "
            f"speech_ratio={precheck.speech_ratio}"
        ),
    )
    if precheck.quarantine_reason:
        diariser = _FallbackDiariser(precheck.duration_sec)
    else:
        diariser = _build_diariser(precheck.duration_sec)
    calendar_title, calendar_attendees = _load_calendar_summary_context(
        recording_id,
        settings,
    )
    asyncio.run(
        run_pipeline(
            audio_path=audio_path,
            cfg=pipeline_settings,
            llm=LLMClient(),
            diariser=diariser,
            recording_id=recording_id,
            precheck=precheck,
            target_summary_language=target_summary_language,
            transcript_language_override=transcript_language_override,
            calendar_title=calendar_title,
            calendar_attendees=calendar_attendees,
        )
    )
    dominant_language, resolved_target_language = _load_transcript_language_payload(
        recording_id,
        settings,
    )
    update_payload: dict[str, str] = {}
    if dominant_language:
        update_payload["language_auto"] = dominant_language
    if has_explicit_summary_target and resolved_target_language:
        update_payload["target_summary_language"] = resolved_target_language
    if update_payload:
        set_recording_language_settings(
            recording_id,
            settings=settings,
            **update_payload,
        )
    _append_step_log(log_path, "pipeline artifacts generated")
    if precheck.quarantine_reason:
        _append_step_log(
            log_path,
            f"quarantined reason={precheck.quarantine_reason}",
        )
        return RECORDING_STATUS_QUARANTINE, precheck.quarantine_reason
    return RECORDING_STATUS_READY, None


def process_job(job_id: str, recording_id: str, job_type: str) -> dict[str, str]:
    """Execute a queue job and persist lifecycle state transitions."""

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

        quarantine_reason: str | None = None
        if job_type == JOB_TYPE_PRECHECK:
            final_status, quarantine_reason = _run_precheck_pipeline(
                recording_id=recording_id,
                settings=settings,
                log_path=log_path,
            )
        else:
            final_status = _success_status(job_type)

        if not set_recording_status(
            recording_id,
            final_status,
            settings=settings,
            quarantine_reason=quarantine_reason,
        ):
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
