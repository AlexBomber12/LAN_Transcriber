from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any

from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.audio_sanitize import (
    AudioSanitizeError,
    sanitize_audio_for_pipeline,
)
from lan_transcriber.compat.call_compat import call_with_supported_kwargs
from lan_transcriber.llm_client import (
    LLMClient,
    LLMEmptyContentError,
    LLMTruncatedResponseError,
)
from lan_transcriber.pipeline import Settings as PipelineSettings
from lan_transcriber.pipeline import run_pipeline, run_precheck
from lan_transcriber.pipeline_steps.diarization_quality import (
    DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
    DEFAULT_DIALOG_RETRY_MIN_TURNS,
    annotation_speaker_count,
    profile_default_speaker_hints,
)

from .config import AppSettings
from .conversation_metrics import refresh_recording_metrics
from .constants import (
    JOB_STATUS_QUEUED,
    JOB_STATUS_STARTED,
    JOB_TYPE_CLEANUP,
    JOB_TYPE_PRECHECK,
    JOB_TYPE_PUBLISH,
    JOB_TYPES,
    RECORDING_STATUSES,
    RECORDING_STATUS_FAILED,
    RECORDING_STATUS_NEEDS_REVIEW,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_PUBLISHED,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_READY,
)
from .db import (
    clear_recording_progress,
    fail_job,
    fail_job_if_started,
    finish_job_if_started,
    get_calendar_match,
    get_job,
    get_recording,
    init_db,
    list_jobs,
    requeue_job_if_started,
    set_recording_progress,
    set_recording_duration,
    set_recording_language_settings,
    set_recording_status,
    set_recording_status_if_current_in_and_job_started,
    start_job,
)
from .diarization_loader import load_pyannote_pipeline
from .routing import refresh_recording_routing

_logger = logging.getLogger(__name__)


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


_TERMINAL_STATUS_RE = re.compile(r"recording_status=([A-Za-z]+)")
_QUARANTINE_REASON_RE = re.compile(r"quarantined reason=([^\s]+)")


def _restore_status_from_precheck_log(
    recording_id: str,
    settings: AppSettings,
) -> tuple[str | None, str | None]:
    """Best-effort recovery of pre-legacy terminal status from precheck logs."""
    precheck_log = _step_log_path(recording_id, JOB_TYPE_PRECHECK, settings)
    if not precheck_log.exists():
        return None, None
    try:
        lines = precheck_log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None

    restored_status: str | None = None
    quarantine_reason: str | None = None
    for line in reversed(lines):
        if quarantine_reason is None:
            reason_match = _QUARANTINE_REASON_RE.search(line)
            if reason_match is not None:
                quarantine_reason = reason_match.group(1).strip() or None
        if restored_status is None:
            status_match = _TERMINAL_STATUS_RE.search(line)
            if status_match is not None:
                status = status_match.group(1).strip()
                if (
                    status in RECORDING_STATUSES
                    and status
                    not in {RECORDING_STATUS_QUEUED, RECORDING_STATUS_PROCESSING}
                ):
                    restored_status = status
                    if restored_status != RECORDING_STATUS_QUARANTINE:
                        return restored_status, None
        if (
            restored_status == RECORDING_STATUS_QUARANTINE
            and quarantine_reason is not None
        ):
            return restored_status, quarantine_reason
    return restored_status, quarantine_reason


def _has_queued_precheck_job(
    recording_id: str,
    *,
    settings: AppSettings,
    exclude_job_id: str,
) -> bool:
    queued_rows, _ = list_jobs(
        settings=settings,
        status=JOB_STATUS_QUEUED,
        recording_id=recording_id,
        limit=500,
        offset=0,
    )
    for row in queued_rows:
        row_id = str(row.get("id") or "")
        if row_id == exclude_job_id:
            continue
        if str(row.get("type") or "") == JOB_TYPE_PRECHECK:
            return True
    return False


def _success_status(job_type: str) -> str:
    if job_type == JOB_TYPE_PUBLISH:
        return RECORDING_STATUS_PUBLISHED
    if job_type == JOB_TYPE_CLEANUP:
        return RECORDING_STATUS_QUARANTINE
    return RECORDING_STATUS_READY


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    backoff_seconds: tuple[int, ...]


_DEFAULT_RETRY_POLICY = RetryPolicy(max_attempts=2, backoff_seconds=(2,))
_JOB_RETRY_POLICIES: dict[str, RetryPolicy] = {
    JOB_TYPE_PRECHECK: RetryPolicy(max_attempts=3, backoff_seconds=(1, 2)),
    JOB_TYPE_PUBLISH: RetryPolicy(max_attempts=2, backoff_seconds=(3,)),
    JOB_TYPE_CLEANUP: RetryPolicy(max_attempts=2, backoff_seconds=(5,)),
}
_MAX_ATTEMPTS_ERROR = "max attempts exceeded"


@dataclass(frozen=True)
class PipelineTerminalState:
    status: str
    quarantine_reason: str | None = None
    review_reason_code: str | None = None
    review_reason_text: str | None = None


def _retry_policy(job_type: str) -> RetryPolicy:
    return _JOB_RETRY_POLICIES.get(job_type, _DEFAULT_RETRY_POLICY)


def _is_retryable_exception(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, RuntimeError))


def _job_attempt(job_id: str, settings: AppSettings) -> int:
    row = get_job(job_id, settings=settings) or {}
    try:
        return int(row.get("attempt") or 0)
    except (TypeError, ValueError):
        return 0


def _job_status(job_id: str, settings: AppSettings) -> str:
    row = get_job(job_id, settings=settings) or {}
    return str(row.get("status") or "").strip()


def _ignored_result(job_id: str, recording_id: str, job_type: str) -> dict[str, str]:
    return {
        "job_id": job_id,
        "recording_id": recording_id,
        "job_type": job_type,
        "status": "ignored",
    }


def _log_stale_inflight_execution(
    *,
    job_id: str,
    job_type: str,
    log_path: Path,
    detail: str,
) -> None:
    try:
        _append_step_log(
            log_path,
            (
                "ignored stale in-flight execution "
                f"job={job_id} type={job_type} {detail}"
            ),
        )
    except OSError:
        pass


def _start_job_or_ignore_stale_execution(
    *,
    job_id: str,
    recording_id: str,
    job_type: str,
    settings: AppSettings,
    log_path: Path,
) -> bool:
    if start_job(job_id, settings=settings):
        return True
    job_row = get_job(job_id, settings=settings) or {}
    status = str(job_row.get("status") or "").strip()
    if status and status != JOB_STATUS_QUEUED:
        try:
            _append_step_log(
                log_path,
                (
                    "ignored stale queue execution "
                    f"job={job_id} type={job_type} status={status}"
                ),
            )
        except OSError:
            pass
        return False
    raise ValueError(f"Job not found: {job_id}")


def _retry_delay_seconds(policy: RetryPolicy, attempt: int) -> int:
    if attempt <= 0:
        return 0
    index = attempt - 1
    if index >= len(policy.backoff_seconds):
        return 0
    return max(int(policy.backoff_seconds[index]), 0)


def _record_retry(
    *,
    job_id: str,
    job_type: str,
    recording_id: str,
    attempt: int,
    max_attempts: int,
    delay_seconds: int,
    settings: AppSettings,
    log_path: Path,
    exc: Exception,
) -> bool:
    error = str(exc)
    try:
        requeued = requeue_job_if_started(
            job_id,
            error=f"retryable failure attempt {attempt}/{max_attempts}: {error}",
            settings=settings,
        )
    except Exception:
        return False
    if not requeued:
        return False
    try:
        set_recording_status(recording_id, RECORDING_STATUS_QUEUED, settings=settings)
    except Exception:
        pass
    try:
        _append_step_log(
            log_path,
            (
                f"retrying job={job_id} type={job_type} "
                f"attempt={attempt}/{max_attempts} "
                f"delay_seconds={delay_seconds} error={error}"
            ),
        )
    except Exception:
        pass
    return True


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


def _record_max_attempts_exceeded(
    *,
    job_id: str,
    job_type: str,
    recording_id: str,
    settings: AppSettings,
    log_path: Path,
    exc: Exception,
) -> None:
    review_reason_code, review_reason_text = _review_reason_from_exception(exc)
    try:
        fail_job(job_id, _MAX_ATTEMPTS_ERROR, settings=settings)
    except Exception:
        pass
    try:
        set_recording_status(
            recording_id,
            RECORDING_STATUS_NEEDS_REVIEW,
            settings=settings,
            review_reason_code=review_reason_code,
            review_reason_text=review_reason_text,
        )
    except Exception:
        pass
    try:
        _append_step_log(
            log_path,
            (
                f"terminal failure job={job_id} type={job_type}: "
                f"{_MAX_ATTEMPTS_ERROR}"
            ),
        )
    except Exception:
        pass


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


def _review_reason_from_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc).strip()
    lowered = message.lower()
    if isinstance(exc, LLMTruncatedResponseError) or "finish_reason=length" in lowered:
        return (
            "llm_truncated",
            "LLM output was truncated repeatedly; manual review required.",
        )
    if isinstance(exc, LLMEmptyContentError) or "empty message.content" in lowered:
        return (
            "llm_empty_content",
            "LLM returned empty content repeatedly; manual review required.",
        )
    if message == _MAX_ATTEMPTS_ERROR:
        return (
            "job_retry_limit_reached",
            "Processing hit the retry limit; manual review required.",
        )
    return (
        "job_retry_limit_reached",
        (
            "Processing hit the retry limit after repeated errors "
            f"({type(exc).__name__}); manual review required."
        ),
    )


def _review_reason_from_routing(
    *,
    recording_id: str,
    settings: AppSettings,
    routing: dict[str, Any],
) -> tuple[str, str]:
    derived_dir = settings.recordings_root / recording_id / "derived"
    summary_payload = _load_json_dict(derived_dir / "summary.json")
    diarization_payload = _load_json_dict(derived_dir / "diarization_metadata.json")

    parse_reason = str(summary_payload.get("parse_error_reason") or "").strip()
    if parse_reason:
        if parse_reason == "json_object_not_found":
            return (
                "llm_empty_content",
                "LLM output was empty or invalid JSON; manual review required.",
            )
        return (
            "llm_output_invalid",
            (
                "LLM output could not be parsed cleanly "
                f"({parse_reason}); manual review required."
            ),
        )

    if bool(diarization_payload.get("degraded")):
        return (
            "diarization_degraded",
            "Diarization ran in degraded mode; manual review required.",
        )

    confidence = float(routing.get("confidence") or 0.0)
    threshold = float(routing.get("threshold") or 0.0)
    return (
        "routing_low_confidence",
        (
            f"Project routing confidence {confidence:.2f} is below "
            f"threshold {threshold:.2f}; manual review required."
        ),
    )


def _resolve_raw_audio_path(recording_id: str, settings: AppSettings) -> Path | None:
    raw_dir = settings.recordings_root / recording_id / "raw"
    candidates = sorted(raw_dir.glob("audio.*"))
    if not candidates:
        return None
    return candidates[0]


def _set_recording_progress_best_effort(
    recording_id: str,
    *,
    stage: str,
    progress: float,
    settings: AppSettings,
) -> None:
    try:
        set_recording_progress(
            recording_id,
            stage=stage,
            progress=progress,
            settings=settings,
        )
    except Exception:
        pass


def _clean_language_value(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _sanitize_audio_for_worker(
    *,
    recording_id: str,
    audio_path: Path,
    settings: AppSettings,
) -> Path:
    derived_dir = settings.recordings_root / recording_id / "derived"
    sanitized_path = derived_dir / "audio_sanitized.wav"
    try:
        working_path = sanitize_audio_for_pipeline(
            audio_path,
            sanitized_path,
        )
    except AudioSanitizeError:
        _logger.exception(
            "audio sanitization failed for recording_id=%s input_path=%s",
            recording_id,
            audio_path,
        )
        raise

    atomic_write_json(
        derived_dir / "audio_sanitize.json",
        {
            "input_path": str(audio_path),
            "output_path": str(working_path),
            "ffmpeg_used": working_path != audio_path,
            "sample_rate": 16000,
            "channels": 1,
            "codec": "pcm_s16le",
        },
    )
    _logger.info("audio sanitized to wav path=%s", working_path)
    return working_path


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
    raw_candidates = row.get("candidates_json")
    if isinstance(raw_candidates, list):
        candidates = raw_candidates
    elif isinstance(raw_candidates, str):
        try:
            parsed = json.loads(raw_candidates or "[]")
        except ValueError:
            return None, []
        if not isinstance(parsed, list):
            return None, []
        candidates = parsed
    else:
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
        self.mode = "fallback"

    async def __call__(self, _audio_path: Path):
        duration = self._duration_sec

        class _Annotation:
            def itertracks(self, yield_label: bool = False):
                if yield_label:
                    yield SimpleNamespace(start=0.0, end=duration), "S1"
                else:  # pragma: no cover - legacy branch
                    yield (SimpleNamespace(start=0.0, end=duration),)

        return _Annotation()


@dataclass(frozen=True)
class _DiarizationRuntimeConfig:
    profile: str
    min_speakers: int | None
    max_speakers: int | None
    dialog_retry_min_duration_seconds: float
    dialog_retry_min_turns: int


class _PyannoteDiariser:
    def __init__(
        self,
        pipeline_model: Any,
        *,
        profile: str = "auto",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        dialog_retry_min_duration_seconds: float = DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
        dialog_retry_min_turns: int = DEFAULT_DIALOG_RETRY_MIN_TURNS,
    ) -> None:
        if pipeline_model is None or not callable(pipeline_model):
            raise TypeError("pipeline_model must be a callable pyannote pipeline.")
        self._pipeline_model = pipeline_model
        self.mode = "pyannote"
        self.profile = str(profile or "auto").strip().lower() or "auto"
        call_kwargs: dict[str, int] = {}
        if min_speakers is not None:
            call_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            call_kwargs["max_speakers"] = max_speakers
        self._base_call_kwargs = call_kwargs
        self.dialog_retry_min_duration_seconds = max(
            dialog_retry_min_duration_seconds,
            0.0,
        )
        self.dialog_retry_min_turns = max(int(dialog_retry_min_turns), 1)
        self.last_run_metadata: dict[str, Any] = {
            "diarization_profile": self.profile,
            "initial_hints": dict(self._base_call_kwargs),
            "effective_hints": dict(self._base_call_kwargs),
            "dialog_retry_used": False,
            "speaker_count_before_retry": None,
            "speaker_count_after_retry": None,
        }

    async def _call_pipeline(
        self,
        audio_path: Path,
        *,
        call_kwargs: dict[str, int],
    ):
        audio_text = str(audio_path)

        def _is_signature_mismatch(exc: TypeError) -> bool:
            message = str(exc).lower()
            signature_markers = (
                "missing required",
                "unexpected keyword",
                "positional argument",
                "got multiple values",
                "takes",
            )
            if any(marker in message for marker in signature_markers):
                return True
            expects_path_like = (
                "expected str" in message
                or "expected bytes" in message
                or "pathlike" in message
                or "os.pathlike" in message
            )
            return expects_path_like and "dict" in message

        def _run_sync():
            try:
                return call_with_supported_kwargs(
                    self._pipeline_model,
                    {"audio": audio_text},
                    **call_kwargs,
                )
            except TypeError as exc:
                if not _is_signature_mismatch(exc):
                    raise
                return call_with_supported_kwargs(
                    self._pipeline_model,
                    audio_text,
                    **call_kwargs,
                )

        return await asyncio.to_thread(_run_sync)

    async def __call__(self, audio_path: Path):
        annotation = await self._call_pipeline(
            audio_path,
            call_kwargs=self._base_call_kwargs,
        )
        speaker_count = annotation_speaker_count(annotation)
        self.last_run_metadata.update(
            {
                "initial_hints": dict(self._base_call_kwargs),
                "effective_hints": dict(self._base_call_kwargs),
                "dialog_retry_used": False,
                "speaker_count_before_retry": speaker_count,
                "speaker_count_after_retry": speaker_count,
            }
        )
        return annotation

    async def retry_dialog(self, audio_path: Path):
        retry_hints = {"min_speakers": 2, "max_speakers": 2}
        annotation = await self._call_pipeline(audio_path, call_kwargs=retry_hints)
        self.last_run_metadata.update(
            {
                "effective_hints": dict(retry_hints),
                "dialog_retry_used": True,
                "speaker_count_after_retry": annotation_speaker_count(annotation),
            }
        )
        return annotation


def _optional_positive_env_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 1:
        return None
    return value


def _optional_nonnegative_env_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0.0:
        return None
    return value


def _resolve_diarization_speaker_hints(
    *,
    settings: AppSettings | None = None,
) -> _DiarizationRuntimeConfig:
    if settings is None:
        profile = str(os.getenv("LAN_DIARIZATION_PROFILE", "auto")).strip().lower() or "auto"
        min_speakers = _optional_positive_env_int("LAN_DIARIZATION_MIN_SPEAKERS")
        max_speakers = _optional_positive_env_int("LAN_DIARIZATION_MAX_SPEAKERS")
        retry_min_duration_seconds = (
            _optional_nonnegative_env_float(
                "LAN_DIARIZATION_DIALOG_RETRY_MIN_DURATION_SECONDS"
            )
            or DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS
        )
        retry_min_turns = (
            _optional_positive_env_int("LAN_DIARIZATION_DIALOG_RETRY_MIN_TURNS")
            or DEFAULT_DIALOG_RETRY_MIN_TURNS
        )
    else:
        profile = settings.diarization_profile
        min_speakers = settings.diarization_min_speakers
        max_speakers = settings.diarization_max_speakers
        retry_min_duration_seconds = settings.diarization_dialog_retry_min_duration_seconds
        retry_min_turns = settings.diarization_dialog_retry_min_turns

    explicit_min_speakers = min_speakers is not None
    explicit_max_speakers = max_speakers is not None
    if profile not in {"auto", "dialog", "meeting"}:
        profile = "auto"
    default_min_speakers, default_max_speakers = profile_default_speaker_hints(profile)
    if min_speakers is None:
        min_speakers = default_min_speakers
    if max_speakers is None:
        max_speakers = default_max_speakers
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        if explicit_min_speakers and not explicit_max_speakers:
            max_speakers = None
        elif explicit_max_speakers and not explicit_min_speakers:
            min_speakers = None
        else:
            min_speakers = None
            max_speakers = None

    return _DiarizationRuntimeConfig(
        profile=profile,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        dialog_retry_min_duration_seconds=retry_min_duration_seconds,
        dialog_retry_min_turns=retry_min_turns,
    )


def _build_pipeline_settings(settings: AppSettings) -> PipelineSettings:
    return PipelineSettings(
        recordings_root=settings.recordings_root,
        voices_dir=settings.data_root / "voices",
        unknown_dir=settings.recordings_root / "unknown",
        tmp_root=settings.data_root / "tmp",
        llm_model=settings.llm_model,
        llm_max_tokens=settings.llm_max_tokens,
        llm_max_tokens_retry=settings.llm_max_tokens_retry,
        llm_chunk_max_chars=settings.llm_chunk_max_chars,
        llm_chunk_overlap_chars=settings.llm_chunk_overlap_chars,
        llm_chunk_timeout_seconds=settings.llm_chunk_timeout_seconds,
        llm_long_transcript_threshold_chars=settings.llm_long_transcript_threshold_chars,
        llm_merge_max_tokens=settings.llm_merge_max_tokens,
        diarization_profile=settings.diarization_profile,
        diarization_min_speakers=settings.diarization_min_speakers,
        diarization_max_speakers=settings.diarization_max_speakers,
        diarization_dialog_retry_min_duration_seconds=(
            settings.diarization_dialog_retry_min_duration_seconds
        ),
        diarization_dialog_retry_min_turns=settings.diarization_dialog_retry_min_turns,
        diarization_merge_gap_seconds=settings.diarization_merge_gap_seconds,
        diarization_min_turn_seconds=settings.diarization_min_turn_seconds,
        vad_method=settings.vad_method,
    )


def _build_diariser(
    duration_sec: float | None,
    *,
    model_id: str | None = None,
    settings: AppSettings | None = None,
):
    diarization_cfg = _resolve_diarization_speaker_hints(settings=settings)
    try:
        model = load_pyannote_pipeline(model_id=model_id)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing == "pyannote":
            return _FallbackDiariser(duration_sec)
        raise
    return _PyannoteDiariser(
        model,
        profile=diarization_cfg.profile,
        min_speakers=diarization_cfg.min_speakers,
        max_speakers=diarization_cfg.max_speakers,
        dialog_retry_min_duration_seconds=(
            diarization_cfg.dialog_retry_min_duration_seconds
        ),
        dialog_retry_min_turns=diarization_cfg.dialog_retry_min_turns,
    )


def _write_diarization_status_artifact(
    *,
    recording_id: str,
    mode: str,
    reason: str | None,
    settings: AppSettings,
) -> None:
    payload: dict[str, Any] = {
        "mode": mode,
        "degraded": mode != "pyannote",
    }
    if reason:
        payload["reason"] = reason
    path = settings.recordings_root / recording_id / "derived" / "diarization_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_json(path, payload)
    except OSError:
        pass


def _run_precheck_pipeline(
    *,
    recording_id: str,
    settings: AppSettings,
    log_path: Path,
) -> PipelineTerminalState:
    recording = get_recording(recording_id, settings=settings) or {}
    transcript_language_override = _clean_language_value(recording.get("language_override"))
    target_summary_language = _clean_language_value(recording.get("target_summary_language"))
    has_explicit_summary_target = target_summary_language is not None

    audio_path = _resolve_raw_audio_path(recording_id, settings)
    if audio_path is None:
        _append_step_log(log_path, "precheck skipped: raw audio not found")
        return PipelineTerminalState(
            status=RECORDING_STATUS_QUARANTINE,
            quarantine_reason="raw_audio_missing",
        )

    _set_recording_progress_best_effort(
        recording_id,
        stage="precheck",
        progress=0.01,
        settings=settings,
    )

    audio_path = _sanitize_audio_for_worker(
        recording_id=recording_id,
        audio_path=audio_path,
        settings=settings,
    )
    _set_recording_progress_best_effort(
        recording_id,
        stage="precheck",
        progress=0.05,
        settings=settings,
    )

    pipeline_settings = _build_pipeline_settings(settings)
    precheck = run_precheck(audio_path, pipeline_settings)
    if precheck.duration_sec is not None:
        set_recording_duration(
            recording_id,
            precheck.duration_sec,
            settings=settings,
        )
    _append_step_log(
        log_path,
        (
            "precheck "
            f"duration_sec={precheck.duration_sec} "
            f"speech_ratio={precheck.speech_ratio}"
        ),
    )
    diarization_mode = "pyannote"
    diarization_reason: str | None = None
    if precheck.quarantine_reason:
        diariser = _FallbackDiariser(precheck.duration_sec)
        diarization_mode = "fallback"
        diarization_reason = "precheck_quarantine"
        _append_step_log(
            log_path,
            "diariser mode=fallback reason=precheck_quarantine",
        )
    else:
        try:
            diariser = _build_diariser(
                precheck.duration_sec,
                model_id=settings.diarization_model_id,
                settings=settings,
            )
            if isinstance(diariser, _FallbackDiariser):
                diarization_mode = "fallback"
                diarization_reason = "pyannote_unavailable"
                _append_step_log(
                    log_path,
                    "diariser mode=fallback reason=pyannote_unavailable",
                )
            else:
                _append_step_log(log_path, "diariser mode=pyannote")
        except Exception as exc:
            _append_step_log(
                log_path,
                (
                    "diariser init failed, falling back: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            diariser = _FallbackDiariser(precheck.duration_sec)
            diarization_mode = "fallback"
            diarization_reason = f"{type(exc).__name__}: {exc}"
            _append_step_log(
                log_path,
                "diariser mode=fallback reason=init_failed",
            )
    _write_diarization_status_artifact(
        recording_id=recording_id,
        mode=diarization_mode,
        reason=diarization_reason,
        settings=settings,
    )
    calendar_title, calendar_attendees = _load_calendar_summary_context(
        recording_id,
        settings,
    )

    def _progress_callback(stage: str, progress: float) -> None:
        _set_recording_progress_best_effort(
            recording_id,
            stage=stage,
            progress=progress,
            settings=settings,
        )

    def _step_log_callback(message: str) -> None:
        _append_step_log(log_path, message)

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
            progress_callback=_progress_callback,
            step_log_callback=_step_log_callback,
        )
    )
    metrics_payload = refresh_recording_metrics(
        recording_id,
        settings=settings,
    )
    _append_step_log(
        log_path,
        (
            "metrics refreshed "
            f"participants={len(metrics_payload.get('participants', []))} "
            f"interruptions={metrics_payload.get('meeting', {}).get('total_interruptions', 0)}"
        ),
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
        return PipelineTerminalState(
            status=RECORDING_STATUS_QUARANTINE,
            quarantine_reason=precheck.quarantine_reason,
        )
    routing = refresh_recording_routing(
        recording_id,
        settings=settings,
        apply_workflow=True,
    )
    _append_step_log(
        log_path,
        (
            "routing "
            f"suggested_project_id={routing.get('suggested_project_id')} "
            f"confidence={float(routing.get('confidence') or 0.0):.2f} "
            f"threshold={float(routing.get('threshold') or 0.0):.2f} "
            f"auto_selected={bool(routing.get('auto_selected'))} "
            f"status_after={routing.get('status_after_routing')}"
        ),
    )
    if routing.get("status_after_routing") == RECORDING_STATUS_NEEDS_REVIEW:
        review_reason_code, review_reason_text = _review_reason_from_routing(
            recording_id=recording_id,
            settings=settings,
            routing=routing,
        )
        return PipelineTerminalState(
            status=RECORDING_STATUS_NEEDS_REVIEW,
            review_reason_code=review_reason_code,
            review_reason_text=review_reason_text,
        )
    return PipelineTerminalState(status=RECORDING_STATUS_READY)


def process_job(job_id: str, recording_id: str, job_type: str) -> dict[str, str]:
    """Execute a queue job and persist lifecycle state transitions."""

    if job_type not in JOB_TYPES:
        raise ValueError(f"Unsupported job type: {job_type}")

    settings = AppSettings()
    init_db(settings)
    log_path = _step_log_path(recording_id, job_type, settings)

    if job_type != JOB_TYPE_PRECHECK:
        recording_before = get_recording(recording_id, settings=settings) or {}
        previous_status = str(recording_before.get("status") or "").strip()
        if not _start_job_or_ignore_stale_execution(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
            settings=settings,
            log_path=log_path,
        ):
            return _ignored_result(job_id, recording_id, job_type)
        unsupported_error = (
            f"unsupported legacy job type under single-job pipeline: {job_type}"
        )
        try:
            _append_step_log(
                log_path,
                (
                    "unsupported job type under single-job mode; "
                    "requeue precheck instead "
                    f"(job={job_id} type={job_type})"
                ),
            )
        except OSError:
            pass
        if not fail_job(job_id, unsupported_error, settings=settings):
            raise ValueError(f"Job not found: {job_id}")
        if previous_status == RECORDING_STATUS_QUEUED:
            if _has_queued_precheck_job(
                recording_id,
                settings=settings,
                exclude_job_id=job_id,
            ):
                try:
                    _append_step_log(
                        log_path,
                        "kept recording status Queued because a precheck job is pending",
                    )
                except OSError:
                    pass
            else:
                restored_status, restored_quarantine_reason = _restore_status_from_precheck_log(
                    recording_id,
                    settings,
                )
                if restored_status is None:
                    restored_status = RECORDING_STATUS_FAILED
                set_recording_status(
                    recording_id,
                    restored_status,
                    settings=settings,
                    quarantine_reason=restored_quarantine_reason,
                )
                try:
                    _append_step_log(
                        log_path,
                        (
                            "restored recording status after unsupported legacy job: "
                            f"{restored_status}"
                        ),
                    )
                except OSError:
                    pass
        return _ignored_result(job_id, recording_id, job_type)

    retry_policy = _retry_policy(job_type)
    max_attempts = max(1, min(retry_policy.max_attempts, settings.max_job_attempts))

    while True:
        try:
            if not _start_job_or_ignore_stale_execution(
                job_id=job_id,
                recording_id=recording_id,
                job_type=job_type,
                settings=settings,
                log_path=log_path,
            ):
                return _ignored_result(job_id, recording_id, job_type)
            attempt = _job_attempt(job_id, settings)
            if attempt > max_attempts:
                raise RuntimeError(_MAX_ATTEMPTS_ERROR)
            if not set_recording_status(
                recording_id,
                RECORDING_STATUS_PROCESSING,
                settings=settings,
            ):
                raise ValueError(f"Recording not found: {recording_id}")
            _append_step_log(log_path, f"started job={job_id} type={job_type}")

            terminal_state = _run_precheck_pipeline(
                recording_id=recording_id,
                settings=settings,
                log_path=log_path,
            )

            current_job_status = _job_status(job_id, settings)
            if current_job_status != JOB_STATUS_STARTED:
                _log_stale_inflight_execution(
                    job_id=job_id,
                    job_type=job_type,
                    log_path=log_path,
                    detail=f"status={current_job_status or 'missing'}",
                )
                return _ignored_result(job_id, recording_id, job_type)

            if not set_recording_status_if_current_in_and_job_started(
                recording_id,
                terminal_state.status,
                job_id=job_id,
                current_statuses=(RECORDING_STATUS_PROCESSING,),
                settings=settings,
                quarantine_reason=terminal_state.quarantine_reason,
                review_reason_code=terminal_state.review_reason_code,
                review_reason_text=terminal_state.review_reason_text,
            ):
                current_job_status = _job_status(job_id, settings)
                if current_job_status != JOB_STATUS_STARTED:
                    _log_stale_inflight_execution(
                        job_id=job_id,
                        job_type=job_type,
                        log_path=log_path,
                        detail=f"status={current_job_status or 'missing'}",
                    )
                    return _ignored_result(job_id, recording_id, job_type)
                recording_row = get_recording(recording_id, settings=settings) or {}
                recording_status = str(recording_row.get("status") or "").strip()
                if recording_status and recording_status != RECORDING_STATUS_PROCESSING:
                    try:
                        fail_job_if_started(
                            job_id,
                            (
                                "stale in-flight execution ignored: "
                                f"recording status changed to {recording_status}"
                            ),
                            settings=settings,
                        )
                    except Exception:
                        pass
                    _log_stale_inflight_execution(
                        job_id=job_id,
                        job_type=job_type,
                        log_path=log_path,
                        detail=f"recording_status={recording_status}",
                    )
                    return _ignored_result(job_id, recording_id, job_type)
                raise ValueError(f"Recording not found: {recording_id}")
            if not finish_job_if_started(job_id, settings=settings):
                job_status = _job_status(job_id, settings)
                if job_status and job_status != JOB_STATUS_STARTED:
                    _log_stale_inflight_execution(
                        job_id=job_id,
                        job_type=job_type,
                        log_path=log_path,
                        detail=f"status={job_status}",
                    )
                    return _ignored_result(job_id, recording_id, job_type)
                raise ValueError(f"Job not found: {job_id}")
            clear_recording_progress(recording_id, settings=settings)
            _append_step_log(
                log_path,
                (
                    f"finished job={job_id} type={job_type} "
                    f"recording_status={terminal_state.status}"
                ),
            )
            break
        except Exception as exc:
            clear_recording_progress(recording_id, settings=settings)
            current_job_status = _job_status(job_id, settings)
            if current_job_status and current_job_status != JOB_STATUS_STARTED:
                _log_stale_inflight_execution(
                    job_id=job_id,
                    job_type=job_type,
                    log_path=log_path,
                    detail=f"status={current_job_status}",
                )
                return _ignored_result(job_id, recording_id, job_type)
            attempt = _job_attempt(job_id, settings)
            if attempt >= max_attempts:
                _record_max_attempts_exceeded(
                    job_id=job_id,
                    job_type=job_type,
                    recording_id=recording_id,
                    settings=settings,
                    log_path=log_path,
                    exc=exc,
                )
                raise
            if (
                _is_retryable_exception(exc)
                and attempt > 0
                and attempt < max_attempts
            ):
                delay_seconds = _retry_delay_seconds(retry_policy, attempt)
                if _record_retry(
                    job_id=job_id,
                    job_type=job_type,
                    recording_id=recording_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_seconds=delay_seconds,
                    settings=settings,
                    log_path=log_path,
                    exc=exc,
                ):
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                current_job_status = _job_status(job_id, settings)
                if current_job_status and current_job_status != JOB_STATUS_STARTED:
                    _log_stale_inflight_execution(
                        job_id=job_id,
                        job_type=job_type,
                        log_path=log_path,
                        detail=f"status={current_job_status}",
                    )
                    return _ignored_result(job_id, recording_id, job_type)
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
