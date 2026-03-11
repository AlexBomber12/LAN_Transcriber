from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any

from lan_transcriber import normalizer
from lan_transcriber.aliases import load_aliases as load_speaker_aliases
from lan_transcriber.aliases import save_aliases as save_speaker_aliases
from lan_transcriber.artifacts import (
    atomic_write_json,
    atomic_write_text,
    build_recording_artifacts,
)
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
from lan_transcriber.pipeline_steps import orchestrator as pipeline_orchestrator
from lan_transcriber.pipeline import PrecheckResult, Settings as PipelineSettings
from lan_transcriber.pipeline import run_pipeline, run_precheck
from lan_transcriber.gpu_policy import (
    collect_cuda_runtime_facts,
    is_gpu_oom_error,
    resolve_scheduler_decision,
)
from lan_transcriber.pipeline_steps.diarization_quality import (
    AUTO_PROFILE_DEFAULT_INITIAL_PROFILE,
    DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
    DEFAULT_DIALOG_RETRY_MIN_TURNS,
    SpeakerTurnSmoothingResult,
    annotation_speaker_count,
    profile_default_speaker_hints,
    smooth_speaker_turns,
)
from lan_transcriber.pipeline_steps.language import (
    analyse_languages,
    resolve_target_summary_language,
)
from lan_transcriber.pipeline_steps.multilingual_asr import run_language_aware_asr
from lan_transcriber.pipeline_steps import summary_builder as pipeline_summary_builder
from lan_transcriber.pipeline_steps.snippets import (
    SnippetExportRequest,
    export_speaker_snippets,
    write_empty_snippets_manifest,
)
from lan_transcriber.pipeline_steps.speaker_turns import (
    _diarization_segments,
    build_speaker_turns,
    normalise_asr_segments,
)
from lan_transcriber.pipeline_steps.summary_builder import (
    build_structured_summary_prompts,
    build_summary_payload,
)
from lan_transcriber.utils import normalise_language_code, safe_float

from .asr_glossary import build_recording_asr_glossary
from .calendar.matching import calendar_summary_context, refresh_recording_calendar_match
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
    clear_recording_pipeline_stages,
    fail_job,
    fail_job_if_started,
    finish_job_if_started,
    get_job,
    get_recording,
    init_db,
    list_jobs,
    list_recording_pipeline_stages,
    mark_recording_pipeline_stage_completed,
    mark_recording_pipeline_stage_failed,
    mark_recording_pipeline_stage_skipped,
    mark_recording_pipeline_stage_started,
    requeue_job_if_started,
    set_recording_progress,
    set_recording_duration,
    set_recording_language_settings,
    set_recording_status,
    set_recording_status_if_current_in_and_job_started,
    start_job,
)
from .diarization_loader import load_pyannote_pipeline
from .pipeline_stages import (
    PIPELINE_STAGE_DEFINITIONS,
    PIPELINE_STAGE_DONE_STATUSES,
    PIPELINE_STAGE_STATUS_SKIPPED,
    validate_stage_artifacts,
)
from .routing import refresh_recording_routing

_logger = logging.getLogger(__name__)
_ORIGINAL_RUN_PIPELINE = run_pipeline


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
    if is_gpu_oom_error(exc):
        return False
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
    review_reason_code, review_reason_text = _review_reason_from_exception(exc)
    terminal_status = (
        RECORDING_STATUS_NEEDS_REVIEW
        if review_reason_code == "gpu_oom"
        else RECORDING_STATUS_FAILED
    )
    try:
        fail_job(job_id, error, settings=settings)
    except Exception:
        pass
    try:
        set_recording_status(
            recording_id,
            terminal_status,
            settings=settings,
            review_reason_code=(
                review_reason_code if terminal_status == RECORDING_STATUS_NEEDS_REVIEW else None
            ),
            review_reason_text=(
                review_reason_text if terminal_status == RECORDING_STATUS_NEEDS_REVIEW else None
            ),
        )
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


def _load_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return list(payload)


def _review_reason_from_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc).strip()
    lowered = message.lower()
    if is_gpu_oom_error(exc):
        return (
            "gpu_oom",
            (
                "The worker ran out of GPU memory while loading or running a heavy "
                "model; manual review required."
            ),
        )
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
    transcript_payload = _load_json_dict(derived_dir / "transcript.json")
    summary_payload = _load_json_dict(derived_dir / "summary.json")
    diarization_payload = _load_json_dict(derived_dir / "diarization_metadata.json")
    transcript_review = transcript_payload.get("review")
    if isinstance(transcript_review, dict) and bool(transcript_review.get("required")):
        return (
            str(transcript_review.get("reason_code") or "").strip()
            or "transcript_review_required",
            str(transcript_review.get("reason_text") or "").strip()
            or "Multilingual transcript review is required.",
        )

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


def _set_recording_duration_best_effort(
    recording_id: str,
    *,
    duration_sec: float | None,
    settings: AppSettings,
) -> None:
    try:
        set_recording_duration(
            recording_id,
            duration_sec,
            settings=settings,
        )
    except Exception:
        _logger.warning(
            "Failed to persist duration for recording %s",
            recording_id,
            exc_info=True,
        )


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
    return calendar_summary_context(
        recording_id,
        settings=settings,
    )


@dataclass(frozen=True)
class _PipelineArtifacts:
    recording_id: str
    derived_dir: Path
    sanitized_audio_path: Path
    audio_sanitize_json_path: Path
    precheck_json_path: Path
    calendar_refresh_json_path: Path
    asr_segments_json_path: Path
    asr_info_json_path: Path
    asr_execution_json_path: Path
    diarization_segments_json_path: Path
    diarization_runtime_json_path: Path
    language_analysis_json_path: Path
    routing_json_path: Path
    recording_artifacts: Any


@dataclass
class _PipelineExecutionContext:
    recording_id: str
    settings: AppSettings
    log_path: Path
    artifacts: _PipelineArtifacts
    pipeline_settings: PipelineSettings
    recording: dict[str, Any]
    transcript_language_override: str | None
    target_summary_language: str | None
    has_explicit_summary_target: bool
    raw_audio_path: Path | None = None
    precheck_result: Any | None = None
    calendar_title: str | None = None
    calendar_attendees: list[str] = field(default_factory=list)
    asr_glossary: dict[str, Any] | None = None
    asr_segments: list[dict[str, Any]] = field(default_factory=list)
    asr_info: dict[str, Any] = field(default_factory=dict)
    asr_execution: dict[str, Any] = field(default_factory=dict)
    language_payload: dict[str, Any] = field(default_factory=dict)
    diarization_segments: list[dict[str, Any]] = field(default_factory=list)
    diarization_runtime: dict[str, Any] = field(default_factory=dict)
    speaker_turns: list[dict[str, Any]] = field(default_factory=list)
    summary_payload: dict[str, Any] | None = None
    clean_text: str = ""
    friendly: int = 0
    routing_payload: dict[str, Any] | None = None


def _build_pipeline_artifacts(
    recording_id: str,
    *,
    settings: AppSettings,
) -> _PipelineArtifacts:
    raw_audio_path = _resolve_raw_audio_path(recording_id, settings)
    recording_artifacts = build_recording_artifacts(
        settings.recordings_root,
        recording_id,
        raw_audio_path.suffix if raw_audio_path is not None else ".bin",
    )
    derived_dir = recording_artifacts.summary_json_path.parent
    return _PipelineArtifacts(
        recording_id=recording_id,
        derived_dir=derived_dir,
        sanitized_audio_path=derived_dir / "audio_sanitized.wav",
        audio_sanitize_json_path=derived_dir / "audio_sanitize.json",
        precheck_json_path=derived_dir / "precheck.json",
        calendar_refresh_json_path=derived_dir / "calendar_refresh.json",
        asr_segments_json_path=derived_dir / "asr_segments.json",
        asr_info_json_path=derived_dir / "asr_info.json",
        asr_execution_json_path=derived_dir / "asr_execution.json",
        diarization_segments_json_path=derived_dir / "diarization_segments.json",
        diarization_runtime_json_path=derived_dir / "diarization_runtime.json",
        language_analysis_json_path=derived_dir / "language_analysis.json",
        routing_json_path=derived_dir / "routing.json",
        recording_artifacts=recording_artifacts,
    )


def _new_pipeline_context(
    *,
    recording_id: str,
    settings: AppSettings,
    log_path: Path,
) -> _PipelineExecutionContext:
    recording = get_recording(recording_id, settings=settings) or {}
    transcript_language_override = _clean_language_value(recording.get("language_override"))
    target_summary_language = _clean_language_value(recording.get("target_summary_language"))
    return _PipelineExecutionContext(
        recording_id=recording_id,
        settings=settings,
        log_path=log_path,
        artifacts=_build_pipeline_artifacts(recording_id, settings=settings),
        pipeline_settings=_build_pipeline_settings(settings),
        recording=recording,
        transcript_language_override=transcript_language_override,
        target_summary_language=target_summary_language,
        has_explicit_summary_target=target_summary_language is not None,
    )


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
    initial_profile: str
    min_speakers: int | None
    max_speakers: int | None
    auto_profile_enabled: bool
    override_reason: str | None
    dialog_retry_min_duration_seconds: float
    dialog_retry_min_turns: int


class _PyannoteDiariser:
    def __init__(
        self,
        pipeline_model: Any | None = None,
        *,
        pipeline_loader: Any | None = None,
        requested_device: str | None = None,
        fallback_duration_sec: float | None = None,
        profile: str = "auto",
        initial_profile: str | None = None,
        auto_profile_enabled: bool = True,
        override_reason: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        dialog_retry_min_duration_seconds: float = DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
        dialog_retry_min_turns: int = DEFAULT_DIALOG_RETRY_MIN_TURNS,
    ) -> None:
        if pipeline_model is not None and not callable(pipeline_model):
            raise TypeError("pipeline_model must be a callable pyannote pipeline.")
        if pipeline_model is None and pipeline_loader is None:
            raise TypeError("pipeline_loader is required when pipeline_model is not provided.")
        self._pipeline_model = pipeline_model
        self._pipeline_loader = pipeline_loader
        self._fallback_diariser: _FallbackDiariser | None = None
        self._fallback_duration_sec = fallback_duration_sec
        normalized_requested_device = str(requested_device or "").strip().lower()
        if normalized_requested_device == "gpu":
            normalized_requested_device = "cuda"
        self._forced_gpu_device_requested = normalized_requested_device.startswith("cuda")
        self.mode = "pyannote"
        self.profile = str(profile or "auto").strip().lower() or "auto"
        self.initial_profile = (
            str(
                initial_profile
                or (self.profile if self.profile in {"dialog", "meeting"} else "")
                or AUTO_PROFILE_DEFAULT_INITIAL_PROFILE
            ).strip().lower()
            or AUTO_PROFILE_DEFAULT_INITIAL_PROFILE
        )
        if self.initial_profile not in {"dialog", "meeting"}:
            self.initial_profile = AUTO_PROFILE_DEFAULT_INITIAL_PROFILE
        self.auto_profile_enabled = bool(auto_profile_enabled and self.profile == "auto")
        self.override_reason = str(override_reason or "").strip() or None
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
            "requested_profile": self.profile,
            "diarization_profile": self.profile,
            "initial_profile": self.initial_profile,
            "selected_profile": self.initial_profile if not self.auto_profile_enabled else None,
            "auto_profile_enabled": self.auto_profile_enabled,
            "override_reason": self.override_reason,
            "initial_hints": dict(self._base_call_kwargs),
            "retry_hints": None,
            "effective_hints": dict(self._base_call_kwargs),
            "profile_selection": None,
            "dialog_retry_used": False,
            "speaker_count_before_retry": None,
            "speaker_count_after_retry": None,
            "effective_device": None,
        }

    def _ensure_pipeline_model(self) -> Any:
        if self._pipeline_model is None:
            if self._pipeline_loader is None:
                raise TypeError("pipeline_loader must be provided for lazy diarization.")
            pipeline_model = self._pipeline_loader()
            if pipeline_model is None or not callable(pipeline_model):
                raise TypeError("pipeline_model must be a callable pyannote pipeline.")
            self._pipeline_model = pipeline_model
            self.last_run_metadata["effective_device"] = str(
                getattr(pipeline_model, "_lan_effective_device", "cpu")
            )
        return self._pipeline_model

    async def _call_pipeline(
        self,
        audio_path: Path,
        *,
        call_kwargs: dict[str, int],
    ):
        audio_text = str(audio_path)
        if self._fallback_diariser is not None:
            return await self._fallback_diariser(audio_path)
        try:
            pipeline_model = self._ensure_pipeline_model()
        except Exception as exc:
            message = str(exc)
            if isinstance(exc, TypeError) and (
                message.startswith("pipeline_loader must be provided")
                or message.startswith("pipeline_model must be a callable")
            ):
                raise
            if (
                isinstance(exc, ValueError)
                and message.startswith("Device must be one of auto, cpu, cuda, or cuda:<index>.")
            ):
                raise
            if (
                self._forced_gpu_device_requested
                and isinstance(exc, RuntimeError)
                and (
                    message.startswith("Requested diarization device ")
                    or message.startswith("Failed to move pyannote diarization pipeline to ")
                )
            ):
                raise
            self.mode = "fallback"
            self.last_run_metadata["effective_device"] = "cpu"
            _logger.warning(
                "pyannote diarization load failed; using fallback diariser: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._fallback_diariser = _FallbackDiariser(self._fallback_duration_sec)
            return await self._fallback_diariser(audio_path)

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
                    pipeline_model,
                    {"audio": audio_text},
                    **call_kwargs,
                )
            except TypeError as exc:
                if not _is_signature_mismatch(exc):
                    raise
                return call_with_supported_kwargs(
                    pipeline_model,
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
                "retry_hints": None,
                "effective_hints": dict(self._base_call_kwargs),
                "selected_profile": self.initial_profile if not self.auto_profile_enabled else None,
                "profile_selection": None,
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
                "retry_hints": dict(retry_hints),
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


def _profile_from_runtime_hints(
    requested_profile: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> str:
    normalized = str(requested_profile or "auto").strip().lower() or "auto"
    if normalized == "dialog":
        return "dialog"
    if normalized == "meeting":
        return "meeting"
    if max_speakers == 2 and (min_speakers is None or min_speakers <= 2):
        return "dialog"
    return AUTO_PROFILE_DEFAULT_INITIAL_PROFILE


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
    default_profile = profile if profile != "auto" else AUTO_PROFILE_DEFAULT_INITIAL_PROFILE
    default_min_speakers, default_max_speakers = profile_default_speaker_hints(
        default_profile
    )
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

    auto_profile_enabled = profile == "auto" and not explicit_min_speakers and not explicit_max_speakers
    override_reason: str | None = None
    if profile == "dialog":
        override_reason = "profile_forced_dialog"
    elif profile == "meeting":
        override_reason = "profile_forced_meeting"
    elif not auto_profile_enabled:
        override_reason = "explicit_speaker_hints"

    return _DiarizationRuntimeConfig(
        profile=profile,
        initial_profile=_profile_from_runtime_hints(profile, min_speakers, max_speakers),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        auto_profile_enabled=auto_profile_enabled,
        override_reason=override_reason,
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
        asr_device=getattr(settings, "asr_device", "auto"),
        diarization_device=getattr(settings, "diarization_device", "auto"),
        gpu_scheduler_mode=getattr(settings, "gpu_scheduler_mode", "auto"),
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
    settings: Any | None = None,
):
    try:
        from pyannote.audio import Pipeline as _Pipeline  # type: ignore
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing == "pyannote":
            return _FallbackDiariser(duration_sec)
        raise
    del _Pipeline

    diarization_cfg = _resolve_diarization_speaker_hints(settings=settings)
    return _PyannoteDiariser(
        pipeline_loader=lambda: load_pyannote_pipeline(
            model_id=model_id,
            device=getattr(settings, "diarization_device", None),
            scheduler_mode=getattr(settings, "gpu_scheduler_mode", None),
        ),
        requested_device=getattr(settings, "diarization_device", None),
        fallback_duration_sec=duration_sec,
        profile=diarization_cfg.profile,
        initial_profile=diarization_cfg.initial_profile,
        auto_profile_enabled=diarization_cfg.auto_profile_enabled,
        override_reason=diarization_cfg.override_reason,
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


def _log_gpu_execution_policy(
    *,
    pipeline_settings: PipelineSettings,
    diariser: Any,
    log_path: Path,
) -> None:
    diariser_mode = str(getattr(diariser, "mode", "") or "").strip().lower()
    scheduler_plan = resolve_scheduler_decision(
        pipeline_settings.gpu_scheduler_mode,
        asr_device=pipeline_settings.asr_device,
        diarization_device=pipeline_settings.diarization_device,
        diarization_is_heavy=diariser_mode == "pyannote",
        cuda_facts=collect_cuda_runtime_facts(),
    )
    facts = scheduler_plan.cuda_facts
    message = (
        "gpu policy "
        f"asr_device={scheduler_plan.asr_device} "
        f"diarization_device={scheduler_plan.diarization_device} "
        f"scheduler_mode={scheduler_plan.effective_mode} "
        f"requested_mode={scheduler_plan.requested_mode} "
        f"reason={scheduler_plan.reason} "
        f"cuda_available={facts.is_available} "
        f"device_count={facts.device_count} "
        f"visible_devices={facts.visible_devices or 'unset'} "
        f"torch_cuda={facts.torch_cuda_version or 'none'}"
    )
    _logger.info(message)
    _append_step_log(log_path, message)


@dataclass(frozen=True)
class _StageResult:
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _stage_rows_by_name(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("stage_name") or ""): row
        for row in list_recording_pipeline_stages(recording_id, settings=settings)
    }


def _stage_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    payload = row.get("metadata_json")
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _log_stage_started(log_path: Path, stage_name: str, *, resumed: bool = False) -> None:
    verb = "resumed" if resumed else "started"
    _append_step_log(log_path, f"stage {verb}: {stage_name}")


def _log_stage_completed(
    log_path: Path,
    stage_name: str,
    *,
    duration_ms: int | None,
) -> None:
    if duration_ms is None:
        _append_step_log(log_path, f"stage completed: {stage_name}")
        return
    _append_step_log(
        log_path,
        f"stage completed: {stage_name} elapsed={duration_ms / 1000:.3f}s",
    )


def _log_stage_skipped(
    log_path: Path,
    stage_name: str,
    *,
    reason: str,
) -> None:
    _append_step_log(log_path, f"stage skipped: {stage_name} reason={reason}")


def _log_stage_invalidated(
    log_path: Path,
    stage_name: str,
    *,
    reason: str,
) -> None:
    _append_step_log(log_path, f"stage invalidated: {stage_name} {reason}, rerunning")


def _recording_pipeline_progress_stage(stage_name: str) -> str:
    return stage_name


def _build_skip_result(reason: str, **metadata: Any) -> _StageResult:
    payload = {"skip_reason": reason}
    payload.update(metadata)
    return _StageResult(status=PIPELINE_STAGE_STATUS_SKIPPED, metadata=payload)


def _load_precheck_artifact(ctx: _PipelineExecutionContext) -> PrecheckResult | None:
    payload = _load_json_dict(ctx.artifacts.precheck_json_path)
    if not payload:
        return None
    return PrecheckResult(
        duration_sec=safe_float(payload.get("duration_sec"), default=0.0)
        if payload.get("duration_sec") is not None
        else None,
        speech_ratio=safe_float(payload.get("speech_ratio"), default=0.0)
        if payload.get("speech_ratio") is not None
        else None,
        quarantine_reason=_clean_language_value(payload.get("quarantine_reason")),
    )


def _load_language_analysis_artifact(ctx: _PipelineExecutionContext) -> dict[str, Any]:
    return _load_json_dict(ctx.artifacts.language_analysis_json_path)


def _load_summary_payload(ctx: _PipelineExecutionContext) -> dict[str, Any]:
    return _load_json_dict(ctx.artifacts.recording_artifacts.summary_json_path)


def _load_diarization_runtime(ctx: _PipelineExecutionContext) -> dict[str, Any]:
    return _load_json_dict(ctx.artifacts.diarization_runtime_json_path)


def _load_asr_execution(ctx: _PipelineExecutionContext) -> dict[str, Any]:
    return _load_json_dict(ctx.artifacts.asr_execution_json_path)


def _working_audio_path(ctx: _PipelineExecutionContext) -> Path | None:
    sanitize_payload = _load_json_dict(ctx.artifacts.audio_sanitize_json_path)
    configured_output = str(sanitize_payload.get("output_path") or "").strip()
    if configured_output:
        output_path = Path(configured_output)
        if output_path.exists():
            return output_path
    if ctx.artifacts.sanitized_audio_path.exists():
        return ctx.artifacts.sanitized_audio_path
    if ctx.raw_audio_path is not None and ctx.raw_audio_path.exists():
        return ctx.raw_audio_path
    resolved_raw = _resolve_raw_audio_path(ctx.recording_id, ctx.settings)
    if resolved_raw is not None and resolved_raw.exists():
        ctx.raw_audio_path = resolved_raw
        return resolved_raw
    return None


def _build_diarization_metadata_payload(
    *,
    runtime: dict[str, Any],
    cfg: PipelineSettings,
    smoothing_result: SpeakerTurnSmoothingResult,
) -> dict[str, Any]:
    diariser_mode = str(runtime.get("mode") or "unknown").strip().lower() or "unknown"
    effective_hints = runtime.get("effective_hints")
    if not isinstance(effective_hints, dict):
        effective_hints = {}
    initial_hints = runtime.get("initial_hints")
    if not isinstance(initial_hints, dict):
        initial_hints = {}
    profile_selection = runtime.get("profile_selection")
    if not isinstance(profile_selection, dict):
        profile_selection = {}
    selected_profile = str(
        profile_selection.get("selected_profile")
        or runtime.get("selected_profile")
        or runtime.get("initial_profile")
        or runtime.get("diarization_profile")
        or cfg.diarization_profile
    )
    initial_metrics = profile_selection.get("initial_metrics")
    if not isinstance(initial_metrics, dict):
        initial_metrics = {}
    payload: dict[str, Any] = {
        "version": 1,
        "mode": diariser_mode,
        "degraded": bool(runtime.get("used_dummy_fallback")) or diariser_mode not in {"pyannote", "unknown"},
        "diarization_profile": str(runtime.get("diarization_profile") or cfg.diarization_profile),
        "requested_profile": str(
            runtime.get("requested_profile")
            or runtime.get("diarization_profile")
            or cfg.diarization_profile
        ),
        "effective_device": str(runtime.get("effective_device") or cfg.diarization_device),
        "scheduler_mode": str(runtime.get("scheduler_mode") or cfg.gpu_scheduler_mode),
        "scheduler_reason": runtime.get("scheduler_reason"),
        "initial_profile": str(runtime.get("initial_profile") or cfg.diarization_profile),
        "selected_profile": selected_profile,
        "selected_result": str(profile_selection.get("selected_result") or "initial_pass"),
        "auto_profile_enabled": bool(runtime.get("auto_profile_enabled", False)),
        "profile_override_reason": runtime.get("override_reason"),
        "hints_applied": effective_hints,
        "dialog_retry_attempted": bool(
            profile_selection.get(
                "dialog_retry_attempted",
                runtime.get("dialog_retry_used", False),
            )
        ),
        "dialog_retry_used": bool(runtime.get("dialog_retry_used", False)),
        "speaker_count_before_retry": runtime.get("speaker_count_before_retry"),
        "speaker_count_after_retry": runtime.get("speaker_count_after_retry"),
        "initial_speaker_count": initial_metrics.get(
            "speaker_count",
            runtime.get("speaker_count_before_retry"),
        ),
        "initial_top_two_coverage": initial_metrics.get("top_two_coverage"),
        "used_dummy_fallback": bool(runtime.get("used_dummy_fallback", False)),
        "smoothing_applied": bool(diariser_mode == "pyannote" and not runtime.get("used_dummy_fallback")),
        "merge_gap_seconds": cfg.diarization_merge_gap_seconds,
        "min_turn_seconds": cfg.diarization_min_turn_seconds,
        "speaker_count_before_smoothing": smoothing_result.speaker_count_before,
        "speaker_count_after_smoothing": smoothing_result.speaker_count_after,
        "turn_count_before_smoothing": smoothing_result.turn_count_before,
        "turn_count_after_smoothing": smoothing_result.turn_count_after,
        "adjacent_merges": smoothing_result.adjacent_merges,
        "micro_turn_absorptions": smoothing_result.micro_turn_absorptions,
    }
    if initial_hints != effective_hints:
        payload["initial_hints"] = initial_hints
    if profile_selection:
        payload["profile_selection"] = profile_selection
    return payload


def _stage_sanitize_audio(ctx: _PipelineExecutionContext) -> _StageResult:
    raw_audio_path = _resolve_raw_audio_path(ctx.recording_id, ctx.settings)
    ctx.raw_audio_path = raw_audio_path
    if raw_audio_path is None:
        return _build_skip_result("raw_audio_missing")
    _set_recording_progress_best_effort(
        ctx.recording_id,
        stage="precheck",
        progress=0.01,
        settings=ctx.settings,
    )
    working_path = _sanitize_audio_for_worker(
        recording_id=ctx.recording_id,
        audio_path=raw_audio_path,
        settings=ctx.settings,
    )
    return _StageResult(
        status="completed",
        metadata={
            "raw_audio_path": str(raw_audio_path),
            "sanitized_path": str(working_path),
        },
    )


def _stage_precheck(ctx: _PipelineExecutionContext) -> _StageResult:
    audio_path = _working_audio_path(ctx)
    if audio_path is None:
        ctx.precheck_result = PrecheckResult(
            duration_sec=None,
            speech_ratio=None,
            quarantine_reason="raw_audio_missing",
        )
    else:
        ctx.precheck_result = run_precheck(audio_path, ctx.pipeline_settings)
        if ctx.precheck_result.duration_sec is not None:
            _set_recording_duration_best_effort(
                ctx.recording_id,
                duration_sec=ctx.precheck_result.duration_sec,
                settings=ctx.settings,
            )
    atomic_write_json(
        ctx.artifacts.precheck_json_path,
        {
            "duration_sec": ctx.precheck_result.duration_sec,
            "speech_ratio": ctx.precheck_result.speech_ratio,
            "quarantine_reason": ctx.precheck_result.quarantine_reason,
        },
    )
    _append_step_log(
        ctx.log_path,
        (
            "precheck "
            f"duration_sec={ctx.precheck_result.duration_sec} "
            f"speech_ratio={ctx.precheck_result.speech_ratio}"
        ),
    )
    return _StageResult(
        status="completed",
        metadata={
            "duration_sec": ctx.precheck_result.duration_sec,
            "speech_ratio": ctx.precheck_result.speech_ratio,
            "quarantine_reason": ctx.precheck_result.quarantine_reason,
        },
    )


def _stage_calendar_refresh(ctx: _PipelineExecutionContext) -> _StageResult:
    warning: str | None = None
    try:
        refresh_recording_calendar_match(
            ctx.recording_id,
            settings=ctx.settings,
        )
    except Exception as exc:
        warning = str(exc) or exc.__class__.__name__
        _logger.warning(
            "calendar matching refresh failed for recording %s",
            ctx.recording_id,
            exc_info=True,
        )
    ctx.calendar_title, ctx.calendar_attendees = _load_calendar_summary_context(
        ctx.recording_id,
        ctx.settings,
    )
    atomic_write_json(
        ctx.artifacts.calendar_refresh_json_path,
        {
            "calendar_title": ctx.calendar_title,
            "calendar_attendees": ctx.calendar_attendees,
            "warning": warning,
        },
    )
    return _StageResult(
        status="completed",
        metadata={
            "calendar_title": ctx.calendar_title,
            "attendee_count": len(ctx.calendar_attendees),
            "warning": warning,
        },
    )


def _stage_asr(ctx: _PipelineExecutionContext) -> _StageResult:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    ctx.precheck_result = precheck_result
    if precheck_result is None:
        raise RuntimeError("Missing precheck artifact")
    if precheck_result.quarantine_reason:
        pipeline_orchestrator._write_asr_glossary_artifact(
            derived_dir=ctx.artifacts.derived_dir,
            recording_id=ctx.recording_id,
            asr_glossary=None,
        )
        return _build_skip_result("quarantined_precheck")

    ctx.calendar_title, ctx.calendar_attendees = _load_calendar_summary_context(
        ctx.recording_id,
        ctx.settings,
    )
    working_audio_path = _working_audio_path(ctx)
    if working_audio_path is None:
        raise RuntimeError("Missing sanitized audio")
    ctx.asr_glossary = build_recording_asr_glossary(
        ctx.recording_id,
        calendar_title=ctx.calendar_title,
        calendar_attendees=ctx.calendar_attendees,
        settings=ctx.settings,
    )
    _append_step_log(
        ctx.log_path,
        (
            "asr glossary "
            f"entries={int(ctx.asr_glossary.get('entry_count') or 0)} "
            f"terms={int(ctx.asr_glossary.get('term_count') or 0)} "
            f"truncated={bool(ctx.asr_glossary.get('truncated'))}"
        ),
    )

    def _step_log_callback(message: str) -> None:
        _append_step_log(ctx.log_path, message)

    def _run_asr_workflow() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        def _transcribe_chunk(
            chunk_audio_path: Path,
            chunk_language_hint: str | None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            return pipeline_orchestrator._whisperx_asr(
                chunk_audio_path,
                override_lang=chunk_language_hint,
                cfg=ctx.pipeline_settings,
                step_log_callback=_step_log_callback,
            )

        if pipeline_orchestrator._whisperx_asr is not pipeline_orchestrator._DEFAULT_WHISPERX_ASR:
            return run_language_aware_asr(
                working_audio_path,
                override_lang=ctx.transcript_language_override,
                configured_mode=ctx.pipeline_settings.asr_multilingual_mode,
                tmp_root=ctx.pipeline_settings.tmp_root,
                transcribe_fn=_transcribe_chunk,
                step_log_callback=_step_log_callback,
            )

        previous_transcriber = getattr(
            pipeline_orchestrator._whisperx_transcriber_state,
            "transcribe_audio",
            None,
        )
        previous_session_flag = bool(
            getattr(pipeline_orchestrator._whisperx_transcriber_state, "use_session_transcriber", False)
        )
        transcribe_audio = pipeline_orchestrator._build_whisperx_transcriber(
            cfg=ctx.pipeline_settings,
            step_log_callback=_step_log_callback,
            asr_glossary=ctx.asr_glossary,
        )
        pipeline_orchestrator._whisperx_transcriber_state.transcribe_audio = transcribe_audio
        pipeline_orchestrator._whisperx_transcriber_state.use_session_transcriber = True
        try:
            segments, info, payload = run_language_aware_asr(
                working_audio_path,
                override_lang=ctx.transcript_language_override,
                configured_mode=ctx.pipeline_settings.asr_multilingual_mode,
                tmp_root=ctx.pipeline_settings.tmp_root,
                transcribe_fn=_transcribe_chunk,
                step_log_callback=_step_log_callback,
            )
            runtime_metadata = pipeline_orchestrator._glossary_runtime_metadata(transcribe_audio)
            if runtime_metadata:
                payload = dict(payload)
                payload["glossary_runtime"] = runtime_metadata
            return segments, info, payload
        finally:
            if previous_transcriber is None:
                if hasattr(pipeline_orchestrator._whisperx_transcriber_state, "transcribe_audio"):
                    delattr(pipeline_orchestrator._whisperx_transcriber_state, "transcribe_audio")
            else:
                pipeline_orchestrator._whisperx_transcriber_state.transcribe_audio = previous_transcriber
            if previous_session_flag:
                pipeline_orchestrator._whisperx_transcriber_state.use_session_transcriber = True
            elif hasattr(pipeline_orchestrator._whisperx_transcriber_state, "use_session_transcriber"):
                delattr(pipeline_orchestrator._whisperx_transcriber_state, "use_session_transcriber")

    raw_segments, info, asr_execution = _run_asr_workflow()
    ctx.asr_segments = normalise_asr_segments(raw_segments)
    ctx.asr_info = dict(info or {})
    ctx.asr_execution = dict(asr_execution or {})
    pipeline_orchestrator._write_asr_glossary_artifact(
        derived_dir=ctx.artifacts.derived_dir,
        recording_id=ctx.recording_id,
        asr_glossary=pipeline_orchestrator._effective_asr_glossary_artifact(
            asr_glossary=ctx.asr_glossary,
            runtime_metadata=(
                ctx.asr_execution.get("glossary_runtime")
                if isinstance(ctx.asr_execution, dict)
                else None
            ),
        ),
    )
    atomic_write_json(ctx.artifacts.asr_segments_json_path, ctx.asr_segments)
    atomic_write_json(ctx.artifacts.asr_info_json_path, ctx.asr_info)
    atomic_write_json(ctx.artifacts.asr_execution_json_path, ctx.asr_execution)
    return _StageResult(
        status="completed",
        metadata={
            "segment_count": len(ctx.asr_segments),
            "used_multilingual_path": bool(ctx.asr_execution.get("used_multilingual_path")),
        },
    )


def _stage_diarization(ctx: _PipelineExecutionContext) -> _StageResult:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    ctx.precheck_result = precheck_result
    if precheck_result is None:
        raise RuntimeError("Missing precheck artifact")
    if precheck_result.quarantine_reason:
        return _build_skip_result("quarantined_precheck")

    diarization_mode = "pyannote"
    diarization_reason: str | None = None
    try:
        diariser = _build_diariser(
            precheck_result.duration_sec,
            model_id=ctx.settings.diarization_model_id,
            settings=ctx.pipeline_settings,
        )
        if isinstance(diariser, _FallbackDiariser):
            diarization_mode = "fallback"
            diarization_reason = "pyannote_unavailable"
            _append_step_log(
                ctx.log_path,
                "diariser mode=fallback reason=pyannote_unavailable",
            )
        else:
            _append_step_log(ctx.log_path, "diariser mode=pyannote")
    except Exception as exc:
        _append_step_log(
            ctx.log_path,
            (
                "diariser init failed, falling back: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        diariser = _FallbackDiariser(precheck_result.duration_sec)
        diarization_mode = "fallback"
        diarization_reason = f"{type(exc).__name__}: {exc}"
        _append_step_log(
            ctx.log_path,
            "diariser mode=fallback reason=init_failed",
        )
    _write_diarization_status_artifact(
        recording_id=ctx.recording_id,
        mode=diarization_mode,
        reason=diarization_reason,
        settings=ctx.settings,
    )
    _log_gpu_execution_policy(
        pipeline_settings=ctx.pipeline_settings,
        diariser=diariser,
        log_path=ctx.log_path,
    )
    working_audio_path = _working_audio_path(ctx)
    if working_audio_path is None:
        raise RuntimeError("Missing sanitized audio")
    ctx.asr_segments = _load_json_list(ctx.artifacts.asr_segments_json_path)

    def _step_log_callback(message: str) -> None:
        _append_step_log(ctx.log_path, message)

    diarization = asyncio.run(diariser(working_audio_path))
    diarization = asyncio.run(
        pipeline_orchestrator._maybe_retry_dialog_diarization(
            diariser=diariser,
            audio_path=working_audio_path,
            diarization=diarization,
            asr_segments=ctx.asr_segments,
            precheck_result=precheck_result,
            step_log_callback=_step_log_callback,
        )
    )
    ctx.diarization_segments = _diarization_segments(diarization)
    used_dummy_fallback = False
    if not ctx.diarization_segments and ctx.asr_segments:
        fallback_end = max(safe_float(seg.get("end")) for seg in ctx.asr_segments)
        used_dummy_fallback = True
        _append_step_log(
            ctx.log_path,
            "diarization output empty; using fallback single-speaker annotation",
        )
        ctx.diarization_segments = _diarization_segments(
            pipeline_orchestrator._fallback_diarization(max(fallback_end, 0.1))
        )
    ctx.diarization_runtime = pipeline_orchestrator._diariser_runtime_metadata(diariser)
    ctx.diarization_runtime["used_dummy_fallback"] = used_dummy_fallback
    ctx.diarization_runtime["mode"] = pipeline_orchestrator._diariser_mode(diariser)
    atomic_write_json(ctx.artifacts.diarization_segments_json_path, ctx.diarization_segments)
    atomic_write_json(ctx.artifacts.diarization_runtime_json_path, ctx.diarization_runtime)
    return _StageResult(
        status="completed",
        metadata={
            "segment_count": len(ctx.diarization_segments),
            "mode": ctx.diarization_runtime.get("mode"),
            "used_dummy_fallback": used_dummy_fallback,
        },
    )


def _stage_language_analysis(ctx: _PipelineExecutionContext) -> _StageResult:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    ctx.precheck_result = precheck_result
    if precheck_result is None:
        raise RuntimeError("Missing precheck artifact")
    if precheck_result.quarantine_reason:
        return _build_skip_result("quarantined_precheck")

    ctx.asr_segments = _load_json_list(ctx.artifacts.asr_segments_json_path)
    ctx.asr_info = _load_json_dict(ctx.artifacts.asr_info_json_path)
    ctx.asr_execution = _load_asr_execution(ctx)
    language_info = pipeline_orchestrator._language_payload(ctx.asr_info)
    detected_language = (
        normalise_language_code(language_info.get("detected"))
        if str(language_info.get("detected") or "") != "unknown"
        else None
    )
    language_analysis = analyse_languages(
        ctx.asr_segments,
        detected_language=(
            None if ctx.asr_execution.get("used_multilingual_path") else detected_language
        ),
        transcript_language_override=ctx.transcript_language_override,
    )
    if (
        ctx.asr_execution.get("used_multilingual_path")
        or language_info.get("detected") == "unknown"
    ) and language_analysis.dominant_language != "unknown":
        language_info["detected"] = language_analysis.dominant_language
        dominant_percent = language_analysis.distribution.get(
            language_analysis.dominant_language
        )
        if dominant_percent is not None:
            language_info["confidence"] = round(dominant_percent / 100.0, 4)
    summary_lang = resolve_target_summary_language(
        ctx.target_summary_language,
        dominant_language=language_analysis.dominant_language,
        detected_language=detected_language,
    )
    ctx.language_payload = {
        "language": language_info,
        "dominant_language": language_analysis.dominant_language,
        "language_distribution": language_analysis.distribution,
        "language_spans": language_analysis.spans,
        "target_summary_language": summary_lang,
        "transcript_language_override": ctx.transcript_language_override,
        "segments": language_analysis.segments,
        "review": {
            "required": bool(language_analysis.review_required),
            "reason_code": language_analysis.review_reason_code,
            "reason_text": language_analysis.review_reason_text,
            "uncertain_segment_count": language_analysis.uncertain_segment_count,
            "conflict_segment_count": language_analysis.conflict_segment_count,
        },
    }
    atomic_write_json(ctx.artifacts.language_analysis_json_path, ctx.language_payload)
    return _StageResult(
        status="completed",
        metadata={
            "dominant_language": language_analysis.dominant_language,
            "review_required": bool(language_analysis.review_required),
        },
    )


def _stage_speaker_turns(ctx: _PipelineExecutionContext) -> _StageResult:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    ctx.precheck_result = precheck_result
    if precheck_result is None:
        raise RuntimeError("Missing precheck artifact")
    if precheck_result.quarantine_reason:
        return _build_skip_result("quarantined_precheck")

    ctx.language_payload = _load_language_analysis_artifact(ctx)
    if not ctx.language_payload:
        raise RuntimeError("Missing language analysis artifact")
    ctx.diarization_segments = _load_json_list(ctx.artifacts.diarization_segments_json_path)
    ctx.diarization_runtime = _load_diarization_runtime(ctx)
    language_segments = [
        row
        for row in ctx.language_payload.get("segments", [])
        if isinstance(row, dict)
    ]
    dominant_language = str(ctx.language_payload.get("dominant_language") or "unknown")
    detected_language = normalise_language_code(
        (ctx.language_payload.get("language") or {}).get("detected")
    )
    unsmoothed_speaker_turns = build_speaker_turns(
        language_segments,
        ctx.diarization_segments,
        default_language=(
            dominant_language if dominant_language != "unknown" else detected_language
        ),
    )
    diariser_mode = str(ctx.diarization_runtime.get("mode") or "unknown").strip().lower()
    if diariser_mode == "pyannote" and not ctx.diarization_runtime.get("used_dummy_fallback"):
        smoothing_result = smooth_speaker_turns(
            unsmoothed_speaker_turns,
            merge_gap_seconds=ctx.pipeline_settings.diarization_merge_gap_seconds,
            min_turn_seconds=ctx.pipeline_settings.diarization_min_turn_seconds,
        )
        ctx.speaker_turns = smoothing_result.turns
    else:
        ctx.speaker_turns = unsmoothed_speaker_turns
        speaker_count = len(
            {
                str(turn.get("speaker") or "S1")
                for turn in ctx.speaker_turns
            }
        )
        smoothing_result = SpeakerTurnSmoothingResult(
            turns=ctx.speaker_turns,
            adjacent_merges=0,
            micro_turn_absorptions=0,
            turn_count_before=len(ctx.speaker_turns),
            turn_count_after=len(ctx.speaker_turns),
            speaker_count_before=speaker_count,
            speaker_count_after=speaker_count,
        )
    diarization_metadata = _build_diarization_metadata_payload(
        runtime=ctx.diarization_runtime,
        cfg=ctx.pipeline_settings,
        smoothing_result=smoothing_result,
    )
    atomic_write_json(
        ctx.artifacts.recording_artifacts.segments_json_path,
        ctx.diarization_segments,
    )
    atomic_write_json(
        ctx.artifacts.recording_artifacts.speaker_turns_json_path,
        ctx.speaker_turns,
    )
    atomic_write_json(
        ctx.artifacts.recording_artifacts.diarization_metadata_json_path,
        diarization_metadata,
    )
    aliases = load_speaker_aliases(ctx.pipeline_settings.speaker_db)
    for row in ctx.diarization_segments:
        aliases.setdefault(str(row.get("speaker") or "S1"), str(row.get("speaker") or "S1"))
    save_speaker_aliases(aliases, ctx.pipeline_settings.speaker_db)
    return _StageResult(
        status="completed",
        metadata={
            "turn_count": len(ctx.speaker_turns),
            "speaker_count": len({str(turn.get('speaker') or 'S1') for turn in ctx.speaker_turns}),
        },
    )


def _stage_llm_extract(ctx: _PipelineExecutionContext) -> _StageResult:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    ctx.precheck_result = precheck_result
    if precheck_result is None:
        raise RuntimeError("Missing precheck artifact")
    if precheck_result.quarantine_reason:
        return _build_skip_result("quarantined_precheck")

    ctx.language_payload = _load_language_analysis_artifact(ctx)
    ctx.speaker_turns = _load_json_list(ctx.artifacts.recording_artifacts.speaker_turns_json_path)
    language_segments = [
        row
        for row in ctx.language_payload.get("segments", [])
        if isinstance(row, dict)
    ]
    asr_text = " ".join(str(seg.get("text") or "").strip() for seg in language_segments).strip()
    ctx.clean_text = normalizer.dedup(asr_text)
    if not ctx.clean_text:
        return _build_skip_result("no_speech")

    llm_model = pipeline_orchestrator._require_llm_model(ctx.pipeline_settings.llm_model)
    summary_lang = str(
        ctx.language_payload.get("target_summary_language")
        or ctx.target_summary_language
        or "en"
    )
    aliases = load_speaker_aliases(ctx.pipeline_settings.speaker_db)
    ctx.friendly = pipeline_orchestrator._sentiment_score(ctx.clean_text)
    llm_prompt_text = pipeline_orchestrator._speaker_turn_prompt_text(
        ctx.speaker_turns,
        aliases=aliases,
    )

    def _progress_callback(stage: str, progress: float) -> None:
        _set_recording_progress_best_effort(
            ctx.recording_id,
            stage=stage,
            progress=progress,
            settings=ctx.settings,
        )

    if pipeline_orchestrator._use_chunked_llm(llm_prompt_text, ctx.pipeline_settings):
        ctx.summary_payload = asyncio.run(
            pipeline_orchestrator._run_chunked_llm_summary(
                transcript_text=llm_prompt_text or ctx.clean_text,
                derived_dir=ctx.artifacts.derived_dir,
                llm=LLMClient(),
                cfg=ctx.pipeline_settings,
                llm_model=llm_model,
                target_summary_language=summary_lang,
                friendly=ctx.friendly,
                default_topic=ctx.calendar_title or "Meeting summary",
                calendar_title=ctx.calendar_title,
                calendar_attendees=ctx.calendar_attendees,
                progress_callback=_progress_callback,
            )
        )
    else:
        sys_prompt, user_prompt = build_structured_summary_prompts(
            ctx.speaker_turns,
            summary_lang,
            calendar_title=ctx.calendar_title,
            calendar_attendees=ctx.calendar_attendees,
        )
        _set_recording_progress_best_effort(
            ctx.recording_id,
            stage="llm",
            progress=0.90,
            settings=ctx.settings,
        )
        msg = asyncio.run(
            pipeline_orchestrator._generate_llm_message(
                LLMClient(),
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                model=llm_model,
                response_format={"type": "json_object"},
                max_tokens=ctx.pipeline_settings.llm_max_tokens,
                max_tokens_retry=ctx.pipeline_settings.llm_max_tokens_retry,
            )
        )
        ctx.summary_payload = build_summary_payload(
            raw_llm_content=str(msg.get("content") or ""),
            model=llm_model,
            target_summary_language=summary_lang,
            friendly=ctx.friendly,
            default_topic=ctx.calendar_title or "Meeting summary",
            derived_dir=ctx.artifacts.derived_dir,
        )
    atomic_write_json(
        ctx.artifacts.recording_artifacts.summary_json_path,
        ctx.summary_payload,
    )
    return _StageResult(
        status="completed",
        metadata={
            "summary_status": str((ctx.summary_payload or {}).get("status") or "ok"),
            "friendly": ctx.friendly,
        },
    )


def _stage_export_artifacts(ctx: _PipelineExecutionContext) -> _StageResult:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    ctx.precheck_result = precheck_result
    if precheck_result is None:
        raise RuntimeError("Missing precheck artifact")
    ctx.calendar_title, ctx.calendar_attendees = _load_calendar_summary_context(
        ctx.recording_id,
        ctx.settings,
    )
    ctx.language_payload = _load_language_analysis_artifact(ctx)
    ctx.asr_execution = _load_asr_execution(ctx)
    ctx.diarization_segments = _load_json_list(ctx.artifacts.diarization_segments_json_path)
    ctx.speaker_turns = _load_json_list(ctx.artifacts.recording_artifacts.speaker_turns_json_path)
    llm_model = pipeline_orchestrator._require_llm_model(ctx.pipeline_settings.llm_model)
    aliases = load_speaker_aliases(ctx.pipeline_settings.speaker_db)
    summary_lang = str(
        ctx.language_payload.get("target_summary_language")
        or ctx.target_summary_language
        or "en"
    )
    working_audio_path = _working_audio_path(ctx)
    language_info = ctx.language_payload.get("language")
    if not isinstance(language_info, dict):
        language_info = {"detected": "unknown", "confidence": None}

    if precheck_result.quarantine_reason:
        write_empty_snippets_manifest(
            snippets_dir=ctx.artifacts.recording_artifacts.snippets_dir,
            pad_seconds=ctx.pipeline_settings.snippet_pad_seconds,
            max_clip_duration_sec=ctx.pipeline_settings.snippet_max_duration_seconds,
            min_clip_duration_sec=ctx.pipeline_settings.snippet_min_duration_seconds,
            max_snippets_per_speaker=ctx.pipeline_settings.snippet_max_per_speaker,
        )
        atomic_write_text(ctx.artifacts.recording_artifacts.transcript_txt_path, "")
        atomic_write_json(
            ctx.artifacts.recording_artifacts.transcript_json_path,
            pipeline_orchestrator._base_transcript_payload(
                recording_id=ctx.recording_id,
                language={"detected": "unknown", "confidence": None},
                dominant_language="unknown",
                language_distribution={},
                language_spans=[],
                target_summary_language=summary_lang,
                transcript_language_override=ctx.transcript_language_override,
                calendar_title=ctx.calendar_title,
                calendar_attendees=ctx.calendar_attendees,
                segments=[],
                speakers=[],
                text="",
            ),
        )
        atomic_write_json(ctx.artifacts.recording_artifacts.segments_json_path, [])
        atomic_write_json(ctx.artifacts.recording_artifacts.speaker_turns_json_path, [])
        atomic_write_json(
            ctx.artifacts.recording_artifacts.summary_json_path,
            pipeline_summary_builder._build_structured_summary_payload(
                model=llm_model,
                target_summary_language=summary_lang,
                friendly=0,
                topic="Quarantined recording",
                summary_bullets=["Recording was quarantined before transcription."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=pipeline_orchestrator._empty_questions(),
                status="quarantined",
                reason=precheck_result.quarantine_reason,
            ),
        )
        return _StageResult(
            status="completed",
            metadata={"output_status": "quarantined"},
        )

    language_segments = [
        row
        for row in ctx.language_payload.get("segments", [])
        if isinstance(row, dict)
    ]
    ctx.clean_text = normalizer.dedup(
        " ".join(str(seg.get("text") or "").strip() for seg in language_segments).strip()
    )
    if not ctx.clean_text:
        write_empty_snippets_manifest(
            snippets_dir=ctx.artifacts.recording_artifacts.snippets_dir,
            pad_seconds=ctx.pipeline_settings.snippet_pad_seconds,
            max_clip_duration_sec=ctx.pipeline_settings.snippet_max_duration_seconds,
            min_clip_duration_sec=ctx.pipeline_settings.snippet_min_duration_seconds,
            max_snippets_per_speaker=ctx.pipeline_settings.snippet_max_per_speaker,
        )
        speakers = sorted(
            {
                aliases.get(str(row.get("speaker") or "S1"), str(row.get("speaker") or "S1"))
                for row in ctx.diarization_segments
            }
        )
        atomic_write_text(ctx.artifacts.recording_artifacts.transcript_txt_path, "")
        atomic_write_json(
            ctx.artifacts.recording_artifacts.transcript_json_path,
            pipeline_orchestrator._base_transcript_payload(
                recording_id=ctx.recording_id,
                language=language_info,
                dominant_language=str(ctx.language_payload.get("dominant_language") or "unknown"),
                language_distribution=dict(ctx.language_payload.get("language_distribution") or {}),
                language_spans=list(ctx.language_payload.get("language_spans") or []),
                target_summary_language=summary_lang,
                transcript_language_override=ctx.transcript_language_override,
                calendar_title=ctx.calendar_title,
                calendar_attendees=ctx.calendar_attendees,
                segments=language_segments,
                speakers=speakers,
                text="",
            ),
        )
        atomic_write_json(ctx.artifacts.recording_artifacts.segments_json_path, ctx.diarization_segments)
        atomic_write_json(ctx.artifacts.recording_artifacts.speaker_turns_json_path, ctx.speaker_turns)
        atomic_write_json(
            ctx.artifacts.recording_artifacts.summary_json_path,
            pipeline_summary_builder._build_structured_summary_payload(
                model=llm_model,
                target_summary_language=summary_lang,
                friendly=0,
                topic="No speech detected",
                summary_bullets=["No speech detected."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=pipeline_orchestrator._empty_questions(),
                status="no_speech",
            ),
        )
        return _StageResult(
            status="completed",
            metadata={"output_status": "no_speech"},
        )

    ctx.summary_payload = _load_summary_payload(ctx)
    snippet_paths = export_speaker_snippets(
        SnippetExportRequest(
            audio_path=working_audio_path or ctx.artifacts.recording_artifacts.raw_audio_path,
            diar_segments=ctx.diarization_segments,
            snippets_dir=ctx.artifacts.recording_artifacts.snippets_dir,
            duration_sec=precheck_result.duration_sec,
            speaker_turns=ctx.speaker_turns,
            degraded_diarization=bool(_load_json_dict(ctx.artifacts.recording_artifacts.diarization_metadata_json_path).get("degraded")),
            pad_seconds=ctx.pipeline_settings.snippet_pad_seconds,
            max_clip_duration_sec=ctx.pipeline_settings.snippet_max_duration_seconds,
            min_clip_duration_sec=ctx.pipeline_settings.snippet_min_duration_seconds,
            max_snippets_per_speaker=ctx.pipeline_settings.snippet_max_per_speaker,
        )
    )
    speaker_lines = pipeline_orchestrator._merge_similar(
        [
            f"[{safe_float(turn.get('start')):.2f}-{safe_float(turn.get('end')):.2f}] **{aliases.get(str(turn.get('speaker') or 'S1'), str(turn.get('speaker') or 'S1'))}:** {str(turn.get('text') or '').strip()}"
            for turn in ctx.speaker_turns
        ],
        ctx.pipeline_settings.merge_similar,
    )
    transcript_payload = pipeline_orchestrator._base_transcript_payload(
        recording_id=ctx.recording_id,
        language=language_info,
        dominant_language=str(ctx.language_payload.get("dominant_language") or "unknown"),
        language_distribution=dict(ctx.language_payload.get("language_distribution") or {}),
        language_spans=list(ctx.language_payload.get("language_spans") or []),
        target_summary_language=summary_lang,
        transcript_language_override=ctx.transcript_language_override,
        calendar_title=ctx.calendar_title,
        calendar_attendees=ctx.calendar_attendees,
        segments=language_segments,
        speakers=sorted(
            {
                aliases.get(str(turn.get("speaker") or "S1"), str(turn.get("speaker") or "S1"))
                for turn in ctx.speaker_turns
            }
        ),
        text=ctx.clean_text,
    )
    transcript_payload["speaker_lines"] = speaker_lines
    transcript_payload["multilingual_asr"] = dict(ctx.asr_execution)
    transcript_payload["review"] = dict(ctx.language_payload.get("review") or {})
    atomic_write_text(ctx.artifacts.recording_artifacts.transcript_txt_path, ctx.clean_text)
    atomic_write_json(ctx.artifacts.recording_artifacts.transcript_json_path, transcript_payload)
    atomic_write_json(ctx.artifacts.recording_artifacts.segments_json_path, ctx.diarization_segments)
    atomic_write_json(ctx.artifacts.recording_artifacts.speaker_turns_json_path, ctx.speaker_turns)
    atomic_write_json(ctx.artifacts.recording_artifacts.summary_json_path, ctx.summary_payload)
    return _StageResult(
        status="completed",
        metadata={
            "output_status": "ok",
            "snippets": len(snippet_paths),
        },
    )


def _stage_metrics(ctx: _PipelineExecutionContext) -> _StageResult:
    metrics_payload = refresh_recording_metrics(
        ctx.recording_id,
        settings=ctx.settings,
    )
    _append_step_log(
        ctx.log_path,
        (
            "metrics refreshed "
            f"participants={len(metrics_payload.get('participants', []))} "
            f"interruptions={metrics_payload.get('meeting', {}).get('total_interruptions', 0)}"
        ),
    )
    dominant_language, resolved_target_language = _load_transcript_language_payload(
        ctx.recording_id,
        ctx.settings,
    )
    update_payload: dict[str, str] = {}
    if dominant_language:
        update_payload["language_auto"] = dominant_language
    if ctx.has_explicit_summary_target and resolved_target_language:
        update_payload["target_summary_language"] = resolved_target_language
    if update_payload:
        set_recording_language_settings(
            ctx.recording_id,
            settings=ctx.settings,
            **update_payload,
        )
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    summary_payload = _load_summary_payload(ctx)
    language_payload = _load_language_analysis_artifact(ctx)
    asr_execution = _load_asr_execution(ctx)
    diar_segments = _load_json_list(ctx.artifacts.recording_artifacts.segments_json_path)
    speaker_turns = _load_json_list(ctx.artifacts.recording_artifacts.speaker_turns_json_path)
    snippets_manifest = _load_json_dict(ctx.artifacts.recording_artifacts.snippets_dir.parent / "snippets_manifest.json")
    metrics_status = "ok"
    if precheck_result is not None and precheck_result.quarantine_reason:
        metrics_status = "quarantined"
    elif str(summary_payload.get("status") or "") == "no_speech":
        metrics_status = "no_speech"
    atomic_write_json(
        ctx.artifacts.recording_artifacts.metrics_json_path,
        {
            "status": metrics_status,
            "version": 1,
            "precheck": {
                "duration_sec": precheck_result.duration_sec if precheck_result else None,
                "speech_ratio": precheck_result.speech_ratio if precheck_result else None,
                "quarantine_reason": (
                    precheck_result.quarantine_reason
                    if metrics_status == "quarantined" and precheck_result is not None
                    else None
                ),
            },
            "language": language_payload.get("language") or {"detected": "unknown", "confidence": None},
            "asr_segments": len(language_payload.get("segments") or []),
            "diar_segments": len(diar_segments),
            "speaker_turns": len(speaker_turns),
            "snippets": len(snippets_manifest.get("speakers") or {}),
            "multilingual_asr": {
                "used_multilingual_path": bool(asr_execution.get("used_multilingual_path")),
                "selected_mode": asr_execution.get("selected_mode"),
            },
            "review_required": bool((language_payload.get("review") or {}).get("required")),
        },
    )
    return _StageResult(
        status="completed",
        metadata={"metrics_status": metrics_status},
    )


def _stage_routing(ctx: _PipelineExecutionContext) -> _StageResult:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    ctx.precheck_result = precheck_result
    if precheck_result is None:
        raise RuntimeError("Missing precheck artifact")
    if precheck_result.quarantine_reason:
        return _build_skip_result(
            "quarantined_precheck",
            status_after_routing=RECORDING_STATUS_QUARANTINE,
        )
    ctx.routing_payload = refresh_recording_routing(
        ctx.recording_id,
        settings=ctx.settings,
        apply_workflow=True,
    )
    _append_step_log(
        ctx.log_path,
        (
            "routing "
            f"suggested_project_id={ctx.routing_payload.get('suggested_project_id')} "
            f"confidence={float(ctx.routing_payload.get('confidence') or 0.0):.2f} "
            f"threshold={float(ctx.routing_payload.get('threshold') or 0.0):.2f} "
            f"auto_selected={bool(ctx.routing_payload.get('auto_selected'))} "
            f"status_after={ctx.routing_payload.get('status_after_routing')}"
        ),
    )
    atomic_write_json(ctx.artifacts.routing_json_path, ctx.routing_payload)
    return _StageResult(
        status="completed",
        metadata={
            "status_after_routing": ctx.routing_payload.get("status_after_routing"),
        },
    )


_PIPELINE_STAGE_RUNNERS: dict[str, Any] = {
    "sanitize_audio": _stage_sanitize_audio,
    "precheck": _stage_precheck,
    "calendar_refresh": _stage_calendar_refresh,
    "asr": _stage_asr,
    "diarization": _stage_diarization,
    "language_analysis": _stage_language_analysis,
    "speaker_turns": _stage_speaker_turns,
    "llm_extract": _stage_llm_extract,
    "export_artifacts": _stage_export_artifacts,
    "metrics": _stage_metrics,
    "routing": _stage_routing,
}


def _clear_later_stage_rows(
    recording_id: str,
    *,
    current_index: int,
    settings: AppSettings,
) -> None:
    if current_index + 1 >= len(PIPELINE_STAGE_DEFINITIONS):
        return
    clear_recording_pipeline_stages(
        recording_id,
        settings=settings,
        from_stage=PIPELINE_STAGE_DEFINITIONS[current_index + 1].name,
    )


def _terminal_state_from_stage_artifacts(
    *,
    ctx: _PipelineExecutionContext,
) -> PipelineTerminalState:
    precheck_result = ctx.precheck_result or _load_precheck_artifact(ctx)
    if precheck_result is not None and precheck_result.quarantine_reason:
        _append_step_log(
            ctx.log_path,
            f"quarantined reason={precheck_result.quarantine_reason}",
        )
        return PipelineTerminalState(
            status=RECORDING_STATUS_QUARANTINE,
            quarantine_reason=precheck_result.quarantine_reason,
        )
    routing_payload = ctx.routing_payload or _load_json_dict(ctx.artifacts.routing_json_path)
    if routing_payload.get("status_after_routing") == RECORDING_STATUS_NEEDS_REVIEW:
        review_reason_code, review_reason_text = _review_reason_from_routing(
            recording_id=ctx.recording_id,
            settings=ctx.settings,
            routing=routing_payload,
        )
        return PipelineTerminalState(
            status=RECORDING_STATUS_NEEDS_REVIEW,
            review_reason_code=review_reason_code,
            review_reason_text=review_reason_text,
        )
    return PipelineTerminalState(status=RECORDING_STATUS_READY)


def _run_precheck_pipeline_legacy(
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
        _set_recording_duration_best_effort(
            recording_id,
            duration_sec=precheck.duration_sec,
            settings=settings,
        )
    try:
        refresh_recording_calendar_match(
            recording_id,
            settings=settings,
        )
    except Exception:
        _logger.warning(
            "calendar matching refresh failed for recording %s",
            recording_id,
            exc_info=True,
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
                settings=pipeline_settings,
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
    _log_gpu_execution_policy(
        pipeline_settings=pipeline_settings,
        diariser=diariser,
        log_path=log_path,
    )
    calendar_title, calendar_attendees = _load_calendar_summary_context(
        recording_id,
        settings,
    )
    asr_glossary = build_recording_asr_glossary(
        recording_id,
        calendar_title=calendar_title,
        calendar_attendees=calendar_attendees,
        settings=settings,
    )
    _append_step_log(
        log_path,
        (
            "asr glossary "
            f"entries={int(asr_glossary.get('entry_count') or 0)} "
            f"terms={int(asr_glossary.get('term_count') or 0)} "
            f"truncated={bool(asr_glossary.get('truncated'))}"
        ),
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
            asr_glossary=asr_glossary,
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

def _run_precheck_pipeline(
    *,
    recording_id: str,
    settings: AppSettings,
    log_path: Path,
) -> PipelineTerminalState:
    if run_pipeline is not _ORIGINAL_RUN_PIPELINE:
        return _run_precheck_pipeline_legacy(
            recording_id=recording_id,
            settings=settings,
            log_path=log_path,
        )

    ctx = _new_pipeline_context(
        recording_id=recording_id,
        settings=settings,
        log_path=log_path,
    )
    stage_rows = _stage_rows_by_name(recording_id, settings=settings)
    start_index: int | None = None

    for index, stage in enumerate(PIPELINE_STAGE_DEFINITIONS):
        row = stage_rows.get(stage.name)
        status = str(row.get("status") or "").strip().lower() if row is not None else ""
        metadata = _stage_metadata(row)
        if status in PIPELINE_STAGE_DONE_STATUSES:
            valid, reason = validate_stage_artifacts(
                recording_id,
                stage_name=stage.name,
                status=status,
                metadata=metadata,
                settings=settings,
            )
            if valid:
                continue
            _log_stage_invalidated(log_path, stage.name, reason=reason or "artifact missing")
            _clear_later_stage_rows(
                recording_id,
                current_index=index,
                settings=settings,
            )
            start_index = index
            break
        start_index = index
        if row is not None:
            _clear_later_stage_rows(
                recording_id,
                current_index=index,
                settings=settings,
            )
        break

    if start_index is None:
        _append_step_log(log_path, "pipeline artifacts generated")
        return _terminal_state_from_stage_artifacts(ctx=ctx)

    for index in range(start_index, len(PIPELINE_STAGE_DEFINITIONS)):
        stage = PIPELINE_STAGE_DEFINITIONS[index]
        resumed = index == start_index and start_index > 0
        _log_stage_started(log_path, stage.name, resumed=resumed)
        mark_recording_pipeline_stage_started(
            recording_id,
            stage_name=stage.name,
            metadata={"label": stage.label},
            settings=settings,
        )
        _set_recording_progress_best_effort(
            recording_id,
            stage=_recording_pipeline_progress_stage(stage.name),
            progress=stage.progress,
            settings=settings,
        )
        runner = _PIPELINE_STAGE_RUNNERS[stage.name]
        try:
            result = runner(ctx)
        except Exception as exc:
            mark_recording_pipeline_stage_failed(
                recording_id,
                stage_name=stage.name,
                error_code=type(exc).__name__,
                error_text=str(exc) or type(exc).__name__,
                metadata={"label": stage.label},
                settings=settings,
            )
            raise

        metadata = dict(result.metadata)
        metadata.setdefault("label", stage.label)
        if result.status == PIPELINE_STAGE_STATUS_SKIPPED:
            row = mark_recording_pipeline_stage_skipped(
                recording_id,
                stage_name=stage.name,
                metadata=metadata,
                settings=settings,
            )
            _log_stage_skipped(
                log_path,
                stage.name,
                reason=str(metadata.get("skip_reason") or "skipped"),
            )
            continue

        row = mark_recording_pipeline_stage_completed(
            recording_id,
            stage_name=stage.name,
            metadata=metadata,
            settings=settings,
        )
        duration_ms = None
        if isinstance(row, dict):
            try:
                duration_ms = int(row.get("duration_ms"))
            except (TypeError, ValueError):
                duration_ms = None
        _log_stage_completed(log_path, stage.name, duration_ms=duration_ms)

    _append_step_log(log_path, "pipeline artifacts generated")
    return _terminal_state_from_stage_artifacts(ctx=ctx)


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
