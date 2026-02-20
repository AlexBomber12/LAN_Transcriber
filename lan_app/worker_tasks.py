from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from .db import fail_job, finish_job, init_db, set_recording_status, start_job


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


class _NoopLLMClient:
    async def generate(self, **_kwargs: Any) -> dict[str, str]:
        return {"content": ""}


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

        return Pipeline.from_pretrained("pyannote/speaker-diarization@3.2")
    except Exception:
        return _FallbackDiariser(duration_sec)


def _run_precheck_pipeline(
    *,
    recording_id: str,
    settings: AppSettings,
    log_path: Path,
) -> tuple[str, str | None]:
    audio_path = _resolve_raw_audio_path(recording_id, settings)
    if audio_path is None:
        _append_step_log(log_path, "precheck skipped: raw audio not found")
        return RECORDING_STATUS_READY, None

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
        _append_step_log(
            log_path,
            f"quarantined reason={precheck.quarantine_reason}",
        )
        return RECORDING_STATUS_QUARANTINE, precheck.quarantine_reason

    diariser = _build_diariser(precheck.duration_sec)
    asyncio.run(
        run_pipeline(
            audio_path=audio_path,
            cfg=pipeline_settings,
            llm=_NoopLLMClient(),
            diariser=diariser,
            recording_id=recording_id,
            precheck=precheck,
        )
    )
    _append_step_log(log_path, "pipeline artifacts generated")
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
