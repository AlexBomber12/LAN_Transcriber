from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings

from lan_app.constants import DEFAULT_RQ_QUEUE_NAME
from lan_transcriber.pipeline_steps.diarization_quality import (
    DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
    DEFAULT_DIALOG_RETRY_MIN_TURNS,
    DEFAULT_DIARIZATION_FLICKER_MAX_CONSECUTIVE,
    DEFAULT_DIARIZATION_FLICKER_MIN_SECONDS,
    DEFAULT_DIARIZATION_MERGE_GAP_SECONDS,
    DEFAULT_DIARIZATION_MIN_TURN_SECONDS,
)
from lan_transcriber.pipeline_steps.speaker_merge import (
    DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS,
    DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD,
    DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
)
from lan_transcriber.pipeline_steps.speaker_turns import (
    DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC,
    DEFAULT_SPEAKER_TURN_MIN_WORDS,
    DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC,
)
from lan_transcriber.runtime_paths import default_data_root, default_recordings_root

from .diarization_loader import DEFAULT_DIARIZATION_MODEL_ID

_DEV_DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
_DEV_DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_UPLOAD_CAPTURE_TIMEZONE = "Europe/Rome"
_LLM_MODEL_REQUIRED_ERROR = (
    "LLM_MODEL is required. Set it in .env (e.g., LLM_MODEL=gpt-oss:120b)."
)
_logger = logging.getLogger(__name__)


def _default_metrics_snapshot_path() -> Path:
    return default_data_root() / "metrics.snap"


def _normalize_optional_env(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class AppSettings(BaseSettings):
    """App-layer runtime settings."""

    lan_env: Literal["dev", "staging", "prod"] = Field(
        default="dev",
        validation_alias=AliasChoices("LAN_ENV"),
    )
    data_root: Path = default_data_root()
    recordings_root: Path = default_recordings_root()
    metrics_snapshot_path: Path = Field(
        default_factory=_default_metrics_snapshot_path,
        validation_alias=AliasChoices(
            "LAN_PROM_SNAPSHOT_PATH",
            "LAN_METRICS_SNAPSHOT_PATH",
            "PROM_SNAPSHOT_PATH",
        ),
    )
    db_path: Path = default_data_root() / "db" / "app.db"
    sqlite_busy_timeout_ms: int = Field(
        default=30000,
        ge=0,
        validation_alias=AliasChoices(
            "LAN_SQLITE_BUSY_TIMEOUT_MS",
            "SQLITE_BUSY_TIMEOUT_MS",
        ),
    )
    redis_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LAN_REDIS_URL", "REDIS_URL"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_BASE_URL"),
    )
    llm_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_MODEL", "LAN_LLM_MODEL"),
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=256,
        validation_alias=AliasChoices("llm_max_tokens", "LLM_MAX_TOKENS"),
    )
    llm_max_tokens_retry: int = Field(
        default=2048,
        ge=256,
        validation_alias=AliasChoices("llm_max_tokens_retry", "LLM_MAX_TOKENS_RETRY"),
    )
    llm_chunk_max_chars: int = Field(
        default=4500,
        ge=1,
        validation_alias=AliasChoices(
            "llm_chunk_max_chars",
            "LAN_LLM_CHUNK_MAX_CHARS",
            "LLM_CHUNK_MAX_CHARS",
        ),
    )
    llm_chunk_overlap_chars: int = Field(
        default=300,
        ge=0,
        validation_alias=AliasChoices(
            "llm_chunk_overlap_chars",
            "LAN_LLM_CHUNK_OVERLAP_CHARS",
            "LLM_CHUNK_OVERLAP_CHARS",
        ),
    )
    llm_chunk_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "llm_chunk_timeout_seconds",
            "LAN_LLM_CHUNK_TIMEOUT_SECONDS",
            "LLM_CHUNK_TIMEOUT_SECONDS",
        ),
    )
    llm_chunk_split_min_chars: int = Field(
        default=1200,
        ge=64,
        validation_alias=AliasChoices(
            "llm_chunk_split_min_chars",
            "LAN_LLM_CHUNK_SPLIT_MIN_CHARS",
            "LLM_CHUNK_SPLIT_MIN_CHARS",
        ),
    )
    llm_chunk_split_max_depth: int = Field(
        default=2,
        ge=0,
        validation_alias=AliasChoices(
            "llm_chunk_split_max_depth",
            "LAN_LLM_CHUNK_SPLIT_MAX_DEPTH",
            "LLM_CHUNK_SPLIT_MAX_DEPTH",
        ),
    )
    llm_long_transcript_threshold_chars: int = Field(
        default=4500,
        ge=1,
        validation_alias=AliasChoices(
            "llm_long_transcript_threshold_chars",
            "LAN_LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS",
            "LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS",
        ),
    )
    llm_merge_max_tokens: int | None = Field(
        default=None,
        ge=256,
        validation_alias=AliasChoices(
            "llm_merge_max_tokens",
            "LAN_LLM_MERGE_MAX_TOKENS",
            "LLM_MERGE_MAX_TOKENS",
        ),
    )
    rq_queue_name: str = Field(
        default=DEFAULT_RQ_QUEUE_NAME,
        validation_alias=AliasChoices("LAN_RQ_QUEUE_NAME", "RQ_QUEUE_NAME"),
    )
    rq_worker_burst: bool = Field(
        default=False,
        validation_alias=AliasChoices("LAN_RQ_WORKER_BURST", "RQ_WORKER_BURST"),
    )
    rq_job_timeout_seconds: int = Field(
        default=7200,
        ge=1,
        validation_alias=AliasChoices(
            "LAN_RQ_JOB_TIMEOUT_SECONDS",
            "RQ_JOB_TIMEOUT_SECONDS",
        ),
    )
    stop_grace_seconds: float = Field(
        default=5.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "stop_grace_seconds",
            "LAN_STOP_GRACE_SECONDS",
            "STOP_GRACE_SECONDS",
        ),
    )
    max_job_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices(
            "LAN_MAX_JOB_ATTEMPTS",
            "MAX_JOB_ATTEMPTS",
        ),
    )
    stuck_job_seconds: int = Field(
        default=7200,
        ge=1,
        validation_alias=AliasChoices(
            "LAN_STUCK_JOB_SECONDS",
            "STUCK_JOB_SECONDS",
        ),
    )
    reaper_interval_seconds: int = Field(
        default=300,
        ge=1,
        validation_alias=AliasChoices(
            "LAN_REAPER_INTERVAL_SECONDS",
            "REAPER_INTERVAL_SECONDS",
        ),
    )

    routing_auto_select_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "ROUTING_AUTO_SELECT_THRESHOLD",
            "LAN_ROUTING_AUTO_SELECT_THRESHOLD",
        ),
    )
    quarantine_retention_days: int = Field(
        default=7,
        ge=1,
        le=3650,
        validation_alias=AliasChoices(
            "QUARANTINE_RETENTION_DAYS",
            "LAN_QUARANTINE_RETENTION_DAYS",
        ),
    )
    api_bearer_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LAN_API_BEARER_TOKEN",
            "API_BEARER_TOKEN",
        ),
    )
    upload_max_bytes: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("UPLOAD_MAX_BYTES"),
    )
    upload_capture_timezone: str = Field(
        default=_DEFAULT_UPLOAD_CAPTURE_TIMEZONE,
        validation_alias=AliasChoices(
            "LAN_UPLOAD_CAPTURE_TIMEZONE",
            "UPLOAD_CAPTURE_TIMEZONE",
        ),
    )
    calendar_expand_past_days: int = Field(
        default=30,
        ge=0,
        validation_alias=AliasChoices(
            "LAN_CALENDAR_EXPAND_PAST_DAYS",
            "CALENDAR_EXPAND_PAST_DAYS",
        ),
    )
    calendar_expand_future_days: int = Field(
        default=180,
        ge=1,
        validation_alias=AliasChoices(
            "LAN_CALENDAR_EXPAND_FUTURE_DAYS",
            "CALENDAR_EXPAND_FUTURE_DAYS",
        ),
    )
    calendar_fetch_timeout_seconds: float = Field(
        default=15.0,
        gt=0.0,
        le=120.0,
        validation_alias=AliasChoices(
            "LAN_CALENDAR_FETCH_TIMEOUT_SECONDS",
            "CALENDAR_FETCH_TIMEOUT_SECONDS",
        ),
    )
    calendar_fetch_max_bytes: int = Field(
        default=2_000_000,
        ge=1024,
        validation_alias=AliasChoices(
            "LAN_CALENDAR_FETCH_MAX_BYTES",
            "CALENDAR_FETCH_MAX_BYTES",
        ),
    )
    calendar_fetch_max_redirects: int = Field(
        default=5,
        ge=0,
        le=20,
        validation_alias=AliasChoices(
            "LAN_CALENDAR_FETCH_MAX_REDIRECTS",
            "CALENDAR_FETCH_MAX_REDIRECTS",
        ),
    )
    diarization_model_id: str = Field(
        default=DEFAULT_DIARIZATION_MODEL_ID,
        validation_alias=AliasChoices("LAN_DIARIZATION_MODEL_ID"),
    )
    asr_device: str = Field(
        default="auto",
        validation_alias=AliasChoices("LAN_ASR_DEVICE"),
    )
    diarization_device: str = Field(
        default="auto",
        validation_alias=AliasChoices("LAN_DIARIZATION_DEVICE"),
    )
    gpu_scheduler_mode: Literal["auto", "sequential", "parallel"] = Field(
        default="auto",
        validation_alias=AliasChoices("LAN_GPU_SCHEDULER_MODE"),
    )
    diarization_profile: Literal["auto", "dialog", "meeting"] = "auto"
    diarization_min_speakers: int | None = Field(default=None, ge=1)
    diarization_max_speakers: int | None = Field(default=None, ge=1)
    diarization_dialog_retry_min_duration_seconds: float = Field(
        default=DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
        ge=0.0,
    )
    diarization_dialog_retry_min_turns: int = Field(
        default=DEFAULT_DIALOG_RETRY_MIN_TURNS,
        ge=1,
    )
    diarization_merge_gap_seconds: float = Field(
        default=DEFAULT_DIARIZATION_MERGE_GAP_SECONDS,
        ge=0.0,
    )
    diarization_min_turn_seconds: float = Field(
        default=DEFAULT_DIARIZATION_MIN_TURN_SECONDS,
        ge=0.0,
    )
    diarization_flicker_min_seconds: float = Field(
        default=DEFAULT_DIARIZATION_FLICKER_MIN_SECONDS,
        ge=0.0,
        validation_alias=AliasChoices(
            "LAN_DIARIZATION_FLICKER_MIN_SECONDS",
            "DIARIZATION_FLICKER_MIN_SECONDS",
        ),
    )
    diarization_flicker_max_consecutive: int = Field(
        default=DEFAULT_DIARIZATION_FLICKER_MAX_CONSECUTIVE,
        ge=0,
        validation_alias=AliasChoices(
            "LAN_DIARIZATION_FLICKER_MAX_CONSECUTIVE",
            "DIARIZATION_FLICKER_MAX_CONSECUTIVE",
        ),
    )
    speaker_turn_merge_gap_sec: float = Field(
        default=DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC,
        ge=0.0,
        validation_alias=AliasChoices(
            "LAN_SPEAKER_TURN_MERGE_GAP_SEC",
            "SPEAKER_TURN_MERGE_GAP_SEC",
        ),
    )
    speaker_turn_short_merge_gap_sec: float = Field(
        default=DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC,
        ge=0.0,
        validation_alias=AliasChoices(
            "LAN_SPEAKER_TURN_SHORT_MERGE_GAP_SEC",
            "SPEAKER_TURN_SHORT_MERGE_GAP_SEC",
        ),
    )
    speaker_turn_min_words: int = Field(
        default=DEFAULT_SPEAKER_TURN_MIN_WORDS,
        ge=0,
        validation_alias=AliasChoices(
            "LAN_SPEAKER_TURN_MIN_WORDS",
            "SPEAKER_TURN_MIN_WORDS",
        ),
    )
    speaker_merge_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "LAN_SPEAKER_MERGE_ENABLED",
            "SPEAKER_MERGE_ENABLED",
        ),
    )
    speaker_merge_similarity_threshold: float = Field(
        default=DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "LAN_SPEAKER_MERGE_SIMILARITY_THRESHOLD",
            "SPEAKER_MERGE_SIMILARITY_THRESHOLD",
        ),
    )
    speaker_merge_no_overlap_similarity_threshold: float = Field(
        default=DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "LAN_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD",
            "SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD",
        ),
    )
    speaker_merge_max_segments: int = Field(
        default=DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS,
        ge=1,
        validation_alias=AliasChoices(
            "LAN_SPEAKER_MERGE_MAX_SEGMENTS",
            "SPEAKER_MERGE_MAX_SEGMENTS",
        ),
    )
    vad_method: Literal["silero", "pyannote"] = "silero"

    @model_validator(mode="after")
    def validate_runtime_environment(self) -> "AppSettings":
        self.redis_url = _normalize_optional_env(self.redis_url)
        self.llm_base_url = _normalize_optional_env(self.llm_base_url)
        self.llm_model = _normalize_optional_env(self.llm_model)
        timezone_name = (
            _normalize_optional_env(self.upload_capture_timezone)
            or _DEFAULT_UPLOAD_CAPTURE_TIMEZONE
        )
        try:
            self.upload_capture_timezone = ZoneInfo(timezone_name).key
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                "UPLOAD_CAPTURE_TIMEZONE must be a valid IANA timezone "
                f"(got {timezone_name!r})"
            ) from exc
        self.diarization_model_id = (
            self.diarization_model_id.strip() or DEFAULT_DIARIZATION_MODEL_ID
        )
        if self.llm_model is None:
            raise ValueError(_LLM_MODEL_REQUIRED_ERROR)

        if self.lan_env == "dev":
            if self.redis_url is None:
                _logger.warning(
                    "LAN_REDIS_URL is not set in LAN_ENV=dev; defaulting to %s",
                    _DEV_DEFAULT_REDIS_URL,
                )
                self.redis_url = _DEV_DEFAULT_REDIS_URL
            if self.llm_base_url is None:
                _logger.warning(
                    "LLM_BASE_URL is not set in LAN_ENV=dev; defaulting to %s",
                    _DEV_DEFAULT_LLM_BASE_URL,
                )
                self.llm_base_url = _DEV_DEFAULT_LLM_BASE_URL
            return self

        missing: list[str] = []
        if self.redis_url is None:
            missing.append("LAN_REDIS_URL")
        if self.llm_base_url is None:
            missing.append("LLM_BASE_URL")
        if missing:
            vars_text = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variable(s) for LAN_ENV={self.lan_env}: {vars_text}"
            )
        return self

    def upload_capture_tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.upload_capture_timezone)

    class Config:
        env_prefix = "LAN_"


__all__ = ["AppSettings"]
