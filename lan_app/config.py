from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings

from lan_app.constants import DEFAULT_RQ_QUEUE_NAME
from lan_transcriber.runtime_paths import default_data_root, default_recordings_root

_DEV_DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
_DEV_DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000"
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

    @model_validator(mode="after")
    def validate_runtime_environment(self) -> "AppSettings":
        self.redis_url = _normalize_optional_env(self.redis_url)
        self.llm_base_url = _normalize_optional_env(self.llm_base_url)

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

    class Config:
        env_prefix = "LAN_"


__all__ = ["AppSettings"]
