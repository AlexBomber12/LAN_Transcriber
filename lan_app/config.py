from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

from lan_app.constants import DEFAULT_RQ_QUEUE_NAME
from lan_transcriber.runtime_paths import default_data_root, default_recordings_root


def _default_metrics_snapshot_path() -> Path:
    return default_data_root() / "metrics.snap"


def _default_msal_cache_path() -> Path:
    return default_data_root() / "auth" / "msal_cache.bin"


class AppSettings(BaseSettings):
    """App-layer runtime settings."""

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
    redis_url: str = Field(
        default="redis://redis:6379/0",
        validation_alias=AliasChoices("LAN_REDIS_URL", "REDIS_URL"),
    )
    rq_queue_name: str = Field(
        default=DEFAULT_RQ_QUEUE_NAME,
        validation_alias=AliasChoices("LAN_RQ_QUEUE_NAME", "RQ_QUEUE_NAME"),
    )
    rq_worker_burst: bool = Field(
        default=False,
        validation_alias=AliasChoices("LAN_RQ_WORKER_BURST", "RQ_WORKER_BURST"),
    )

    # Google Drive ingest
    gdrive_sa_json_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("GDRIVE_SA_JSON_PATH", "LAN_GDRIVE_SA_JSON_PATH"),
    )
    gdrive_inbox_folder_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GDRIVE_INBOX_FOLDER_ID", "LAN_GDRIVE_INBOX_FOLDER_ID"
        ),
    )
    gdrive_poll_interval_seconds: int = Field(
        default=60,
        validation_alias=AliasChoices(
            "GDRIVE_POLL_INTERVAL_SECONDS", "LAN_GDRIVE_POLL_INTERVAL_SECONDS"
        ),
    )

    # Microsoft Graph delegated auth (Device Code Flow)
    ms_tenant_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MS_TENANT_ID", "LAN_MS_TENANT_ID"),
    )
    ms_client_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MS_CLIENT_ID", "LAN_MS_CLIENT_ID"),
    )
    ms_scopes: str = Field(
        default="offline_access User.Read Notes.ReadWrite Calendars.Read",
        validation_alias=AliasChoices("MS_SCOPES", "LAN_MS_SCOPES"),
    )
    msal_cache_path: Path = Field(
        default_factory=_default_msal_cache_path,
        validation_alias=AliasChoices("MSAL_CACHE_PATH", "LAN_MSAL_CACHE_PATH"),
    )
    calendar_match_window_minutes: int = Field(
        default=45,
        ge=5,
        le=24 * 60,
        validation_alias=AliasChoices(
            "CALENDAR_MATCH_WINDOW_MINUTES",
            "LAN_CALENDAR_MATCH_WINDOW_MINUTES",
        ),
    )
    calendar_auto_match_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "CALENDAR_AUTO_MATCH_THRESHOLD",
            "LAN_CALENDAR_AUTO_MATCH_THRESHOLD",
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

    @property
    def ms_scopes_list(self) -> list[str]:
        return [s.strip() for s in self.ms_scopes.replace(",", " ").split() if s.strip()]

    class Config:
        env_prefix = "LAN_"


__all__ = ["AppSettings"]
