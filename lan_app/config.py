from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

from lan_app.constants import DEFAULT_RQ_QUEUE_NAME
from lan_transcriber.runtime_paths import default_data_root, default_recordings_root


def _default_metrics_snapshot_path() -> Path:
    return default_data_root() / "metrics.snap"


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

    class Config:
        env_prefix = "LAN_"


__all__ = ["AppSettings"]
