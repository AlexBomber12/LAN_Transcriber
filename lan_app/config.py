from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

from lan_transcriber.runtime_paths import default_data_root


def _default_metrics_snapshot_path() -> Path:
    return default_data_root() / "metrics.snap"


class AppSettings(BaseSettings):
    """App-layer runtime settings."""

    data_root: Path = default_data_root()
    metrics_snapshot_path: Path = Field(
        default_factory=_default_metrics_snapshot_path,
        validation_alias=AliasChoices(
            "LAN_PROM_SNAPSHOT_PATH",
            "LAN_METRICS_SNAPSHOT_PATH",
            "PROM_SNAPSHOT_PATH",
        ),
    )
    db_path: Path = default_data_root() / "db" / "app.db"

    class Config:
        env_prefix = "LAN_"


__all__ = ["AppSettings"]
