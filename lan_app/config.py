from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

from lan_transcriber.runtime_paths import default_data_root


def _default_metrics_snapshot_path() -> Path:
    legacy = os.getenv("PROM_SNAPSHOT_PATH")
    if legacy:
        return Path(legacy)
    return default_data_root() / "metrics.snap"


class AppSettings(BaseSettings):
    """App-layer runtime settings."""

    data_root: Path = default_data_root()
    metrics_snapshot_path: Path = _default_metrics_snapshot_path()
    db_path: Path = default_data_root() / "db" / "app.db"

    class Config:
        env_prefix = "LAN_"


__all__ = ["AppSettings"]
