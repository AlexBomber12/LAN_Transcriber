from __future__ import annotations

import os
from pathlib import Path


def default_data_root() -> Path:
    """Return runtime data root, preferring /data with local fallback."""
    configured = os.getenv("LAN_DATA_ROOT")
    if configured:
        return Path(configured)

    data_root = Path("/data")
    if data_root.exists() and os.access(data_root, os.W_OK):
        return data_root
    return Path("data")


def default_alias_path() -> Path:
    return default_data_root() / "db" / "speaker_bank.yaml"


def default_recordings_root() -> Path:
    return default_data_root() / "recordings"


def default_voices_dir() -> Path:
    return default_data_root() / "voices"


def default_unknown_dir() -> Path:
    return default_recordings_root() / "unknown"


def default_tmp_root() -> Path:
    return default_data_root() / "tmp"

