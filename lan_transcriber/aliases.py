from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import yaml

from .runtime_paths import default_alias_path

ALIAS_PATH = Path(os.getenv("LAN_SPEAKER_DB", str(default_alias_path())))


def load_aliases(path: Path = ALIAS_PATH) -> Dict[str, str]:
    """Load speaker aliases from ``path`` if it exists."""
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    return {}


def save_aliases(aliases: Dict[str, str], path: Path = ALIAS_PATH) -> None:
    """Persist ``aliases`` to ``path`` as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(sorted(aliases.items()))), encoding="utf-8")


__all__ = ["ALIAS_PATH", "load_aliases", "save_aliases"]
