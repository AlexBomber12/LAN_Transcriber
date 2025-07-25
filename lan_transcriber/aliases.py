from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

ALIAS_PATH = Path(__file__).resolve().parent / "speaker_bank.yaml"


def load_aliases(path: Path = ALIAS_PATH) -> Dict[str, str]:
    """Load speaker aliases from ``path`` if it exists."""
    if path.exists():
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    return {}


def save_aliases(aliases: Dict[str, str], path: Path = ALIAS_PATH) -> None:
    """Persist ``aliases`` to ``path`` as YAML."""
    path.write_text(yaml.safe_dump(dict(sorted(aliases.items()))))


__all__ = ["ALIAS_PATH", "load_aliases", "save_aliases"]
