from __future__ import annotations

from pathlib import Path

from .config import AppSettings


class DBNotReadyError(RuntimeError):
    """Raised while DB integration is still intentionally stubbed."""


def db_path(settings: AppSettings | None = None) -> Path:
    cfg = settings or AppSettings()
    return cfg.db_path


def connect(_settings: AppSettings | None = None) -> None:
    """Stub connection hook reserved for PR-DB-QUEUE-01."""
    raise DBNotReadyError("DB integration is scheduled for PR-DB-QUEUE-01.")


__all__ = ["DBNotReadyError", "db_path", "connect"]
