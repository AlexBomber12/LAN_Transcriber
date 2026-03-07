from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import shutil
from typing import Any

from .config import AppSettings
from .constants import RECORDING_STATUS_QUARANTINE
from .db import delete_recording, list_recordings


class RecordingDeleteError(RuntimeError):
    """Raised when recording deletion cannot safely remove disk artifacts."""


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    normalised = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iter_recordings(
    *,
    settings: AppSettings,
    status: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        rows, total = list_recordings(
            settings=settings,
            status=status,
            limit=500,
            offset=offset,
        )
        if not rows:
            break
        out.extend(rows)
        offset += len(rows)
        if offset >= total:
            break
    return out


def _delete_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink(missing_ok=True)
        except TypeError:  # pragma: no cover - Python < 3.8 compatibility shim
            if path.exists():
                path.unlink()
    return True


def _recording_root_path(recording_id: str, settings: AppSettings) -> Path:
    raw_recording_id = str(recording_id or "").strip()
    if not raw_recording_id:
        raise RecordingDeleteError("Delete failed: recording id is required.")
    if raw_recording_id in {".", ".."}:
        raise RecordingDeleteError("Delete failed: invalid recording id.")
    if Path(raw_recording_id).parts != (raw_recording_id,):
        raise RecordingDeleteError("Delete failed: invalid recording id.")

    recordings_root = settings.recordings_root.resolve(strict=False)
    recording_root = recordings_root / raw_recording_id
    if recording_root.is_symlink():
        raise RecordingDeleteError("Delete failed: invalid recording path.")
    return recording_root


def _delete_path_strict(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    shutil.rmtree(path)
    return True


def delete_recording_with_artifacts(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> bool:
    cfg = settings or AppSettings()
    recording_root = _recording_root_path(recording_id, cfg)
    try:
        deleted = delete_recording(recording_id, settings=cfg)
    except sqlite3.Error as exc:
        raise RecordingDeleteError(
            f"Delete failed during database cleanup: {exc}"
        ) from exc
    if not deleted:
        return False

    try:
        if recording_root.exists() or recording_root.is_symlink():
            if recording_root.is_symlink() or recording_root.is_file():
                _delete_path_strict(recording_root)
            else:
                for child_name in ("raw", "derived", "logs"):
                    _delete_path_strict(recording_root / child_name)
                for entry in list(recording_root.iterdir()):
                    _delete_path_strict(entry)
                recording_root.rmdir()
    except OSError as exc:
        raise RecordingDeleteError(
            f"Delete failed during disk cleanup: {exc}"
        ) from exc

    return True


def run_retention_cleanup(
    *,
    settings: AppSettings | None = None,
) -> dict[str, int]:
    """Delete old quarantine recordings and stale temporary artifacts."""

    cfg = settings or AppSettings()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=cfg.quarantine_retention_days)

    summary = {
        "quarantine_recordings_deleted": 0,
        "quarantine_directories_deleted": 0,
        "tmp_entries_deleted": 0,
    }

    quarantined = _iter_recordings(settings=cfg, status=RECORDING_STATUS_QUARANTINE)
    for row in quarantined:
        recording_id = str(row.get("id") or "").strip()
        if not recording_id:
            continue
        updated_at = _parse_utc(row.get("updated_at")) or _parse_utc(row.get("created_at"))
        if updated_at is None or updated_at > cutoff:
            continue
        recording_root = cfg.recordings_root / recording_id
        if _delete_path(recording_root):
            summary["quarantine_directories_deleted"] += 1
        if delete_recording(recording_id, settings=cfg):
            summary["quarantine_recordings_deleted"] += 1

    tmp_root = cfg.data_root / "tmp"
    if tmp_root.exists() and tmp_root.is_dir():
        for entry in tmp_root.iterdir():
            try:
                modified = datetime.fromtimestamp(
                    entry.stat().st_mtime,
                    tz=timezone.utc,
                )
            except OSError:
                continue
            if modified > cutoff:
                continue
            if _delete_path(entry):
                summary["tmp_entries_deleted"] += 1

    return summary


__all__ = [
    "RecordingDeleteError",
    "delete_recording_with_artifacts",
    "run_retention_cleanup",
]
