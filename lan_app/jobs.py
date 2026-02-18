from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordingJob:
    """In-memory job envelope until DB-backed queue lands."""

    recording_id: str
    audio_path: Path


__all__ = ["RecordingJob"]
