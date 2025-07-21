from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel


class SpeakerSegment(BaseModel):
    """A piece of transcript belonging to a single speaker."""

    start: float
    end: float
    speaker: str
    text: str


class TranscriptResult(BaseModel):
    """Full result of a transcription pass."""

    summary: str
    body: str
    friendly: int
    speakers: List[str]
    summary_path: Path
    body_path: Path
    unknown_chunks: List[Path]


__all__ = ["SpeakerSegment", "TranscriptResult"]
