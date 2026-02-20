from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordingArtifacts:
    """Canonical v1 artifact layout for a processed recording."""

    recording_id: str
    root_dir: Path
    raw_audio_path: Path
    transcript_json_path: Path
    transcript_txt_path: Path
    segments_json_path: Path
    speaker_turns_json_path: Path
    snippets_dir: Path
    summary_json_path: Path
    metrics_json_path: Path
    logs_dir: Path


def _normalize_recording_id(recording_id: str) -> str:
    slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in recording_id.strip()
    )
    slug = slug.strip("-_").lower()
    return slug or "recording"


def _normalize_audio_ext(audio_ext: str | None) -> str:
    if not audio_ext:
        return ".bin"
    return audio_ext if audio_ext.startswith(".") else f".{audio_ext}"


def build_recording_artifacts(
    recordings_root: Path,
    recording_id: str,
    audio_ext: str | None = None,
) -> RecordingArtifacts:
    """Create and return canonical artifact paths for ``recording_id``."""
    rid = _normalize_recording_id(recording_id)
    ext = _normalize_audio_ext(audio_ext)
    root_dir = recordings_root / rid
    raw_dir = root_dir / "raw"
    derived_dir = root_dir / "derived"
    snippets_dir = derived_dir / "snippets"
    logs_dir = root_dir / "logs"

    for directory in (raw_dir, derived_dir, snippets_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return RecordingArtifacts(
        recording_id=rid,
        root_dir=root_dir,
        raw_audio_path=raw_dir / f"audio{ext}",
        transcript_json_path=derived_dir / "transcript.json",
        transcript_txt_path=derived_dir / "transcript.txt",
        segments_json_path=derived_dir / "segments.json",
        speaker_turns_json_path=derived_dir / "speaker_turns.json",
        snippets_dir=snippets_dir,
        summary_json_path=derived_dir / "summary.json",
        metrics_json_path=derived_dir / "metrics.json",
        logs_dir=logs_dir,
    )


def stage_raw_audio(source: Path, destination: Path) -> Path:
    """Copy source audio to canonical raw artifact location."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.exists() and source.resolve() == destination.resolve():
            return destination
    except FileNotFoundError:
        pass
    shutil.copy2(source, destination)
    return destination


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_write_bytes(path, f"{payload}\n".encode("utf-8"))


__all__ = [
    "RecordingArtifacts",
    "build_recording_artifacts",
    "stage_raw_audio",
    "atomic_write_text",
    "atomic_write_json",
]
