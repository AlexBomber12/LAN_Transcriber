from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lan_transcriber.artifacts import atomic_write_json, atomic_write_text


@dataclass(frozen=True)
class LlmDebugArtifacts:
    derived_dir: Path
    raw_output: str
    extracted_payload: dict[str, Any] | None
    validation_error: dict[str, Any]


def write_json_artifact(path: Path, payload: Any) -> Path:
    atomic_write_json(path, payload)
    return path


def write_llm_debug_artifacts(spec: LlmDebugArtifacts) -> None:
    write_json_artifact(spec.derived_dir / "llm_extract.json", spec.extracted_payload or {})
    write_json_artifact(spec.derived_dir / "llm_validation_error.json", spec.validation_error)
    atomic_write_text(spec.derived_dir / "llm_raw.txt", spec.raw_output)


__all__ = ["LlmDebugArtifacts", "write_json_artifact", "write_llm_debug_artifacts"]
