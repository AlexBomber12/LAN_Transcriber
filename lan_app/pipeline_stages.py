from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import AppSettings

PIPELINE_STAGE_STATUS_PENDING = "pending"
PIPELINE_STAGE_STATUS_RUNNING = "running"
PIPELINE_STAGE_STATUS_COMPLETED = "completed"
PIPELINE_STAGE_STATUS_FAILED = "failed"
PIPELINE_STAGE_STATUS_SKIPPED = "skipped"
PIPELINE_STAGE_STATUS_CANCELLED = "cancelled"

PIPELINE_STAGE_STATUSES = (
    PIPELINE_STAGE_STATUS_PENDING,
    PIPELINE_STAGE_STATUS_RUNNING,
    PIPELINE_STAGE_STATUS_COMPLETED,
    PIPELINE_STAGE_STATUS_FAILED,
    PIPELINE_STAGE_STATUS_SKIPPED,
    PIPELINE_STAGE_STATUS_CANCELLED,
)

PIPELINE_STAGE_DONE_STATUSES = frozenset(
    {
        PIPELINE_STAGE_STATUS_COMPLETED,
        PIPELINE_STAGE_STATUS_SKIPPED,
    }
)


@dataclass(frozen=True)
class PipelineStageDefinition:
    name: str
    order: int
    progress: float
    label: str


PIPELINE_STAGE_DEFINITIONS = (
    PipelineStageDefinition("sanitize_audio", 10, 0.02, "Sanitize Audio"),
    PipelineStageDefinition("precheck", 20, 0.05, "Precheck"),
    PipelineStageDefinition("calendar_refresh", 30, 0.08, "Calendar Refresh"),
    PipelineStageDefinition("asr", 40, 0.35, "ASR"),
    PipelineStageDefinition("diarization", 50, 0.60, "Diarization"),
    PipelineStageDefinition("language_analysis", 60, 0.75, "Language Analysis"),
    PipelineStageDefinition("speaker_turns", 70, 0.80, "Speaker Turns"),
    PipelineStageDefinition("snippet_export", 75, 0.84, "Snippet Export"),
    PipelineStageDefinition("llm_extract", 80, 0.90, "LLM Extract"),
    PipelineStageDefinition("export_artifacts", 90, 0.95, "Export Artifacts"),
    PipelineStageDefinition("metrics", 100, 0.98, "Metrics"),
    PipelineStageDefinition("routing", 110, 0.99, "Routing"),
)

PIPELINE_STAGE_BY_NAME = {
    stage.name: stage for stage in PIPELINE_STAGE_DEFINITIONS
}


def validate_pipeline_stage_name(stage_name: str) -> str:
    normalized = str(stage_name or "").strip()
    if normalized not in PIPELINE_STAGE_BY_NAME:
        allowed = ", ".join(stage.name for stage in PIPELINE_STAGE_DEFINITIONS)
        raise ValueError(f"Unsupported pipeline stage: {normalized} ({allowed})")
    return normalized


def validate_pipeline_stage_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized not in PIPELINE_STAGE_STATUSES:
        allowed = ", ".join(PIPELINE_STAGE_STATUSES)
        raise ValueError(f"Unsupported pipeline stage status: {normalized} ({allowed})")
    return normalized


def stage_order(stage_name: str) -> int:
    return PIPELINE_STAGE_BY_NAME[validate_pipeline_stage_name(stage_name)].order


def stage_progress(stage_name: str) -> float:
    return PIPELINE_STAGE_BY_NAME[validate_pipeline_stage_name(stage_name)].progress


def stage_label(stage_name: str) -> str:
    return PIPELINE_STAGE_BY_NAME[validate_pipeline_stage_name(stage_name)].label


def derived_dir(recording_id: str, *, settings: AppSettings | None = None) -> Path:
    cfg = settings or AppSettings()
    return cfg.recordings_root / recording_id / "derived"


def stage_artifact_paths(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, tuple[Path, ...]]:
    derived = derived_dir(recording_id, settings=settings)
    return {
        "sanitize_audio": (
            derived / "audio_sanitized.wav",
            derived / "audio_sanitize.json",
        ),
        "precheck": (derived / "precheck.json",),
        "calendar_refresh": (derived / "calendar_refresh.json",),
        "asr": (
            derived / "asr_segments.json",
            derived / "asr_info.json",
            derived / "asr_execution.json",
        ),
        "diarization": (
            derived / "diarization_segments.json",
            derived / "diarization_runtime.json",
            derived / "diarization_status.json",
        ),
        "language_analysis": (derived / "language_analysis.json",),
        "speaker_turns": (
            derived / "speaker_turns.json",
            derived / "segments.json",
            derived / "diarization_metadata.json",
        ),
        "snippet_export": (derived / "snippets_manifest.json",),
        "llm_extract": (derived / "summary.json",),
        "export_artifacts": (
            derived / "transcript.json",
            derived / "transcript.txt",
            derived / "summary.json",
            derived / "segments.json",
            derived / "speaker_turns.json",
        ),
        "metrics": (derived / "metrics.json",),
        "routing": (derived / "routing.json",),
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _validate_path_payload(path: Path) -> bool:
    if not path.exists():
        return False
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
        return payload is not None
    return path.is_file()


def validate_stage_artifacts(
    recording_id: str,
    *,
    stage_name: str,
    status: str,
    metadata: dict[str, Any] | None = None,
    settings: AppSettings | None = None,
) -> tuple[bool, str | None]:
    normalized_stage = validate_pipeline_stage_name(stage_name)
    normalized_status = validate_pipeline_stage_status(status)
    payload = metadata if isinstance(metadata, dict) else {}

    if normalized_status == PIPELINE_STAGE_STATUS_SKIPPED:
        skip_reason = str(payload.get("skip_reason") or "").strip()
        if skip_reason:
            return True, None
        return False, f"{normalized_stage} missing skip_reason metadata"

    if normalized_stage == "sanitize_audio" and bool(payload.get("raw_audio_missing")):
        return True, None
    if normalized_stage == "sanitize_audio":
        sanitize_json = stage_artifact_paths(recording_id, settings=settings)[normalized_stage][1]
        sanitize_payload = _load_json(sanitize_json)
        output_path = ""
        if isinstance(sanitize_payload, dict):
            output_path = str(sanitize_payload.get("output_path") or "").strip()
        if _validate_path_payload(sanitize_json) and output_path and Path(output_path).exists():
            return True, None

    for artifact_path in stage_artifact_paths(recording_id, settings=settings)[normalized_stage]:
        if _validate_path_payload(artifact_path):
            continue
        return (
            False,
            f"{normalized_stage} missing artifact {artifact_path.name}",
        )
    return True, None


__all__ = [
    "PIPELINE_STAGE_BY_NAME",
    "PIPELINE_STAGE_DEFINITIONS",
    "PIPELINE_STAGE_DONE_STATUSES",
    "PIPELINE_STAGE_STATUSES",
    "PIPELINE_STAGE_STATUS_CANCELLED",
    "PIPELINE_STAGE_STATUS_COMPLETED",
    "PIPELINE_STAGE_STATUS_FAILED",
    "PIPELINE_STAGE_STATUS_PENDING",
    "PIPELINE_STAGE_STATUS_RUNNING",
    "PIPELINE_STAGE_STATUS_SKIPPED",
    "PipelineStageDefinition",
    "derived_dir",
    "stage_artifact_paths",
    "stage_label",
    "stage_order",
    "stage_progress",
    "validate_pipeline_stage_name",
    "validate_pipeline_stage_status",
    "validate_stage_artifacts",
]
