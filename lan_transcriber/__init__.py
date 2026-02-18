"""LAN Transcriber package."""

from .aliases import ALIAS_PATH, load_aliases, save_aliases
from .artifacts import (
    RecordingArtifacts,
    atomic_write_json,
    atomic_write_text,
    build_recording_artifacts,
    stage_raw_audio,
)
from .llm_client import LLMClient, generate
from .metrics import (
    error_rate_total,
    llm_timeouts_total,
    p95_latency_seconds,
    write_metrics_snapshot,
)
from .models import SpeakerSegment, TranscriptResult
from .normalizer import dedup
from .pipeline import Diariser, Settings, run_pipeline

__all__ = [
    "ALIAS_PATH",
    "load_aliases",
    "save_aliases",
    "RecordingArtifacts",
    "build_recording_artifacts",
    "stage_raw_audio",
    "atomic_write_text",
    "atomic_write_json",
    "LLMClient",
    "generate",
    "p95_latency_seconds",
    "error_rate_total",
    "llm_timeouts_total",
    "write_metrics_snapshot",
    "Settings",
    "run_pipeline",
    "Diariser",
    "SpeakerSegment",
    "TranscriptResult",
    "dedup",
]
