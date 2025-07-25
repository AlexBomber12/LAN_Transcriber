"""LAN Transcriber package."""

from .llm_client import LLMClient, generate
from .pipeline import Settings, run_pipeline, Diariser
from .models import SpeakerSegment, TranscriptResult
from .normalizer import dedup
from . import aliases, metrics
from .aliases import *
from .metrics import *

__all__ = [
    "LLMClient",
    "generate",
    "Settings",
    "run_pipeline",
    "Diariser",
    "SpeakerSegment",
    "TranscriptResult",
    "dedup",
    *aliases.__all__,
    *metrics.__all__,
]
