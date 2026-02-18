from __future__ import annotations

from pathlib import Path

from lan_transcriber.llm_client import LLMClient
from lan_transcriber.models import TranscriptResult
from lan_transcriber.pipeline import Diariser, Settings, run_pipeline


async def process_recording(
    audio_path: Path,
    diariser: Diariser,
    recording_id: str | None = None,
    cfg: Settings | None = None,
    llm_client: LLMClient | None = None,
) -> TranscriptResult:
    """Run a single recording through the core pipeline."""
    settings = cfg or Settings()
    llm = llm_client or LLMClient()
    return await run_pipeline(
        audio_path=audio_path,
        cfg=settings,
        llm=llm,
        diariser=diariser,
        recording_id=recording_id,
    )


__all__ = ["process_recording"]
