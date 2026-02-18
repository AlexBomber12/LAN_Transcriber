from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Iterable, List, Protocol

from pydantic_settings import BaseSettings

from . import normalizer
from .aliases import ALIAS_PATH, load_aliases as _load_aliases, save_aliases as _save_aliases
from .artifacts import (
    atomic_write_json,
    atomic_write_text,
    build_recording_artifacts,
    stage_raw_audio,
)
from .llm_client import LLMClient
from .metrics import error_rate_total, p95_latency_seconds
from .models import SpeakerSegment, TranscriptResult
from .runtime_paths import (
    default_recordings_root,
    default_tmp_root,
    default_unknown_dir,
    default_voices_dir,
)


class Diariser(Protocol):
    """Minimal interface for speaker diarisation."""

    async def __call__(self, audio_path: Path): ...


class Settings(BaseSettings):
    """Runtime configuration for the transcription pipeline."""

    speaker_db: Path = ALIAS_PATH
    recordings_root: Path = default_recordings_root()
    voices_dir: Path = default_voices_dir()
    unknown_dir: Path = default_unknown_dir()
    tmp_root: Path = default_tmp_root()
    llm_model: str = "llama3:8b"
    embed_threshold: float = 0.65
    merge_similar: float = 0.9

    class Config:
        env_prefix = "LAN_"


def _merge_similar(
    lines: Iterable[str], threshold: float
) -> List[str]:  # pragma: no cover - simple heuristic
    out: List[str] = []
    for line in lines:
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        sim = sum(a == b for a, b in zip(prev, line)) / max(len(prev), len(line))
        if sim >= threshold:
            continue
        out.append(line)
    return out


def _sentiment_score(text: str) -> int:  # pragma: no cover - trivial wrapper
    from transformers import pipeline as hf_pipeline

    sent = hf_pipeline("sentiment-analysis")(text[:4000])[0]
    if sent["label"] == "positive":
        return int(sent["score"] * 100)
    if sent["label"] == "negative":
        return int((1 - sent["score"]) * 100)
    return 50


def refresh_aliases(result: TranscriptResult, alias_path: Path = ALIAS_PATH) -> None:
    """Reload aliases from disk and update ``result`` in-place."""
    aliases = _load_aliases(alias_path)
    result.speakers = sorted({aliases.get(s.speaker, s.speaker) for s in result.segments})


def _default_recording_id(audio_path: Path) -> str:
    stem = audio_path.stem.strip()
    return stem or "recording"


async def run_pipeline(
    audio_path: Path,
    cfg: Settings,
    llm: LLMClient,
    diariser: Diariser,
    recording_id: str | None = None,
) -> TranscriptResult:
    """Transcribe ``audio_path`` and return a structured result."""
    start = time.perf_counter()
    import whisperx

    artifact_paths = build_recording_artifacts(
        cfg.recordings_root,
        recording_id=recording_id or _default_recording_id(audio_path),
        audio_ext=audio_path.suffix,
    )
    stage_raw_audio(audio_path, artifact_paths.raw_audio_path)
    atomic_write_json(
        artifact_paths.metrics_json_path,
        {
            "status": "placeholder",
            "version": 1,
        },
    )

    aliases = _load_aliases(cfg.speaker_db)

    def _asr() -> tuple[List[dict], dict]:
        segments, info = whisperx.transcribe(
            str(audio_path), vad_filter=True, language="auto"
        )
        return list(segments), info

    asr_task = asyncio.to_thread(_asr)
    diar_task = diariser(audio_path)
    asr_result, diarization = await asyncio.gather(asr_task, diar_task)
    segments, _info = asr_result

    asr_text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
    clean_text = normalizer.dedup(asr_text)
    if not clean_text:
        atomic_write_text(artifact_paths.transcript_txt_path, "")
        atomic_write_json(artifact_paths.transcript_json_path, {"recording_id": artifact_paths.recording_id, "speakers": [], "text": ""})
        atomic_write_json(artifact_paths.segments_json_path, [])
        atomic_write_json(
            artifact_paths.summary_json_path,
            {
                "friendly": 0,
                "model": cfg.llm_model,
                "summary": "No speech detected",
            },
        )
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(
            summary="No speech detected",
            body="",
            friendly=0,
            speakers=[],
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=[],
            segments=[],
        )

    lines: List[str] = []
    speakers: List[str] = []
    segs: List[SpeakerSegment] = []
    for seg, label in diarization.itertracks(yield_label=True):
        text = whisperx.utils.get_segments(
            {"segments": segments}, seg.start, seg.end
        ).strip()
        if not text:
            continue
        segs.append(SpeakerSegment(start=seg.start, end=seg.end, speaker=label, text=text))
        name = aliases.get(label, label)
        if label not in aliases:
            aliases[label] = name
        speakers.append(name)
        lines.append(f"[{seg.start:.2f}–{seg.end:.2f}] **{name}:** {text}")

    if not speakers:
        fallback = aliases.get("S1", "S1")
        aliases.setdefault("S1", fallback)
        speakers.append(fallback)

    _save_aliases(aliases, cfg.speaker_db)
    lines = _merge_similar(lines, cfg.merge_similar)
    body = clean_text
    friendly = _sentiment_score(body)

    sys_prompt = (
        "You are an assistant who writes concise 5-8 bullet summaries of any audio transcript. "
        "Return only the list without extra explanation."
    )
    user_prompt = f"{sys_prompt}\n\nTRANSCRIPT:\n{body}\n\nSUMMARY:"
    try:
        msg = await llm.generate(
            system_prompt=sys_prompt, user_prompt=user_prompt, model=cfg.llm_model
        )
        summary = msg.get("content", "") if isinstance(msg, dict) else str(msg)

        serialised_segments = [segment.model_dump() for segment in segs]
        atomic_write_text(artifact_paths.transcript_txt_path, body)
        atomic_write_json(
            artifact_paths.transcript_json_path,
            {
                "recording_id": artifact_paths.recording_id,
                "speaker_lines": lines,
                "speakers": sorted(set(speakers)),
                "text": body,
            },
        )
        atomic_write_json(artifact_paths.segments_json_path, serialised_segments)
        atomic_write_json(
            artifact_paths.summary_json_path,
            {
                "friendly": friendly,
                "model": cfg.llm_model,
                "summary": summary,
            },
        )

        result = TranscriptResult(
            summary=summary,
            body=body,
            friendly=friendly,
            speakers=sorted(set(speakers)),
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=[],
            segments=segs,
        )
    except Exception:
        error_rate_total.inc()
        atomic_write_json(
            artifact_paths.summary_json_path,
            {
                "friendly": friendly,
                "model": cfg.llm_model,
                "summary": "",
                "status": "failed",
            },
        )
        raise
    finally:
        p95_latency_seconds.observe(time.perf_counter() - start)

    return result


__all__ = ["run_pipeline", "Settings", "Diariser", "refresh_aliases"]
