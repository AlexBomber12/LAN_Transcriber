from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Iterable, List, Literal, Protocol, Sequence

import httpx

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

from .. import normalizer
from ..aliases import ALIAS_PATH, load_aliases as _load_aliases, save_aliases as _save_aliases
from ..artifacts import atomic_write_json, atomic_write_text, build_recording_artifacts, stage_raw_audio
from ..compat.call_compat import (
    clear_last_supported_kwargs_call_details,
    call_with_supported_kwargs,
    filter_kwargs_for_callable,
    last_supported_kwargs_call_details,
)
from ..compat.pyannote_compat import patch_pyannote_inference_ignore_use_auth_token
from ..llm_chunking import (
    TranscriptChunk,
    build_compact_transcript,
    build_chunk_prompt,
    build_merge_prompt,
    merge_chunk_results,
    parse_chunk_extract,
    plan_compact_transcript_chunks,
    plan_transcript_chunks,
)
from ..llm_client import LLMClient
from ..metrics import error_rate_total, p95_latency_seconds
from ..models import SpeakerSegment, TranscriptResult
from ..native_fixups import ensure_ctranslate2_no_execstack
from ..torch_safe_globals import (
    diarization_safe_globals_for_torch_load,
    import_trusted_diarization_symbol,
    omegaconf_safe_globals_for_torch_load,
    unsupported_global_diarization_fqn_from_error,
    unsupported_global_omegaconf_fqn_from_error,
)
from .language import analyse_languages, resolve_target_summary_language, segment_language
from ..gpu_policy import (
    SchedulerDecision,
    collect_cuda_runtime_facts,
    cuda_memory_info,
    is_gpu_device,
    is_gpu_oom_error,
    normalize_device,
    resolve_effective_device,
    resolve_scheduler_decision,
)
from .precheck import PrecheckResult, run_precheck as _run_precheck
from .diarization_quality import (
    DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
    DEFAULT_DIALOG_RETRY_MIN_TURNS,
    DEFAULT_DIARIZATION_FLICKER_MAX_CONSECUTIVE,
    DEFAULT_DIARIZATION_FLICKER_MIN_SECONDS,
    DEFAULT_DIARIZATION_MERGE_GAP_SECONDS,
    DEFAULT_DIARIZATION_MIN_TURN_SECONDS,
    SpeakerTurnSmoothingResult,
    choose_dialog_retry_winner,
    classify_diarization_profile,
    filter_flickering_speakers,
    smooth_speaker_turns,
)
from .snippets import SnippetExportRequest, export_speaker_snippets, write_empty_snippets_manifest
from .speaker_merge import (
    DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS,
    DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
    EmbeddingModel,
    merge_similar_speakers,
)
from .speaker_turns import (
    DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC,
    DEFAULT_SPEAKER_TURN_MIN_WORDS,
    DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC,
    _diarization_segments,
    build_speaker_turns,
    merge_short_turns,
    normalise_asr_segments,
)
from .multilingual_asr import run_language_aware_asr
from .summary_builder import (
    _build_structured_summary_payload,
    build_structured_summary_prompts,
    build_summary_payload,
    build_summary_prompts,
)
from ..runtime_paths import default_recordings_root, default_tmp_root, default_unknown_dir, default_voices_dir
from ..utils import normalise_language_code, safe_float

_logger = logging.getLogger(__name__)
_whisperx_transcriber_state = threading.local()
_ASR_MODEL_CACHE: dict[tuple[str, str, str, str, Any], Any] = {}
_ASR_MODEL_CACHE_LOCK = threading.Lock()
_LLM_MODEL_REQUIRED_ERROR = (
    "LLM_MODEL is required. Set it in .env (e.g., LLM_MODEL=gpt-oss:120b)."
)
_LLM_TIMEOUT_SENTINEL = "**LLM timeout**"


@dataclass(frozen=True)
class _CachedAsrModel:
    model: Any
    compute_type: str


class Diariser(Protocol):
    """Minimal interface for speaker diarisation."""

    async def __call__(self, audio_path: Path): ...


class ChunkStateStore(Protocol):
    def list_states(self, *, chunk_group: str) -> list[dict[str, Any]]: ...

    def upsert_state(
        self,
        *,
        chunk_group: str,
        chunk_index: str,
        chunk_total: int,
        status: str,
        attempt: int = 0,
        started_at: str | None = None,
        finished_at: str | None = None,
        duration_ms: int | None = None,
        error_code: str | None = None,
        error_text: str | None = None,
        parent_chunk_index: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def mark_started(
        self,
        *,
        chunk_group: str,
        chunk_index: str,
        chunk_total: int,
        parent_chunk_index: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def mark_completed(
        self,
        *,
        chunk_group: str,
        chunk_index: str,
        chunk_total: int,
        parent_chunk_index: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def mark_failed(
        self,
        *,
        chunk_group: str,
        chunk_index: str,
        chunk_total: int,
        error_code: str | None = None,
        error_text: str | None = None,
        parent_chunk_index: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def mark_split(
        self,
        *,
        chunk_group: str,
        chunk_index: str,
        chunk_total: int,
        error_code: str | None = None,
        error_text: str | None = None,
        parent_chunk_index: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def clear_states(self, *, chunk_group: str | None = None) -> int: ...


@dataclass(frozen=True)
class _ChunkRuntime:
    chunk_id: str
    parent_chunk_id: str | None
    depth: int
    order_path: tuple[int, ...]
    text: str
    base_text: str
    overlap_prefix: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    source_kind: str = "root"

    def metadata_payload(self, *, transcript_hash: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "depth": self.depth,
            "order_path": list(self.order_path),
            "text": self.text,
            "base_text": self.base_text,
            "overlap_prefix": self.overlap_prefix,
            "transcript_hash": transcript_hash,
            "source_kind": self.source_kind,
        }
        if self.start_seconds is not None:
            payload["start_seconds"] = round(self.start_seconds, 3)
        if self.end_seconds is not None:
            payload["end_seconds"] = round(self.end_seconds, 3)
        return payload

    def plan_payload(
        self,
        *,
        position: int,
        total: int,
        transcript_hash: str,
        status: str | None,
        attempt: int | None,
        error_code: str | None,
    ) -> dict[str, Any]:
        chunk = TranscriptChunk(
            index=position,
            total=total,
            text=self.text,
            base_text=self.base_text,
            overlap_prefix=self.overlap_prefix,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )
        payload = chunk.plan_payload()
        payload["chunk_id"] = self.chunk_id
        payload["parent_chunk_index"] = self.parent_chunk_id
        payload["split_depth"] = self.depth
        payload["order_path"] = list(self.order_path)
        payload["source_kind"] = self.source_kind
        payload["transcript_hash"] = transcript_hash
        if status:
            payload["status"] = status
        if attempt is not None:
            payload["attempt"] = attempt
        if error_code:
            payload["error_code"] = error_code
        return payload


class Settings(BaseSettings):
    """Runtime configuration for the transcription pipeline."""

    speaker_db: Path = ALIAS_PATH
    recordings_root: Path = default_recordings_root()
    voices_dir: Path = default_voices_dir()
    unknown_dir: Path = default_unknown_dir()
    tmp_root: Path = default_tmp_root()
    llm_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("llm_model", "LLM_MODEL", "LAN_LLM_MODEL"),
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=256,
        validation_alias=AliasChoices("llm_max_tokens", "LLM_MAX_TOKENS", "LAN_LLM_MAX_TOKENS"),
    )
    llm_max_tokens_retry: int = Field(
        default=2048,
        ge=256,
        validation_alias=AliasChoices(
            "llm_max_tokens_retry",
            "LLM_MAX_TOKENS_RETRY",
            "LAN_LLM_MAX_TOKENS_RETRY",
        ),
    )
    llm_chunk_max_chars: int = Field(
        default=4500,
        ge=1,
        validation_alias=AliasChoices(
            "llm_chunk_max_chars",
            "LLM_CHUNK_MAX_CHARS",
            "LAN_LLM_CHUNK_MAX_CHARS",
        ),
    )
    llm_chunk_overlap_chars: int = Field(
        default=300,
        ge=0,
        validation_alias=AliasChoices(
            "llm_chunk_overlap_chars",
            "LLM_CHUNK_OVERLAP_CHARS",
            "LAN_LLM_CHUNK_OVERLAP_CHARS",
        ),
    )
    llm_chunk_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "llm_chunk_timeout_seconds",
            "LLM_CHUNK_TIMEOUT_SECONDS",
            "LAN_LLM_CHUNK_TIMEOUT_SECONDS",
        ),
    )
    llm_chunk_split_min_chars: int = Field(
        default=1200,
        ge=64,
        validation_alias=AliasChoices(
            "llm_chunk_split_min_chars",
            "LLM_CHUNK_SPLIT_MIN_CHARS",
            "LAN_LLM_CHUNK_SPLIT_MIN_CHARS",
        ),
    )
    llm_chunk_split_max_depth: int = Field(
        default=2,
        ge=0,
        validation_alias=AliasChoices(
            "llm_chunk_split_max_depth",
            "LLM_CHUNK_SPLIT_MAX_DEPTH",
            "LAN_LLM_CHUNK_SPLIT_MAX_DEPTH",
        ),
    )
    llm_long_transcript_threshold_chars: int = Field(
        default=4500,
        ge=1,
        validation_alias=AliasChoices(
            "llm_long_transcript_threshold_chars",
            "LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS",
            "LAN_LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS",
        ),
    )
    llm_merge_max_tokens: int | None = Field(
        default=None,
        ge=256,
        validation_alias=AliasChoices(
            "llm_merge_max_tokens",
            "LLM_MERGE_MAX_TOKENS",
            "LAN_LLM_MERGE_MAX_TOKENS",
        ),
    )
    asr_model: str = "large-v3"
    asr_device: str = Field(
        default="auto",
        validation_alias=AliasChoices("asr_device", "ASR_DEVICE", "LAN_ASR_DEVICE"),
    )
    asr_compute_type: str | None = None
    asr_batch_size: int = 16
    asr_enable_align: bool = True
    asr_multilingual_mode: Literal[
        "auto", "force_single_language", "force_multilingual"
    ] = Field(
        default="auto",
        validation_alias=AliasChoices(
            "asr_multilingual_mode",
            "ASR_MULTILINGUAL_MODE",
            "LAN_ASR_MULTILINGUAL_MODE",
        ),
    )
    vad_method: Literal["silero", "pyannote"] = "silero"
    diarization_device: str = Field(
        default="auto",
        validation_alias=AliasChoices(
            "diarization_device",
            "DIARIZATION_DEVICE",
            "LAN_DIARIZATION_DEVICE",
        ),
    )
    gpu_scheduler_mode: Literal["auto", "sequential", "parallel"] = Field(
        default="auto",
        validation_alias=AliasChoices(
            "gpu_scheduler_mode",
            "GPU_SCHEDULER_MODE",
            "LAN_GPU_SCHEDULER_MODE",
        ),
    )
    embed_threshold: float = 0.65
    merge_similar: float = 0.9
    precheck_min_duration_sec: float = 20.0
    precheck_min_speech_ratio: float = 0.10
    diarization_profile: Literal["auto", "dialog", "meeting"] = "auto"
    diarization_min_speakers: int | None = Field(default=None, ge=1)
    diarization_max_speakers: int | None = Field(default=None, ge=1)
    diarization_dialog_retry_min_duration_seconds: float = Field(
        default=DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
        ge=0.0,
    )
    diarization_dialog_retry_min_turns: int = Field(
        default=DEFAULT_DIALOG_RETRY_MIN_TURNS,
        ge=1,
    )
    diarization_merge_gap_seconds: float = Field(
        default=DEFAULT_DIARIZATION_MERGE_GAP_SECONDS,
        ge=0.0,
    )
    diarization_min_turn_seconds: float = Field(
        default=DEFAULT_DIARIZATION_MIN_TURN_SECONDS,
        ge=0.0,
    )
    diarization_flicker_min_seconds: float = Field(
        default=DEFAULT_DIARIZATION_FLICKER_MIN_SECONDS,
        ge=0.0,
        validation_alias=AliasChoices(
            "diarization_flicker_min_seconds",
            "DIARIZATION_FLICKER_MIN_SECONDS",
            "LAN_DIARIZATION_FLICKER_MIN_SECONDS",
        ),
    )
    diarization_flicker_max_consecutive: int = Field(
        default=DEFAULT_DIARIZATION_FLICKER_MAX_CONSECUTIVE,
        ge=0,
        validation_alias=AliasChoices(
            "diarization_flicker_max_consecutive",
            "DIARIZATION_FLICKER_MAX_CONSECUTIVE",
            "LAN_DIARIZATION_FLICKER_MAX_CONSECUTIVE",
        ),
    )
    speaker_turn_merge_gap_sec: float = Field(
        default=DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC,
        ge=0.0,
        validation_alias=AliasChoices(
            "speaker_turn_merge_gap_sec",
            "SPEAKER_TURN_MERGE_GAP_SEC",
            "LAN_SPEAKER_TURN_MERGE_GAP_SEC",
        ),
    )
    speaker_turn_short_merge_gap_sec: float = Field(
        default=DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC,
        ge=0.0,
        validation_alias=AliasChoices(
            "speaker_turn_short_merge_gap_sec",
            "SPEAKER_TURN_SHORT_MERGE_GAP_SEC",
            "LAN_SPEAKER_TURN_SHORT_MERGE_GAP_SEC",
        ),
    )
    speaker_turn_min_words: int = Field(
        default=DEFAULT_SPEAKER_TURN_MIN_WORDS,
        ge=0,
        validation_alias=AliasChoices(
            "speaker_turn_min_words",
            "SPEAKER_TURN_MIN_WORDS",
            "LAN_SPEAKER_TURN_MIN_WORDS",
        ),
    )
    snippet_pad_seconds: float = Field(
        default=0.25,
        ge=0.0,
    )
    snippet_max_duration_seconds: float = Field(
        default=8.0,
        gt=0.0,
    )
    snippet_min_duration_seconds: float = Field(
        default=0.8,
        gt=0.0,
    )
    snippet_max_per_speaker: int = Field(
        default=3,
        ge=0,
    )
    speaker_merge_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "speaker_merge_enabled",
            "SPEAKER_MERGE_ENABLED",
            "LAN_SPEAKER_MERGE_ENABLED",
        ),
    )
    speaker_merge_similarity_threshold: float = Field(
        default=DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "speaker_merge_similarity_threshold",
            "SPEAKER_MERGE_SIMILARITY_THRESHOLD",
            "LAN_SPEAKER_MERGE_SIMILARITY_THRESHOLD",
        ),
    )
    speaker_merge_max_segments: int = Field(
        default=DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS,
        ge=1,
        validation_alias=AliasChoices(
            "speaker_merge_max_segments",
            "SPEAKER_MERGE_MAX_SEGMENTS",
            "LAN_SPEAKER_MERGE_MAX_SEGMENTS",
        ),
    )

    class Config:
        env_prefix = "LAN_"


def _merge_similar(lines: Iterable[str], threshold: float) -> List[str]:
    out: List[str] = []
    for line in lines:
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        sim = sum(a == b for a, b in zip(prev, line)) / max(len(prev), len(line))
        if sim < threshold:
            out.append(line)
    return out


def _sentiment_score(text: str) -> int | float:
    from transformers import pipeline as hf_pipeline

    sentiment = hf_pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )
    try:
        sent = sentiment(
            text[:4000],
            truncation=True,
            max_length=512,
        )[0]
    except Exception as exc:
        _logger.warning("Sentiment scoring failed (%s); using neutral score", type(exc).__name__)
        return 0.0
    if sent["label"] == "positive":
        return int(sent["score"] * 100)
    if sent["label"] == "negative":
        return int((1 - sent["score"]) * 100)
    return 50


def refresh_aliases(result: TranscriptResult, alias_path: Path = ALIAS_PATH) -> None:
    aliases = _load_aliases(alias_path)
    result.speakers = sorted({aliases.get(s.speaker, s.speaker) for s in result.segments})


def _default_recording_id(audio_path: Path) -> str:
    stem = audio_path.stem.strip()
    return stem or "recording"


def _fallback_diarization(duration_sec: float | None):
    duration = max(duration_sec or 0.0, 0.1)

    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            if yield_label:
                yield SimpleNamespace(start=0.0, end=duration), "S1"
            else:  # pragma: no cover - legacy branch
                yield (SimpleNamespace(start=0.0, end=duration),)

    return _Annotation()


def _language_payload(info: dict[str, Any]) -> dict[str, Any]:
    detected_raw = str(info.get("language") or info.get("detected_language") or info.get("lang") or "unknown")
    detected = normalise_language_code(detected_raw) or "unknown"
    confidence_raw = None
    for key in ("language_probability", "language_confidence", "language_score", "probability"):
        if key in info and info[key] is not None:
            confidence_raw = info[key]
            break
    confidence = None if confidence_raw is None else round(safe_float(confidence_raw, default=0.0), 4)
    return {"detected": detected, "confidence": confidence}


def _select_asr_device(cfg: Settings) -> str:
    cuda_facts = collect_cuda_runtime_facts()
    _logger.info(
        "Torch CUDA runtime: is_available=%s device_count=%s visible_devices=%s torch.version.cuda=%s",
        cuda_facts.is_available,
        cuda_facts.device_count,
        cuda_facts.visible_devices,
        cuda_facts.torch_cuda_version,
    )
    return resolve_effective_device(
        cfg.asr_device,
        cuda_facts=cuda_facts,
        label="ASR device",
    )


def _select_diarization_device(cfg: Settings) -> str:
    return resolve_effective_device(
        cfg.diarization_device,
        cuda_facts=collect_cuda_runtime_facts(),
        label="diarization device",
    )


def _resolve_scheduler_plan(cfg: Settings, diariser: Diariser) -> SchedulerDecision:
    return resolve_scheduler_decision(
        cfg.gpu_scheduler_mode,
        asr_device=cfg.asr_device,
        diarization_device=cfg.diarization_device,
        diarization_is_heavy=_diariser_mode(diariser) == "pyannote",
    )


def _select_compute_type(cfg: Settings, device: str) -> str:
    configured = str(cfg.asr_compute_type or "").strip()
    if configured:
        return configured
    return "float16" if is_gpu_device(device) else "int8"


def clear_asr_model_cache() -> None:
    with _ASR_MODEL_CACHE_LOCK:
        _ASR_MODEL_CACHE.clear()
    _cleanup_cuda_memory("cuda")


def _asr_model_cache_key(
    *,
    cfg: Settings,
    device: str,
    compute_type: str,
    load_model_callable: Any,
) -> tuple[str, str, str, str, Any]:
    return (
        str(cfg.asr_model or "").strip(),
        str(device).strip(),
        str(compute_type).strip(),
        str(cfg.vad_method or "").strip(),
        load_model_callable,
    )


def _cached_asr_model_entry(
    cached_entry: Any,
    *,
    default_compute_type: str,
) -> tuple[Any | None, str]:
    if cached_entry is None:
        return None, default_compute_type
    if isinstance(cached_entry, _CachedAsrModel):
        return cached_entry.model, cached_entry.compute_type
    return cached_entry, default_compute_type


def _log_cuda_memory_snapshot(
    *,
    label: str,
    device: str,
) -> None:
    memory_info = cuda_memory_info(device)
    if memory_info is None:
        return
    free_bytes, total_bytes = memory_info
    _logger.info(
        "%s VRAM snapshot: device=%s free_bytes=%s total_bytes=%s",
        label,
        device,
        free_bytes,
        total_bytes,
    )


def _cleanup_cuda_memory(device: str) -> None:
    if not is_gpu_device(device):
        return
    gc.collect()
    try:
        import torch
    except Exception:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        cuda.empty_cache()
    except Exception:
        return


def _require_llm_model(llm_model: str | None) -> str:
    resolved = str(llm_model or "").strip()
    if not resolved:
        raise RuntimeError(_LLM_MODEL_REQUIRED_ERROR)
    return resolved


def _log_dropped_kwargs(
    *,
    callback: Callable[[str], Any] | None,
    scope: str,
    attempted: dict[str, Any],
    filtered: dict[str, Any],
) -> None:
    if callback is None:
        return
    dropped = [key for key in attempted if key not in filtered]
    if not dropped:
        return
    try:
        callback(f"{scope}: dropped unsupported kwargs: {', '.join(dropped)}")
    except Exception:
        # Step log append is best-effort and must not break processing.
        pass


def _best_effort_step_log(
    callback: Callable[[str], Any] | None,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        pass


def _glossary_transcribe_kwargs(asr_glossary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(asr_glossary, dict):
        return {}
    initial_prompt = " ".join(str(asr_glossary.get("initial_prompt") or "").split())
    hotwords = " ".join(str(asr_glossary.get("hotwords") or "").split())
    kwargs: dict[str, Any] = {}
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    if hotwords:
        kwargs["hotwords"] = hotwords
    return kwargs


def _new_glossary_runtime_state(glossary_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_keys": tuple(sorted(glossary_kwargs)),
        "applied_keys": set(),
        "dropped_keys": set(),
        "checked": False,
    }


def _update_glossary_runtime_state(
    *,
    state: dict[str, Any],
    glossary_kwargs: dict[str, Any],
    final_kwargs: dict[str, Any],
    dropped_kwargs: Iterable[str],
) -> None:
    if not glossary_kwargs:
        return
    glossary_keys = set(glossary_kwargs)
    applied_keys = {key for key in final_kwargs if key in glossary_keys}
    dropped_keys = glossary_keys - applied_keys
    dropped_keys.update(key for key in dropped_kwargs if key in glossary_keys)

    state["checked"] = True
    raw_applied = state.get("applied_keys")
    if isinstance(raw_applied, set):
        raw_applied.update(applied_keys)
    raw_dropped = state.get("dropped_keys")
    if isinstance(raw_dropped, set):
        raw_dropped.update(dropped_keys)


def _glossary_runtime_metadata(transcribe_audio: Any) -> dict[str, Any]:
    raw = getattr(transcribe_audio, "glossary_runtime_state", None)
    if not isinstance(raw, dict):
        return {}

    requested_keys_raw = raw.get("requested_keys")
    applied_keys_raw = raw.get("applied_keys")
    dropped_keys_raw = raw.get("dropped_keys")
    requested_keys = (
        sorted(
            key
            for key in requested_keys_raw
            if isinstance(key, str) and key
        )
        if isinstance(requested_keys_raw, (tuple, list, set))
        else []
    )
    applied_keys = (
        sorted(
            key
            for key in applied_keys_raw
            if isinstance(key, str) and key in requested_keys
        )
        if isinstance(applied_keys_raw, set)
        else []
    )
    dropped_keys = (
        sorted(
            key
            for key in dropped_keys_raw
            if isinstance(key, str) and key in requested_keys and key not in applied_keys
        )
        if isinstance(dropped_keys_raw, set)
        else []
    )
    if not requested_keys:
        return {}
    return {
        "checked": bool(raw.get("checked")),
        "requested_keys": requested_keys,
        "applied_keys": applied_keys,
        "dropped_keys": dropped_keys,
    }


def _effective_asr_glossary_artifact(
    *,
    asr_glossary: dict[str, Any] | None,
    runtime_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(asr_glossary, dict):
        return None
    payload = dict(asr_glossary)
    if not isinstance(runtime_metadata, dict) or not runtime_metadata:
        return payload

    requested_raw = runtime_metadata.get("requested_keys")
    applied_raw = runtime_metadata.get("applied_keys")
    dropped_raw = runtime_metadata.get("dropped_keys")
    if not isinstance(requested_raw, list) or not isinstance(applied_raw, list):
        return payload

    requested_keys = [
        key for key in requested_raw if key in {"initial_prompt", "hotwords"}
    ]
    applied_keys = [
        key
        for key in applied_raw
        if key in requested_keys and " ".join(str(payload.get(key) or "").split())
    ]
    if requested_keys and not applied_keys:
        return None

    for key in ("initial_prompt", "hotwords"):
        if key not in applied_keys and key in payload:
            payload.pop(key, None)

    if applied_keys:
        payload["applied_kwargs"] = applied_keys
    dropped_keys = (
        [
            key
            for key in dropped_raw
            if key in requested_keys and key not in applied_keys
        ]
        if isinstance(dropped_raw, list)
        else []
    )
    if dropped_keys:
        payload["dropped_kwargs"] = dropped_keys
    return payload


def _log_glossary_context_unsupported(
    *,
    callback: Callable[[str], Any] | None,
    glossary_kwargs: dict[str, Any],
    filtered_kwargs: dict[str, Any],
    dropped_kwargs: Iterable[str] = (),
    state: dict[str, bool],
) -> None:
    if callback is None or not glossary_kwargs or state.get("unsupported"):
        return
    glossary_keys = set(glossary_kwargs)
    remaining_keys = {key for key in glossary_keys if key in filtered_kwargs}
    runtime_dropped = {key for key in dropped_kwargs if key in glossary_keys}
    effective_keys = remaining_keys - runtime_dropped
    if effective_keys == glossary_keys:
        return
    state["unsupported"] = True
    if effective_keys:
        message = (
            "whisperx transcribe: glossary context unsupported for some kwargs; "
            "continuing with supported ASR hints only"
        )
    else:
        message = "whisperx transcribe: glossary context unsupported; continuing without ASR hints"
    _best_effort_step_log(
        callback,
        message,
    )


def _write_asr_glossary_artifact(
    *,
    derived_dir: Path,
    recording_id: str,
    asr_glossary: dict[str, Any] | None,
) -> None:
    artifact_path = derived_dir / "asr_glossary.json"
    if not isinstance(asr_glossary, dict):
        try:
            artifact_path.unlink()
        except FileNotFoundError:
            pass
        return
    payload = dict(asr_glossary)
    payload.setdefault("version", 1)
    payload["recording_id"] = recording_id
    atomic_write_json(artifact_path, payload)


def _diariser_runtime_metadata(diariser: Diariser) -> dict[str, Any]:
    raw = getattr(diariser, "last_run_metadata", None)
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _update_diariser_runtime_metadata(
    diariser: Diariser,
    **updates: Any,
) -> None:
    raw = getattr(diariser, "last_run_metadata", None)
    if not isinstance(raw, dict):
        return
    raw.update(updates)


def _diariser_mode(diariser: Diariser) -> str:
    raw_mode = getattr(diariser, "mode", None)
    if raw_mode is None:
        return "unknown"
    mode = str(raw_mode).strip().lower()
    return mode or "unknown"


def _is_degraded_diarization(
    diariser: Diariser,
    *,
    used_dummy_fallback: bool,
) -> bool:
    if used_dummy_fallback:
        return True
    return _diariser_mode(diariser) not in {"pyannote", "unknown"}


def _diariser_pipeline_model(diariser: Diariser) -> Any | None:
    """Best-effort access to the underlying pyannote pipeline model."""
    for attr in ("_pipeline_model", "pipeline_model", "pipeline"):
        model = getattr(diariser, attr, None)
        if model is not None and model is not diariser:
            return model
    return None


_DEFAULT_SPEAKER_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
_SPEAKER_EMBEDDING_SAFE_GLOBAL_ATTEMPTS = 3


def _build_pyannote_inference(
    model_or_name: Any, *, device: str | None = None
) -> Any | None:
    """Construct a pyannote ``Inference`` wrapper, returning ``None`` on failure.

    Loading the wespeaker checkpoint goes through ``torch.load`` which now
    defaults to ``weights_only=True``. Wrap the constructor in the same trusted
    safe-globals context that the diarization pipeline loader uses, with the
    same bounded retry on ``Unsupported global`` errors.
    """
    try:
        from pyannote.audio import Inference  # type: ignore
    except Exception as exc:
        _logger.warning(
            "speaker_merge: pyannote.audio.Inference unavailable; "
            "skipping merge step (%s)",
            exc,
        )
        return None

    kwargs: dict[str, Any] = {"window": "whole"}
    if device and device != "cpu":
        try:
            import torch  # type: ignore

            kwargs["device"] = torch.device(device)
        except Exception as exc:  # pragma: no cover - torch always present in prod
            _logger.debug(
                "speaker_merge: unable to bind embedding model to device %s: %s",
                device,
                exc,
            )

    extra_fqns: list[str] = []
    last_error: Exception | None = None
    for _ in range(_SPEAKER_EMBEDDING_SAFE_GLOBAL_ATTEMPTS):
        try:
            with diarization_safe_globals_for_torch_load(extra_fqns=extra_fqns):
                return Inference(model_or_name, **kwargs)
        except Exception as exc:
            last_error = exc
        retry_fqn = unsupported_global_diarization_fqn_from_error(last_error)
        if retry_fqn is None or retry_fqn in extra_fqns:
            break
        if import_trusted_diarization_symbol(retry_fqn) is None:
            break
        extra_fqns.append(retry_fqn)

    _logger.warning(
        "speaker_merge: failed to load embedding model %s: %s",
        model_or_name,
        last_error,
    )
    return None


def _resolve_pyannote_embedding_model(diariser: Diariser) -> EmbeddingModel | None:
    """Resolve a speaker embedding callable from the loaded diarization pipeline.

    Returns a callable ``(audio_path, start, end) -> np.ndarray`` or ``None`` if
    no usable embedding model can be obtained. Failures are logged and swallowed
    so the pipeline still runs when the merge step cannot be enabled.

    pyannote-audio >= 3.1 stores the embedding sub-model on
    ``SpeakerDiarization._embedding``, but that attribute is a
    :class:`PretrainedSpeakerEmbedding` callable — it does **not** expose the
    ``.crop(file, segment)`` method we need. We therefore only reuse the
    pipeline's attribute when it already looks like an ``Inference`` wrapper
    (i.e. exposes ``crop``). Otherwise we construct a fresh
    ``Inference(embedding_model, window="whole")`` — using the model identifier
    stored on the pipeline when available and falling back to the
    wespeaker checkpoint that pyannote-audio bundles by default.
    """

    cached = getattr(diariser, "_lan_speaker_embedding_model", None)
    if cached is not None:
        return cached
    if getattr(diariser, "_lan_speaker_embedding_unavailable", False):
        return None
    pipeline_model = _diariser_pipeline_model(diariser)
    if pipeline_model is None:
        setattr(diariser, "_lan_speaker_embedding_unavailable", True)
        return None

    inference: Any = None
    resolution_source: str | None = None
    for attr in ("_embedding", "embedding"):
        candidate = getattr(pipeline_model, attr, None)
        if candidate is not None and callable(getattr(candidate, "crop", None)):
            inference = candidate
            resolution_source = "pipeline_attribute"
            break

    if inference is None:
        raw_embedding_attr = getattr(pipeline_model, "embedding", None)
        if isinstance(raw_embedding_attr, (str, Path)):
            model_name: Any = str(raw_embedding_attr)
        else:
            model_name = _DEFAULT_SPEAKER_EMBEDDING_MODEL
        # ``_lan_effective_device`` is set on the pyannote pipeline model by
        # ``load_pyannote_pipeline``. Fall back to ``diariser`` for forward
        # compatibility with future code that may copy the attribute up.
        effective_device = getattr(
            diariser, "_lan_effective_device", None
        ) or getattr(pipeline_model, "_lan_effective_device", None)
        inference = _build_pyannote_inference(model_name, device=effective_device)
        if inference is None:
            setattr(diariser, "_lan_speaker_embedding_unavailable", True)
            return None
        resolution_source = "standalone_inference"

    _logger.info(
        "speaker_merge: embedding model ready (source=%s, device=%s)",
        resolution_source,
        getattr(inference, "device", "unknown"),
    )

    def _embed(audio_path: Path, start: float, end: float):
        try:
            from pyannote.core import Segment  # type: ignore
        except Exception:  # pragma: no cover - pyannote missing at runtime
            return None
        try:
            crop = inference.crop(str(audio_path), Segment(float(start), float(end)))
        except Exception as exc:
            _logger.debug(
                "speaker_merge: embedding inference failed: %s (%.3f-%.3f)",
                exc,
                start,
                end,
            )
            return None
        return crop

    setattr(diariser, "_lan_speaker_embedding_model", _embed)
    return _embed


async def _maybe_retry_dialog_diarization(
    *,
    diariser: Diariser,
    audio_path: Path,
    diarization: Any,
    asr_segments: Sequence[dict[str, Any]],
    precheck_result: PrecheckResult,
    step_log_callback: Callable[[str], Any] | None,
) -> Any:
    metadata = _diariser_runtime_metadata(diariser)
    initial_hints = metadata.get("initial_hints")
    if not isinstance(initial_hints, dict):
        initial_hints = {}
    auto_profile_enabled = bool(metadata.get("auto_profile_enabled", False))
    initial_profile = str(
        metadata.get("initial_profile")
        or metadata.get("diarization_profile")
        or "meeting"
    )
    retry_profile_enabled = auto_profile_enabled or initial_profile == "dialog"
    initial_decision = classify_diarization_profile(
        diarization,
        speech_turn_count=len(asr_segments),
        duration_sec=precheck_result.duration_sec,
        min_turns=int(
            getattr(
                diariser,
                "dialog_retry_min_turns",
                DEFAULT_DIALOG_RETRY_MIN_TURNS,
            )
        ),
        min_duration_seconds=safe_float(
            getattr(
                diariser,
                "dialog_retry_min_duration_seconds",
                DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
            ),
            default=DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
        ),
    )
    profile_selection: dict[str, Any] = {
        "requested_profile": str(metadata.get("diarization_profile") or "auto"),
        "initial_profile": initial_profile,
        "auto_profile_enabled": auto_profile_enabled,
        "override_reason": metadata.get("override_reason"),
        "selected_profile": (
            initial_profile
            if not auto_profile_enabled
            else initial_decision.selected_profile
        ),
        "classification_reason": initial_decision.reason,
        "dialog_retry_attempted": False,
        "selected_result": "initial_pass",
        "winner_reason": metadata.get("override_reason") or initial_decision.reason,
        "initial_metrics": initial_decision.metrics.as_dict(),
        "initial_dialog_score": initial_decision.dialog_score,
    }
    selected_profile_without_retry = profile_selection["selected_profile"]
    if not retry_profile_enabled:
        _update_diariser_runtime_metadata(
            diariser,
            effective_hints=dict(initial_hints),
            selected_profile=selected_profile_without_retry,
            profile_selection=profile_selection,
        )
        return diarization

    retry_dialog = getattr(diariser, "retry_dialog", None)
    if not callable(retry_dialog):
        profile_selection["winner_reason"] = "dialog_retry_unavailable"
        _update_diariser_runtime_metadata(
            diariser,
            selected_profile=selected_profile_without_retry,
            profile_selection=profile_selection,
        )
        return diarization
    if initial_decision.selected_profile != "dialog":
        _update_diariser_runtime_metadata(
            diariser,
            selected_profile=selected_profile_without_retry,
            profile_selection=profile_selection,
        )
        return diarization

    _best_effort_step_log(
        step_log_callback,
        (
            f"{'diarization auto-profile retry' if auto_profile_enabled else 'diarization forced-dialog retry'} "
            f"classification={initial_decision.reason} "
            "min_speakers=2 max_speakers=2"
        ),
    )
    profile_selection["dialog_retry_attempted"] = True
    try:
        retry_result = await retry_dialog(audio_path)
    except Exception as exc:
        profile_selection["selected_profile"] = selected_profile_without_retry
        profile_selection["winner_reason"] = "dialog_retry_failed"
        profile_selection["retry_error"] = str(exc) or exc.__class__.__name__
        _update_diariser_runtime_metadata(
            diariser,
            effective_hints=dict(initial_hints),
            selected_profile=profile_selection["selected_profile"],
            profile_selection=profile_selection,
        )
        _best_effort_step_log(
            step_log_callback,
            f"diarization dialog retry failed: {exc}",
        )
        return diarization
    retry_decision = classify_diarization_profile(
        retry_result,
        speech_turn_count=len(asr_segments),
        duration_sec=precheck_result.duration_sec,
        min_turns=int(
            getattr(
                diariser,
                "dialog_retry_min_turns",
                DEFAULT_DIALOG_RETRY_MIN_TURNS,
            )
        ),
        min_duration_seconds=safe_float(
            getattr(
                diariser,
                "dialog_retry_min_duration_seconds",
                DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
            ),
            default=DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
        ),
    )
    retry_selection = choose_dialog_retry_winner(initial_decision, retry_decision)
    winner_is_retry = retry_selection.selected_result == "dialog_retry"
    retry_hints = _diariser_runtime_metadata(diariser).get("retry_hints")
    if not isinstance(retry_hints, dict):
        retry_hints = {"min_speakers": 2, "max_speakers": 2}
    profile_selection.update(
        {
            "selected_profile": (
                retry_decision.selected_profile
                if winner_is_retry
                else selected_profile_without_retry
            ),
            "selected_result": retry_selection.selected_result,
            "winner_reason": retry_selection.winner_reason,
            "retry_metrics": retry_decision.metrics.as_dict(),
            "retry_dialog_score": retry_decision.dialog_score,
        }
    )
    _update_diariser_runtime_metadata(
        diariser,
        effective_hints=dict(retry_hints if winner_is_retry else initial_hints),
        dialog_retry_used=winner_is_retry,
        selected_profile=profile_selection["selected_profile"],
        profile_selection=profile_selection,
    )
    return retry_result if winner_is_retry else diarization


def _write_diarization_metadata_artifact(
    *,
    artifacts,
    diariser: Diariser,
    cfg: Settings,
    smoothing_result,
    used_dummy_fallback: bool,
    speaker_merges: dict[str, str] | None = None,
    speaker_merge_diagnostics: dict[str, Any] | None = None,
) -> None:
    metadata = _diariser_runtime_metadata(diariser)
    diariser_mode = _diariser_mode(diariser)
    effective_hints = metadata.get("effective_hints")
    if not isinstance(effective_hints, dict):
        effective_hints = {}
    initial_hints = metadata.get("initial_hints")
    if not isinstance(initial_hints, dict):
        initial_hints = {}
    profile_selection = metadata.get("profile_selection")
    if not isinstance(profile_selection, dict):
        profile_selection = {}
    selected_profile = str(
        profile_selection.get("selected_profile")
        or metadata.get("selected_profile")
        or metadata.get("initial_profile")
        or metadata.get("diarization_profile")
        or cfg.diarization_profile
    )
    initial_metrics = profile_selection.get("initial_metrics")
    if not isinstance(initial_metrics, dict):
        initial_metrics = {}

    payload: dict[str, Any] = {
        "version": 1,
        "mode": diariser_mode,
        "degraded": _is_degraded_diarization(
            diariser,
            used_dummy_fallback=used_dummy_fallback,
        ),
        "diarization_profile": str(metadata.get("diarization_profile") or cfg.diarization_profile),
        "requested_profile": str(
            metadata.get("requested_profile")
            or metadata.get("diarization_profile")
            or cfg.diarization_profile
        ),
        "effective_device": str(metadata.get("effective_device") or cfg.diarization_device),
        "scheduler_mode": str(metadata.get("scheduler_mode") or cfg.gpu_scheduler_mode),
        "scheduler_reason": metadata.get("scheduler_reason"),
        "initial_profile": str(metadata.get("initial_profile") or cfg.diarization_profile),
        "selected_profile": selected_profile,
        "selected_result": str(profile_selection.get("selected_result") or "initial_pass"),
        "auto_profile_enabled": bool(metadata.get("auto_profile_enabled", False)),
        "profile_override_reason": metadata.get("override_reason"),
        "hints_applied": effective_hints,
        "dialog_retry_attempted": bool(
            profile_selection.get(
                "dialog_retry_attempted",
                metadata.get("dialog_retry_used", False),
            )
        ),
        "dialog_retry_used": bool(metadata.get("dialog_retry_used", False)),
        "speaker_count_before_retry": metadata.get("speaker_count_before_retry"),
        "speaker_count_after_retry": metadata.get("speaker_count_after_retry"),
        "initial_speaker_count": initial_metrics.get(
            "speaker_count",
            metadata.get("speaker_count_before_retry"),
        ),
        "initial_top_two_coverage": initial_metrics.get("top_two_coverage"),
        "used_dummy_fallback": used_dummy_fallback,
        "smoothing_applied": bool(diariser_mode == "pyannote" and not used_dummy_fallback),
        "merge_gap_seconds": cfg.diarization_merge_gap_seconds,
        "min_turn_seconds": cfg.diarization_min_turn_seconds,
        "speaker_count_before_smoothing": smoothing_result.speaker_count_before,
        "speaker_count_after_smoothing": smoothing_result.speaker_count_after,
        "turn_count_before_smoothing": smoothing_result.turn_count_before,
        "turn_count_after_smoothing": smoothing_result.turn_count_after,
        "adjacent_merges": smoothing_result.adjacent_merges,
        "micro_turn_absorptions": smoothing_result.micro_turn_absorptions,
    }
    if initial_hints != effective_hints:
        payload["initial_hints"] = initial_hints
    if profile_selection:
        payload["profile_selection"] = profile_selection
    payload["speaker_merges"] = dict(speaker_merges or {})
    payload["speaker_merge_diagnostics"] = dict(speaker_merge_diagnostics or {})
    atomic_write_json(artifacts.diarization_metadata_json_path, payload)


def _load_cached_whisperx_model(
    *,
    cfg: Settings,
    wx_asr: Any,
    device: str,
    compute_type: str,
    step_log_callback: Callable[[str], Any] | None,
) -> tuple[Any, str]:
    load_model_callable = getattr(wx_asr, "load_model", None)
    cache_key = _asr_model_cache_key(
        cfg=cfg,
        device=device,
        compute_type=compute_type,
        load_model_callable=load_model_callable,
    )
    with _ASR_MODEL_CACHE_LOCK:
        cached_model, cached_compute_type = _cached_asr_model_entry(
            _ASR_MODEL_CACHE.get(cache_key),
            default_compute_type=compute_type,
        )
    if cached_model is not None:
        _best_effort_step_log(
            step_log_callback,
            (
                "asr model cache hit "
                f"model={cfg.asr_model} device={device} compute_type={cached_compute_type}"
            ),
        )
        return cached_model, cached_compute_type

    model_load_kwargs = {"compute_type": compute_type, "vad_method": cfg.vad_method}

    def _load_model(selected_compute_type: str) -> Any:
        load_kwargs = dict(model_load_kwargs)
        load_kwargs["compute_type"] = selected_compute_type
        _log_cuda_memory_snapshot(label="ASR load", device=device)
        return call_with_supported_kwargs(
            wx_asr.load_model,
            cfg.asr_model,
            device,
            **load_kwargs,
        )

    selected_compute_type = compute_type
    with omegaconf_safe_globals_for_torch_load():
        try:
            model = _load_model(selected_compute_type)
        except Exception as first_error:
            retry_fqn = unsupported_global_omegaconf_fqn_from_error(first_error)
            if retry_fqn is not None:
                with omegaconf_safe_globals_for_torch_load(extra_fqns=[retry_fqn]):
                    try:
                        model = _load_model(selected_compute_type)
                    except Exception:
                        raise first_error
            elif (
                is_gpu_oom_error(first_error)
                and normalize_device(cfg.asr_device) == "auto"
                and not str(cfg.asr_compute_type or "").strip()
                and selected_compute_type != "int8_float16"
            ):
                clear_asr_model_cache()
                selected_compute_type = "int8_float16"
                _best_effort_step_log(
                    step_log_callback,
                    (
                        "asr GPU OOM during model load; retrying once with "
                        "compute_type=int8_float16"
                    ),
                )
                model = _load_model(selected_compute_type)
            else:
                raise

    resolved_cache_key = _asr_model_cache_key(
        cfg=cfg,
        device=device,
        compute_type=selected_compute_type,
        load_model_callable=load_model_callable,
    )
    cache_entry = _CachedAsrModel(model=model, compute_type=selected_compute_type)
    with _ASR_MODEL_CACHE_LOCK:
        cached_model, cached_compute_type = _cached_asr_model_entry(
            _ASR_MODEL_CACHE.get(resolved_cache_key),
            default_compute_type=selected_compute_type,
        )
        if cached_model is None:
            _ASR_MODEL_CACHE[resolved_cache_key] = cache_entry
            cached_model = model
            cached_compute_type = selected_compute_type
        if selected_compute_type != compute_type:
            _ASR_MODEL_CACHE.setdefault(
                cache_key,
                _CachedAsrModel(
                    model=cached_model,
                    compute_type=cached_compute_type,
                ),
            )
    return cached_model, cached_compute_type


def _build_whisperx_transcriber(
    cfg: Settings,
    step_log_callback: Callable[[str], Any] | None = None,
    asr_glossary: dict[str, Any] | None = None,
) -> Callable[[Path, str | None], tuple[list[dict[str, Any]], dict[str, Any]]]:
    glossary_kwargs = _glossary_transcribe_kwargs(asr_glossary)
    glossary_log_state = {"unsupported": False}
    glossary_runtime_state = _new_glossary_runtime_state(glossary_kwargs)
    patched_paths = ensure_ctranslate2_no_execstack()
    if patched_paths and step_log_callback is not None:
        try:
            count = len(patched_paths)
            noun = "library" if count == 1 else "libraries"
            step_log_callback(
                f"native fixup: cleared executable-stack flag on {count} ctranslate2 {noun}"
            )
        except Exception:
            # Step log append is best-effort and must not break processing.
            pass
    if cfg.vad_method == "pyannote" and patch_pyannote_inference_ignore_use_auth_token() and step_log_callback is not None:
        try:
            step_log_callback("pyannote compat: ignore unsupported Inference use_auth_token")
        except Exception:
            # Step log append is best-effort and must not break processing.
            pass

    import whisperx
    try:
        import whisperx.asr as wx_asr
    except Exception:
        # Test doubles may provide only whisperx.load_model without submodules.
        wx_asr = whisperx

    legacy_transcribe = getattr(whisperx, "transcribe", None)
    if legacy_transcribe is not None and not callable(legacy_transcribe):
        raise TypeError(
            f"whisperx.transcribe must be callable or None, got {type(legacy_transcribe).__name__}"
        )

    if callable(legacy_transcribe):
        def _legacy_transcribe(
            audio_path: Path,
            override_lang: str | None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            kwargs: dict[str, Any] = {
                "vad_filter": True,
                "language": override_lang or "auto",
                "word_timestamps": True,
                **glossary_kwargs,
            }
            filtered_kwargs = filter_kwargs_for_callable(legacy_transcribe, kwargs)
            _log_dropped_kwargs(
                callback=step_log_callback,
                scope="whisperx transcribe",
                attempted=kwargs,
                filtered=filtered_kwargs,
            )
            attempted_runtime_kwargs = filtered_kwargs
            try:
                clear_last_supported_kwargs_call_details()
                segments, info = call_with_supported_kwargs(
                    legacy_transcribe,
                    str(audio_path),
                    **kwargs,
                )
                final_kwargs, dropped_kwargs = last_supported_kwargs_call_details()
            except TypeError:
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("word_timestamps", None)
                _log_dropped_kwargs(
                    callback=step_log_callback,
                    scope="whisperx transcribe",
                    attempted=kwargs,
                    filtered=retry_kwargs,
                )
                attempted_runtime_kwargs = filter_kwargs_for_callable(legacy_transcribe, retry_kwargs)
                clear_last_supported_kwargs_call_details()
                segments, info = call_with_supported_kwargs(
                    legacy_transcribe,
                    str(audio_path),
                    **retry_kwargs,
                )
                final_kwargs, dropped_kwargs = last_supported_kwargs_call_details()
            if final_kwargs is None:
                final_kwargs = dict(attempted_runtime_kwargs)
            if dropped_kwargs is None:
                dropped_kwargs = ()
            _update_glossary_runtime_state(
                state=glossary_runtime_state,
                glossary_kwargs=glossary_kwargs,
                final_kwargs=final_kwargs,
                dropped_kwargs=dropped_kwargs,
            )
            _log_dropped_kwargs(
                callback=step_log_callback,
                scope="whisperx transcribe",
                attempted=attempted_runtime_kwargs,
                filtered=final_kwargs,
            )
            _log_glossary_context_unsupported(
                callback=step_log_callback,
                glossary_kwargs=glossary_kwargs,
                filtered_kwargs=final_kwargs,
                dropped_kwargs=dropped_kwargs,
                state=glossary_log_state,
            )
            return list(segments), dict(info or {})

        _legacy_transcribe.glossary_runtime_state = glossary_runtime_state  # type: ignore[attr-defined]
        return _legacy_transcribe

    device = _select_asr_device(cfg)
    compute_type = _select_compute_type(cfg, device)
    _logger.info("ASR VAD method: %s", cfg.vad_method)
    model, compute_type = _load_cached_whisperx_model(
        cfg=cfg,
        wx_asr=wx_asr,
        device=device,
        compute_type=compute_type,
        step_log_callback=step_log_callback,
    )

    vad = getattr(model, "vad_model", None)
    if not callable(vad):
        raise RuntimeError(
            "WhisperX VAD misconfigured: "
            f"vad_method={cfg.vad_method!r}, "
            f"type(vad_model)={type(vad)!r}; expected callable model.vad_model"
        )

    def _modern_transcribe(
        audio_path: Path,
        override_lang: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        audio = whisperx.load_audio(str(audio_path))
        transcribe_kwargs: dict[str, Any] = {
            "batch_size": cfg.asr_batch_size,
            "vad_filter": True,
            "language": (override_lang if override_lang else None),
            **glossary_kwargs,
        }
        filtered_kwargs = filter_kwargs_for_callable(model.transcribe, transcribe_kwargs)
        _log_dropped_kwargs(
            callback=step_log_callback,
            scope="whisperx transcribe",
            attempted=transcribe_kwargs,
            filtered=filtered_kwargs,
        )
        clear_last_supported_kwargs_call_details()
        result = call_with_supported_kwargs(
            model.transcribe,
            audio,
            **transcribe_kwargs,
        )
        final_kwargs, dropped_kwargs = last_supported_kwargs_call_details()
        if final_kwargs is None:
            final_kwargs = dict(filtered_kwargs)
        if dropped_kwargs is None:
            dropped_kwargs = ()
        _update_glossary_runtime_state(
            state=glossary_runtime_state,
            glossary_kwargs=glossary_kwargs,
            final_kwargs=final_kwargs,
            dropped_kwargs=dropped_kwargs,
        )
        _log_dropped_kwargs(
            callback=step_log_callback,
            scope="whisperx transcribe",
            attempted=filtered_kwargs,
            filtered=final_kwargs,
        )
        _log_glossary_context_unsupported(
            callback=step_log_callback,
            glossary_kwargs=glossary_kwargs,
            filtered_kwargs=final_kwargs,
            dropped_kwargs=dropped_kwargs,
            state=glossary_log_state,
        )
        segments = list(result.get("segments", []))
        info: dict[str, Any] = {
            "language": result.get("language") or (override_lang or "unknown")
        }

        if cfg.asr_enable_align:
            try:
                align_lang = normalise_language_code(info.get("language")) or "en"
                model_a, metadata = whisperx.load_align_model(
                    language_code=align_lang,
                    device=device,
                )
                try:
                    aligned = whisperx.align(
                        segments,
                        model_a,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=False,
                    )
                except TypeError:
                    aligned = whisperx.align(segments, model_a, metadata, audio, device)
                segments = list(aligned.get("segments", segments))
            except Exception:
                pass

        return segments, info

    _modern_transcribe.glossary_runtime_state = glossary_runtime_state  # type: ignore[attr-defined]
    return _modern_transcribe


def _whisperx_asr(
    audio_path: Path,
    *,
    override_lang: str | None,
    cfg: Settings,
    step_log_callback: Callable[[str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transcribe_audio = (
        getattr(_whisperx_transcriber_state, "transcribe_audio", None)
        if bool(getattr(_whisperx_transcriber_state, "use_session_transcriber", False))
        else None
    )
    if transcribe_audio is None:
        transcribe_audio = _build_whisperx_transcriber(
            cfg=cfg,
            step_log_callback=step_log_callback,
        )
    return transcribe_audio(audio_path, override_lang)


_DEFAULT_WHISPERX_ASR = _whisperx_asr


def _empty_questions() -> dict[str, Any]:
    return {
        "total_count": 0,
        "types": {"open": 0, "yes_no": 0, "clarification": 0, "status": 0, "decision_seeking": 0},
        "extracted": [],
    }


def _clear_dir(path: Path) -> None:
    for child in path.iterdir() if path.exists() else []:
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)


def _base_transcript_payload(
    *,
    recording_id: str,
    language: dict[str, Any],
    dominant_language: str,
    language_distribution: dict[str, float],
    language_spans: list[dict[str, Any]],
    target_summary_language: str,
    transcript_language_override: str | None,
    calendar_title: str | None,
    calendar_attendees: list[str],
    segments: list[dict[str, Any]],
    speakers: list[str],
    text: str,
) -> dict[str, Any]:
    return {
        "recording_id": recording_id,
        "language": language,
        "dominant_language": dominant_language,
        "language_distribution": language_distribution,
        "language_spans": language_spans,
        "target_summary_language": target_summary_language,
        "transcript_language_override": transcript_language_override,
        "calendar_title": calendar_title,
        "calendar_attendees": calendar_attendees,
        "segments": segments,
        "speakers": speakers,
        "text": text,
    }


def run_precheck(audio_path: Path, cfg: Settings | None = None) -> PrecheckResult:
    settings = cfg or Settings()
    return _run_precheck(
        audio_path,
        min_duration_sec=settings.precheck_min_duration_sec,
        min_speech_ratio=settings.precheck_min_speech_ratio,
    )


ProgressCallback = Callable[[str, float], Any | Awaitable[Any]]


async def _emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    progress: float,
) -> None:
    if callback is None:
        return
    try:
        out = callback(stage, progress)
        if inspect.isawaitable(out):
            await out
    except Exception:
        # Progress reporting is best-effort and must not break processing.
        return


async def _await_result(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _normalise_llm_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    return {"role": "assistant", "content": str(message)}


async def _generate_llm_message(
    llm: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format: dict[str, Any] | None,
    max_tokens: int,
    max_tokens_retry: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    request = call_with_supported_kwargs(
        llm.generate,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        response_format=response_format,
        max_tokens=max_tokens,
        max_tokens_retry=max_tokens_retry,
    )
    awaitable = _await_result(request)
    if timeout_seconds is not None:
        return _normalise_llm_message(await asyncio.wait_for(awaitable, timeout=timeout_seconds))
    return _normalise_llm_message(await awaitable)


def _llm_message_timed_out(message: dict[str, Any]) -> bool:
    return str(message.get("content") or "").strip() == _LLM_TIMEOUT_SENTINEL


def _llm_timeout_message(timeout_seconds: float | None) -> str:
    if timeout_seconds is None or timeout_seconds <= 0:
        return "timed out"
    return f"timed out after {timeout_seconds:g}s"


def _llm_chunk_progress(chunk_index: int, total_chunks: int) -> float:
    total = max(total_chunks, 1)
    start = 0.82
    span = 0.08
    return min(0.90, start + (max(chunk_index, 1) / total) * span)


_LLM_CHUNK_GROUP_EXTRACT = "extract"
_LLM_CHUNK_REASON_TIMEOUT = "llm_chunk_timeout"
_LLM_CHUNK_REASON_REQUEST_TIMEOUT = "llm_chunk_request_timeout"
_LLM_CHUNK_REASON_PARSE_ERROR = "llm_chunk_parse_error"
_LLM_CHUNK_REASON_CONNECTION_ERROR = "llm_chunk_connection_error"
_LLM_CHUNK_REASON_RUNTIME_ERROR = "llm_chunk_runtime_error"
_LLM_MERGE_REASON_TIMEOUT = "llm_merge_timeout"
_LLM_MERGE_REASON_REQUEST_TIMEOUT = "llm_merge_request_timeout"
_LLM_MERGE_REASON_CONNECTION_ERROR = "llm_merge_connection_error"
_LLM_MERGE_REASON_PARSE_ERROR = "llm_merge_parse_error"


def _llm_request_timeout_message(timeout_seconds: float | None) -> str:
    base = _llm_timeout_message(timeout_seconds)
    return f"request {base}"


async def _emit_step_log(
    callback: Callable[[str], Any] | None,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        out = callback(message)
        if inspect.isawaitable(out):
            await out
    except Exception:
        return


def _read_json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _compact_transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_request_timeout_seconds(llm: Any) -> float | None:
    timeout = safe_float(getattr(llm, "timeout", None), default=0.0)
    return timeout if timeout > 0 else None


def _chunk_artifact_token(chunk_id: str) -> str:
    normalized = str(chunk_id or "").strip().lower()
    digits: list[str] = []
    suffix: list[str] = []
    seen_suffix = False
    for char in normalized:
        if not seen_suffix and char.isdigit():
            digits.append(char)
            continue
        seen_suffix = True
        suffix.append(char if char.isalnum() else "_")
    if digits:
        prefix = f"{int(''.join(digits)):03d}"
    else:
        prefix = "chunk"
    suffix_text = "".join(suffix)
    return f"{prefix}{suffix_text}"


def _chunk_artifact_paths(
    derived_dir: Path,
    chunk_id: str,
) -> tuple[Path, Path, Path]:
    token = _chunk_artifact_token(chunk_id)
    return (
        derived_dir / f"llm_chunk_{token}_raw.json",
        derived_dir / f"llm_chunk_{token}_extract.json",
        derived_dir / f"llm_chunk_{token}_error.json",
    )


def _chunk_runtime_from_state_row(
    row: dict[str, Any],
    *,
    transcript_hash: str,
) -> _ChunkRuntime | None:
    metadata = row.get("metadata_json")
    if not isinstance(metadata, dict):
        return None
    if str(metadata.get("transcript_hash") or "").strip() != transcript_hash:
        return None
    chunk_id = str(row.get("chunk_index") or metadata.get("chunk_id") or "").strip().lower()
    text = str(metadata.get("text") or "").strip()
    base_text = str(metadata.get("base_text") or "").strip()
    if not chunk_id or not text or not base_text:
        return None
    order_path_raw = metadata.get("order_path")
    if not isinstance(order_path_raw, list) or not order_path_raw:
        return None
    try:
        order_path = tuple(max(int(item), 0) for item in order_path_raw)
    except (TypeError, ValueError):
        return None
    if any(item <= 0 for item in order_path):
        return None
    return _ChunkRuntime(
        chunk_id=chunk_id,
        parent_chunk_id=(
            str(row.get("parent_chunk_index") or "").strip().lower() or None
        ),
        depth=max(int(safe_float(metadata.get("depth"), default=float(len(order_path) - 1))), 0),
        order_path=order_path,
        text=text,
        base_text=base_text,
        overlap_prefix=str(metadata.get("overlap_prefix") or ""),
        start_seconds=(
            round(safe_float(metadata.get("start_seconds"), default=0.0), 3)
            if metadata.get("start_seconds") is not None
            else None
        ),
        end_seconds=(
            round(safe_float(metadata.get("end_seconds"), default=0.0), 3)
            if metadata.get("end_seconds") is not None
            else None
        ),
        source_kind=str(metadata.get("source_kind") or "root"),
    )


def _build_root_chunk_runtime(chunks: Sequence[TranscriptChunk]) -> list[_ChunkRuntime]:
    runtime_chunks: list[_ChunkRuntime] = []
    for chunk in chunks:
        runtime_chunks.append(
            _ChunkRuntime(
                chunk_id=str(chunk.index),
                parent_chunk_id=None,
                depth=0,
                order_path=(int(safe_float(chunk.index, default=0.0)),),
                text=chunk.text,
                base_text=chunk.base_text,
                overlap_prefix=chunk.overlap_prefix,
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                source_kind="root",
            )
        )
    return runtime_chunks


def _active_chunk_rows_by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_index") or "").strip().lower()
        if chunk_id:
            mapping[chunk_id] = dict(row)
    return mapping


def _load_active_chunk_runtime(
    rows: Sequence[dict[str, Any]],
    *,
    transcript_hash: str,
) -> list[_ChunkRuntime] | None:
    active_chunks: list[_ChunkRuntime] = []
    for row in rows:
        if str(row.get("status") or "").strip().lower() == "split":
            runtime = _chunk_runtime_from_state_row(row, transcript_hash=transcript_hash)
            if runtime is None:
                return None
            continue
        runtime = _chunk_runtime_from_state_row(row, transcript_hash=transcript_hash)
        if runtime is None:
            return None
        active_chunks.append(runtime)
    if not active_chunks:
        return None
    active_chunks.sort(key=lambda item: item.order_path)
    return active_chunks


def _runtime_chunk_row_payload(
    chunk: _ChunkRuntime,
    *,
    total: int,
    transcript_hash: str,
    existing_row: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "chunk_group": _LLM_CHUNK_GROUP_EXTRACT,
        "chunk_index": chunk.chunk_id,
        "chunk_total": total,
        "status": str(existing_row.get("status") or "planned") if existing_row else "planned",
        "attempt": max(int(existing_row.get("attempt") or 0), 0) if existing_row else 0,
        "started_at": str(existing_row.get("started_at") or "").strip() or None if existing_row else None,
        "finished_at": str(existing_row.get("finished_at") or "").strip() or None if existing_row else None,
        "duration_ms": (
            int(existing_row.get("duration_ms"))
            if existing_row and existing_row.get("duration_ms") is not None
            else None
        ),
        "error_code": str(existing_row.get("error_code") or "").strip() or None if existing_row else None,
        "error_text": str(existing_row.get("error_text") or "").strip() or None if existing_row else None,
        "parent_chunk_index": chunk.parent_chunk_id,
        "metadata": chunk.metadata_payload(transcript_hash=transcript_hash),
    }


def _sync_chunk_rows(
    *,
    chunk_state_store: ChunkStateStore | None,
    active_chunks: Sequence[_ChunkRuntime],
    rows_by_id: dict[str, dict[str, Any]],
    transcript_hash: str,
) -> dict[str, dict[str, Any]]:
    if chunk_state_store is None:
        return rows_by_id
    synced = dict(rows_by_id)
    total = len(active_chunks)
    for chunk in active_chunks:
        row = chunk_state_store.upsert_state(
            **_runtime_chunk_row_payload(
                chunk,
                total=total,
                transcript_hash=transcript_hash,
                existing_row=synced.get(chunk.chunk_id),
            )
        )
        if row is not None:
            synced[chunk.chunk_id] = row
    return synced


def _split_parent_rows(
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rows_by_id.values():
        if str(row.get("status") or "").strip().lower() != "split":
            continue
        rows.append(dict(row))
    rows.sort(key=lambda item: str(item.get("chunk_index") or ""))
    return rows


def _write_chunk_plan_artifact(
    *,
    derived_dir: Path,
    compact_transcript: Any,
    transcript_text: str,
    transcript_hash: str,
    cfg: Settings,
    active_chunks: Sequence[_ChunkRuntime],
    rows_by_id: dict[str, dict[str, Any]],
) -> None:
    total = len(active_chunks)
    chunk_rows = [
        chunk.plan_payload(
            position=index,
            total=total,
            transcript_hash=transcript_hash,
            status=str(rows_by_id.get(chunk.chunk_id, {}).get("status") or "").strip() or None,
            attempt=(
                int(rows_by_id[chunk.chunk_id]["attempt"])
                if chunk.chunk_id in rows_by_id and rows_by_id[chunk.chunk_id].get("attempt") is not None
                else None
            ),
            error_code=str(rows_by_id.get(chunk.chunk_id, {}).get("error_code") or "").strip() or None,
        )
        for index, chunk in enumerate(active_chunks, start=1)
    ]
    split_rows: list[dict[str, Any]] = []
    for row in _split_parent_rows(rows_by_id):
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        split_rows.append(
            {
                "chunk_index": str(row.get("chunk_index") or ""),
                "parent_chunk_index": str(row.get("parent_chunk_index") or "").strip() or None,
                "status": "split",
                "attempt": max(int(row.get("attempt") or 0), 0),
                "error_code": str(row.get("error_code") or "").strip() or None,
                "child_chunk_indexes": list(metadata.get("split_child_chunk_indexes") or []),
            }
        )
    atomic_write_json(
        derived_dir / "llm_chunks_plan.json",
        {
            "version": 2,
            "source_chars": compact_transcript.source_chars,
            "compact_chars": compact_transcript.compact_chars,
            "transcript_hash": transcript_hash,
            "chunk_max_chars": cfg.llm_chunk_max_chars,
            "chunk_overlap_chars": cfg.llm_chunk_overlap_chars,
            "chunk_timeout_seconds": cfg.llm_chunk_timeout_seconds,
            "chunk_split_min_chars": cfg.llm_chunk_split_min_chars,
            "chunk_split_max_depth": cfg.llm_chunk_split_max_depth,
            "long_transcript_threshold_chars": cfg.llm_long_transcript_threshold_chars,
            "source_transcript_chars": len(transcript_text.strip()),
            "compaction": {
                "merge_gap_seconds": compact_transcript.merge_gap_seconds,
                "source_turn_count": compact_transcript.source_turn_count,
                "compact_turn_count": compact_transcript.compact_turn_count,
                "speaker_mapping": [
                    item.artifact_payload() for item in compact_transcript.speaker_mapping
                ],
            },
            "chunks": chunk_rows,
            "split_chunks": split_rows,
            "state_summary": {
                "planned": sum(1 for row in rows_by_id.values() if str(row.get("status") or "") == "planned"),
                "running": sum(1 for row in rows_by_id.values() if str(row.get("status") or "") == "running"),
                "completed": sum(1 for row in rows_by_id.values() if str(row.get("status") or "") == "completed"),
                "failed": sum(1 for row in rows_by_id.values() if str(row.get("status") or "") == "failed"),
                "split": sum(1 for row in rows_by_id.values() if str(row.get("status") or "") == "split"),
            },
        },
    )


def _chunk_error_payload(
    *,
    chunk: _ChunkRuntime,
    position: int,
    total: int,
    attempt: int,
    reason_code: str,
    message: str,
    status: str,
    timeout_seconds: float | None = None,
    child_chunk_indexes: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunk_index": chunk.chunk_id,
        "chunk_position": position,
        "chunk_total": total,
        "parent_chunk_index": chunk.parent_chunk_id,
        "reason_code": reason_code,
        "error": message,
        "attempt": attempt,
        "status": status,
    }
    if timeout_seconds is not None and timeout_seconds > 0:
        payload["timeout_seconds"] = round(timeout_seconds, 3)
    if child_chunk_indexes:
        payload["child_chunk_indexes"] = [str(item) for item in child_chunk_indexes]
    return payload


def _best_chunk_split_index(parts: Sequence[TranscriptChunk]) -> int:
    if len(parts) <= 1:
        return 0
    total_chars = sum(len(part.base_text.strip()) for part in parts)
    left_chars = 0
    best_index = 1
    best_delta = total_chars
    for index in range(1, len(parts)):
        left_chars += len(parts[index - 1].base_text.strip())
        delta = abs((total_chars - left_chars) - left_chars)
        if delta < best_delta:
            best_index = index
            best_delta = delta
    return best_index


def _split_runtime_chunk(
    chunk: _ChunkRuntime,
    *,
    cfg: Settings,
) -> list[_ChunkRuntime]:
    base_text = chunk.base_text.strip()
    if chunk.depth >= cfg.llm_chunk_split_max_depth:
        return []
    if len(base_text) < max(cfg.llm_chunk_split_min_chars * 2, 2):
        return []
    planned_parts = plan_transcript_chunks(
        base_text,
        max_chars=max(cfg.llm_chunk_split_min_chars, len(base_text) // 2),
        overlap_chars=0,
    )
    if len(planned_parts) < 2:
        return []
    split_index = _best_chunk_split_index(planned_parts)
    left_parts = planned_parts[:split_index]
    right_parts = planned_parts[split_index:]
    if not left_parts or not right_parts:
        return []
    left_base = "\n".join(part.base_text.strip() for part in left_parts if part.base_text.strip()).strip()
    right_base = "\n".join(part.base_text.strip() for part in right_parts if part.base_text.strip()).strip()
    if not left_base or not right_base or max(len(left_base), len(right_base)) >= len(base_text):
        return []
    start_seconds = chunk.start_seconds
    end_seconds = chunk.end_seconds
    left_start = start_seconds
    left_end = None
    right_start = None
    right_end = end_seconds
    if start_seconds is not None and end_seconds is not None and end_seconds >= start_seconds:
        span = end_seconds - start_seconds
        left_ratio = len(left_base) / max(len(left_base) + len(right_base), 1)
        midpoint = round(start_seconds + (span * left_ratio), 3)
        left_end = midpoint
        right_start = midpoint
    return [
        _ChunkRuntime(
            chunk_id=f"{chunk.chunk_id}a",
            parent_chunk_id=chunk.chunk_id,
            depth=chunk.depth + 1,
            order_path=(*chunk.order_path, 1),
            text=left_base,
            base_text=left_base,
            overlap_prefix="",
            start_seconds=left_start,
            end_seconds=left_end,
            source_kind="split",
        ),
        _ChunkRuntime(
            chunk_id=f"{chunk.chunk_id}b",
            parent_chunk_id=chunk.chunk_id,
            depth=chunk.depth + 1,
            order_path=(*chunk.order_path, 2),
            text=right_base,
            base_text=right_base,
            overlap_prefix="",
            start_seconds=right_start,
            end_seconds=right_end,
            source_kind="split",
        ),
    ]


def _validated_completed_chunk_extract(
    *,
    derived_dir: Path,
    chunk: _ChunkRuntime,
    position: int,
    total: int,
) -> tuple[dict[str, Any] | None, str | None]:
    raw_path, extract_path, _error_path = _chunk_artifact_paths(derived_dir, chunk.chunk_id)
    raw_payload = _read_json_artifact(raw_path)
    if raw_payload is None:
        return None, f"missing raw artifact {raw_path.name}"
    extract_payload = _read_json_artifact(extract_path)
    if extract_payload is None:
        return None, f"missing extract artifact {extract_path.name}"
    recorded_chunk_id = str(extract_payload.get("chunk_id") or chunk.chunk_id).strip().lower()
    if recorded_chunk_id != chunk.chunk_id:
        return None, f"extract chunk_id mismatch for {chunk.chunk_id}"
    extract_payload["chunk_id"] = chunk.chunk_id
    extract_payload["chunk_index"] = position
    extract_payload["chunk_total"] = total
    extract_payload["parent_chunk_index"] = chunk.parent_chunk_id
    extract_payload["split_depth"] = chunk.depth
    return extract_payload, None


def _speaker_turn_prompt_text(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    aliases: dict[str, str],
) -> str:
    rows: list[str] = []
    for turn in speaker_turns:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker_key = str(turn.get("speaker") or "S1")
        speaker = aliases.get(speaker_key, speaker_key)
        start = safe_float(turn.get("start"), default=0.0)
        end = safe_float(turn.get("end"), default=0.0)
        rows.append(f"[{start:.2f}-{end:.2f}] {speaker}: {text}")
    return "\n".join(rows).strip()


def _use_chunked_llm(transcript_text: str, cfg: Settings) -> bool:
    return len(transcript_text.strip()) > cfg.llm_long_transcript_threshold_chars


async def _run_chunked_llm_summary(
    *,
    transcript_text: str,
    speaker_turns: Sequence[dict[str, Any]],
    aliases: dict[str, str],
    derived_dir: Path,
    llm: LLMClient,
    cfg: Settings,
    llm_model: str,
    target_summary_language: str,
    friendly: int,
    default_topic: str,
    calendar_title: str | None,
    calendar_attendees: Sequence[str],
    progress_callback: ProgressCallback | None,
    chunk_state_store: ChunkStateStore | None = None,
    step_log_callback: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    try:
        compact_transcript = build_compact_transcript(
            speaker_turns,
            aliases=aliases,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    speaker_mapping = compact_transcript.prompt_speaker_mapping()
    transcript_hash = _compact_transcript_hash(compact_transcript.text)
    atomic_write_text(derived_dir / "llm_compact_transcript.txt", compact_transcript.text)
    atomic_write_json(
        derived_dir / "llm_compact_transcript.json",
        compact_transcript.artifact_payload(),
    )
    state_rows = (
        chunk_state_store.list_states(chunk_group=_LLM_CHUNK_GROUP_EXTRACT)
        if chunk_state_store is not None
        else []
    )
    rows_by_id = _active_chunk_rows_by_id(state_rows)
    active_chunks = _load_active_chunk_runtime(
        state_rows,
        transcript_hash=transcript_hash,
    )
    if state_rows and active_chunks is None:
        await _emit_step_log(
            step_log_callback,
            "llm chunk state reset reason=transcript_or_metadata_mismatch",
        )
        assert chunk_state_store is not None
        chunk_state_store.clear_states(chunk_group=_LLM_CHUNK_GROUP_EXTRACT)
        state_rows = []
        rows_by_id = {}

    if active_chunks is None:
        planned_chunks = plan_compact_transcript_chunks(
            compact_transcript,
            max_chars=cfg.llm_chunk_max_chars,
            overlap_chars=cfg.llm_chunk_overlap_chars,
        )
        if not planned_chunks:
            raise RuntimeError("LLM chunk planning produced no chunks")
        active_chunks = _build_root_chunk_runtime(planned_chunks)

    rows_by_id = _sync_chunk_rows(
        chunk_state_store=chunk_state_store,
        active_chunks=active_chunks,
        rows_by_id=rows_by_id,
        transcript_hash=transcript_hash,
    )
    _write_chunk_plan_artifact(
        derived_dir=derived_dir,
        compact_transcript=compact_transcript,
        transcript_text=transcript_text,
        transcript_hash=transcript_hash,
        cfg=cfg,
        active_chunks=active_chunks,
        rows_by_id=rows_by_id,
    )

    chunk_cursor = 0
    while chunk_cursor < len(active_chunks):
        total = len(active_chunks)
        position = chunk_cursor + 1
        chunk = active_chunks[chunk_cursor]
        row = rows_by_id.get(chunk.chunk_id, {})
        status = str(row.get("status") or "").strip().lower()
        raw_path, extract_path, error_path = _chunk_artifact_paths(derived_dir, chunk.chunk_id)
        await _emit_progress(
            progress_callback,
            stage=f"llm_chunk_{chunk.chunk_id}_of_{total}",
            progress=_llm_chunk_progress(position, total),
        )

        if status == "completed":
            reused_extract, reuse_error = _validated_completed_chunk_extract(
                derived_dir=derived_dir,
                chunk=chunk,
                position=position,
                total=total,
            )
            if reused_extract is not None:
                atomic_write_json(extract_path, reused_extract)
                await _emit_step_log(
                    step_log_callback,
                    f"llm chunk resumed index={chunk.chunk_id}",
                )
                chunk_cursor += 1
                continue
            await _emit_step_log(
                step_log_callback,
                f"llm chunk invalidated index={chunk.chunk_id} reason={reuse_error}",
            )
            assert chunk_state_store is not None
            invalidated_row = dict(row)
            invalidated_row["status"] = "planned"
            invalidated_row["finished_at"] = None
            invalidated_row["duration_ms"] = None
            invalidated_row["error_code"] = "llm_chunk_artifact_invalid"
            invalidated_row["error_text"] = reuse_error
            reset_row = chunk_state_store.upsert_state(
                **_runtime_chunk_row_payload(
                    chunk,
                    total=total,
                    transcript_hash=transcript_hash,
                    existing_row=invalidated_row,
                )
            )
            if reset_row is not None:
                rows_by_id[chunk.chunk_id] = reset_row
            _write_chunk_plan_artifact(
                derived_dir=derived_dir,
                compact_transcript=compact_transcript,
                transcript_text=transcript_text,
                transcript_hash=transcript_hash,
                cfg=cfg,
                active_chunks=active_chunks,
                rows_by_id=rows_by_id,
            )

        start_row = (
            chunk_state_store.mark_started(
                chunk_group=_LLM_CHUNK_GROUP_EXTRACT,
                chunk_index=chunk.chunk_id,
                chunk_total=total,
                parent_chunk_index=chunk.parent_chunk_id,
                metadata=chunk.metadata_payload(transcript_hash=transcript_hash),
            )
            if chunk_state_store is not None
            else None
        )
        if start_row is not None:
            rows_by_id[chunk.chunk_id] = start_row
            row = start_row
        attempt = max(int(row.get("attempt") or 0), 1)
        await _emit_step_log(
            step_log_callback,
            (
                f"llm chunk {'resumed' if status in {'failed', 'running'} else 'started'} "
                f"index={chunk.chunk_id} total={total} attempt={attempt} chars={len(chunk.text)}"
            ),
        )

        prompt_chunk = TranscriptChunk(
            index=position,
            total=total,
            text=chunk.text,
            base_text=chunk.base_text,
            overlap_prefix=chunk.overlap_prefix,
            start_seconds=chunk.start_seconds,
            end_seconds=chunk.end_seconds,
        )
        chunk_sys_prompt, chunk_user_prompt = build_chunk_prompt(
            prompt_chunk,
            target_summary_language=target_summary_language,
            calendar_title=calendar_title,
            calendar_attendees=calendar_attendees,
            speaker_mapping=speaker_mapping,
            chunk_id=chunk.chunk_id,
        )
        raw_chunk: dict[str, Any] | None = None
        extract: dict[str, Any] | None = None
        reason_code: str | None = None
        message: str | None = None
        start_perf = time.perf_counter()

        try:
            request_context = getattr(llm, "request_context", None)
            context_manager = (
                request_context(
                    checkpoint="llm_chunk_request",
                    chunk_index=chunk.chunk_id,
                    chunk_total=total,
                )
                if callable(request_context)
                else nullcontext()
            )
            with context_manager:
                raw_chunk = await _generate_llm_message(
                    llm,
                    system_prompt=chunk_sys_prompt,
                    user_prompt=chunk_user_prompt,
                    model=llm_model,
                    response_format={"type": "json_object"},
                    max_tokens=cfg.llm_max_tokens,
                    max_tokens_retry=cfg.llm_max_tokens_retry,
                    timeout_seconds=cfg.llm_chunk_timeout_seconds,
                )
        except asyncio.TimeoutError:
            reason_code = _LLM_CHUNK_REASON_TIMEOUT
            message = _llm_timeout_message(cfg.llm_chunk_timeout_seconds)
        except (TimeoutError, httpx.TimeoutException):
            reason_code = _LLM_CHUNK_REASON_REQUEST_TIMEOUT
            message = _llm_request_timeout_message(_chunk_request_timeout_seconds(llm))
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, ConnectionError) as exc:
            reason_code = _LLM_CHUNK_REASON_CONNECTION_ERROR
            message = str(exc) or exc.__class__.__name__
        except Exception as exc:
            reason_code = _LLM_CHUNK_REASON_RUNTIME_ERROR
            message = str(exc) or exc.__class__.__name__

        if raw_chunk is not None:
            atomic_write_json(raw_path, raw_chunk)
            if _llm_message_timed_out(raw_chunk):
                reason_code = _LLM_CHUNK_REASON_REQUEST_TIMEOUT
                message = _llm_request_timeout_message(_chunk_request_timeout_seconds(llm))
            else:
                try:
                    extract = parse_chunk_extract(str(raw_chunk.get("content") or ""))
                except Exception as exc:
                    reason_code = _LLM_CHUNK_REASON_PARSE_ERROR
                    message = str(exc) or exc.__class__.__name__

        if reason_code is not None:
            child_chunks = (
                _split_runtime_chunk(chunk, cfg=cfg)
                if reason_code in {_LLM_CHUNK_REASON_TIMEOUT, _LLM_CHUNK_REASON_REQUEST_TIMEOUT}
                else []
            )
            if child_chunks:
                child_ids = [item.chunk_id for item in child_chunks]
                split_metadata = chunk.metadata_payload(transcript_hash=transcript_hash)
                split_metadata["split_child_chunk_indexes"] = child_ids
                split_metadata["last_reason_code"] = reason_code
                split_metadata["last_error"] = message
                atomic_write_json(
                    error_path,
                    _chunk_error_payload(
                        chunk=chunk,
                        position=position,
                        total=total,
                        attempt=attempt,
                        reason_code=reason_code,
                        message=message or reason_code,
                        status="split",
                        timeout_seconds=(
                            cfg.llm_chunk_timeout_seconds
                            if reason_code == _LLM_CHUNK_REASON_TIMEOUT
                            else _chunk_request_timeout_seconds(llm)
                        ),
                        child_chunk_indexes=child_ids,
                    ),
                )
                if chunk_state_store is not None:
                    split_row = chunk_state_store.mark_split(
                        chunk_group=_LLM_CHUNK_GROUP_EXTRACT,
                        chunk_index=chunk.chunk_id,
                        chunk_total=total + len(child_chunks) - 1,
                        error_code=reason_code,
                        error_text=message,
                        parent_chunk_index=chunk.parent_chunk_id,
                        metadata=split_metadata,
                    )
                    if split_row is not None:
                        rows_by_id[chunk.chunk_id] = split_row
                active_chunks = [
                    *active_chunks[:chunk_cursor],
                    *child_chunks,
                    *active_chunks[chunk_cursor + 1 :],
                ]
                rows_by_id = _sync_chunk_rows(
                    chunk_state_store=chunk_state_store,
                    active_chunks=active_chunks,
                    rows_by_id=rows_by_id,
                    transcript_hash=transcript_hash,
                )
                _write_chunk_plan_artifact(
                    derived_dir=derived_dir,
                    compact_transcript=compact_transcript,
                    transcript_text=transcript_text,
                    transcript_hash=transcript_hash,
                    cfg=cfg,
                    active_chunks=active_chunks,
                    rows_by_id=rows_by_id,
                )
                await _emit_step_log(
                    step_log_callback,
                    f"llm chunk split index={chunk.chunk_id} into {'/'.join(child_ids)}",
                )
                continue

            atomic_write_json(
                error_path,
                _chunk_error_payload(
                    chunk=chunk,
                    position=position,
                    total=total,
                    attempt=attempt,
                    reason_code=reason_code,
                    message=message or reason_code,
                    status="failed",
                    timeout_seconds=(
                        cfg.llm_chunk_timeout_seconds
                        if reason_code == _LLM_CHUNK_REASON_TIMEOUT
                        else _chunk_request_timeout_seconds(llm)
                        if reason_code == _LLM_CHUNK_REASON_REQUEST_TIMEOUT
                        else None
                    ),
                ),
            )
            if chunk_state_store is not None:
                failed_row = chunk_state_store.mark_failed(
                    chunk_group=_LLM_CHUNK_GROUP_EXTRACT,
                    chunk_index=chunk.chunk_id,
                    chunk_total=total,
                    error_code=reason_code,
                    error_text=message,
                    parent_chunk_index=chunk.parent_chunk_id,
                    metadata=chunk.metadata_payload(transcript_hash=transcript_hash),
                )
                if failed_row is not None:
                    rows_by_id[chunk.chunk_id] = failed_row
            _write_chunk_plan_artifact(
                derived_dir=derived_dir,
                compact_transcript=compact_transcript,
                transcript_text=transcript_text,
                transcript_hash=transcript_hash,
                cfg=cfg,
                active_chunks=active_chunks,
                rows_by_id=rows_by_id,
            )
            await _emit_step_log(
                step_log_callback,
                f"llm chunk failed index={chunk.chunk_id} reason={reason_code}",
            )
            raise RuntimeError(
                f"LLM chunk {chunk.chunk_id}/{total} failed [{reason_code}]: {message}"
            )

        assert extract is not None
        extract["chunk_id"] = chunk.chunk_id
        extract["chunk_index"] = position
        extract["chunk_total"] = total
        extract["parent_chunk_index"] = chunk.parent_chunk_id
        extract["split_depth"] = chunk.depth
        atomic_write_json(extract_path, extract)
        completed_row = (
            chunk_state_store.mark_completed(
                chunk_group=_LLM_CHUNK_GROUP_EXTRACT,
                chunk_index=chunk.chunk_id,
                chunk_total=total,
                parent_chunk_index=chunk.parent_chunk_id,
                metadata=chunk.metadata_payload(transcript_hash=transcript_hash),
            )
            if chunk_state_store is not None
            else None
        )
        if completed_row is not None:
            rows_by_id[chunk.chunk_id] = completed_row
            duration_ms = (
                int(completed_row.get("duration_ms"))
                if completed_row.get("duration_ms") is not None
                else int((time.perf_counter() - start_perf) * 1000)
            )
        else:
            duration_ms = int((time.perf_counter() - start_perf) * 1000)
        _write_chunk_plan_artifact(
            derived_dir=derived_dir,
            compact_transcript=compact_transcript,
            transcript_text=transcript_text,
            transcript_hash=transcript_hash,
            cfg=cfg,
            active_chunks=active_chunks,
            rows_by_id=rows_by_id,
        )
        await _emit_step_log(
            step_log_callback,
            f"llm chunk completed index={chunk.chunk_id} elapsed={duration_ms / 1000:.3f}s",
        )
        chunk_cursor += 1

    chunk_results: list[dict[str, Any]] = []
    total = len(active_chunks)
    for position, chunk in enumerate(active_chunks, start=1):
        extract_payload, reuse_error = _validated_completed_chunk_extract(
            derived_dir=derived_dir,
            chunk=chunk,
            position=position,
            total=total,
        )
        if extract_payload is None:
            raise RuntimeError(
                f"LLM merge failed [{_LLM_MERGE_REASON_PARSE_ERROR}]: {reuse_error}"
            )
        atomic_write_json(_chunk_artifact_paths(derived_dir, chunk.chunk_id)[1], extract_payload)
        chunk_results.append(extract_payload)

    merge_input = merge_chunk_results(chunk_results)
    atomic_write_json(derived_dir / "llm_merge_input.json", merge_input)
    await _emit_progress(progress_callback, stage="llm_merge", progress=0.94)
    await _emit_step_log(
        step_log_callback,
        f"llm merge started chunk_count={len(chunk_results)}",
    )
    merge_sys_prompt, merge_user_prompt = build_merge_prompt(
        merge_input,
        target_summary_language=target_summary_language,
        calendar_title=calendar_title,
        calendar_attendees=calendar_attendees,
    )
    merge_max_tokens = cfg.llm_merge_max_tokens or cfg.llm_max_tokens
    raw_merge: dict[str, Any] | None = None
    merge_reason_code: str | None = None
    merge_message: str | None = None
    try:
        request_context = getattr(llm, "request_context", None)
        context_manager = (
            request_context(
                checkpoint="llm_merge_request",
                chunk_total=len(chunk_results),
            )
            if callable(request_context)
            else nullcontext()
        )
        with context_manager:
            raw_merge = await _generate_llm_message(
                llm,
                system_prompt=merge_sys_prompt,
                user_prompt=merge_user_prompt,
                model=llm_model,
                response_format={"type": "json_object"},
                max_tokens=merge_max_tokens,
                max_tokens_retry=max(cfg.llm_max_tokens_retry, merge_max_tokens),
            )
    except asyncio.TimeoutError:
        merge_reason_code = _LLM_MERGE_REASON_TIMEOUT
        merge_message = _llm_timeout_message(cfg.llm_chunk_timeout_seconds)
    except (TimeoutError, httpx.TimeoutException):
        merge_reason_code = _LLM_MERGE_REASON_REQUEST_TIMEOUT
        merge_message = _llm_request_timeout_message(_chunk_request_timeout_seconds(llm))
    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, ConnectionError) as exc:
        merge_reason_code = _LLM_MERGE_REASON_CONNECTION_ERROR
        merge_message = str(exc) or exc.__class__.__name__
    except Exception as exc:
        merge_reason_code = _LLM_MERGE_REASON_PARSE_ERROR
        merge_message = str(exc) or exc.__class__.__name__

    if raw_merge is not None:
        atomic_write_json(derived_dir / "llm_merge_raw.json", raw_merge)
        if _llm_message_timed_out(raw_merge):
            merge_reason_code = _LLM_MERGE_REASON_REQUEST_TIMEOUT
            merge_message = _llm_request_timeout_message(_chunk_request_timeout_seconds(llm))

    if merge_reason_code is not None:
        atomic_write_json(
            derived_dir / "llm_merge_error.json",
            {
                "reason_code": merge_reason_code,
                "error": merge_message,
                "chunk_count": len(chunk_results),
            },
        )
        await _emit_step_log(
            step_log_callback,
            f"llm merge failed reason={merge_reason_code}",
        )
        raise RuntimeError(f"LLM merge failed [{merge_reason_code}]: {merge_message}")

    try:
        summary_payload = build_summary_payload(
            raw_llm_content=str(raw_merge.get("content") or ""),
            model=llm_model,
            target_summary_language=target_summary_language,
            friendly=friendly,
            default_topic=default_topic,
            derived_dir=derived_dir,
        )
    except Exception as exc:
        merge_reason_code = _LLM_MERGE_REASON_PARSE_ERROR
        merge_message = str(exc) or exc.__class__.__name__
        atomic_write_json(
            derived_dir / "llm_merge_error.json",
            {
                "reason_code": merge_reason_code,
                "error": merge_message,
                "chunk_count": len(chunk_results),
            },
        )
        await _emit_step_log(
            step_log_callback,
            f"llm merge failed reason={merge_reason_code}",
        )
        raise RuntimeError(f"LLM merge failed [{merge_reason_code}]: {merge_message}") from exc

    await _emit_step_log(
        step_log_callback,
        f"llm merge completed chunk_count={len(chunk_results)}",
    )
    return summary_payload


async def run_pipeline(
    audio_path: Path,
    cfg: Settings,
    llm: LLMClient,
    diariser: Diariser,
    recording_id: str | None = None,
    precheck: PrecheckResult | None = None,
    target_summary_language: str | None = None,
    transcript_language_override: str | None = None,
    calendar_title: str | None = None,
    calendar_attendees: Sequence[str] | None = None,
    asr_glossary: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    step_log_callback: Callable[[str], Any] | None = None,
) -> TranscriptResult:
    llm_model = _require_llm_model(cfg.llm_model)
    start = time.perf_counter()
    artifacts = build_recording_artifacts(cfg.recordings_root, recording_id or _default_recording_id(audio_path), audio_path.suffix)
    stage_raw_audio(audio_path, artifacts.raw_audio_path)

    await _emit_progress(progress_callback, stage="precheck", progress=0.05)
    precheck_result = precheck or run_precheck(audio_path, cfg)
    override_lang = normalise_language_code(transcript_language_override)
    summary_lang = resolve_target_summary_language(target_summary_language, dominant_language=override_lang or "unknown", detected_language=None)
    cal_title = str(calendar_title or "").strip() or None
    cal_attendees = [str(item).strip() for item in (calendar_attendees or []) if str(item).strip()]
    language_info: dict[str, Any] = {
        "detected": override_lang or "unknown",
        "confidence": None,
    }
    language_analysis = analyse_languages(
        [],
        detected_language=override_lang,
        transcript_language_override=override_lang,
    )
    diar_segments: list[dict[str, Any]] = []
    speaker_turns: list[dict[str, Any]] = []
    friendly = 0

    atomic_write_json(
        artifacts.metrics_json_path,
        {"status": "running", "version": 1, "precheck": precheck_result.__dict__},
    )

    if precheck_result.quarantine_reason:
        _write_asr_glossary_artifact(
            derived_dir=artifacts.transcript_json_path.parent,
            recording_id=artifacts.recording_id,
            asr_glossary=None,
        )
        await _emit_progress(progress_callback, stage="metrics", progress=0.98)
        _clear_dir(artifacts.snippets_dir)
        atomic_write_text(artifacts.transcript_txt_path, "")
        atomic_write_json(
            artifacts.transcript_json_path,
            _base_transcript_payload(
                recording_id=artifacts.recording_id,
                language={"detected": "unknown", "confidence": None},
                dominant_language=override_lang or "unknown",
                language_distribution={},
                language_spans=[],
                target_summary_language=summary_lang,
                transcript_language_override=override_lang,
                calendar_title=cal_title,
                calendar_attendees=cal_attendees,
                segments=[],
                speakers=[],
                text="",
            ),
        )
        atomic_write_json(artifacts.segments_json_path, [])
        atomic_write_json(artifacts.speaker_turns_json_path, [])
        atomic_write_json(
            artifacts.summary_json_path,
            _build_structured_summary_payload(
                model=llm_model,
                target_summary_language=summary_lang,
                friendly=0,
                topic="Quarantined recording",
                summary_bullets=["Recording was quarantined before transcription."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=_empty_questions(),
                status="quarantined",
                reason=precheck_result.quarantine_reason,
            ),
        )
        atomic_write_json(artifacts.metrics_json_path, {"status": "quarantined", "version": 1, "precheck": precheck_result.__dict__})
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(summary="Quarantined", body="", friendly=0, speakers=[], summary_path=artifacts.summary_json_path, body_path=artifacts.transcript_txt_path, unknown_chunks=[], segments=[])

    _write_asr_glossary_artifact(
        derived_dir=artifacts.transcript_json_path.parent,
        recording_id=artifacts.recording_id,
        asr_glossary=None,
    )

    try:
        await _emit_progress(progress_callback, stage="stt", progress=0.10)
        def _run_asr_workflow() -> tuple[
            list[dict[str, Any]],
            dict[str, Any],
            dict[str, Any],
        ]:
            def _transcribe_chunk(
                chunk_audio_path: Path,
                chunk_language_hint: str | None,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                return _whisperx_asr(
                    chunk_audio_path,
                    override_lang=chunk_language_hint,
                    cfg=cfg,
                    step_log_callback=step_log_callback,
                )

            if _whisperx_asr is not _DEFAULT_WHISPERX_ASR:
                return run_language_aware_asr(
                    audio_path,
                    override_lang=override_lang,
                    configured_mode=cfg.asr_multilingual_mode,
                    tmp_root=cfg.tmp_root,
                    transcribe_fn=_transcribe_chunk,
                    step_log_callback=step_log_callback,
            )

            previous_transcriber = getattr(_whisperx_transcriber_state, "transcribe_audio", None)
            previous_session_flag = bool(
                getattr(_whisperx_transcriber_state, "use_session_transcriber", False)
            )
            transcribe_audio = _build_whisperx_transcriber(
                cfg=cfg,
                step_log_callback=step_log_callback,
                asr_glossary=asr_glossary,
            )
            _whisperx_transcriber_state.transcribe_audio = transcribe_audio
            _whisperx_transcriber_state.use_session_transcriber = True
            try:
                segments, info, payload = run_language_aware_asr(
                    audio_path,
                    override_lang=override_lang,
                    configured_mode=cfg.asr_multilingual_mode,
                    tmp_root=cfg.tmp_root,
                    transcribe_fn=_transcribe_chunk,
                    step_log_callback=step_log_callback,
                )
                runtime_metadata = _glossary_runtime_metadata(transcribe_audio)
                if runtime_metadata:
                    payload = dict(payload)
                    payload["glossary_runtime"] = runtime_metadata
                return segments, info, payload
            finally:
                if previous_transcriber is None:
                    if hasattr(_whisperx_transcriber_state, "transcribe_audio"):
                        delattr(_whisperx_transcriber_state, "transcribe_audio")
                else:
                    _whisperx_transcriber_state.transcribe_audio = previous_transcriber
                if previous_session_flag:
                    _whisperx_transcriber_state.use_session_transcriber = True
                elif hasattr(_whisperx_transcriber_state, "use_session_transcriber"):
                    delattr(_whisperx_transcriber_state, "use_session_transcriber")

        scheduler_plan = _resolve_scheduler_plan(cfg, diariser)
        _update_diariser_runtime_metadata(
            diariser,
            scheduler_mode=scheduler_plan.effective_mode,
            scheduler_reason=scheduler_plan.reason,
            requested_scheduler_mode=scheduler_plan.requested_mode,
            effective_device=scheduler_plan.diarization_device,
        )
        _best_effort_step_log(
            step_log_callback,
            (
                "gpu scheduler "
                f"asr_device={scheduler_plan.asr_device} "
                f"diarization_device={scheduler_plan.diarization_device} "
                f"mode={scheduler_plan.effective_mode} "
                f"reason={scheduler_plan.reason}"
            ),
        )
        if scheduler_plan.effective_mode == "parallel":
            (raw_segments, info, asr_execution), diarization = await asyncio.gather(
                asyncio.to_thread(_run_asr_workflow),
                diariser(audio_path),
            )
        else:
            raw_segments, info, asr_execution = await asyncio.to_thread(_run_asr_workflow)
            await _emit_progress(progress_callback, stage="stt", progress=0.35)
            _cleanup_cuda_memory(scheduler_plan.asr_device)
            if is_gpu_device(scheduler_plan.asr_device):
                _best_effort_step_log(
                    step_log_callback,
                    "gpu scheduler cleared CUDA cache before diarization stage",
                )
            diarization = await diariser(audio_path)
        _write_asr_glossary_artifact(
            derived_dir=artifacts.transcript_json_path.parent,
            recording_id=artifacts.recording_id,
            asr_glossary=_effective_asr_glossary_artifact(
                asr_glossary=asr_glossary,
                runtime_metadata=(
                    asr_execution.get("glossary_runtime")
                    if isinstance(asr_execution, dict)
                    else None
                ),
            ),
        )
        if scheduler_plan.effective_mode == "parallel":
            await _emit_progress(progress_callback, stage="stt", progress=0.35)
        asr_segments = normalise_asr_segments(raw_segments)
        diarization = await _maybe_retry_dialog_diarization(
            diariser=diariser,
            audio_path=audio_path,
            diarization=diarization,
            asr_segments=asr_segments,
            precheck_result=precheck_result,
            step_log_callback=step_log_callback,
        )
        await _emit_progress(progress_callback, stage="diarize", progress=0.60)
        await _emit_progress(progress_callback, stage="align", progress=0.68)
        language_info = _language_payload(info)
        detected_language = normalise_language_code(language_info["detected"]) if language_info["detected"] != "unknown" else None
        language_analysis = analyse_languages(
            asr_segments,
            detected_language=(
                None if asr_execution.get("used_multilingual_path") else detected_language
            ),
            transcript_language_override=override_lang,
        )
        if (
            asr_execution.get("used_multilingual_path")
            or language_info["detected"] == "unknown"
        ) and language_analysis.dominant_language != "unknown":
            language_info["detected"] = language_analysis.dominant_language
            dominant_percent = language_analysis.distribution.get(
                language_analysis.dominant_language
            )
            if dominant_percent is not None:
                language_info["confidence"] = round(dominant_percent / 100.0, 4)
        summary_lang = resolve_target_summary_language(target_summary_language, dominant_language=language_analysis.dominant_language, detected_language=detected_language)
        await _emit_progress(progress_callback, stage="language", progress=0.75)

        asr_text = " ".join(seg.get("text", "").strip() for seg in language_analysis.segments).strip()
        clean_text = normalizer.dedup(asr_text)
        diar_segments = _diarization_segments(diarization)
        used_dummy_fallback = False
        if not diar_segments and language_analysis.segments:
            fallback_end = max(safe_float(seg.get("end")) for seg in language_analysis.segments)
            used_dummy_fallback = True
            _best_effort_step_log(step_log_callback, "diarization output empty; using fallback single-speaker annotation")
            diar_segments = _diarization_segments(_fallback_diarization(max(fallback_end, 0.1)))
        diar_segments_before_flicker = sorted(
            diar_segments,
            key=lambda row: (
                safe_float(row.get("start"), default=0.0),
                safe_float(row.get("end"), default=0.0),
                str(row.get("speaker") or ""),
            ),
        )
        diar_segments = filter_flickering_speakers(
            diar_segments_before_flicker,
            min_total_seconds=cfg.diarization_flicker_min_seconds,
            max_consecutive_segments=cfg.diarization_flicker_max_consecutive,
        )
        flicker_stats: dict[str, dict[str, float]] = {}
        for before_row, after_row in zip(
            diar_segments_before_flicker, diar_segments
        ):
            before_speaker = str(before_row.get("speaker") or "")
            after_speaker = str(after_row.get("speaker") or "")
            if before_speaker == after_speaker:
                continue
            stats = flicker_stats.setdefault(
                before_speaker,
                {"segments": 0.0, "total_seconds": 0.0},
            )
            stats["segments"] += 1.0
            start = safe_float(before_row.get("start"), default=0.0)
            end = safe_float(before_row.get("end"), default=start)
            stats["total_seconds"] += max(end - start, 0.0)
        for speaker, stats in flicker_stats.items():
            _logger.warning(
                "Diarization flicker speaker reassigned: speaker=%s total_seconds=%.3f segments_reassigned=%d",
                speaker,
                stats["total_seconds"],
                int(stats["segments"]),
            )
        speaker_merge_map: dict[str, str] = {}
        speaker_merge_diagnostics: dict[str, Any] = {
            "embedding_model_available": False,
            "speakers_found": [],
            "centroids_computed": [],
            "pairwise_scores": [],
            "merges_applied": {},
            "skipped_reason": None,
        }
        if not cfg.speaker_merge_enabled:
            speaker_merge_diagnostics["skipped_reason"] = "disabled_by_config"
        elif used_dummy_fallback:
            speaker_merge_diagnostics["skipped_reason"] = "dummy_fallback"
        elif _diariser_mode(diariser) != "pyannote":
            speaker_merge_diagnostics["skipped_reason"] = "non_pyannote_diariser"
        elif len({str(row.get("speaker") or "") for row in diar_segments}) < 2:
            speaker_merge_diagnostics["skipped_reason"] = "single_speaker"
        else:
            embedding_model = _resolve_pyannote_embedding_model(diariser)
            if embedding_model is None:
                speaker_merge_diagnostics["skipped_reason"] = (
                    "embedding_model_unavailable"
                )
                _best_effort_step_log(
                    step_log_callback,
                    "speaker_merge skipped: embedding model unavailable",
                )
            else:
                (
                    diar_segments,
                    speaker_merge_map,
                    merge_run_diagnostics,
                ) = merge_similar_speakers(
                    diar_segments,
                    audio_path=audio_path,
                    embedding_model=embedding_model,
                    similarity_threshold=cfg.speaker_merge_similarity_threshold,
                    max_segments_per_speaker=cfg.speaker_merge_max_segments,
                )
                speaker_merge_diagnostics.update(merge_run_diagnostics)
                speaker_merge_diagnostics["skipped_reason"] = None
                if speaker_merge_map:
                    _best_effort_step_log(
                        step_log_callback,
                        (
                            "speaker_merge applied merges="
                            + ",".join(
                                f"{src}->{dst}"
                                for src, dst in sorted(speaker_merge_map.items())
                            )
                        ),
                    )
        unsmoothed_speaker_turns = build_speaker_turns(
            language_analysis.segments,
            diar_segments,
            default_language=language_analysis.dominant_language if language_analysis.dominant_language != "unknown" else detected_language,
            merge_gap_sec=cfg.speaker_turn_merge_gap_sec,
        )
        unsmoothed_speaker_turns = merge_short_turns(
            unsmoothed_speaker_turns,
            min_words=cfg.speaker_turn_min_words,
            merge_gap_sec=cfg.speaker_turn_short_merge_gap_sec,
        )
        diariser_mode = _diariser_mode(diariser)
        if diariser_mode == "pyannote" and not used_dummy_fallback:
            smoothing_result = smooth_speaker_turns(
                unsmoothed_speaker_turns,
                merge_gap_seconds=cfg.diarization_merge_gap_seconds,
                min_turn_seconds=cfg.diarization_min_turn_seconds,
            )
            speaker_turns = smoothing_result.turns
        else:
            speaker_turns = unsmoothed_speaker_turns
            speaker_count = len({str(turn.get("speaker") or "S1") for turn in speaker_turns})
            smoothing_result = SpeakerTurnSmoothingResult(
                turns=speaker_turns,
                adjacent_merges=0,
                micro_turn_absorptions=0,
                turn_count_before=len(speaker_turns),
                turn_count_after=len(speaker_turns),
                speaker_count_before=speaker_count,
                speaker_count_after=speaker_count,
            )
        _write_diarization_metadata_artifact(
            artifacts=artifacts,
            diariser=diariser,
            cfg=cfg,
            smoothing_result=smoothing_result,
            used_dummy_fallback=used_dummy_fallback,
            speaker_merges=speaker_merge_map,
            speaker_merge_diagnostics=speaker_merge_diagnostics,
        )

        aliases = _load_aliases(cfg.speaker_db)
        for row in diar_segments:
            aliases.setdefault(str(row["speaker"]), str(row["speaker"]))
        _save_aliases(aliases, cfg.speaker_db)
    except Exception as exc:
        error_rate_total.inc()
        atomic_write_json(
            artifacts.summary_json_path,
            _build_structured_summary_payload(
                model=llm_model,
                target_summary_language=summary_lang,
                friendly=0,
                topic="Summary generation failed",
                summary_bullets=["Unable to produce a summary due to a processing error."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=_empty_questions(),
                status="failed",
                error=str(exc) or exc.__class__.__name__,
            ),
        )
        atomic_write_json(
            artifacts.metrics_json_path,
            {"status": "failed", "version": 1, "precheck": {**precheck_result.__dict__, "quarantine_reason": None}, "error": str(exc) or exc.__class__.__name__},
        )
        raise

    if not clean_text:
        write_empty_snippets_manifest(
            snippets_dir=artifacts.snippets_dir,
            pad_seconds=cfg.snippet_pad_seconds,
            max_clip_duration_sec=cfg.snippet_max_duration_seconds,
            min_clip_duration_sec=cfg.snippet_min_duration_seconds,
            max_snippets_per_speaker=cfg.snippet_max_per_speaker,
        )
        atomic_write_text(artifacts.transcript_txt_path, "")
        speakers = sorted({aliases.get(row["speaker"], row["speaker"]) for row in diar_segments})
        atomic_write_json(
            artifacts.transcript_json_path,
            _base_transcript_payload(
                recording_id=artifacts.recording_id,
                language=language_info,
                dominant_language=language_analysis.dominant_language,
                language_distribution=language_analysis.distribution,
                language_spans=language_analysis.spans,
                target_summary_language=summary_lang,
                transcript_language_override=override_lang,
                calendar_title=cal_title,
                calendar_attendees=cal_attendees,
                segments=language_analysis.segments,
                speakers=speakers,
                text="",
            ),
        )
        atomic_write_json(artifacts.segments_json_path, diar_segments)
        atomic_write_json(artifacts.speaker_turns_json_path, speaker_turns)
        atomic_write_json(
            artifacts.summary_json_path,
            _build_structured_summary_payload(
                model=llm_model,
                target_summary_language=summary_lang,
                friendly=0,
                topic="No speech detected",
                summary_bullets=["No speech detected."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=_empty_questions(),
                status="no_speech",
            ),
        )
        await _emit_progress(progress_callback, stage="metrics", progress=0.98)
        atomic_write_json(artifacts.metrics_json_path, {"status": "no_speech", "version": 1, "precheck": {**precheck_result.__dict__, "quarantine_reason": None}, "language": language_info, "asr_segments": len(language_analysis.segments), "diar_segments": len(diar_segments), "speaker_turns": len(speaker_turns)})
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(summary="No speech detected", body="", friendly=0, speakers=speakers, summary_path=artifacts.summary_json_path, body_path=artifacts.transcript_txt_path, unknown_chunks=[], segments=[])

    try:
        snippet_paths = export_speaker_snippets(
            SnippetExportRequest(
                audio_path=audio_path,
                diar_segments=diar_segments,
                snippets_dir=artifacts.snippets_dir,
                duration_sec=precheck_result.duration_sec,
                speaker_turns=speaker_turns,
                degraded_diarization=_is_degraded_diarization(
                    diariser,
                    used_dummy_fallback=used_dummy_fallback,
                ),
                pad_seconds=cfg.snippet_pad_seconds,
                max_clip_duration_sec=cfg.snippet_max_duration_seconds,
                min_clip_duration_sec=cfg.snippet_min_duration_seconds,
                max_snippets_per_speaker=cfg.snippet_max_per_speaker,
            )
        )
        speaker_lines = _merge_similar(
            [
                f"[{turn['start']:.2f}-{turn['end']:.2f}] **{aliases.get(turn['speaker'], turn['speaker'])}:** {turn['text']}"
                for turn in speaker_turns
            ],
            cfg.merge_similar,
        )
        friendly = _sentiment_score(clean_text)
        llm_prompt_text = _speaker_turn_prompt_text(speaker_turns, aliases=aliases)
        if _use_chunked_llm(llm_prompt_text, cfg):
            summary_payload = await _run_chunked_llm_summary(
                transcript_text=llm_prompt_text or clean_text,
                speaker_turns=speaker_turns,
                aliases=aliases,
                derived_dir=artifacts.summary_json_path.parent,
                llm=llm,
                cfg=cfg,
                llm_model=llm_model,
                target_summary_language=summary_lang,
                friendly=friendly,
                default_topic=cal_title or "Meeting summary",
                calendar_title=cal_title,
                calendar_attendees=cal_attendees,
                progress_callback=progress_callback,
                step_log_callback=step_log_callback,
            )
        else:
            sys_prompt, user_prompt = build_structured_summary_prompts(
                speaker_turns,
                summary_lang,
                calendar_title=cal_title,
                calendar_attendees=cal_attendees,
            )
            await _emit_progress(progress_callback, stage="llm", progress=0.90)
            msg = await _generate_llm_message(
                llm,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                model=llm_model,
                response_format={"type": "json_object"},
                max_tokens=cfg.llm_max_tokens,
                max_tokens_retry=cfg.llm_max_tokens_retry,
            )
            summary_payload = build_summary_payload(
                raw_llm_content=str(msg.get("content") or ""),
                model=llm_model,
                target_summary_language=summary_lang,
                friendly=friendly,
                default_topic=cal_title or "Meeting summary",
                derived_dir=artifacts.summary_json_path.parent,
            )
        serialised_segments = [SpeakerSegment(start=safe_float(turn["start"]), end=safe_float(turn["end"]), speaker=str(turn["speaker"]), text=str(turn["text"])) for turn in speaker_turns]
        speakers = sorted(set(aliases.get(turn["speaker"], turn["speaker"]) for turn in speaker_turns))
        atomic_write_text(artifacts.transcript_txt_path, clean_text)
        payload = _base_transcript_payload(
            recording_id=artifacts.recording_id,
            language=language_info,
            dominant_language=language_analysis.dominant_language,
            language_distribution=language_analysis.distribution,
            language_spans=language_analysis.spans,
            target_summary_language=summary_lang,
            transcript_language_override=override_lang,
            calendar_title=cal_title,
            calendar_attendees=cal_attendees,
            segments=language_analysis.segments,
            speakers=speakers,
            text=clean_text,
        )
        payload["speaker_lines"] = speaker_lines
        payload["multilingual_asr"] = dict(asr_execution)
        payload["review"] = {
            "required": bool(language_analysis.review_required),
            "reason_code": language_analysis.review_reason_code,
            "reason_text": language_analysis.review_reason_text,
            "uncertain_segment_count": language_analysis.uncertain_segment_count,
            "conflict_segment_count": language_analysis.conflict_segment_count,
        }
        atomic_write_json(artifacts.transcript_json_path, payload)
        atomic_write_json(artifacts.segments_json_path, diar_segments)
        atomic_write_json(artifacts.speaker_turns_json_path, speaker_turns)
        atomic_write_json(artifacts.summary_json_path, summary_payload)
        await _emit_progress(progress_callback, stage="metrics", progress=0.98)
        atomic_write_json(
            artifacts.metrics_json_path,
            {
                "status": "ok",
                "version": 1,
                "precheck": {
                    **precheck_result.__dict__,
                    "quarantine_reason": None,
                },
                "language": language_info,
                "asr_segments": len(language_analysis.segments),
                "diar_segments": len(diar_segments),
                "speaker_turns": len(speaker_turns),
                "snippets": len(snippet_paths),
                "multilingual_asr": {
                    "used_multilingual_path": bool(
                        asr_execution.get("used_multilingual_path")
                    ),
                    "selected_mode": asr_execution.get("selected_mode"),
                },
                "review_required": bool(language_analysis.review_required),
            },
        )
        return TranscriptResult(summary=str(summary_payload.get("summary") or ""), body=clean_text, friendly=friendly, speakers=speakers, summary_path=artifacts.summary_json_path, body_path=artifacts.transcript_txt_path, unknown_chunks=snippet_paths, segments=serialised_segments)
    except Exception as exc:
        error_rate_total.inc()
        atomic_write_json(
            artifacts.summary_json_path,
            _build_structured_summary_payload(
                model=llm_model,
                target_summary_language=summary_lang,
                friendly=friendly,
                topic="Summary generation failed",
                summary_bullets=["Unable to produce a summary due to a processing error."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=_empty_questions(),
                status="failed",
                error=str(exc) or exc.__class__.__name__,
            ),
        )
        atomic_write_json(
            artifacts.metrics_json_path,
            {
                "status": "failed",
                "version": 1,
                "precheck": {**precheck_result.__dict__, "quarantine_reason": None},
                "language": language_info,
                "asr_segments": len(language_analysis.segments),
                "diar_segments": len(diar_segments),
                "speaker_turns": len(speaker_turns),
                "error": str(exc) or exc.__class__.__name__,
            },
        )
        raise
    finally:
        p95_latency_seconds.observe(time.perf_counter() - start)


_segment_language = segment_language

__all__ = [
    "run_pipeline",
    "run_precheck",
    "PrecheckResult",
    "Settings",
    "Diariser",
    "refresh_aliases",
    "build_summary_prompts",
    "build_structured_summary_prompts",
    "build_summary_payload",
    "clear_asr_model_cache",
]
