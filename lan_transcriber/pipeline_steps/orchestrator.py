from __future__ import annotations

import asyncio
import inspect
import logging
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Iterable, List, Literal, Protocol, Sequence

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
    build_chunk_prompt,
    build_merge_prompt,
    merge_chunk_results,
    parse_chunk_extract,
    plan_transcript_chunks,
)
from ..llm_client import LLMClient
from ..metrics import error_rate_total, p95_latency_seconds
from ..models import SpeakerSegment, TranscriptResult
from ..native_fixups import ensure_ctranslate2_no_execstack
from ..torch_safe_globals import (
    omegaconf_safe_globals_for_torch_load,
    unsupported_global_omegaconf_fqn_from_error,
)
from .language import analyse_languages, resolve_target_summary_language, segment_language
from .precheck import PrecheckResult, run_precheck as _run_precheck
from .diarization_quality import (
    DEFAULT_DIALOG_RETRY_MIN_DURATION_SECONDS,
    DEFAULT_DIALOG_RETRY_MIN_TURNS,
    DEFAULT_DIARIZATION_MERGE_GAP_SECONDS,
    DEFAULT_DIARIZATION_MIN_TURN_SECONDS,
    SpeakerTurnSmoothingResult,
    choose_dialog_retry_winner,
    classify_diarization_profile,
    smooth_speaker_turns,
)
from .snippets import SnippetExportRequest, export_speaker_snippets, write_empty_snippets_manifest
from .speaker_turns import (
    _diarization_segments,
    build_speaker_turns,
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
_LLM_MODEL_REQUIRED_ERROR = (
    "LLM_MODEL is required. Set it in .env (e.g., LLM_MODEL=gpt-oss:120b)."
)
_LLM_TIMEOUT_SENTINEL = "**LLM timeout**"


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
        default=6000,
        ge=1,
        validation_alias=AliasChoices(
            "llm_chunk_max_chars",
            "LLM_CHUNK_MAX_CHARS",
            "LAN_LLM_CHUNK_MAX_CHARS",
        ),
    )
    llm_chunk_overlap_chars: int = Field(
        default=600,
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
    llm_long_transcript_threshold_chars: int = Field(
        default=6000,
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
    asr_device: str = "auto"
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
    preferred = str(cfg.asr_device or "auto").strip().lower()
    if preferred in {"cpu", "cuda"}:
        return preferred
    try:
        import torch
    except Exception:
        _logger.info(
            "Torch CUDA runtime: is_available=%s device_count=%s torch.version.cuda=%s",
            False,
            0,
            None,
        )
        return "cpu"
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    _logger.info(
        "Torch CUDA runtime: is_available=%s device_count=%s torch.version.cuda=%s",
        cuda_available,
        device_count,
        cuda_version,
    )
    return "cuda" if cuda_available else "cpu"


def _select_compute_type(cfg: Settings, device: str) -> str:
    configured = str(cfg.asr_compute_type or "").strip()
    if configured:
        return configured
    return "float16" if device == "cuda" else "int8"


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
    atomic_write_json(artifacts.diarization_metadata_json_path, payload)


def _build_whisperx_transcriber(
    cfg: Settings,
    step_log_callback: Callable[[str], Any] | None = None,
    asr_glossary: dict[str, Any] | None = None,
) -> Callable[[Path, str | None], tuple[list[dict[str, Any]], dict[str, Any]]]:
    glossary_kwargs = _glossary_transcribe_kwargs(asr_glossary)
    glossary_log_state = {"unsupported": False}
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

        return _legacy_transcribe

    device = _select_asr_device(cfg)
    compute_type = _select_compute_type(cfg, device)
    _logger.info("ASR VAD method: %s", cfg.vad_method)
    model_load_kwargs = {"compute_type": compute_type, "vad_method": cfg.vad_method}

    def _load_model() -> Any:
        return call_with_supported_kwargs(wx_asr.load_model, cfg.asr_model, device, **model_load_kwargs)

    with omegaconf_safe_globals_for_torch_load():
        try:
            model = _load_model()
        except Exception as first_error:
            retry_fqn = unsupported_global_omegaconf_fqn_from_error(first_error)
            if retry_fqn is None:
                raise
            with omegaconf_safe_globals_for_torch_load(extra_fqns=[retry_fqn]):
                try:
                    model = _load_model()
                except Exception:
                    raise first_error

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

    return _modern_transcribe


def _whisperx_asr(
    audio_path: Path,
    *,
    override_lang: str | None,
    cfg: Settings,
    step_log_callback: Callable[[str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transcribe_audio = getattr(_whisperx_transcriber_state, "transcribe_audio", None)
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
) -> dict[str, Any]:
    chunks = plan_transcript_chunks(
        transcript_text,
        max_chars=cfg.llm_chunk_max_chars,
        overlap_chars=cfg.llm_chunk_overlap_chars,
    )
    atomic_write_json(
        derived_dir / "llm_chunks_plan.json",
        {
            "chunk_max_chars": cfg.llm_chunk_max_chars,
            "chunk_overlap_chars": cfg.llm_chunk_overlap_chars,
            "long_transcript_threshold_chars": cfg.llm_long_transcript_threshold_chars,
            "chunks": [chunk.plan_payload() for chunk in chunks],
        },
    )
    if not chunks:
        raise RuntimeError("LLM chunk planning produced no chunks")

    chunk_results: list[dict[str, Any]] = []
    for chunk in chunks:
        await _emit_progress(
            progress_callback,
            stage=f"llm_chunk_{chunk.index}_of_{chunk.total}",
            progress=_llm_chunk_progress(chunk.index, chunk.total),
        )
        chunk_sys_prompt, chunk_user_prompt = build_chunk_prompt(
            chunk,
            target_summary_language=target_summary_language,
            calendar_title=calendar_title,
            calendar_attendees=calendar_attendees,
        )
        error_path = derived_dir / f"llm_chunk_{chunk.index:03d}_error.json"
        try:
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
            atomic_write_json(derived_dir / f"llm_chunk_{chunk.index:03d}_raw.json", raw_chunk)
            if _llm_message_timed_out(raw_chunk):
                raise TimeoutError(_LLM_TIMEOUT_SENTINEL)
            extract = parse_chunk_extract(str(raw_chunk.get("content") or ""))
        except (TimeoutError, asyncio.TimeoutError) as exc:
            message = _llm_timeout_message(cfg.llm_chunk_timeout_seconds)
            atomic_write_json(error_path, {"error": message})
            raise RuntimeError(f"LLM chunk {chunk.index}/{chunk.total} failed: {message}") from exc
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            atomic_write_json(error_path, {"error": message})
            raise RuntimeError(f"LLM chunk {chunk.index}/{chunk.total} failed: {message}") from exc

        extract["chunk_index"] = chunk.index
        extract["chunk_total"] = chunk.total
        atomic_write_json(derived_dir / f"llm_chunk_{chunk.index:03d}_extract.json", extract)
        chunk_results.append(extract)

    merge_input = merge_chunk_results(chunk_results)
    atomic_write_json(derived_dir / "llm_merge_input.json", merge_input)
    await _emit_progress(progress_callback, stage="llm_merge", progress=0.94)
    merge_sys_prompt, merge_user_prompt = build_merge_prompt(
        merge_input,
        target_summary_language=target_summary_language,
        calendar_title=calendar_title,
        calendar_attendees=calendar_attendees,
    )
    merge_max_tokens = cfg.llm_merge_max_tokens or cfg.llm_max_tokens
    raw_merge = await _generate_llm_message(
        llm,
        system_prompt=merge_sys_prompt,
        user_prompt=merge_user_prompt,
        model=llm_model,
        response_format={"type": "json_object"},
        max_tokens=merge_max_tokens,
        max_tokens_retry=max(cfg.llm_max_tokens_retry, merge_max_tokens),
    )
    atomic_write_json(derived_dir / "llm_merge_raw.json", raw_merge)
    if _llm_message_timed_out(raw_merge):
        message = _llm_timeout_message(getattr(llm, "timeout", None))
        raise RuntimeError(f"LLM merge failed: {message}")
    return build_summary_payload(
        raw_llm_content=str(raw_merge.get("content") or ""),
        model=llm_model,
        target_summary_language=target_summary_language,
        friendly=friendly,
        default_topic=default_topic,
        derived_dir=derived_dir,
    )


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
        asr_glossary=asr_glossary,
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
            _whisperx_transcriber_state.transcribe_audio = _build_whisperx_transcriber(
                cfg=cfg,
                step_log_callback=step_log_callback,
                asr_glossary=asr_glossary,
            )
            try:
                return run_language_aware_asr(
                    audio_path,
                    override_lang=override_lang,
                    configured_mode=cfg.asr_multilingual_mode,
                    tmp_root=cfg.tmp_root,
                    transcribe_fn=_transcribe_chunk,
                    step_log_callback=step_log_callback,
                )
            finally:
                if previous_transcriber is None:
                    if hasattr(_whisperx_transcriber_state, "transcribe_audio"):
                        delattr(_whisperx_transcriber_state, "transcribe_audio")
                else:
                    _whisperx_transcriber_state.transcribe_audio = previous_transcriber

        (raw_segments, info, asr_execution), diarization = await asyncio.gather(
            asyncio.to_thread(
                _run_asr_workflow,
            ),
            diariser(audio_path),
        )
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
        unsmoothed_speaker_turns = build_speaker_turns(
            language_analysis.segments,
            diar_segments,
            default_language=language_analysis.dominant_language if language_analysis.dominant_language != "unknown" else detected_language,
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

    try:
        if _use_chunked_llm(llm_prompt_text, cfg):
            summary_payload = await _run_chunked_llm_summary(
                transcript_text=llm_prompt_text or clean_text,
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
]
