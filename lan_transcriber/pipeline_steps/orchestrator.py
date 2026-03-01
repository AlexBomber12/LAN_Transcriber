from __future__ import annotations

import asyncio
import inspect
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Iterable, List, Protocol, Sequence

from pydantic_settings import BaseSettings

from .. import normalizer
from ..aliases import ALIAS_PATH, load_aliases as _load_aliases, save_aliases as _save_aliases
from ..artifacts import atomic_write_json, atomic_write_text, build_recording_artifacts, stage_raw_audio
from ..compat.call_compat import call_with_supported_kwargs, filter_kwargs_for_callable
from ..compat.pyannote_compat import patch_pyannote_inference_ignore_use_auth_token
from ..llm_client import LLMClient
from ..metrics import error_rate_total, p95_latency_seconds
from ..models import SpeakerSegment, TranscriptResult
from ..native_fixups import ensure_ctranslate2_no_execstack
from .language import analyse_languages, resolve_target_summary_language, segment_language
from .precheck import PrecheckResult, run_precheck as _run_precheck
from .snippets import SnippetExportRequest, export_speaker_snippets
from .speaker_turns import (
    _diarization_segments,
    build_speaker_turns,
    normalise_asr_segments,
)
from .summary_builder import (
    _build_structured_summary_payload,
    build_structured_summary_prompts,
    build_summary_payload,
    build_summary_prompts,
)
from ..runtime_paths import default_recordings_root, default_tmp_root, default_unknown_dir, default_voices_dir
from ..utils import normalise_language_code, safe_float


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
    asr_model: str = "large-v3"
    asr_device: str = "auto"
    asr_compute_type: str | None = None
    asr_batch_size: int = 16
    asr_enable_align: bool = True
    embed_threshold: float = 0.65
    merge_similar: float = 0.9
    precheck_min_duration_sec: float = 20.0
    precheck_min_speech_ratio: float = 0.10

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


def _sentiment_score(text: str) -> int:  # pragma: no cover - trivial wrapper
    from transformers import pipeline as hf_pipeline

    sent = hf_pipeline("sentiment-analysis")(text[:4000])[0]
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
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_compute_type(cfg: Settings, device: str) -> str:
    configured = str(cfg.asr_compute_type or "").strip()
    if configured:
        return configured
    return "float16" if device == "cuda" else "int8"


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


def _whisperx_asr(
    audio_path: Path,
    *,
    override_lang: str | None,
    cfg: Settings,
    step_log_callback: Callable[[str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    if patch_pyannote_inference_ignore_use_auth_token() and step_log_callback is not None:
        try:
            step_log_callback("pyannote compat: ignore unsupported Inference use_auth_token")
        except Exception:
            # Step log append is best-effort and must not break processing.
            pass

    import whisperx

    legacy_transcribe = getattr(whisperx, "transcribe", None)
    if legacy_transcribe is not None and not callable(legacy_transcribe):
        raise TypeError(
            f"whisperx.transcribe must be callable or None, got {type(legacy_transcribe).__name__}"
        )

    if callable(legacy_transcribe):
        kwargs: dict[str, Any] = {
            "vad_filter": True,
            "language": override_lang or "auto",
            "word_timestamps": True,
        }
        filtered_kwargs = filter_kwargs_for_callable(legacy_transcribe, kwargs)
        _log_dropped_kwargs(
            callback=step_log_callback,
            scope="whisperx transcribe",
            attempted=kwargs,
            filtered=filtered_kwargs,
        )
        try:
            segments, info = call_with_supported_kwargs(legacy_transcribe, str(audio_path), **kwargs)
        except TypeError:
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("word_timestamps", None)
            _log_dropped_kwargs(
                callback=step_log_callback,
                scope="whisperx transcribe",
                attempted=kwargs,
                filtered=retry_kwargs,
            )
            segments, info = call_with_supported_kwargs(legacy_transcribe, str(audio_path), **retry_kwargs)
        return list(segments), dict(info or {})

    device = _select_asr_device(cfg)
    compute_type = _select_compute_type(cfg, device)
    audio = whisperx.load_audio(str(audio_path))
    try:
        model = whisperx.load_model(cfg.asr_model, device, compute_type=compute_type)
    except TypeError:
        model = whisperx.load_model(cfg.asr_model, device)

    transcribe_kwargs: dict[str, Any] = {
        "batch_size": cfg.asr_batch_size,
        "vad_filter": True,
        "language": (override_lang if override_lang else None),
    }
    filtered_kwargs = filter_kwargs_for_callable(model.transcribe, transcribe_kwargs)
    _log_dropped_kwargs(
        callback=step_log_callback,
        scope="whisperx transcribe",
        attempted=transcribe_kwargs,
        filtered=filtered_kwargs,
    )
    result = call_with_supported_kwargs(model.transcribe, audio, **transcribe_kwargs)
    segments = list(result.get("segments", []))
    info: dict[str, Any] = {"language": result.get("language") or (override_lang or "unknown")}

    if cfg.asr_enable_align:
        try:
            align_lang = normalise_language_code(info.get("language")) or "en"
            model_a, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
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
    progress_callback: ProgressCallback | None = None,
    step_log_callback: Callable[[str], Any] | None = None,
) -> TranscriptResult:
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
        await _emit_progress(progress_callback, stage="metrics", progress=0.95)
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
                model=cfg.llm_model,
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
        await _emit_progress(progress_callback, stage="done", progress=1.0)
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(summary="Quarantined", body="", friendly=0, speakers=[], summary_path=artifacts.summary_json_path, body_path=artifacts.transcript_txt_path, unknown_chunks=[], segments=[])

    try:
        await _emit_progress(progress_callback, stage="stt", progress=0.30)

        await _emit_progress(progress_callback, stage="diarize", progress=0.50)
        (raw_segments, info), diarization = await asyncio.gather(
            asyncio.to_thread(
                _whisperx_asr,
                audio_path,
                override_lang=override_lang,
                cfg=cfg,
                step_log_callback=step_log_callback,
            ),
            diariser(audio_path),
        )
        await _emit_progress(progress_callback, stage="align", progress=0.60)
        asr_segments = normalise_asr_segments(raw_segments)
        language_info = _language_payload(info)
        await _emit_progress(progress_callback, stage="language", progress=0.70)
        detected_language = normalise_language_code(language_info["detected"]) if language_info["detected"] != "unknown" else None
        language_analysis = analyse_languages(asr_segments, detected_language=detected_language, transcript_language_override=override_lang)
        if language_info["detected"] == "unknown" and language_analysis.dominant_language != "unknown":
            language_info["detected"] = language_analysis.dominant_language
        summary_lang = resolve_target_summary_language(target_summary_language, dominant_language=language_analysis.dominant_language, detected_language=detected_language)

        asr_text = " ".join(seg.get("text", "").strip() for seg in language_analysis.segments).strip()
        clean_text = normalizer.dedup(asr_text)
        diar_segments = _diarization_segments(diarization)
        if not diar_segments and language_analysis.segments:
            fallback_end = max(safe_float(seg.get("end")) for seg in language_analysis.segments)
            diar_segments = _diarization_segments(_fallback_diarization(max(fallback_end, 0.1)))
        speaker_turns = build_speaker_turns(
            language_analysis.segments,
            diar_segments,
            default_language=language_analysis.dominant_language if language_analysis.dominant_language != "unknown" else detected_language,
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
                model=cfg.llm_model,
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
        _clear_dir(artifacts.snippets_dir)
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
                model=cfg.llm_model,
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
        await _emit_progress(progress_callback, stage="metrics", progress=0.95)
        atomic_write_json(artifacts.metrics_json_path, {"status": "no_speech", "version": 1, "precheck": {**precheck_result.__dict__, "quarantine_reason": None}, "language": language_info, "asr_segments": len(language_analysis.segments), "diar_segments": len(diar_segments), "speaker_turns": len(speaker_turns)})
        await _emit_progress(progress_callback, stage="done", progress=1.0)
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(summary="No speech detected", body="", friendly=0, speakers=speakers, summary_path=artifacts.summary_json_path, body_path=artifacts.transcript_txt_path, unknown_chunks=[], segments=[])

    snippet_paths = export_speaker_snippets(SnippetExportRequest(audio_path=audio_path, diar_segments=diar_segments, snippets_dir=artifacts.snippets_dir, duration_sec=precheck_result.duration_sec))
    speaker_lines = _merge_similar(
        [
            f"[{turn['start']:.2f}-{turn['end']:.2f}] **{aliases.get(turn['speaker'], turn['speaker'])}:** {turn['text']}"
            for turn in speaker_turns
        ],
        cfg.merge_similar,
    )
    friendly = _sentiment_score(clean_text)
    sys_prompt, user_prompt = build_structured_summary_prompts(speaker_turns, summary_lang, calendar_title=cal_title, calendar_attendees=cal_attendees)

    try:
        await _emit_progress(progress_callback, stage="llm", progress=0.85)
        msg = await llm.generate(system_prompt=sys_prompt, user_prompt=user_prompt, model=cfg.llm_model, response_format={"type": "json_object"})
        raw_summary = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        summary_payload = build_summary_payload(
            raw_llm_content=raw_summary,
            model=cfg.llm_model,
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
        atomic_write_json(artifacts.transcript_json_path, payload)
        atomic_write_json(artifacts.segments_json_path, diar_segments)
        atomic_write_json(artifacts.speaker_turns_json_path, speaker_turns)
        atomic_write_json(artifacts.summary_json_path, summary_payload)
        await _emit_progress(progress_callback, stage="metrics", progress=0.95)
        atomic_write_json(artifacts.metrics_json_path, {"status": "ok", "version": 1, "precheck": {**precheck_result.__dict__, "quarantine_reason": None}, "language": language_info, "asr_segments": len(language_analysis.segments), "diar_segments": len(diar_segments), "speaker_turns": len(speaker_turns), "snippets": len(snippet_paths)})
        await _emit_progress(progress_callback, stage="done", progress=1.0)
        return TranscriptResult(summary=str(summary_payload.get("summary") or ""), body=clean_text, friendly=friendly, speakers=speakers, summary_path=artifacts.summary_json_path, body_path=artifacts.transcript_txt_path, unknown_chunks=snippet_paths, segments=serialised_segments)
    except Exception as exc:
        error_rate_total.inc()
        atomic_write_json(
            artifacts.summary_json_path,
            _build_structured_summary_payload(
                model=cfg.llm_model,
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
