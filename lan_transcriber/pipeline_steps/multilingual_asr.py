from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from lan_transcriber.utils import normalise_language_code, safe_float

from .language import LanguageAnalysis, analyse_languages

AUTO_MIN_SWITCH_SPAN_SECONDS = 3.0
AUTO_MIN_SWITCH_SHARE_PERCENT = 15.0
CHUNK_MERGE_GAP_SECONDS = 0.75
HIGH_CONFIDENCE_THRESHOLD = 0.75

MultilingualMode = Literal["auto", "force_single_language", "force_multilingual"]
TranscribeFn = Callable[[Path, str | None], tuple[list[dict[str, Any]], dict[str, Any]]]


@dataclass(frozen=True)
class ChunkPlan:
    start: float
    end: float
    language: str
    confidence: float | None
    language_hint: str | None
    uncertain: bool
    conflict: bool
    segment_count: int

    def to_payload(self, *, index: int, total: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": index,
            "total": total,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "language": self.language,
            "segment_count": self.segment_count,
            "language_hint": self.language_hint,
            "hint_applied": self.language_hint is not None,
            "uncertain": self.uncertain,
            "conflict": self.conflict,
        }
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 4)
        return payload


def _log(callback: Callable[[str], Any] | None, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        pass


def _language_from_info(info: dict[str, Any]) -> tuple[str | None, float | None]:
    language = normalise_language_code(
        info.get("language") or info.get("detected_language") or info.get("lang")
    )
    confidence_raw = None
    for key in (
        "language_probability",
        "language_confidence",
        "language_score",
        "probability",
    ):
        if key in info and info[key] is not None:
            confidence_raw = info[key]
            break
    if confidence_raw is None:
        return language, None
    return language, round(safe_float(confidence_raw, default=0.0), 4)


def should_use_multilingual_path(
    analysis: LanguageAnalysis,
    *,
    configured_mode: MultilingualMode,
) -> tuple[bool, str]:
    if configured_mode == "force_single_language":
        return False, "forced_single_language"
    if not analysis.segments:
        return False, "no_segments"
    if configured_mode == "force_multilingual":
        return True, "forced_multilingual"

    significant_languages = {
        lang
        for lang, percent in analysis.distribution.items()
        if lang != "unknown" and percent >= AUTO_MIN_SWITCH_SHARE_PERCENT
    }
    switched_languages = {
        str(span.get("lang") or "unknown")
        for span in analysis.spans
        if str(span.get("lang") or "unknown") != "unknown"
        and (
            safe_float(span.get("end"), default=0.0)
            - safe_float(span.get("start"), default=0.0)
        )
        >= AUTO_MIN_SWITCH_SPAN_SECONDS
    }
    if len(significant_languages) >= 2 and len(switched_languages) >= 2:
        return True, "credible_language_switches"
    return False, "dominant_single_language"


def plan_multilingual_chunks(
    analysis: LanguageAnalysis,
    *,
    merge_gap_seconds: float = CHUNK_MERGE_GAP_SECONDS,
) -> list[ChunkPlan]:
    chunks: list[ChunkPlan] = []
    for row in analysis.segments:
        start = safe_float(row.get("start"), default=0.0)
        end = safe_float(row.get("end"), default=start)
        if end < start:
            end = start
        language = normalise_language_code(row.get("language")) or "unknown"
        confidence_raw = row.get("language_confidence")
        confidence = (
            None
            if confidence_raw is None
            else round(safe_float(confidence_raw, default=0.0), 4)
        )
        uncertain = bool(row.get("language_uncertain"))
        conflict = bool(row.get("language_conflict"))
        language_hint = None if uncertain or language == "unknown" else language

        if (
            chunks
            and chunks[-1].language == language
            and chunks[-1].language_hint == language_hint
            and chunks[-1].uncertain == uncertain
            and start <= chunks[-1].end + max(merge_gap_seconds, 0.0)
        ):
            previous = chunks[-1]
            chunks[-1] = ChunkPlan(
                start=previous.start,
                end=max(previous.end, end),
                language=previous.language,
                confidence=previous.confidence if previous.confidence is not None else confidence,
                language_hint=previous.language_hint,
                uncertain=previous.uncertain,
                conflict=previous.conflict or conflict,
                segment_count=previous.segment_count + 1,
            )
            continue

        chunks.append(
            ChunkPlan(
                start=start,
                end=end,
                language=language,
                confidence=confidence,
                language_hint=language_hint,
                uncertain=uncertain,
                conflict=conflict,
                segment_count=1,
            )
        )
    return chunks


def _write_wav_chunk(
    source_audio_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> Path:
    with wave.open(str(source_audio_path), "rb") as source:
        params = source.getparams()
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        start_frame = max(0, min(int(start_sec * frame_rate), total_frames))
        end_frame = max(0, min(int(end_sec * frame_rate), total_frames))
        if end_frame <= start_frame:
            end_frame = min(total_frames, start_frame + 1)
        source.setpos(start_frame)
        frames = source.readframes(max(end_frame - start_frame, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)
    return output_path


def _offset_segment_words(words: Any, *, offset_seconds: float) -> list[dict[str, Any]]:
    if not isinstance(words, list):
        return []
    shifted: list[dict[str, Any]] = []
    for row in words:
        if not isinstance(row, dict):
            continue
        payload = dict(row)
        if "start" in payload:
            payload["start"] = round(
                safe_float(payload.get("start"), default=0.0) + offset_seconds,
                3,
            )
        if "end" in payload:
            payload["end"] = round(
                safe_float(payload.get("end"), default=0.0) + offset_seconds,
                3,
            )
        shifted.append(payload)
    return shifted


def _annotate_chunk_segments(
    raw_segments: list[dict[str, Any]],
    *,
    offset_seconds: float,
    language: str,
    confidence: float | None,
    language_hint: str | None,
    uncertain: bool,
    conflict: bool,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in raw_segments:
        payload = dict(row)
        payload["start"] = round(
            safe_float(payload.get("start"), default=0.0) + offset_seconds,
            3,
        )
        payload["end"] = round(
            safe_float(payload.get("end"), default=payload["start"]) + offset_seconds,
            3,
        )
        if payload["end"] < payload["start"]:
            payload["end"] = payload["start"]
        payload["words"] = _offset_segment_words(
            payload.get("words"),
            offset_seconds=offset_seconds,
        )

        seg_language = normalise_language_code(payload.get("language")) or language
        if seg_language:
            payload["language"] = seg_language
        if confidence is not None:
            payload["language_confidence"] = round(confidence, 4)
        payload["language_source"] = (
            "chunk_hint" if language_hint is not None else "chunk_detected"
        )
        if language_hint is not None:
            payload["language_hint"] = language_hint
            payload["language_hint_applied"] = True
        if uncertain:
            payload["language_uncertain"] = True
        if conflict:
            payload["language_conflict"] = True
            payload["language_uncertain"] = True
        annotated.append(payload)
    return annotated


def _select_segments_for_range(
    raw_segments: list[dict[str, Any]],
    *,
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in raw_segments:
        if not isinstance(row, dict):
            continue
        seg_start = safe_float(row.get("start"), default=0.0)
        seg_end = safe_float(row.get("end"), default=seg_start)
        if seg_end < seg_start:
            seg_end = seg_start
        if seg_end <= start_seconds or seg_start >= end_seconds:
            continue
        selected.append(dict(row))
    return selected


def _merge_result_language_info(
    chunk_payloads: list[dict[str, Any]],
    *,
    fallback_info: dict[str, Any],
) -> dict[str, Any]:
    weighted: dict[str, float] = {}
    for chunk in chunk_payloads:
        language = normalise_language_code(
            chunk.get("result_language")
            or chunk.get("language")
            or chunk.get("language_hint")
        )
        if language is None:
            continue
        duration = max(
            0.0,
            safe_float(chunk.get("end"), default=0.0)
            - safe_float(chunk.get("start"), default=0.0),
        )
        if duration <= 0.0:
            duration = 0.001
        weighted[language] = weighted.get(language, 0.0) + duration

    merged = dict(fallback_info)
    if not weighted:
        return merged

    ordered = sorted(weighted.items(), key=lambda item: (-item[1], item[0]))
    detected_language = next(
        (language for language, _weight in ordered if language != "unknown"),
        ordered[0][0],
    )
    total = sum(weighted.values())
    dominant_weight = weighted.get(detected_language, 0.0)
    merged["language"] = detected_language
    merged["language_probability"] = round(
        (dominant_weight / total) if total > 0 else 0.0,
        4,
    )
    return merged


def run_language_aware_asr(
    audio_path: Path,
    *,
    override_lang: str | None,
    configured_mode: MultilingualMode,
    tmp_root: Path,
    transcribe_fn: TranscribeFn,
    step_log_callback: Callable[[str], Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if override_lang:
        segments, info = transcribe_fn(audio_path, override_lang)
        payload = {
            "configured_mode": configured_mode,
            "selected_mode": "single_language",
            "selection_reason": "transcript_language_override",
            "used_multilingual_path": False,
            "chunks": [],
        }
        _log(
            step_log_callback,
            "multilingual asr: single_language reason=transcript_language_override",
        )
        return segments, info, payload

    initial_segments, initial_info = transcribe_fn(audio_path, None)
    detected_language, _confidence = _language_from_info(initial_info)
    initial_analysis = analyse_languages(
        initial_segments,
        detected_language=detected_language,
        transcript_language_override=None,
    )
    use_multilingual, selection_reason = should_use_multilingual_path(
        initial_analysis,
        configured_mode=configured_mode,
    )
    if not use_multilingual:
        payload = {
            "configured_mode": configured_mode,
            "selected_mode": "single_language",
            "selection_reason": selection_reason,
            "used_multilingual_path": False,
            "initial_detected_language": detected_language or "unknown",
            "initial_language_distribution": initial_analysis.distribution,
            "chunks": [],
        }
        _log(
            step_log_callback,
            f"multilingual asr: single_language reason={selection_reason}",
        )
        return initial_segments, initial_info, payload

    chunks = plan_multilingual_chunks(initial_analysis)
    if not chunks:
        payload = {
            "configured_mode": configured_mode,
            "selected_mode": "single_language",
            "selection_reason": "empty_chunk_plan",
            "used_multilingual_path": False,
            "initial_detected_language": detected_language or "unknown",
            "initial_language_distribution": initial_analysis.distribution,
            "chunks": [],
        }
        _log(
            step_log_callback,
            "multilingual asr: single_language reason=empty_chunk_plan",
        )
        return initial_segments, initial_info, payload

    if audio_path.suffix.lower() != ".wav":
        if configured_mode == "force_multilingual":
            raise ValueError(
                "force_multilingual requires a WAV input; sanitize or transcode before chunking"
            )
        payload = {
            "configured_mode": configured_mode,
            "selected_mode": "single_language",
            "selection_reason": "multilingual_chunking_requires_wav",
            "used_multilingual_path": False,
            "initial_detected_language": detected_language or "unknown",
            "initial_language_distribution": initial_analysis.distribution,
            "chunks": [],
        }
        _log(
            step_log_callback,
            "multilingual asr: single_language reason=multilingual_chunking_requires_wav",
        )
        return initial_segments, initial_info, payload

    tmp_root.mkdir(parents=True, exist_ok=True)
    merged_segments: list[dict[str, Any]] = []
    chunk_payloads: list[dict[str, Any]] = []
    _log(
        step_log_callback,
        (
            "multilingual asr: multilingual "
            f"reason={selection_reason} chunks={len(chunks)}"
        ),
    )
    with tempfile.TemporaryDirectory(
        dir=str(tmp_root),
        prefix="multilingual-asr-",
    ) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for index, chunk in enumerate(chunks, start=1):
            chunk_audio_path = _write_wav_chunk(
                audio_path,
                start_sec=chunk.start,
                end_sec=chunk.end,
                output_path=tmp_dir / f"chunk-{index:03d}.wav",
            )
            chunk_segments, chunk_info = transcribe_fn(chunk_audio_path, chunk.language_hint)
            result_language, result_confidence = _language_from_info(chunk_info)
            conflict = bool(
                chunk.language_hint is not None
                and result_language is not None
                and result_language != chunk.language_hint
            )
            confident_result = (
                result_language is not None
                and result_confidence is not None
                and result_confidence >= HIGH_CONFIDENCE_THRESHOLD
            )
            uncertain = bool(chunk.uncertain and not confident_result) or conflict
            effective_language = result_language or chunk.language
            effective_confidence = (
                result_confidence
                if result_confidence is not None
                else chunk.confidence
            )
            annotated_segments = _annotate_chunk_segments(
                chunk_segments,
                offset_seconds=chunk.start,
                language=effective_language,
                confidence=effective_confidence,
                language_hint=chunk.language_hint,
                uncertain=uncertain,
                conflict=conflict,
            )
            if not annotated_segments:
                annotated_segments = _annotate_chunk_segments(
                    _select_segments_for_range(
                        initial_segments,
                        start_seconds=chunk.start,
                        end_seconds=chunk.end,
                    ),
                    offset_seconds=0.0,
                    language=effective_language,
                    confidence=effective_confidence,
                    language_hint=chunk.language_hint,
                    uncertain=uncertain,
                    conflict=conflict,
                )

            merged_segments.extend(annotated_segments)
            payload = chunk.to_payload(index=index, total=len(chunks))
            payload["result_language"] = effective_language
            payload["result_uncertain"] = uncertain
            payload["conflict"] = conflict
            if effective_confidence is not None:
                payload["result_confidence"] = round(effective_confidence, 4)
            chunk_payloads.append(payload)

    payload = {
        "configured_mode": configured_mode,
        "selected_mode": "multilingual",
        "selection_reason": selection_reason,
        "used_multilingual_path": True,
        "initial_detected_language": detected_language or "unknown",
        "initial_language_distribution": initial_analysis.distribution,
        "chunks": chunk_payloads,
    }
    return merged_segments, _merge_result_language_info(
        chunk_payloads,
        fallback_info=initial_info,
    ), payload


__all__ = [
    "ChunkPlan",
    "run_language_aware_asr",
    "plan_multilingual_chunks",
    "should_use_multilingual_path",
]
