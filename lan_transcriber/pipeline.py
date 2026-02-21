from __future__ import annotations

import asyncio
import audioop
import json
import re
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List, Protocol, Sequence

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


@dataclass(frozen=True)
class PrecheckResult:
    duration_sec: float | None
    speech_ratio: float | None
    quarantine_reason: str | None = None


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
    precheck_min_duration_sec: float = 20.0
    precheck_min_speech_ratio: float = 0.10

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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_word(word: dict[str, Any], seg_start: float, seg_end: float) -> dict[str, Any] | None:
    text = str(word.get("word") or word.get("text") or "").strip()
    if not text:
        return None
    start = _safe_float(word.get("start"), default=seg_start)
    end = _safe_float(word.get("end"), default=max(start, seg_end))
    if end < start:
        end = start
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "word": text,
    }


def _normalise_asr_segments(raw_segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_segments):
        start = _safe_float(raw.get("start"), default=float(idx))
        end = _safe_float(raw.get("end"), default=start)
        if end < start:
            end = start
        text = str(raw.get("text") or "").strip()
        words_raw = raw.get("words")
        words: list[dict[str, Any]] = []
        if isinstance(words_raw, list):
            for word in words_raw:
                if not isinstance(word, dict):
                    continue
                normalised = _normalise_word(word, start, end)
                if normalised is not None:
                    words.append(normalised)
        if not words and text:
            words = [{"start": round(start, 3), "end": round(end, 3), "word": text}]
        payload: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "words": words,
        }
        language = raw.get("language")
        if isinstance(language, str) and language.strip():
            payload["language"] = language.strip()
        out.append(payload)
    return out


_LANGUAGE_NAME_MAP: dict[str, str] = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
}

_LANGUAGE_CODE_MAP: dict[str, str] = {
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "fre": "fr",
    "deu": "de",
    "ger": "de",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "ukr": "uk",
    "jpn": "ja",
    "kor": "ko",
    "zho": "zh",
    "chi": "zh",
}

_EN_STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "is",
    "are",
    "we",
    "you",
    "hello",
    "thanks",
    "meeting",
    "team",
    "today",
}

_ES_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "de",
    "que",
    "y",
    "en",
    "para",
    "con",
    "es",
    "somos",
    "hola",
    "gracias",
    "reunion",
    "equipo",
    "hoy",
}


def _normalise_language_code(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    cleaned = raw.replace("_", "-")
    base = cleaned.split("-", 1)[0]
    if not base.isalpha():
        return None
    if len(base) == 2:
        return base
    if len(base) == 3:
        return _LANGUAGE_CODE_MAP.get(base, None)
    return None


def _language_name(code: str) -> str:
    return _LANGUAGE_NAME_MAP.get(code, code.upper())


def _guess_language_from_text(text: str) -> str | None:
    sample = text.strip().lower()
    if not sample:
        return None
    tokens = re.findall(r"[a-zA-Z\u00c0-\u017f]+", sample)
    if not tokens:
        return None
    en_score = sum(1 for token in tokens if token in _EN_STOPWORDS)
    es_score = sum(1 for token in tokens if token in _ES_STOPWORDS)
    if any(ch in sample for ch in "áéíóúñ¿¡"):
        es_score += 2
    if en_score == 0 and es_score == 0:
        return None
    if es_score > en_score:
        return "es"
    if en_score > es_score:
        return "en"
    return None


def _segment_language(
    segment: dict[str, Any],
    *,
    detected_language: str | None,
    transcript_language_override: str | None,
) -> str:
    if transcript_language_override:
        return transcript_language_override
    seg_language = _normalise_language_code(segment.get("language"))
    if seg_language:
        return seg_language
    if detected_language:
        return detected_language
    text_language = _guess_language_from_text(str(segment.get("text") or ""))
    if text_language:
        return text_language
    return "unknown"


def _duration_weight(start: float, end: float, text: str) -> float:
    duration = max(0.0, end - start)
    if duration > 0:
        return duration
    tokens = max(len(text.split()), 1)
    return tokens * 0.01


def _language_stats(
    asr_segments: Sequence[dict[str, Any]],
    *,
    detected_language: str | None,
    transcript_language_override: str | None,
) -> tuple[list[dict[str, Any]], str, dict[str, float], list[dict[str, Any]]]:
    if not asr_segments:
        fallback = transcript_language_override or detected_language or "unknown"
        return [], fallback, {}, []

    enriched = sorted(
        (dict(seg) for seg in asr_segments),
        key=lambda row: (
            _safe_float(row.get("start"), default=0.0),
            _safe_float(row.get("end"), default=0.0),
        ),
    )

    weighted_totals: dict[str, float] = {}
    spans: list[dict[str, Any]] = []
    for segment in enriched:
        start = _safe_float(segment.get("start"), default=0.0)
        end = _safe_float(segment.get("end"), default=start)
        if end < start:
            end = start
        text = str(segment.get("text") or "").strip()
        lang = _segment_language(
            segment,
            detected_language=detected_language,
            transcript_language_override=transcript_language_override,
        )
        segment["language"] = lang

        weight = _duration_weight(start, end, text)
        weighted_totals[lang] = weighted_totals.get(lang, 0.0) + weight

        if spans and spans[-1]["lang"] == lang and start <= _safe_float(spans[-1]["end"]) + 0.5:
            spans[-1]["end"] = round(max(_safe_float(spans[-1]["end"]), end), 3)
        else:
            spans.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "lang": lang,
                }
            )

    ordered = sorted(weighted_totals.items(), key=lambda row: (-row[1], row[0]))
    dominant = "unknown"
    for lang, _weight in ordered:
        if lang != "unknown":
            dominant = lang
            break
    if dominant == "unknown" and ordered:
        dominant = ordered[0][0]

    total_weight = sum(weighted_totals.values())
    distribution: dict[str, float] = {}
    if total_weight > 0:
        for lang, weight in ordered:
            distribution[lang] = round((weight / total_weight) * 100.0, 2)

    return enriched, dominant, distribution, spans


def _resolve_target_summary_language(
    requested_language: str | None,
    *,
    dominant_language: str,
    detected_language: str | None,
) -> str:
    requested = _normalise_language_code(requested_language)
    if requested:
        return requested
    if dominant_language and dominant_language != "unknown":
        return dominant_language
    if detected_language:
        return detected_language
    return "en"


_QUESTION_TYPE_KEYS = (
    "open",
    "yes_no",
    "clarification",
    "status",
    "decision_seeking",
)


def _chunk_text_for_prompt(text: str, *, max_chars: int = 500) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    words = normalized.split(" ")
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if not current:
            if word_len > max_chars:
                for start in range(0, word_len, max_chars):
                    chunks.append(word[start : start + max_chars])
                continue
            current = [word]
            current_len = word_len
            continue

        next_len = current_len + 1 + word_len
        if next_len > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len = next_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def _normalise_prompt_speaker_turns(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    max_turns: int = 300,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in speaker_turns:
        start = round(_safe_float(row.get("start"), default=0.0), 3)
        end = round(_safe_float(row.get("end"), default=0.0), 3)
        speaker = str(row.get("speaker") or "S1")
        lang = _normalise_language_code(row.get("language"))
        for chunk in _chunk_text_for_prompt(str(row.get("text") or "").strip()):
            if len(out) >= max_turns:
                return out
            payload: dict[str, Any] = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": chunk,
            }
            if lang:
                payload["language"] = lang
            out.append(payload)
    return out


def build_structured_summary_prompts(
    speaker_turns: Sequence[dict[str, Any]],
    target_summary_language: str,
    *,
    calendar_title: str | None = None,
    calendar_attendees: Sequence[str] | None = None,
) -> tuple[str, str]:
    language_name = _language_name(target_summary_language)
    sys_prompt = (
        "You are an assistant that summarizes meeting transcripts. "
        f"Write topic, summary_bullets, decisions, action_items, emotional_summary, and questions in {language_name}. "
        "Keep names, quotes, and domain terms in their original language when needed. "
        "Return strict JSON only, with no markdown fences."
    )
    prompt_payload = {
        "target_summary_language": target_summary_language,
        "calendar": {
            "title": (calendar_title or "").strip() or None,
            "attendees": [str(item).strip() for item in (calendar_attendees or []) if str(item).strip()],
        },
        "speaker_turns": _normalise_prompt_speaker_turns(speaker_turns),
        "required_schema": {
            "topic": "string",
            "summary_bullets": ["string"],
            "decisions": ["string"],
            "action_items": [
                {
                    "task": "string",
                    "owner": "string|null",
                    "deadline": "string|null",
                    "confidence": "number [0,1]",
                }
            ],
            "emotional_summary": "1-3 short lines as a string",
            "questions": {
                "total_count": "integer >= 0",
                "types": {key: "integer >= 0" for key in _QUESTION_TYPE_KEYS},
                "extracted": ["string"],
            },
        },
    }
    user_prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    return sys_prompt, user_prompt


def build_summary_prompts(clean_text: str, target_summary_language: str) -> tuple[str, str]:
    pseudo_turns = []
    stripped = clean_text.strip()
    if stripped:
        pseudo_turns.append({"start": 0.0, "end": 0.0, "speaker": "S1", "text": stripped})
    return build_structured_summary_prompts(
        pseudo_turns,
        target_summary_language,
    )


def _normalise_text_list(value: Any, *, max_items: int) -> list[str]:
    rows: list[Any]
    if isinstance(value, list):
        rows = value
    elif isinstance(value, str):
        rows = [line.strip() for line in value.splitlines() if line.strip()]
    else:
        return []

    out: list[str] = []
    for item in rows:
        if len(out) >= max_items:
            break
        text = str(item).strip()
        if not text:
            continue
        text = re.sub(r"^[\-\*\u2022]+\s*", "", text).strip()
        if text:
            out.append(text)
    return out


def _extract_json_dict(raw_content: str) -> dict[str, Any] | None:
    text = raw_content.strip()
    if not text:
        return None

    candidates: list[str] = [text]
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(match.strip() for match in fenced_matches if match.strip())

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalise_confidence(value: Any, *, default: float = 0.5) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return round(min(max(confidence, 0.0), 1.0), 2)


def _normalise_action_items(value: Any) -> list[dict[str, Any]]:
    rows: list[Any]
    if isinstance(value, list):
        rows = value
    elif value is None:
        rows = []
    else:
        rows = [value]

    out: list[dict[str, Any]] = []
    for row in rows:
        if len(out) >= 30:
            break
        if isinstance(row, dict):
            task = str(row.get("task") or row.get("action") or row.get("title") or "").strip()
            owner_raw = row.get("owner")
            deadline_raw = row.get("deadline") or row.get("due")
            confidence_raw = row.get("confidence", row.get("score"))
        else:
            task = str(row).strip()
            owner_raw = None
            deadline_raw = None
            confidence_raw = None
        if not task:
            continue
        owner = str(owner_raw).strip() if owner_raw is not None else ""
        deadline = str(deadline_raw).strip() if deadline_raw is not None else ""
        out.append(
            {
                "task": task,
                "owner": owner or None,
                "deadline": deadline or None,
                "confidence": _normalise_confidence(confidence_raw),
            }
        )
    return out


def _normalise_question_types(value: Any) -> dict[str, int]:
    out = {key: 0 for key in _QUESTION_TYPE_KEYS}
    if not isinstance(value, dict):
        return out
    for key in _QUESTION_TYPE_KEYS:
        out[key] = max(0, int(_safe_float(value.get(key), default=0.0)))
    return out


def _normalise_questions(value: Any) -> dict[str, Any]:
    total_count = 0
    question_types = {key: 0 for key in _QUESTION_TYPE_KEYS}
    extracted: list[str] = []

    if isinstance(value, dict):
        total_count = max(0, int(_safe_float(value.get("total_count"), default=0.0)))
        question_types = _normalise_question_types(value.get("types"))
        if sum(question_types.values()) == 0:
            question_types = _normalise_question_types(value)
        extracted = _normalise_text_list(value.get("extracted"), max_items=20)

    inferred_total = max(sum(question_types.values()), len(extracted))
    if total_count == 0:
        total_count = inferred_total

    return {
        "total_count": total_count,
        "types": question_types,
        "extracted": extracted,
    }


def _normalise_emotional_summary(value: Any) -> str:
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
    elif isinstance(value, list):
        lines = _normalise_text_list(value, max_items=3)
    else:
        lines = []
    if not lines:
        lines = ["Neutral and focused discussion."]
    return "\n".join(lines[:3])


def _summary_text_from_bullets(summary_bullets: Sequence[str]) -> str:
    return "\n".join(f"- {bullet}" for bullet in summary_bullets)


def _build_structured_summary_payload(
    *,
    model: str,
    target_summary_language: str,
    friendly: int,
    topic: str,
    summary_bullets: Sequence[str],
    decisions: Sequence[str],
    action_items: Sequence[dict[str, Any]],
    emotional_summary: str,
    questions: dict[str, Any],
    status: str | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "friendly": int(friendly),
        "model": model,
        "target_summary_language": target_summary_language,
        "topic": topic,
        "summary_bullets": list(summary_bullets),
        "summary": _summary_text_from_bullets(summary_bullets),
        "decisions": list(decisions),
        "action_items": list(action_items),
        "emotional_summary": emotional_summary,
        "questions": questions,
    }
    if status:
        payload["status"] = status
    if reason:
        payload["reason"] = reason
    if error:
        payload["error"] = error
    return payload


def build_summary_payload(
    *,
    raw_llm_content: str,
    model: str,
    target_summary_language: str,
    friendly: int,
    default_topic: str = "Meeting summary",
) -> dict[str, Any]:
    parsed = _extract_json_dict(raw_llm_content) or {}

    summary_bullets = _normalise_text_list(parsed.get("summary_bullets"), max_items=12)
    if not summary_bullets:
        summary_bullets = _normalise_text_list(parsed.get("summary"), max_items=12)
    if not summary_bullets:
        summary_bullets = _normalise_text_list(raw_llm_content, max_items=12)
    if not summary_bullets:
        summary_bullets = ["No summary available."]

    topic = str(parsed.get("topic") or "").strip()
    if not topic:
        topic = summary_bullets[0][:120] if summary_bullets else default_topic
    topic = topic or default_topic

    return _build_structured_summary_payload(
        model=model,
        target_summary_language=target_summary_language,
        friendly=friendly,
        topic=topic,
        summary_bullets=summary_bullets,
        decisions=_normalise_text_list(parsed.get("decisions"), max_items=20),
        action_items=_normalise_action_items(parsed.get("action_items")),
        emotional_summary=_normalise_emotional_summary(parsed.get("emotional_summary")),
        questions=_normalise_questions(parsed.get("questions")),
    )


def _language_payload(info: dict[str, Any]) -> dict[str, Any]:
    detected_raw = str(
        info.get("language")
        or info.get("detected_language")
        or info.get("lang")
        or "unknown"
    )
    detected = _normalise_language_code(detected_raw) or "unknown"
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
    confidence = None
    if confidence_raw is not None:
        confidence = round(_safe_float(confidence_raw, default=0.0), 4)
    return {"detected": detected, "confidence": confidence}


def _safe_diarization_segments(diarization: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if diarization is None or not hasattr(diarization, "itertracks"):
        return out
    tracks = diarization.itertracks(yield_label=True)
    for item in tracks:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        seg = item[0]
        if len(item) == 2:
            label = item[1]
        else:
            label = item[-1]
        start = _safe_float(getattr(seg, "start", 0.0), default=0.0)
        end = _safe_float(getattr(seg, "end", start), default=start)
        if end < start:
            end = start
        out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": str(label),
            }
        )
    out.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    return out


def _overlap_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _pick_speaker(start: float, end: float, diar_segments: Sequence[dict[str, Any]]) -> str:
    if not diar_segments:
        return "S1"
    best_speaker = str(diar_segments[0]["speaker"])
    best_overlap = -1.0
    midpoint = (start + end) / 2.0
    best_distance = float("inf")
    for seg in diar_segments:
        d_start = _safe_float(seg.get("start"), default=0.0)
        d_end = _safe_float(seg.get("end"), default=d_start)
        overlap = _overlap_seconds(start, end, d_start, d_end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(seg.get("speaker", "S1"))
        if overlap == 0.0:
            if midpoint < d_start:
                distance = d_start - midpoint
            elif midpoint > d_end:
                distance = midpoint - d_end
            else:
                distance = 0.0
            if best_overlap <= 0.0 and distance < best_distance:
                best_distance = distance
                best_speaker = str(seg.get("speaker", "S1"))
    return best_speaker


def _words_from_segments(
    asr_segments: Sequence[dict[str, Any]],
    *,
    default_language: str | None,
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for seg in asr_segments:
        seg_start = _safe_float(seg.get("start"), default=0.0)
        seg_end = _safe_float(seg.get("end"), default=seg_start)
        seg_language = seg.get("language")
        language = (
            str(seg_language)
            if isinstance(seg_language, str) and seg_language.strip()
            else default_language
        )
        seg_words = seg.get("words")
        if not isinstance(seg_words, list):
            seg_words = []
        if not seg_words and seg.get("text"):
            seg_words = [
                {
                    "start": seg_start,
                    "end": seg_end,
                    "word": str(seg.get("text")),
                }
            ]
        for raw_word in seg_words:
            if not isinstance(raw_word, dict):
                continue
            text = str(raw_word.get("word") or "").strip()
            if not text:
                continue
            start = _safe_float(raw_word.get("start"), default=seg_start)
            end = _safe_float(raw_word.get("end"), default=max(start, seg_end))
            if end < start:
                end = start
            payload: dict[str, Any] = {
                "start": round(start, 3),
                "end": round(end, 3),
                "word": text,
            }
            if language:
                payload["language"] = language
            words.append(payload)
    words.sort(key=lambda row: (row["start"], row["end"], row["word"]))
    return words


def _build_speaker_turns(
    asr_segments: Sequence[dict[str, Any]],
    diar_segments: Sequence[dict[str, Any]],
    *,
    default_language: str | None,
) -> list[dict[str, Any]]:
    words = _words_from_segments(asr_segments, default_language=default_language)
    if not words:
        return []

    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for word in words:
        start = _safe_float(word.get("start"), default=0.0)
        end = _safe_float(word.get("end"), default=start)
        speaker = _pick_speaker(start, end, diar_segments)
        language = word.get("language")
        if (
            current is not None
            and current["speaker"] == speaker
            and start - _safe_float(current["end"], default=start) <= 1.0
        ):
            current["end"] = round(max(_safe_float(current["end"]), end), 3)
            current["text"] = f"{current['text']} {word['word']}".strip()
        else:
            if current is not None:
                turns.append(current)
            current = {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "text": str(word["word"]),
            }
            if isinstance(language, str) and language:
                current["language"] = language
    if current is not None:
        turns.append(current)
    return turns


def _speaker_slug(label: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    slug = slug.strip("_")
    return slug or "speaker"


def _clear_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _snippet_window(
    start: float,
    end: float,
    *,
    duration_sec: float | None,
) -> tuple[float, float]:
    seg_duration = max(0.0, end - start)
    target = min(20.0, max(10.0, seg_duration if seg_duration > 0 else 10.0))
    center = start + (seg_duration / 2.0 if seg_duration > 0 else 0.0)
    clip_start = max(0.0, center - (target / 2.0))
    clip_end = clip_start + target
    if duration_sec is not None and duration_sec > 0:
        if clip_end > duration_sec:
            clip_end = duration_sec
            clip_start = max(0.0, clip_end - target)
    if clip_end <= clip_start:
        clip_end = clip_start + min(target, 1.0)
    return round(clip_start, 3), round(clip_end, 3)


def _extract_wav_snippet_with_wave(
    audio_path: Path,
    out_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> bool:
    if audio_path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            start_frame = max(0, int(start_sec * rate))
            end_frame = max(start_frame + 1, int(end_sec * rate))
            src.setpos(min(start_frame, src.getnframes()))
            frames = src.readframes(max(0, end_frame - start_frame))
        if not frames:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(rate)
            dst.writeframes(frames)
        return True
    except Exception:
        return False


def _extract_wav_snippet_with_ffmpeg(
    audio_path: Path,
    out_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    duration = max(0.1, end_sec - start_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return False
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 44


def _write_silence_wav(path: Path, duration_sec: float = 1.0) -> None:
    samples = max(int(16000 * max(duration_sec, 0.1)), 1)
    payload = b"\x00\x00" * samples
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(16000)
        wav_out.writeframes(payload)


def _export_speaker_snippets(
    *,
    audio_path: Path,
    diar_segments: Sequence[dict[str, Any]],
    snippets_dir: Path,
    duration_sec: float | None,
) -> list[Path]:
    _clear_dir(snippets_dir)

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in diar_segments:
        speaker = str(segment.get("speaker", "S1"))
        by_speaker.setdefault(speaker, []).append(segment)

    all_outputs: list[Path] = []
    for speaker in sorted(by_speaker):
        ranked = sorted(
            by_speaker[speaker],
            key=lambda row: (
                -(_safe_float(row.get("end")) - _safe_float(row.get("start"))),
                _safe_float(row.get("start")),
                _safe_float(row.get("end")),
            ),
        )
        chosen: list[dict[str, Any]] = []
        for candidate in ranked:
            start = _safe_float(candidate.get("start"))
            end = _safe_float(candidate.get("end"), default=start)
            if end <= start:
                continue
            overlaps = any(
                _overlap_seconds(
                    start,
                    end,
                    _safe_float(existing.get("start")),
                    _safe_float(existing.get("end")),
                )
                > 0.5
                for existing in chosen
            )
            if overlaps:
                continue
            chosen.append(candidate)
            if len(chosen) == 3:
                break
        if len(chosen) < 2:
            for candidate in ranked:
                if candidate in chosen:
                    continue
                chosen.append(candidate)
                if len(chosen) == 2:
                    break

        speaker_dir = snippets_dir / _speaker_slug(speaker)
        for idx, segment in enumerate(
            sorted(chosen, key=lambda row: _safe_float(row.get("start"))), start=1
        ):
            seg_start = _safe_float(segment.get("start"))
            seg_end = _safe_float(segment.get("end"), default=seg_start)
            clip_start, clip_end = _snippet_window(
                seg_start,
                seg_end,
                duration_sec=duration_sec,
            )
            out_path = speaker_dir / f"{idx}.wav"
            written = _extract_wav_snippet_with_wave(
                audio_path,
                out_path,
                start_sec=clip_start,
                end_sec=clip_end,
            )
            if not written:
                written = _extract_wav_snippet_with_ffmpeg(
                    audio_path,
                    out_path,
                    start_sec=clip_start,
                    end_sec=clip_end,
                )
            if not written:
                _write_silence_wav(out_path, duration_sec=min(max(clip_end - clip_start, 1.0), 2.0))
            all_outputs.append(out_path)
    all_outputs.sort()
    return all_outputs


def _audio_duration_from_wave(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            frames = src.getnframes()
        if rate <= 0:
            return None
        return frames / float(rate)
    except Exception:
        return None


def _audio_duration_from_ffprobe(audio_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _speech_ratio_from_wave(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            sample_width = src.getsampwidth()
            frame_samples = max(int(rate * 0.03), 1)
            frame_bytes = frame_samples * channels * sample_width
            voiced = 0
            total = 0
            while True:
                chunk = src.readframes(frame_samples)
                if not chunk:
                    break
                if len(chunk) < frame_bytes // 2:
                    break
                total += 1
                if audioop.rms(chunk, sample_width) >= 350:
                    voiced += 1
            if total == 0:
                return 0.0
            return voiced / float(total)
    except Exception:
        return None


def _speech_ratio_from_ffmpeg(audio_path: Path) -> float | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            if proc.stdout is None:
                return None
            frame_bytes = 960  # 30ms @ 16kHz * 2 bytes sample
            voiced = 0
            total = 0
            while True:
                chunk = proc.stdout.read(frame_bytes)
                if not chunk:
                    break
                if len(chunk) < frame_bytes:
                    break
                total += 1
                if audioop.rms(chunk, 2) >= 300:
                    voiced += 1
            # Avoid a fixed wait cap; long/slower ffmpeg runs should not
            # downgrade valid audio into "metrics unavailable" quarantine.
            proc.wait()
            if proc.returncode != 0:
                return None
            if total == 0:
                return 0.0
            return voiced / float(total)
    except Exception:
        return None


def run_precheck(audio_path: Path, cfg: Settings | None = None) -> PrecheckResult:
    """Compute duration + VAD speech ratio and decide quarantine status."""
    settings = cfg or Settings()
    duration_sec = _audio_duration_from_wave(audio_path)
    if duration_sec is None:
        duration_sec = _audio_duration_from_ffprobe(audio_path)

    speech_ratio = _speech_ratio_from_wave(audio_path)
    if speech_ratio is None:
        speech_ratio = _speech_ratio_from_ffmpeg(audio_path)

    quarantine_reason: str | None = None
    if duration_sec is None or speech_ratio is None:
        quarantine_reason = "precheck_metrics_unavailable"
    elif duration_sec < settings.precheck_min_duration_sec:
        quarantine_reason = f"duration_lt_{settings.precheck_min_duration_sec:.0f}s"
    elif speech_ratio < settings.precheck_min_speech_ratio:
        quarantine_reason = (
            f"speech_ratio_lt_{settings.precheck_min_speech_ratio:.2f}"
        )

    return PrecheckResult(
        duration_sec=round(duration_sec, 3) if duration_sec is not None else None,
        speech_ratio=round(speech_ratio, 4) if speech_ratio is not None else None,
        quarantine_reason=quarantine_reason,
    )


def _fallback_diarization(duration_sec: float | None) -> Any:
    duration = max(duration_sec or 0.0, 0.1)

    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            if yield_label:
                yield SimpleNamespace(start=0.0, end=duration), "S1"
            else:  # pragma: no cover - legacy compatibility branch
                yield (SimpleNamespace(start=0.0, end=duration),)

    return _Annotation()


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
) -> TranscriptResult:
    """Transcribe ``audio_path`` and return a structured result."""
    start = time.perf_counter()

    artifact_paths = build_recording_artifacts(
        cfg.recordings_root,
        recording_id=recording_id or _default_recording_id(audio_path),
        audio_ext=audio_path.suffix,
    )
    stage_raw_audio(audio_path, artifact_paths.raw_audio_path)

    precheck_result = precheck or run_precheck(audio_path, cfg)
    normalized_transcript_language_override = _normalise_language_code(
        transcript_language_override
    )
    resolved_summary_language = _resolve_target_summary_language(
        target_summary_language,
        dominant_language=normalized_transcript_language_override or "unknown",
        detected_language=None,
    )
    normalized_calendar_title = str(calendar_title or "").strip() or None
    normalized_calendar_attendees = [
        str(attendee).strip()
        for attendee in (calendar_attendees or [])
        if str(attendee).strip()
    ]

    atomic_write_json(
        artifact_paths.metrics_json_path,
        {
            "status": "running",
            "version": 1,
            "precheck": {
                "duration_sec": precheck_result.duration_sec,
                "speech_ratio": precheck_result.speech_ratio,
                "quarantine_reason": precheck_result.quarantine_reason,
            },
        },
    )

    if precheck_result.quarantine_reason:
        _clear_dir(artifact_paths.snippets_dir)
        atomic_write_text(artifact_paths.transcript_txt_path, "")
        atomic_write_json(
            artifact_paths.transcript_json_path,
            {
                "recording_id": artifact_paths.recording_id,
                "language": {"detected": "unknown", "confidence": None},
                "dominant_language": normalized_transcript_language_override or "unknown",
                "language_distribution": {},
                "language_spans": [],
                "target_summary_language": resolved_summary_language,
                "transcript_language_override": normalized_transcript_language_override,
                "calendar_title": normalized_calendar_title,
                "calendar_attendees": normalized_calendar_attendees,
                "segments": [],
                "speakers": [],
                "text": "",
            },
        )
        atomic_write_json(artifact_paths.segments_json_path, [])
        atomic_write_json(artifact_paths.speaker_turns_json_path, [])
        atomic_write_json(
            artifact_paths.summary_json_path,
            _build_structured_summary_payload(
                model=cfg.llm_model,
                target_summary_language=resolved_summary_language,
                friendly=0,
                topic="Quarantined recording",
                summary_bullets=["Recording was quarantined before transcription."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=_normalise_questions(None),
                status="quarantined",
                reason=precheck_result.quarantine_reason,
            ),
        )
        atomic_write_json(
            artifact_paths.metrics_json_path,
            {
                "status": "quarantined",
                "version": 1,
                "precheck": {
                    "duration_sec": precheck_result.duration_sec,
                    "speech_ratio": precheck_result.speech_ratio,
                    "quarantine_reason": precheck_result.quarantine_reason,
                },
            },
        )
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(
            summary="Quarantined",
            body="",
            friendly=0,
            speakers=[],
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=[],
            segments=[],
        )

    def _write_failed_artifacts(
        exc: Exception,
        *,
        friendly_score: int = 0,
        language_payload: dict[str, Any] | None = None,
        asr_count: int = 0,
        diar_count: int = 0,
        speaker_turn_count: int = 0,
    ) -> None:
        atomic_write_json(
            artifact_paths.summary_json_path,
            _build_structured_summary_payload(
                model=cfg.llm_model,
                target_summary_language=resolved_summary_language,
                friendly=friendly_score,
                topic="Summary generation failed",
                summary_bullets=["Unable to produce a summary due to a processing error."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=_normalise_questions(None),
                status="failed",
                error=str(exc) or exc.__class__.__name__,
            ),
        )
        atomic_write_json(
            artifact_paths.metrics_json_path,
            {
                "status": "failed",
                "version": 1,
                "precheck": {
                    "duration_sec": precheck_result.duration_sec,
                    "speech_ratio": precheck_result.speech_ratio,
                    "quarantine_reason": None,
                },
                "language": language_payload or {"detected": "unknown", "confidence": None},
                "asr_segments": asr_count,
                "diar_segments": diar_count,
                "speaker_turns": speaker_turn_count,
                "error": str(exc) or exc.__class__.__name__,
            },
        )

    try:
        import whisperx

        def _asr() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            asr_language = normalized_transcript_language_override or "auto"
            kwargs: dict[str, Any] = {"vad_filter": True, "language": asr_language}
            try:
                segments, info = whisperx.transcribe(
                    str(audio_path),
                    word_timestamps=True,
                    **kwargs,
                )
            except TypeError:
                segments, info = whisperx.transcribe(str(audio_path), **kwargs)
            return list(segments), dict(info or {})

        asr_task = asyncio.to_thread(_asr)
        diar_task = diariser(audio_path)
        (raw_segments, info), diarization = await asyncio.gather(asr_task, diar_task)

        asr_segments = _normalise_asr_segments(raw_segments)
        language_info = _language_payload(info)
        detected_language = (
            _normalise_language_code(language_info["detected"])
            if language_info["detected"] != "unknown"
            else None
        )
        (
            asr_segments,
            dominant_language,
            language_distribution,
            language_spans,
        ) = _language_stats(
            asr_segments,
            detected_language=detected_language,
            transcript_language_override=normalized_transcript_language_override,
        )
        if language_info["detected"] == "unknown" and dominant_language != "unknown":
            language_info["detected"] = dominant_language
        resolved_summary_language = _resolve_target_summary_language(
            target_summary_language,
            dominant_language=dominant_language,
            detected_language=detected_language,
        )

        asr_text = " ".join(seg.get("text", "").strip() for seg in asr_segments).strip()
        clean_text = normalizer.dedup(asr_text)
        diar_segments = _safe_diarization_segments(diarization)
        if not diar_segments and asr_segments:
            fallback_end = max(_safe_float(seg.get("end")) for seg in asr_segments)
            diar_segments = [
                {"start": 0.0, "end": round(max(fallback_end, 0.1), 3), "speaker": "S1"}
            ]

        speaker_turns = _build_speaker_turns(
            asr_segments,
            diar_segments,
            default_language=dominant_language if dominant_language != "unknown" else detected_language,
        )

        aliases = _load_aliases(cfg.speaker_db)
        for row in diar_segments:
            label = str(row["speaker"])
            aliases.setdefault(label, label)
        _save_aliases(aliases, cfg.speaker_db)
    except Exception as exc:
        error_rate_total.inc()
        _write_failed_artifacts(exc)
        raise

    if not clean_text:
        _clear_dir(artifact_paths.snippets_dir)
        atomic_write_text(artifact_paths.transcript_txt_path, "")
        atomic_write_json(
            artifact_paths.transcript_json_path,
            {
                "recording_id": artifact_paths.recording_id,
                "language": language_info,
                "dominant_language": dominant_language,
                "language_distribution": language_distribution,
                "language_spans": language_spans,
                "target_summary_language": resolved_summary_language,
                "transcript_language_override": normalized_transcript_language_override,
                "calendar_title": normalized_calendar_title,
                "calendar_attendees": normalized_calendar_attendees,
                "segments": asr_segments,
                "speakers": sorted({aliases.get(row["speaker"], row["speaker"]) for row in diar_segments}),
                "text": "",
            },
        )
        atomic_write_json(artifact_paths.segments_json_path, diar_segments)
        atomic_write_json(artifact_paths.speaker_turns_json_path, speaker_turns)
        atomic_write_json(
            artifact_paths.summary_json_path,
            _build_structured_summary_payload(
                model=cfg.llm_model,
                target_summary_language=resolved_summary_language,
                friendly=0,
                topic="No speech detected",
                summary_bullets=["No speech detected."],
                decisions=[],
                action_items=[],
                emotional_summary="No emotional summary available.",
                questions=_normalise_questions(None),
                status="no_speech",
            ),
        )
        atomic_write_json(
            artifact_paths.metrics_json_path,
            {
                "status": "no_speech",
                "version": 1,
                "precheck": {
                    "duration_sec": precheck_result.duration_sec,
                    "speech_ratio": precheck_result.speech_ratio,
                    "quarantine_reason": None,
                },
                "language": language_info,
                "asr_segments": len(asr_segments),
                "diar_segments": len(diar_segments),
                "speaker_turns": len(speaker_turns),
            },
        )
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(
            summary="No speech detected",
            body="",
            friendly=0,
            speakers=sorted({aliases.get(row["speaker"], row["speaker"]) for row in diar_segments}),
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=[],
            segments=[],
        )

    snippet_paths = _export_speaker_snippets(
        audio_path=audio_path,
        diar_segments=diar_segments,
        snippets_dir=artifact_paths.snippets_dir,
        duration_sec=precheck_result.duration_sec,
    )

    speaker_lines = [
        f"[{turn['start']:.2f}-{turn['end']:.2f}] **{aliases.get(turn['speaker'], turn['speaker'])}:** {turn['text']}"
        for turn in speaker_turns
    ]
    speaker_lines = _merge_similar(speaker_lines, cfg.merge_similar)

    friendly = _sentiment_score(clean_text)
    sys_prompt, user_prompt = build_structured_summary_prompts(
        speaker_turns,
        resolved_summary_language,
        calendar_title=normalized_calendar_title,
        calendar_attendees=normalized_calendar_attendees,
    )

    try:
        msg = await llm.generate(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            model=cfg.llm_model,
            response_format={"type": "json_object"},
        )
        raw_summary = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        summary_payload = build_summary_payload(
            raw_llm_content=raw_summary,
            model=cfg.llm_model,
            target_summary_language=resolved_summary_language,
            friendly=friendly,
            default_topic=normalized_calendar_title or "Meeting summary",
        )
        summary = str(summary_payload.get("summary") or "")

        serialised_segments = [
            SpeakerSegment(
                start=_safe_float(turn["start"]),
                end=_safe_float(turn["end"]),
                speaker=str(turn["speaker"]),
                text=str(turn["text"]),
            )
            for turn in speaker_turns
        ]
        atomic_write_text(artifact_paths.transcript_txt_path, clean_text)
        atomic_write_json(
            artifact_paths.transcript_json_path,
            {
                "recording_id": artifact_paths.recording_id,
                "language": language_info,
                "dominant_language": dominant_language,
                "language_distribution": language_distribution,
                "language_spans": language_spans,
                "target_summary_language": resolved_summary_language,
                "transcript_language_override": normalized_transcript_language_override,
                "calendar_title": normalized_calendar_title,
                "calendar_attendees": normalized_calendar_attendees,
                "segments": asr_segments,
                "speaker_lines": speaker_lines,
                "speakers": sorted(set(aliases.get(turn["speaker"], turn["speaker"]) for turn in speaker_turns)),
                "text": clean_text,
            },
        )
        atomic_write_json(artifact_paths.segments_json_path, diar_segments)
        atomic_write_json(artifact_paths.speaker_turns_json_path, speaker_turns)
        atomic_write_json(
            artifact_paths.summary_json_path,
            summary_payload,
        )
        atomic_write_json(
            artifact_paths.metrics_json_path,
            {
                "status": "ok",
                "version": 1,
                "precheck": {
                    "duration_sec": precheck_result.duration_sec,
                    "speech_ratio": precheck_result.speech_ratio,
                    "quarantine_reason": None,
                },
                "language": language_info,
                "asr_segments": len(asr_segments),
                "diar_segments": len(diar_segments),
                "speaker_turns": len(speaker_turns),
                "snippets": len(snippet_paths),
            },
        )

        result = TranscriptResult(
            summary=summary,
            body=clean_text,
            friendly=friendly,
            speakers=sorted(set(aliases.get(turn["speaker"], turn["speaker"]) for turn in speaker_turns)),
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=snippet_paths,
            segments=serialised_segments,
        )
    except Exception as exc:
        error_rate_total.inc()
        _write_failed_artifacts(
            exc,
            friendly_score=friendly,
            language_payload=language_info,
            asr_count=len(asr_segments),
            diar_count=len(diar_segments),
            speaker_turn_count=len(speaker_turns),
        )
        raise
    finally:
        p95_latency_seconds.observe(time.perf_counter() - start)

    return result


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
