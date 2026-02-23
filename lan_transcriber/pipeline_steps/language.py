from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from lan_transcriber.utils import normalise_language_code, safe_float

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


@dataclass(frozen=True)
class LanguageAnalysis:
    segments: list[dict[str, Any]]
    dominant_language: str
    distribution: dict[str, float]
    spans: list[dict[str, Any]]


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


def segment_language(
    segment: dict[str, Any],
    *,
    detected_language: str | None,
    transcript_language_override: str | None,
) -> str:
    if transcript_language_override:
        return transcript_language_override
    seg_language = normalise_language_code(segment.get("language"))
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


def analyse_languages(
    asr_segments: Sequence[dict[str, Any]],
    *,
    detected_language: str | None,
    transcript_language_override: str | None,
) -> LanguageAnalysis:
    if not asr_segments:
        fallback = transcript_language_override or detected_language or "unknown"
        return LanguageAnalysis([], fallback, {}, [])

    enriched = sorted(
        (dict(seg) for seg in asr_segments),
        key=lambda row: (
            safe_float(row.get("start"), default=0.0),
            safe_float(row.get("end"), default=0.0),
        ),
    )

    weighted_totals: dict[str, float] = {}
    spans: list[dict[str, Any]] = []
    for segment in enriched:
        start = safe_float(segment.get("start"), default=0.0)
        end = safe_float(segment.get("end"), default=start)
        if end < start:
            end = start
        text = str(segment.get("text") or "").strip()
        lang = segment_language(
            segment,
            detected_language=detected_language,
            transcript_language_override=transcript_language_override,
        )
        segment["language"] = lang

        weight = _duration_weight(start, end, text)
        weighted_totals[lang] = weighted_totals.get(lang, 0.0) + weight

        if spans and spans[-1]["lang"] == lang and start <= safe_float(spans[-1]["end"]) + 0.5:
            spans[-1]["end"] = round(max(safe_float(spans[-1]["end"]), end), 3)
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

    return LanguageAnalysis(enriched, dominant, distribution, spans)


def resolve_target_summary_language(
    requested_language: str | None,
    *,
    dominant_language: str,
    detected_language: str | None,
) -> str:
    requested = normalise_language_code(requested_language)
    if requested:
        return requested
    if dominant_language and dominant_language != "unknown":
        return dominant_language
    if detected_language:
        return detected_language
    return "en"


__all__ = ["LanguageAnalysis", "segment_language", "analyse_languages", "resolve_target_summary_language"]
