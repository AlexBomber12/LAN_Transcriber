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

_LANGUAGE_CONFIDENCE_KEYS = (
    "language_confidence",
    "language_probability",
    "language_score",
    "probability",
)
_TEXT_GUESS_HIGH_CONFIDENCE = 0.75
_REVIEW_UNCERTAIN_SEGMENT_THRESHOLD = 2


@dataclass(frozen=True)
class LanguageAnalysis:
    segments: list[dict[str, Any]]
    dominant_language: str
    distribution: dict[str, float]
    spans: list[dict[str, Any]]
    review_required: bool = False
    review_reason_code: str | None = None
    review_reason_text: str | None = None
    uncertain_segment_count: int = 0
    conflict_segment_count: int = 0


def _language_scores_from_text(text: str) -> dict[str, float]:
    sample = text.strip().lower()
    if not sample:
        return {}
    tokens = re.findall(r"[a-zA-Z\u00c0-\u017f]+", sample)
    if not tokens:
        return {}
    en_score = float(sum(1 for token in tokens if token in _EN_STOPWORDS))
    es_score = float(sum(1 for token in tokens if token in _ES_STOPWORDS))
    if any(ch in sample for ch in "áéíóúñ¿¡"):
        es_score += 2.0
    scores = {
        "en": en_score,
        "es": es_score,
    }
    return {code: value for code, value in scores.items() if value > 0.0}


def _guess_language_with_confidence(text: str) -> tuple[str | None, float | None]:
    scores = _language_scores_from_text(text)
    if not scores:
        return None, None
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None, round(ordered[0][1] / (ordered[0][1] + ordered[1][1]), 4)
    top_language, top_score = ordered[0]
    total_score = sum(scores.values())
    confidence = 1.0 if total_score <= 0 else round(top_score / total_score, 4)
    return top_language, confidence


def _guess_language_from_text(text: str) -> str | None:
    language, _confidence = _guess_language_with_confidence(text)
    return language


def _extract_language_confidence(segment: dict[str, Any]) -> float | None:
    for key in _LANGUAGE_CONFIDENCE_KEYS:
        if key in segment and segment[key] is not None:
            return round(safe_float(segment[key], default=0.0), 4)
    return None


def _segment_language_details(
    segment: dict[str, Any],
    *,
    detected_language: str | None,
    transcript_language_override: str | None,
    prefer_detected_language: bool,
) -> dict[str, Any]:
    if transcript_language_override:
        return {
            "language": transcript_language_override,
            "confidence": 1.0,
            "source": "override",
            "uncertain": False,
            "conflict": False,
        }

    segment_language_value = normalise_language_code(segment.get("language"))
    segment_confidence = _extract_language_confidence(segment)
    text_language, text_confidence = _guess_language_with_confidence(
        str(segment.get("text") or "")
    )

    if segment_language_value:
        resolved_confidence = (
            segment_confidence if segment_confidence is not None else 0.95
        )
        conflict = bool(
            text_language
            and text_confidence is not None
            and text_confidence >= _TEXT_GUESS_HIGH_CONFIDENCE
            and text_language != segment_language_value
        )
        return {
            "language": segment_language_value,
            "confidence": resolved_confidence,
            "source": "segment",
            "uncertain": resolved_confidence < _TEXT_GUESS_HIGH_CONFIDENCE or conflict,
            "conflict": conflict,
        }

    if prefer_detected_language and detected_language:
        return {
            "language": detected_language,
            "confidence": None,
            "source": "recording_detected",
            "uncertain": bool(text_language and text_language != detected_language),
            "conflict": False,
        }

    if text_language:
        conflict = bool(
            detected_language
            and text_confidence is not None
            and text_confidence >= _TEXT_GUESS_HIGH_CONFIDENCE
            and text_language != detected_language
        )
        return {
            "language": text_language,
            "confidence": text_confidence,
            "source": "text_guess",
            "uncertain": (
                text_confidence is None
                or text_confidence < _TEXT_GUESS_HIGH_CONFIDENCE
            ),
            "conflict": conflict,
        }

    if detected_language:
        return {
            "language": detected_language,
            "confidence": None,
            "source": "recording_detected",
            "uncertain": True,
            "conflict": False,
        }

    return {
        "language": "unknown",
        "confidence": None,
        "source": "unknown",
        "uncertain": True,
        "conflict": False,
    }


def segment_language(
    segment: dict[str, Any],
    *,
    detected_language: str | None,
    transcript_language_override: str | None,
) -> str:
    return str(
        _segment_language_details(
            segment,
            detected_language=detected_language,
            transcript_language_override=transcript_language_override,
            prefer_detected_language=True,
        )["language"]
    )


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
    uncertain_segment_count = 0
    conflict_segment_count = 0
    for segment in enriched:
        start = safe_float(segment.get("start"), default=0.0)
        end = safe_float(segment.get("end"), default=start)
        if end < start:
            end = start
        text = str(segment.get("text") or "").strip()
        preexisting_uncertain = bool(segment.get("language_uncertain"))
        preexisting_conflict = bool(segment.get("language_conflict"))
        preexisting_source = str(segment.get("language_source") or "").strip()
        details = _segment_language_details(
            segment,
            detected_language=detected_language,
            transcript_language_override=transcript_language_override,
            prefer_detected_language=False,
        )
        lang = str(details["language"])
        segment["language"] = lang
        segment["language_source"] = preexisting_source or str(details["source"])
        if details["confidence"] is not None:
            segment["language_confidence"] = round(
                safe_float(details["confidence"], default=0.0),
                4,
            )
        if details["uncertain"] or preexisting_uncertain:
            segment["language_uncertain"] = True
            uncertain_segment_count += 1
        if details["conflict"] or preexisting_conflict:
            segment["language_conflict"] = True
            conflict_segment_count += 1

        weight = _duration_weight(start, end, text)
        weighted_totals[lang] = weighted_totals.get(lang, 0.0) + weight

        if spans and spans[-1]["lang"] == lang and start <= safe_float(spans[-1]["end"]) + 0.5:
            spans[-1]["end"] = round(max(safe_float(spans[-1]["end"]), end), 3)
            if details["uncertain"] or preexisting_uncertain:
                spans[-1]["uncertain"] = True
        else:
            payload = {
                "start": round(start, 3),
                "end": round(end, 3),
                "lang": lang,
            }
            if details["uncertain"] or preexisting_uncertain:
                payload["uncertain"] = True
            spans.append(payload)

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

    non_unknown_languages = {
        lang for lang in distribution if lang != "unknown"
    }
    review_required = False
    review_reason_code: str | None = None
    review_reason_text: str | None = None
    if conflict_segment_count > 0:
        review_required = True
        review_reason_code = "multilingual_uncertain"
        review_reason_text = (
            "Segment-level language detection conflicted across multilingual "
            "chunks; manual review required."
        )
    elif (
        len(non_unknown_languages) >= 2
        and uncertain_segment_count >= _REVIEW_UNCERTAIN_SEGMENT_THRESHOLD
    ):
        review_required = True
        review_reason_code = "multilingual_uncertain"
        review_reason_text = (
            "Language detection remained low-confidence across multiple "
            "multilingual chunks; manual review required."
        )

    return LanguageAnalysis(
        enriched,
        dominant,
        distribution,
        spans,
        review_required=review_required,
        review_reason_code=review_reason_code,
        review_reason_text=review_reason_text,
        uncertain_segment_count=uncertain_segment_count,
        conflict_segment_count=conflict_segment_count,
    )


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
