from __future__ import annotations

from lan_transcriber.pipeline_steps.language import (
    analyse_languages,
    resolve_target_summary_language,
    segment_language,
)


def test_segment_language_prefers_override_then_detected_then_guess():
    segment = {"text": "hola equipo y gracias"}
    assert segment_language(segment, detected_language="fr", transcript_language_override="es") == "es"
    assert segment_language(segment, detected_language="fr", transcript_language_override=None) == "fr"
    assert segment_language(segment, detected_language=None, transcript_language_override=None) == "es"


def test_analyse_languages_returns_dominant_distribution_and_spans():
    rows = [
        {"start": 0.0, "end": 3.0, "text": "hello team", "language": "en"},
        {"start": 3.0, "end": 11.0, "text": "hola equipo", "language": "es"},
    ]

    analysis = analyse_languages(rows, detected_language="en", transcript_language_override=None)

    assert analysis.dominant_language == "es"
    assert set(analysis.distribution) == {"en", "es"}
    assert analysis.spans[0]["lang"] == "en"
    assert analysis.spans[1]["lang"] == "es"


def test_resolve_target_summary_language_fallback_order():
    assert resolve_target_summary_language("fr", dominant_language="en", detected_language="es") == "fr"
    assert resolve_target_summary_language(None, dominant_language="en", detected_language="es") == "en"
    assert resolve_target_summary_language(None, dominant_language="unknown", detected_language="es") == "es"
    assert resolve_target_summary_language(None, dominant_language="unknown", detected_language=None) == "en"
