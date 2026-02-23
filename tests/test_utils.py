from __future__ import annotations

from lan_transcriber.utils import normalise_language_code, normalise_text_items, safe_float


def test_safe_float_handles_invalid_and_bounds():
    assert safe_float("1.25") == 1.25
    assert safe_float("NaN", default=0.7) == 0.7
    assert safe_float("-5", default=0.1, min_value=0.0) == 0.1
    assert safe_float("6", default=0.2, max_value=5.0) == 0.2


def test_normalise_language_code_handles_iso639_1_and_3():
    assert normalise_language_code("en") == "en"
    assert normalise_language_code("EN-us") == "en"
    assert normalise_language_code("spa") == "es"
    assert normalise_language_code("___") is None


def test_normalise_text_items_strips_bullets_and_caps_items():
    value = ["- one", "* two", "  ", "three", "four"]
    assert normalise_text_items(value, max_items=3) == ["one", "two", "three"]
