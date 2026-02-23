from __future__ import annotations

import math
import re
from typing import Any

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


def safe_float(
    value: Any,
    default: float = 0.0,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Best-effort numeric parse with optional bounds."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if min_value is not None and parsed < min_value:
        return default
    if max_value is not None and parsed > max_value:
        return default
    return parsed


def normalise_language_code(value: Any) -> str | None:
    """Convert language values like EN-us / eng into ISO-639-1 codes."""
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    token = raw.replace("_", "-").split("-", 1)[0]
    if not token.isalpha():
        return None
    if len(token) == 2:
        return token
    if len(token) == 3:
        return _LANGUAGE_CODE_MAP.get(token)
    return None


def normalise_text_items(
    value: Any,
    *,
    max_items: int,
    strip_bullets: bool = True,
) -> list[str]:
    """Normalise list-like or multiline text payloads into clean lines."""
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
        if strip_bullets:
            text = re.sub(r"^[\-*\u2022]+\s*", "", text).strip()
        if text:
            out.append(text)
    return out


__all__ = ["safe_float", "normalise_language_code", "normalise_text_items"]
