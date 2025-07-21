import re
from collections import deque
from typing import Deque

from rapidfuzz import fuzz as rfuzz


def dedup(text: str, *, window: int = 3, fuzz: int = 90) -> str:
    """Collapse consecutive duplicate sentences."""
    text = text.strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1:
        sentences = text.split()

    out = []
    history: Deque[str] = deque(maxlen=window)
    for s in sentences:
        duplicate = False
        for prev in history:
            if len(prev.split()) <= 25 and len(s.split()) <= 25:
                if rfuzz.ratio(prev, s) >= fuzz:
                    duplicate = True
                    break
        if not duplicate:
            out.append(s)
            history.append(s)

    return " ".join(out).strip()


__all__ = ["dedup"]
