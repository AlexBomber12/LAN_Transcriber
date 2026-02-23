from __future__ import annotations

from typing import Any, Sequence

from lan_transcriber.utils import normalise_language_code, safe_float

DEFAULT_INTERRUPTION_OVERLAP_SEC = 0.3


def _normalise_word(word: dict[str, Any], seg_start: float, seg_end: float) -> dict[str, Any] | None:
    text = str(word.get("word") or word.get("text") or "").strip()
    if not text:
        return None
    start = safe_float(word.get("start"), default=seg_start)
    end = safe_float(word.get("end"), default=max(start, seg_end))
    if end < start:
        end = start
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "word": text,
    }


def normalise_asr_segments(raw_segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_segments):
        start = safe_float(raw.get("start"), default=float(idx))
        end = safe_float(raw.get("end"), default=start)
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


def _diarization_segments(diarization: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if diarization is None or not hasattr(diarization, "itertracks"):
        return out
    tracks = diarization.itertracks(yield_label=True)
    for item in tracks:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        seg = item[0]
        label = item[1] if len(item) == 2 else item[-1]
        start = safe_float(getattr(seg, "start", 0.0), default=0.0)
        end = safe_float(getattr(seg, "end", start), default=start)
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
        d_start = safe_float(seg.get("start"), default=0.0)
        d_end = safe_float(seg.get("end"), default=d_start)
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
        seg_start = safe_float(seg.get("start"), default=0.0)
        seg_end = safe_float(seg.get("end"), default=seg_start)
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
            start = safe_float(raw_word.get("start"), default=seg_start)
            end = safe_float(raw_word.get("end"), default=max(start, seg_end))
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


def build_speaker_turns(
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
        start = safe_float(word.get("start"), default=0.0)
        end = safe_float(word.get("end"), default=start)
        speaker = _pick_speaker(start, end, diar_segments)
        language = normalise_language_code(word.get("language"))
        if (
            current is not None
            and current["speaker"] == speaker
            and start - safe_float(current["end"], default=start) <= 1.0
        ):
            current["end"] = round(max(safe_float(current["end"]), end), 3)
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
            if language:
                current["language"] = language
    if current is not None:
        turns.append(current)
    return turns


def _normalise_turns(turns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in turns:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        start = safe_float(row.get("start"), default=0.0)
        end = safe_float(row.get("end"), default=start)
        if end < start:
            end = start
        speaker = str(row.get("speaker") or "S1").strip() or "S1"
        payload: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
            "text": text,
        }
        language = normalise_language_code(row.get("language"))
        if language:
            payload["language"] = language
        out.append(payload)
    out.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return out


def count_interruptions(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    overlap_threshold: float = DEFAULT_INTERRUPTION_OVERLAP_SEC,
) -> dict[str, Any]:
    """Count interruption events and per-speaker done/received counters."""
    turns = _normalise_turns(speaker_turns)
    safe_overlap = max(overlap_threshold, 0.0)

    done: dict[str, int] = {}
    received: dict[str, int] = {}
    total = 0

    for idx, turn in enumerate(turns):
        interrupter = str(turn["speaker"])
        done.setdefault(interrupter, 0)
        received.setdefault(interrupter, 0)

        turn_start = safe_float(turn.get("start"))
        turn_end = safe_float(turn.get("end"), default=turn_start)
        if turn_end <= turn_start:
            continue

        seen_receivers: set[str] = set()
        for previous in turns[:idx]:
            receiver = str(previous["speaker"])
            if receiver == interrupter or receiver in seen_receivers:
                continue

            previous_start = safe_float(previous.get("start"), default=0.0)
            previous_end = safe_float(previous.get("end"), default=safe_float(previous.get("start")))
            if turn_start <= previous_start:
                continue
            if turn_start >= previous_end:
                continue

            overlap = min(previous_end, turn_end) - turn_start
            if overlap < safe_overlap:
                continue

            done[interrupter] = done.get(interrupter, 0) + 1
            received[receiver] = received.get(receiver, 0) + 1
            total += 1
            seen_receivers.add(receiver)

    return {
        "total": total,
        "done": done,
        "received": received,
    }


__all__ = [
    "DEFAULT_INTERRUPTION_OVERLAP_SEC",
    "normalise_asr_segments",
    "build_speaker_turns",
    "count_interruptions",
]
