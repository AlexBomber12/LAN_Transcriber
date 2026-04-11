from __future__ import annotations

from typing import Any, Sequence

from lan_transcriber.utils import normalise_language_code, safe_float

DEFAULT_INTERRUPTION_OVERLAP_SEC = 0.3
DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC = 4.0
DEFAULT_SPEAKER_TURN_MIN_WORDS = 6
# The post-pass that folds short turns into adjacent same-speaker turns must
# use a strictly larger gap than DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC. By the
# time merge_short_turns runs, build_speaker_turns has already collapsed every
# adjacent same-speaker pair whose gap is <= DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC,
# so any same-speaker neighbours that remain have gaps strictly above that
# threshold. A larger threshold here lets the short-turn pass actually fire
# for brief interjections (e.g. "uh-huh", "yeah okay") that sit between
# longer same-speaker stretches separated by a noticeable silence.
DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC = 8.0


def _decapitalize_join(existing_text: str, appended_text: str) -> str:
    """Join two text fragments, lowercasing the appended fragment's first letter
    when it looks like a false sentence start from segment splitting."""
    if not existing_text or not appended_text:
        return (existing_text + " " + appended_text).strip()
    last_char = existing_text.rstrip()[-1] if existing_text.strip() else ""
    if last_char in ".!?":
        return f"{existing_text} {appended_text}"
    first_word = appended_text.split()[0] if appended_text.split() else ""
    if not first_word or not first_word[0].isupper():
        return f"{existing_text} {appended_text}"
    # Preserve acronym-leading words. This covers plain acronyms ("ISO",
    # "API") as well as acronym-leading compounds like "API-based" or
    # "NASA's": the leading run of alphabetic characters, taken before any
    # punctuation or suffix, must be at least two characters long and fully
    # uppercase.
    leading_letters = ""
    for char in first_word:
        if char.isalpha():
            leading_letters += char
        else:
            break
    if len(leading_letters) >= 2 and leading_letters.isupper():
        return f"{existing_text} {appended_text}"
    if first_word == "I":
        return f"{existing_text} {appended_text}"
    fixed = appended_text[0].lower() + appended_text[1:]
    return f"{existing_text} {fixed}"


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
        if raw.get("language_confidence") is not None:
            payload["language_confidence"] = round(
                safe_float(raw.get("language_confidence"), default=0.0),
                4,
            )
        language_source = raw.get("language_source")
        if isinstance(language_source, str) and language_source.strip():
            payload["language_source"] = language_source.strip()
        if "language_uncertain" in raw:
            payload["language_uncertain"] = bool(raw.get("language_uncertain"))
        if "language_conflict" in raw:
            payload["language_conflict"] = bool(raw.get("language_conflict"))
        language_hint = raw.get("language_hint")
        if isinstance(language_hint, str) and language_hint.strip():
            payload["language_hint"] = language_hint.strip()
        if "language_hint_applied" in raw:
            payload["language_hint_applied"] = bool(raw.get("language_hint_applied"))
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
        seen_first_word = False
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
            if not seen_first_word:
                payload["segment_start"] = True
                seen_first_word = True
            words.append(payload)
    words.sort(key=lambda row: (row["start"], row["end"], row["word"]))
    return words


def build_speaker_turns(
    asr_segments: Sequence[dict[str, Any]],
    diar_segments: Sequence[dict[str, Any]],
    *,
    default_language: str | None,
    merge_gap_sec: float = DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC,
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
            and start - safe_float(current["end"], default=start) <= merge_gap_sec
        ):
            current["end"] = round(max(safe_float(current["end"]), end), 3)
            appended = str(word["word"])
            if word.get("segment_start"):
                current["text"] = _decapitalize_join(current["text"], appended)
            else:
                current["text"] = f"{current['text']} {appended}".strip()
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


def _word_count(text: str) -> int:
    return len(text.split())


def merge_short_turns(
    turns: Sequence[dict[str, Any]],
    *,
    min_words: int = DEFAULT_SPEAKER_TURN_MIN_WORDS,
    merge_gap_sec: float = DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC,
) -> list[dict[str, Any]]:
    """Merge short speaker turns into adjacent turns from the same speaker.

    A turn is considered "short" when its text contains fewer than ``min_words``
    whitespace-separated words. Short turns are merged backward into the
    immediately preceding turn (preferred) or forward into the immediately
    following turn when the previous turn cannot absorb them, but only when the
    neighbour shares the same speaker and the inter-turn gap is below
    ``merge_gap_sec`` seconds. Short turns whose neighbours are different
    speakers are kept as-is so no transcript content is discarded.

    ``merge_gap_sec`` should be strictly larger than the gap used by
    :func:`build_speaker_turns`, otherwise this post-pass is a no-op: every
    same-speaker pair within the base merge gap has already been collapsed by
    that earlier pass.
    """
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = [dict(turn) for turn in turns if isinstance(turn, dict)]
    if not pending:
        return out

    idx = 0
    while idx < len(pending):
        turn = pending[idx]
        text = str(turn.get("text") or "").strip()
        if _word_count(text) >= min_words:
            out.append(turn)
            idx += 1
            continue

        prev_turn = out[-1] if out else None
        prev_same_speaker = (
            prev_turn is not None
            and str(prev_turn.get("speaker")) == str(turn.get("speaker"))
        )
        prev_gap = (
            safe_float(turn.get("start"), default=0.0)
            - safe_float(prev_turn.get("end"), default=0.0)
            if prev_turn is not None
            else float("inf")
        )

        next_turn = pending[idx + 1] if idx + 1 < len(pending) else None
        next_same_speaker = (
            next_turn is not None
            and str(next_turn.get("speaker")) == str(turn.get("speaker"))
        )
        next_gap = (
            safe_float(next_turn.get("start"), default=0.0)
            - safe_float(turn.get("end"), default=0.0)
            if next_turn is not None
            else float("inf")
        )

        if prev_same_speaker and prev_gap < merge_gap_sec:
            prev_text = str(prev_turn.get("text") or "").strip()
            merged_text = _decapitalize_join(prev_text, text) if text else prev_text
            prev_turn["text"] = merged_text
            prev_turn["end"] = round(
                max(
                    safe_float(prev_turn.get("end"), default=0.0),
                    safe_float(turn.get("end"), default=0.0),
                ),
                3,
            )
            idx += 1
            continue

        if next_same_speaker and next_gap < merge_gap_sec:
            next_text = str(next_turn.get("text") or "").strip()
            merged_text = _decapitalize_join(text, next_text) if next_text else text
            next_turn["text"] = merged_text
            next_turn["start"] = round(
                min(
                    safe_float(next_turn.get("start"), default=0.0),
                    safe_float(turn.get("start"), default=0.0),
                ),
                3,
            )
            idx += 1
            continue

        out.append(turn)
        idx += 1

    return out


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
    "DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC",
    "DEFAULT_SPEAKER_TURN_MIN_WORDS",
    "DEFAULT_SPEAKER_TURN_SHORT_MERGE_GAP_SEC",
    "normalise_asr_segments",
    "build_speaker_turns",
    "merge_short_turns",
    "count_interruptions",
]
