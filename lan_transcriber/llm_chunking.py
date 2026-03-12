from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lan_transcriber.utils import normalise_text_items, safe_float

_QUESTION_TYPE_KEYS = (
    "open",
    "yes_no",
    "clarification",
    "status",
    "decision_seeking",
)
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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_GENERIC_SPEAKER_RE = re.compile(r"^(?:speaker|spk|s)[\s_-]*\d+$", flags=re.IGNORECASE)
_MAX_CONCISE_SPEAKER_LABEL_CHARS = 10
_MAX_CONCISE_SPEAKER_LABEL_WORDS = 2
DEFAULT_COMPACT_MERGE_GAP_SECONDS = 1.0


@dataclass(frozen=True)
class CompactSpeakerMapping:
    compact_label: str
    original_speaker: str
    display_name: str

    def prompt_payload(self) -> dict[str, str]:
        payload = {"label": self.compact_label, "speaker": self.display_name}
        if self.original_speaker != self.display_name:
            payload["original_speaker"] = self.original_speaker
        return payload

    def artifact_payload(self) -> dict[str, str]:
        return {
            "compact_label": self.compact_label,
            "original_speaker": self.original_speaker,
            "display_name": self.display_name,
        }


@dataclass(frozen=True)
class CompactTranscriptUnit:
    speaker_label: str
    original_speaker: str
    display_name: str
    start_seconds: float
    end_seconds: float
    text: str
    source_turn_count: int

    @property
    def line(self) -> str:
        return f"{self.speaker_label}: {self.text}"

    def artifact_payload(self) -> dict[str, Any]:
        return {
            "speaker_label": self.speaker_label,
            "original_speaker": self.original_speaker,
            "display_name": self.display_name,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "text": self.text,
            "chars": len(self.line),
            "source_turn_count": self.source_turn_count,
        }


@dataclass(frozen=True)
class CompactTranscript:
    text: str
    source_chars: int
    compact_chars: int
    merge_gap_seconds: float
    source_turn_count: int
    compact_turn_count: int
    speaker_mapping: tuple[CompactSpeakerMapping, ...]
    units: tuple[CompactTranscriptUnit, ...]

    def prompt_speaker_mapping(self) -> list[dict[str, str]]:
        return [
            item.prompt_payload()
            for item in self.speaker_mapping
            if item.compact_label != item.display_name or item.original_speaker != item.display_name
        ]

    def artifact_payload(self) -> dict[str, Any]:
        reduction_chars = max(self.source_chars - self.compact_chars, 0)
        reduction_ratio = 0.0 if self.source_chars <= 0 else round(reduction_chars / self.source_chars, 4)
        return {
            "source_chars": self.source_chars,
            "compact_chars": self.compact_chars,
            "reduction_chars": reduction_chars,
            "reduction_ratio": reduction_ratio,
            "merge_gap_seconds": self.merge_gap_seconds,
            "source_turn_count": self.source_turn_count,
            "compact_turn_count": self.compact_turn_count,
            "speaker_mapping": [item.artifact_payload() for item in self.speaker_mapping],
            "units": [item.artifact_payload() for item in self.units],
        }


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    total: int
    text: str
    base_text: str
    overlap_prefix: str = ""
    start_seconds: float | None = None
    end_seconds: float | None = None

    def plan_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_index": self.index,
            "chunk_total": self.total,
            "chars": len(self.text),
            "effective_chars": len(self.text),
            "base_chars": len(self.base_text),
            "overlap_chars": len(self.overlap_prefix),
            "text": self.text,
        }
        if self.start_seconds is not None and self.end_seconds is not None:
            payload["time_range"] = {
                "start_seconds": round(self.start_seconds, 3),
                "end_seconds": round(self.end_seconds, 3),
            }
        return payload


@dataclass(frozen=True)
class _PlannedChunkUnit:
    text: str
    start_seconds: float
    end_seconds: float


def _language_name(code: str) -> str:
    return _LANGUAGE_NAME_MAP.get(code, code.upper())


def _normalise_compact_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()


def _has_substantive_content(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _normalise_source_turns(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    aliases: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    resolved_aliases = aliases or {}
    turns: list[dict[str, Any]] = []
    for row in speaker_turns:
        speaker_key = str(row.get("speaker") or "S1").strip() or "S1"
        display_name = str(resolved_aliases.get(speaker_key, speaker_key)).strip() or speaker_key
        text = _normalise_compact_text(row.get("text") or "")
        if not text:
            continue
        start = round(safe_float(row.get("start"), default=0.0), 3)
        end = round(max(start, safe_float(row.get("end"), default=start)), 3)
        turns.append(
            {
                "speaker": speaker_key,
                "display_name": display_name,
                "start": start,
                "end": end,
                "text": text,
            }
        )
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return turns


def _render_source_turn_text(turns: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{turn['start']:.2f}-{turn['end']:.2f}] {turn['display_name']}: {turn['text']}"
        for turn in turns
    ).strip()


def _is_concise_speaker_label(label: str) -> bool:
    normalized = " ".join(str(label).split())
    if not normalized:
        return False
    if len(normalized) > _MAX_CONCISE_SPEAKER_LABEL_CHARS:
        return False
    if len(normalized.split()) > _MAX_CONCISE_SPEAKER_LABEL_WORDS:
        return False
    if _GENERIC_SPEAKER_RE.match(normalized):
        return False
    return all(char.isalnum() or char in {" ", "-", "_", "'"} for char in normalized)


def build_compact_transcript(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    aliases: Mapping[str, str] | None = None,
    merge_gap_seconds: float = DEFAULT_COMPACT_MERGE_GAP_SECONDS,
) -> CompactTranscript:
    normalized_turns = _normalise_source_turns(speaker_turns, aliases=aliases)
    if not normalized_turns:
        raise ValueError("LLM transcript compaction produced no usable content")

    safe_gap = max(merge_gap_seconds, 0.0)
    speaker_mapping_by_key: dict[str, CompactSpeakerMapping] = {}
    used_labels: set[str] = set()
    compact_units: list[CompactTranscriptUnit] = []
    current: CompactTranscriptUnit | None = None

    def _mapping_for(turn: dict[str, Any]) -> CompactSpeakerMapping:
        speaker_key = str(turn["speaker"])
        existing = speaker_mapping_by_key.get(speaker_key)
        if existing is not None:
            return existing

        display_name = str(turn["display_name"])
        order = len(speaker_mapping_by_key) + 1
        preferred = display_name if _is_concise_speaker_label(display_name) else ""
        compact_label = preferred if preferred and preferred not in used_labels else f"S{order}"
        used_labels.add(compact_label)
        mapping = CompactSpeakerMapping(
            compact_label=compact_label,
            original_speaker=speaker_key,
            display_name=display_name,
        )
        speaker_mapping_by_key[speaker_key] = mapping
        return mapping

    def _flush(unit: CompactTranscriptUnit | None) -> None:
        if unit is None:
            return
        cleaned = _normalise_compact_text(unit.text)
        if not cleaned:
            return
        compact_units.append(
            CompactTranscriptUnit(
                speaker_label=unit.speaker_label,
                original_speaker=unit.original_speaker,
                display_name=unit.display_name,
                start_seconds=round(unit.start_seconds, 3),
                end_seconds=round(max(unit.start_seconds, unit.end_seconds), 3),
                text=cleaned,
                source_turn_count=unit.source_turn_count,
            )
        )

    for turn in normalized_turns:
        mapping = _mapping_for(turn)
        text = str(turn["text"])
        if not _has_substantive_content(text):
            continue
        start_seconds = safe_float(turn["start"], default=0.0)
        end_seconds = max(start_seconds, safe_float(turn["end"], default=start_seconds))
        if (
            current is not None
            and current.original_speaker == mapping.original_speaker
            and start_seconds - current.end_seconds <= safe_gap
        ):
            current = CompactTranscriptUnit(
                speaker_label=current.speaker_label,
                original_speaker=current.original_speaker,
                display_name=current.display_name,
                start_seconds=current.start_seconds,
                end_seconds=max(current.end_seconds, end_seconds),
                text=f"{current.text} {text}".strip(),
                source_turn_count=current.source_turn_count + 1,
            )
            continue

        _flush(current)
        current = CompactTranscriptUnit(
            speaker_label=mapping.compact_label,
            original_speaker=mapping.original_speaker,
            display_name=mapping.display_name,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            text=text,
            source_turn_count=1,
        )

    _flush(current)

    if not compact_units:
        raise ValueError("LLM transcript compaction produced no usable content")

    compact_text = "\n".join(unit.line for unit in compact_units).strip()
    source_text = _render_source_turn_text(normalized_turns)
    return CompactTranscript(
        text=compact_text,
        source_chars=len(source_text),
        compact_chars=len(compact_text),
        merge_gap_seconds=safe_gap,
        source_turn_count=len(normalized_turns),
        compact_turn_count=len(compact_units),
        speaker_mapping=tuple(speaker_mapping_by_key[key] for key in speaker_mapping_by_key),
        units=tuple(compact_units),
    )


def _split_words_to_fit(text: str, *, max_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    words = normalized.split(" ")
    chunks: list[str] = []
    current: list[str] = []

    def _append_oversized_word(word: str) -> None:
        for start in range(0, len(word), max_chars):
            chunks.append(word[start : start + max_chars])

    for word in words:
        if not current:
            if len(word) > max_chars:
                _append_oversized_word(word)
                continue
            current = [word]
            continue

        candidate = " ".join([*current, word])
        if len(candidate) > max_chars:
            chunks.append(" ".join(current))
            if len(word) > max_chars:
                _append_oversized_word(word)
                current = []
                continue
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_unit_to_fit(unit: str, *, max_chars: int) -> list[str]:
    normalized = " ".join(unit.split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(normalized) if item.strip()]
    if len(sentences) <= 1:
        return _split_words_to_fit(normalized, max_chars=max_chars)

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(_split_words_to_fit(sentence, max_chars=max_chars))
            continue

        candidate = " ".join([*current, sentence]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current = [*current, sentence]

    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text

    tail = text[-max_chars:]
    newline = tail.find("\n")
    if newline > 0:
        trimmed = tail[newline + 1 :].lstrip()
        if trimmed:
            return trimmed
    return tail.lstrip()


def plan_transcript_chunks(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[TranscriptChunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    safe_overlap = min(overlap_chars, max(max_chars - 1, 0))
    base_units: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        base_units.extend(_split_unit_to_fit(stripped, max_chars=max_chars))

    if not base_units:
        return []

    base_chunks: list[str] = []
    current: list[str] = []
    for unit in base_units:
        candidate = "\n".join([*current, unit]).strip() if current else unit
        if current and len(candidate) > max_chars:
            base_chunks.append("\n".join(current))
            current = [unit]
        else:
            current = [*current, unit]

    base_chunks.append("\n".join(current))

    total = len(base_chunks)
    chunks: list[TranscriptChunk] = []
    for index, base_text in enumerate(base_chunks, start=1):
        overlap_prefix = ""
        if safe_overlap > 0 and index > 1:
            overlap_prefix = _tail_text(base_chunks[index - 2], max_chars=safe_overlap)
        text_with_overlap = base_text if not overlap_prefix else f"{overlap_prefix}\n{base_text}"
        chunks.append(
            TranscriptChunk(
                index=index,
                total=total,
                text=text_with_overlap,
                base_text=base_text,
                overlap_prefix=overlap_prefix,
            )
        )
    return chunks


def _split_compact_unit_to_fit(
    unit: CompactTranscriptUnit,
    *,
    max_chars: int,
) -> list[_PlannedChunkUnit]:
    line = unit.line
    if len(line) <= max_chars:
        return [
            _PlannedChunkUnit(
                text=line,
                start_seconds=unit.start_seconds,
                end_seconds=unit.end_seconds,
            )
        ]

    prefix = f"{unit.speaker_label}: "
    content_limit = max_chars - len(prefix)
    if content_limit <= 0:
        return [
            _PlannedChunkUnit(
                text=fragment,
                start_seconds=unit.start_seconds,
                end_seconds=unit.end_seconds,
            )
            for fragment in _split_words_to_fit(line, max_chars=max_chars)
        ]

    return [
        _PlannedChunkUnit(
            text=f"{prefix}{fragment}",
            start_seconds=unit.start_seconds,
            end_seconds=unit.end_seconds,
        )
        for fragment in _split_unit_to_fit(unit.text, max_chars=content_limit)
    ]


def plan_compact_transcript_chunks(
    compact_transcript: CompactTranscript,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[TranscriptChunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if not compact_transcript.text.strip():
        return []

    safe_overlap = min(overlap_chars, max(max_chars - 1, 0))
    planned_units: list[_PlannedChunkUnit] = []
    for unit in compact_transcript.units:
        planned_units.extend(_split_compact_unit_to_fit(unit, max_chars=max_chars))

    if not planned_units:
        return []

    base_chunks: list[tuple[str, float, float]] = []
    current_lines: list[str] = []
    current_start: float | None = None
    current_end: float | None = None
    for unit in planned_units:
        candidate = "\n".join([*current_lines, unit.text]).strip() if current_lines else unit.text
        if current_lines and len(candidate) > max_chars:
            base_chunks.append(("\n".join(current_lines), current_start or 0.0, current_end or current_start or 0.0))
            current_lines = [unit.text]
            current_start = unit.start_seconds
            current_end = unit.end_seconds
            continue

        current_lines.append(unit.text)
        current_start = unit.start_seconds if current_start is None else current_start
        current_end = unit.end_seconds

    base_chunks.append(("\n".join(current_lines), current_start or 0.0, current_end or current_start or 0.0))

    total = len(base_chunks)
    chunks: list[TranscriptChunk] = []
    for index, (base_text, start_seconds, end_seconds) in enumerate(base_chunks, start=1):
        overlap_prefix = ""
        if safe_overlap > 0 and index > 1:
            overlap_prefix = _tail_text(base_chunks[index - 2][0], max_chars=safe_overlap)
        text_with_overlap = base_text if not overlap_prefix else f"{overlap_prefix}\n{base_text}"
        chunks.append(
            TranscriptChunk(
                index=index,
                total=total,
                text=text_with_overlap,
                base_text=base_text,
                overlap_prefix=overlap_prefix,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return chunks


def split_transcript_for_llm(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    return [chunk.text for chunk in plan_transcript_chunks(text, max_chars=max_chars, overlap_chars=overlap_chars)]


def build_chunk_prompt(
    chunk: TranscriptChunk,
    *,
    target_summary_language: str,
    calendar_title: str | None = None,
    calendar_attendees: Sequence[str] | None = None,
    speaker_mapping: Sequence[dict[str, str]] | None = None,
) -> tuple[str, str]:
    language_name = _language_name(target_summary_language)
    sys_prompt = (
        "You extract structured facts from one chunk of a meeting transcript. "
        f"Write concise structured fields in {language_name}. "
        "Prefer short factual bullets over prose. "
        "Return strict JSON only, with no markdown fences."
    )
    payload = {
        "target_summary_language": target_summary_language,
        "chunk": {
            "index": chunk.index,
            "total": chunk.total,
            "time_range": (
                {
                    "start_seconds": round(chunk.start_seconds, 3),
                    "end_seconds": round(chunk.end_seconds, 3),
                }
                if chunk.start_seconds is not None and chunk.end_seconds is not None
                else None
            ),
        },
        "calendar": {
            "title": (calendar_title or "").strip() or None,
            "attendees": [str(item).strip() for item in (calendar_attendees or []) if str(item).strip()],
        },
        "speaker_mapping": [dict(item) for item in (speaker_mapping or [])],
        "transcript_chunk": chunk.text,
        "required_schema": {
            "topic_candidates": ["string"],
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
            "emotional_cues": ["string"],
            "questions": {
                "total_count": "integer >= 0",
                "types": {key: "integer >= 0" for key in _QUESTION_TYPE_KEYS},
                "extracted": ["string"],
            },
        },
    }
    return sys_prompt, json.dumps(payload, ensure_ascii=False, indent=2)


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

    for candidate in dict.fromkeys(candidates):
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalise_action_items(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else ([] if value is None else [value])
    items: list[dict[str, Any]] = []
    for row in rows:
        if len(items) >= 30:
            break
        if isinstance(row, dict):
            task = str(row.get("task") or row.get("action") or row.get("title") or "").strip()
            owner = str(row.get("owner") or "").strip() or None
            deadline = str(row.get("deadline") or row.get("due") or "").strip() or None
            confidence = round(min(max(safe_float(row.get("confidence"), default=0.5), 0.0), 1.0), 2)
        else:
            task = str(row).strip()
            owner = None
            deadline = None
            confidence = 0.5
        if not task:
            continue
        items.append(
            {
                "task": task,
                "owner": owner,
                "deadline": deadline,
                "confidence": confidence,
            }
        )
    return items


def _normalise_questions(value: Any) -> dict[str, Any]:
    total_count = 0
    types = {key: 0 for key in _QUESTION_TYPE_KEYS}
    extracted = normalise_text_items([], max_items=20)

    if isinstance(value, dict):
        total_count = max(0, int(safe_float(value.get("total_count"), default=0.0)))
        raw_types = value.get("types") if isinstance(value.get("types"), dict) else value
        for key in _QUESTION_TYPE_KEYS:
            types[key] = max(0, int(safe_float(raw_types.get(key), default=0.0)))
        extracted = normalise_text_items(value.get("extracted"), max_items=20)

    if total_count == 0:
        total_count = max(sum(types.values()), len(extracted))
    return {"total_count": total_count, "types": types, "extracted": extracted}


def parse_chunk_extract(raw_content: str) -> dict[str, Any]:
    extracted = _extract_json_dict(raw_content)
    if extracted is None:
        raise ValueError("json_object_not_found")

    topic_candidates = normalise_text_items(
        extracted.get("topic_candidates") or extracted.get("topics") or extracted.get("topic"),
        max_items=8,
    )
    summary_bullets = normalise_text_items(
        extracted.get("summary_bullets") or extracted.get("summary"),
        max_items=12,
    )
    decisions = normalise_text_items(extracted.get("decisions"), max_items=20)
    emotional_cues = normalise_text_items(
        extracted.get("emotional_cues") or extracted.get("emotional_summary"),
        max_items=6,
    )
    return {
        "topic_candidates": topic_candidates,
        "summary_bullets": summary_bullets,
        "decisions": decisions,
        "action_items": _normalise_action_items(extracted.get("action_items")),
        "emotional_cues": emotional_cues,
        "questions": _normalise_questions(extracted.get("questions")),
    }


def _dedupe_text_items(items: Sequence[str], *, max_items: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if len(deduped) >= max_items:
            break
    return deduped


def _dedupe_action_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        task = str(item.get("task") or "").strip()
        owner = str(item.get("owner") or "").strip()
        deadline = str(item.get("deadline") or "").strip()
        if not task:
            continue
        key = (task.casefold(), owner.casefold(), deadline.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "task": task,
                "owner": owner or None,
                "deadline": deadline or None,
                "confidence": round(min(max(safe_float(item.get("confidence"), default=0.5), 0.0), 1.0), 2),
            }
        )
        if len(deduped) >= 30:
            break
    return deduped


def merge_chunk_results(chunk_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    topics: list[str] = []
    summary_bullets: list[str] = []
    decisions: list[str] = []
    emotional_cues: list[str] = []
    action_items: list[dict[str, Any]] = []
    question_extracted: list[str] = []
    question_types = {key: 0 for key in _QUESTION_TYPE_KEYS}
    total_count_hint = 0
    chunks: list[dict[str, Any]] = []

    for offset, row in enumerate(chunk_results, start=1):
        topic_candidates = normalise_text_items(row.get("topic_candidates"), max_items=8)
        chunk_summary = normalise_text_items(row.get("summary_bullets"), max_items=12)
        chunk_decisions = normalise_text_items(row.get("decisions"), max_items=20)
        chunk_emotional_cues = normalise_text_items(row.get("emotional_cues"), max_items=6)
        chunk_action_items = _normalise_action_items(row.get("action_items"))
        chunk_questions = _normalise_questions(row.get("questions"))

        topics.extend(topic_candidates)
        summary_bullets.extend(chunk_summary)
        decisions.extend(chunk_decisions)
        emotional_cues.extend(chunk_emotional_cues)
        action_items.extend(chunk_action_items)
        question_extracted.extend(chunk_questions["extracted"])
        total_count_hint += int(chunk_questions["total_count"])
        for key in _QUESTION_TYPE_KEYS:
            question_types[key] += int(chunk_questions["types"].get(key, 0))

        chunk_index = max(1, int(safe_float(row.get("chunk_index"), default=float(offset))))
        chunk_total = max(chunk_index, int(safe_float(row.get("chunk_total"), default=float(offset))))
        chunks.append(
            {
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
                "topic_candidates": topic_candidates,
                "summary_bullets": chunk_summary,
                "decisions": chunk_decisions,
                "action_items": chunk_action_items,
                "emotional_cues": chunk_emotional_cues,
                "questions": chunk_questions,
            }
        )

    return {
        "chunk_count": len(chunks),
        "topic_candidates": _dedupe_text_items(topics, max_items=12),
        "summary_bullets": _dedupe_text_items(summary_bullets, max_items=40),
        "decisions": _dedupe_text_items(decisions, max_items=40),
        "action_items": _dedupe_action_items(action_items),
        "emotional_cues": _dedupe_text_items(emotional_cues, max_items=12),
        "questions": {
            "total_count_hint": max(total_count_hint, len(_dedupe_text_items(question_extracted, max_items=30))),
            "type_hints": question_types,
            "extracted": _dedupe_text_items(question_extracted, max_items=30),
        },
        "chunks": chunks,
    }


def build_merge_prompt(
    merge_input: dict[str, Any],
    *,
    target_summary_language: str,
    calendar_title: str | None = None,
    calendar_attendees: Sequence[str] | None = None,
) -> tuple[str, str]:
    language_name = _language_name(target_summary_language)
    sys_prompt = (
        "You merge structured extracts from multiple overlapping transcript chunks into one final meeting summary. "
        f"Write topic, summary_bullets, decisions, action_items, emotional_summary, and questions in {language_name}. "
        "Deduplicate repeated content caused by overlapping chunks and reconcile conflicts conservatively. "
        "Return strict JSON only, with no markdown fences."
    )
    payload = {
        "target_summary_language": target_summary_language,
        "calendar": {
            "title": (calendar_title or "").strip() or None,
            "attendees": [str(item).strip() for item in (calendar_attendees or []) if str(item).strip()],
        },
        "overlap_warning": "Adjacent transcript chunks overlap. Do not double-count repeated evidence.",
        "merge_input": merge_input,
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
    return sys_prompt, json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "CompactSpeakerMapping",
    "CompactTranscript",
    "CompactTranscriptUnit",
    "DEFAULT_COMPACT_MERGE_GAP_SECONDS",
    "TranscriptChunk",
    "build_compact_transcript",
    "build_chunk_prompt",
    "build_merge_prompt",
    "merge_chunk_results",
    "parse_chunk_extract",
    "plan_compact_transcript_chunks",
    "plan_transcript_chunks",
    "split_transcript_for_llm",
]
