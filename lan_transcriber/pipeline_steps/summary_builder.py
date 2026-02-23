from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from lan_transcriber.pipeline_steps.artifacts import LlmDebugArtifacts, write_llm_debug_artifacts
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


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str
    owner: str | None = None
    deadline: str | None = None
    confidence: float = Field(default=0.5)

    @field_validator("task", mode="before")
    @classmethod
    def _clean_task(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("task is required")
        return text

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _clean_nullable_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clean_confidence(cls, value: Any) -> float:
        parsed = safe_float(value, default=0.5)
        return round(min(max(parsed, 0.0), 1.0), 2)


class Question(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_count: int = 0
    types: dict[str, int] = Field(default_factory=lambda: {key: 0 for key in _QUESTION_TYPE_KEYS})
    extracted: list[str] = Field(default_factory=list)

    @field_validator("total_count", mode="before")
    @classmethod
    def _clean_total(cls, value: Any) -> int:
        return max(0, int(safe_float(value, default=0.0)))

    @field_validator("types", mode="before")
    @classmethod
    def _clean_types(cls, value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            return {key: 0 for key in _QUESTION_TYPE_KEYS}
        out = {key: 0 for key in _QUESTION_TYPE_KEYS}
        for key in _QUESTION_TYPE_KEYS:
            out[key] = max(0, int(safe_float(value.get(key), default=0.0)))
        return out

    @field_validator("extracted", mode="before")
    @classmethod
    def _clean_extracted(cls, value: Any) -> list[str]:
        return normalise_text_items(value, max_items=20)

    @model_validator(mode="after")
    def _ensure_total_count(self) -> "Question":
        inferred = max(sum(self.types.values()), len(self.extracted))
        if self.total_count == 0:
            self.total_count = inferred
        return self


class SummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topic: str
    summary_bullets: list[str]
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    emotional_summary: str
    questions: Question = Field(default_factory=Question)

    @field_validator("topic", mode="before")
    @classmethod
    def _clean_topic(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("summary_bullets", mode="before")
    @classmethod
    def _clean_summary_bullets(cls, value: Any) -> list[str]:
        items = normalise_text_items(value, max_items=12)
        if not items:
            raise ValueError("summary_bullets must not be empty")
        return items

    @field_validator("decisions", mode="before")
    @classmethod
    def _clean_decisions(cls, value: Any) -> list[str]:
        return normalise_text_items(value, max_items=20)

    @field_validator("emotional_summary", mode="before")
    @classmethod
    def _clean_emotional_summary(cls, value: Any) -> str:
        lines: list[str]
        if isinstance(value, str):
            lines = [line.strip() for line in value.splitlines() if line.strip()]
        else:
            lines = normalise_text_items(value, max_items=3)
        if not lines:
            return "Neutral and focused discussion."
        return "\n".join(lines[:3])


def _language_name(code: str) -> str:
    return _LANGUAGE_NAME_MAP.get(code, code.upper())


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
    max_turns: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in speaker_turns:
        start = round(safe_float(row.get("start"), default=0.0), 3)
        end = round(safe_float(row.get("end"), default=0.0), 3)
        speaker = str(row.get("speaker") or "S1")
        language = row.get("language")
        for chunk in _chunk_text_for_prompt(str(row.get("text") or "").strip()):
            if max_turns is not None and max_turns >= 0 and len(out) >= max_turns:
                return out
            payload: dict[str, Any] = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": chunk,
            }
            if isinstance(language, str) and language.strip():
                payload["language"] = language.strip()
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


def _summary_text_from_bullets(summary_bullets: Sequence[str]) -> str:
    return "\n".join(f"- {bullet}" for bullet in summary_bullets)


def _normalise_action_items_fallback(value: Any) -> list[dict[str, Any]]:
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
        confidence = safe_float(confidence_raw, default=0.5)
        out.append(
            {
                "task": task,
                "owner": owner or None,
                "deadline": deadline or None,
                "confidence": round(min(max(confidence, 0.0), 1.0), 2),
            }
        )
    return out


def _normalise_questions_fallback(value: Any) -> dict[str, Any]:
    total_count = 0
    question_types = {key: 0 for key in _QUESTION_TYPE_KEYS}
    extracted: list[str] = []

    if isinstance(value, dict):
        total_count = max(0, int(safe_float(value.get("total_count"), default=0.0)))
        types_payload = value.get("types") if isinstance(value.get("types"), dict) else value
        for key in _QUESTION_TYPE_KEYS:
            question_types[key] = max(0, int(safe_float(types_payload.get(key), default=0.0)))
        extracted = normalise_text_items(value.get("extracted"), max_items=20)

    inferred_total = max(sum(question_types.values()), len(extracted))
    if total_count == 0:
        total_count = inferred_total

    return {
        "total_count": total_count,
        "types": question_types,
        "extracted": extracted,
    }


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
    parse_error: bool = False,
    parse_error_reason: str | None = None,
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
    if parse_error:
        payload["parse_error"] = True
        payload["parse_error_reason"] = parse_error_reason or "validation_failed"
    return payload


def _fallback_payload(
    *,
    raw_llm_content: str,
    extracted: dict[str, Any],
    model: str,
    target_summary_language: str,
    friendly: int,
    default_topic: str,
    parse_error_reason: str,
) -> dict[str, Any]:
    summary_bullets = normalise_text_items(extracted.get("summary_bullets"), max_items=12)
    if not summary_bullets:
        summary_bullets = normalise_text_items(extracted.get("summary"), max_items=12)
    if not summary_bullets:
        summary_bullets = normalise_text_items(raw_llm_content, max_items=12)
    if not summary_bullets:
        summary_bullets = ["No summary available."]

    topic = str(extracted.get("topic") or "").strip()
    if not topic:
        topic = summary_bullets[0][:120] if summary_bullets else default_topic
    topic = topic or default_topic

    emotional_lines = normalise_text_items(extracted.get("emotional_summary"), max_items=3)
    emotional_summary = "\n".join(emotional_lines) if emotional_lines else "Neutral and focused discussion."

    return _build_structured_summary_payload(
        model=model,
        target_summary_language=target_summary_language,
        friendly=friendly,
        topic=topic,
        summary_bullets=summary_bullets,
        decisions=normalise_text_items(extracted.get("decisions"), max_items=20),
        action_items=_normalise_action_items_fallback(extracted.get("action_items")),
        emotional_summary=emotional_summary,
        questions=_normalise_questions_fallback(extracted.get("questions")),
        parse_error=True,
        parse_error_reason=parse_error_reason,
    )


def _validation_reason(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "validation_failed"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", []))
    message = str(first.get("msg") or "invalid value")
    return f"{location}: {message}" if location else message


def build_summary_payload(
    *,
    raw_llm_content: str,
    model: str,
    target_summary_language: str,
    friendly: int,
    default_topic: str = "Meeting summary",
    derived_dir: Path | None = None,
) -> dict[str, Any]:
    extracted = _extract_json_dict(raw_llm_content)
    if extracted is None:
        extracted = {}
        reason = "json_object_not_found"
        if derived_dir is not None:
            write_llm_debug_artifacts(
                LlmDebugArtifacts(
                    derived_dir=derived_dir,
                    raw_output=raw_llm_content,
                    extracted_payload=extracted,
                    validation_error={"reason": reason},
                )
            )
        return _fallback_payload(
            raw_llm_content=raw_llm_content,
            extracted=extracted,
            model=model,
            target_summary_language=target_summary_language,
            friendly=friendly,
            default_topic=default_topic,
            parse_error_reason=reason,
        )

    candidate = dict(extracted)
    if "summary_bullets" not in candidate and "summary" in candidate:
        candidate["summary_bullets"] = normalise_text_items(candidate.get("summary"), max_items=12)
    if "topic" not in candidate:
        candidate["topic"] = default_topic

    try:
        validated = SummaryResponse.model_validate(candidate)
    except ValidationError as exc:
        reason = _validation_reason(exc)
        if derived_dir is not None:
            write_llm_debug_artifacts(
                LlmDebugArtifacts(
                    derived_dir=derived_dir,
                    raw_output=raw_llm_content,
                    extracted_payload=extracted,
                    validation_error={
                        "reason": reason,
                        "errors": json.loads(exc.json()),
                    },
                )
            )
        return _fallback_payload(
            raw_llm_content=raw_llm_content,
            extracted=extracted,
            model=model,
            target_summary_language=target_summary_language,
            friendly=friendly,
            default_topic=default_topic,
            parse_error_reason=reason,
        )

    return _build_structured_summary_payload(
        model=model,
        target_summary_language=target_summary_language,
        friendly=friendly,
        topic=validated.topic or default_topic,
        summary_bullets=validated.summary_bullets,
        decisions=validated.decisions,
        action_items=[item.model_dump() for item in validated.action_items],
        emotional_summary=validated.emotional_summary,
        questions=validated.questions.model_dump(),
    )


__all__ = [
    "ActionItem",
    "Question",
    "SummaryResponse",
    "build_summary_payload",
    "build_summary_prompts",
    "build_structured_summary_prompts",
]
