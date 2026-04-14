from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .calendar.matching import calendar_summary_context
from .config import AppSettings
from .conversation_metrics import refresh_recording_metrics
from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.llm_client import LLMClient
from lan_transcriber.pipeline import (
    build_structured_summary_prompts,
    build_summary_payload,
)
from lan_transcriber.utils import normalise_language_code


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(payload)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_json_list(path: Path) -> list[Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(payload)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def _chunk_text_for_turns(text: str, *, chunk_size: int = 450) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    words = normalized.split(" ")
    current: list[str] = []
    current_len = 0
    for word in words:
        word_len = len(word)
        separator_len = 1 if current else 0
        if current and current_len + separator_len + word_len > chunk_size:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
            continue
        if not current and word_len > chunk_size:
            start = 0
            while start < word_len:
                end = min(start + chunk_size, word_len)
                chunks.append(word[start:end])
                start = end
            continue
        if separator_len:
            current_len += separator_len
        current.append(word)
        current_len += word_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def _fallback_speaker_turns_from_transcript(
    transcript_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    segments_payload = transcript_payload.get("segments")
    if isinstance(segments_payload, list):
        segment_turns: list[dict[str, Any]] = []
        for row in segments_payload:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(row.get("start") or 0.0)
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(row.get("end") or row.get("start") or start)
            except (TypeError, ValueError):
                end = start
            segment_turns.append(
                {
                    "start": start,
                    "end": max(end, start),
                    "speaker": "S1",
                    "text": text,
                    "language": row.get("language"),
                }
            )
        if segment_turns:
            return segment_turns

    transcript_text = str(transcript_payload.get("text") or "").strip()
    return [
        {
            "start": float(index),
            "end": float(index + 1),
            "speaker": "S1",
            "text": chunk,
        }
        for index, chunk in enumerate(_chunk_text_for_turns(transcript_text))
    ]


def _recording_derived_paths(
    recording_id: str,
    settings: AppSettings,
) -> tuple[Path, Path]:
    derived_dir = settings.recordings_root / recording_id / "derived"
    return derived_dir / "transcript.json", derived_dir / "summary.json"


def _llm_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(message)


def resummarize_recording(
    recording_id: str,
    *,
    settings: AppSettings,
    target_summary_language: str | None,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    transcript_path, summary_path = _recording_derived_paths(recording_id, settings)
    speaker_turns_path = transcript_path.parent / "speaker_turns.json"
    transcript_payload = _load_json_dict(transcript_path)
    if not transcript_payload:
        raise ValueError("No transcript.json found for this recording")

    transcript_text = str(transcript_payload.get("text") or "").strip()
    if not transcript_text:
        raise ValueError("Transcript text is empty; re-transcribe first")

    language_payload = transcript_payload.get("language")
    language_obj = language_payload if isinstance(language_payload, dict) else {}
    resolved_target = (
        normalise_language_code(target_summary_language)
        or normalise_language_code(transcript_payload.get("target_summary_language"))
        or normalise_language_code(transcript_payload.get("dominant_language"))
        or normalise_language_code(language_obj.get("detected"))
        or "en"
    )

    speaker_turns_raw = _load_json_list(speaker_turns_path)
    speaker_turns = [row for row in speaker_turns_raw if isinstance(row, dict)]
    if not speaker_turns:
        speaker_turns = _fallback_speaker_turns_from_transcript(transcript_payload)
    if not speaker_turns:
        speaker_turns = [
            {"start": 0.0, "end": 0.0, "speaker": "S1", "text": transcript_text}
        ]

    calendar_title, calendar_attendees = calendar_summary_context(
        recording_id,
        settings=settings,
    )
    if calendar_title is None and not calendar_attendees:
        calendar_title = (
            str(transcript_payload.get("calendar_title") or "").strip() or None
        )
        attendees_payload = transcript_payload.get("calendar_attendees")
        if isinstance(attendees_payload, list):
            calendar_attendees = [
                str(attendee).strip()
                for attendee in attendees_payload
                if str(attendee).strip()
            ]

    system_prompt, user_prompt = build_structured_summary_prompts(
        speaker_turns,
        resolved_target,
        calendar_title=calendar_title,
        calendar_attendees=calendar_attendees,
    )
    client = llm_client or LLMClient()
    message = client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=settings.llm_model,
        response_format={"type": "json_object"},
    )
    if asyncio.iscoroutine(message):
        message = asyncio.run(message)
    raw_summary = _llm_message_content(message)

    summary_payload = _load_json_dict(summary_path)
    friendly = summary_payload.get("friendly")
    if not isinstance(friendly, int):
        friendly = 0
    structured_payload = build_summary_payload(
        raw_llm_content=raw_summary,
        model=settings.llm_model,
        target_summary_language=resolved_target,
        friendly=friendly,
        default_topic=calendar_title or "Meeting summary",
    )
    summary_payload.update(structured_payload)
    atomic_write_json(summary_path, summary_payload)

    transcript_payload["target_summary_language"] = resolved_target
    atomic_write_json(transcript_path, transcript_payload)
    refresh_recording_metrics(recording_id, settings=settings)
    return summary_payload


__all__ = ["resummarize_recording"]
