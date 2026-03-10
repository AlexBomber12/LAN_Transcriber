from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any

from lan_app.config import AppSettings
from lan_app.db import (
    get_calendar_match,
    get_recording,
    list_calendar_events,
    upsert_calendar_match,
)

_AUTO_SELECT_MIN_MARGIN = 0.12
_AUTO_SELECT_MIN_SCORE = 0.78
_SEARCH_WINDOW_AFTER = timedelta(hours=12)
_SEARCH_WINDOW_BEFORE = timedelta(hours=8)
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_STOPWORDS = {"and", "audio", "meeting", "recording", "sync", "the"}


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_duration_seconds(value: object) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return duration


def _tokenize(value: object) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    tokens = {token for token in _TOKEN_RE.findall(text) if token not in _STOPWORDS}
    return tokens


def _recording_tokens(recording: dict[str, Any]) -> set[str]:
    filename = Path(str(recording.get("source_filename") or "").strip()).stem
    return _tokenize(filename)


def _candidate_rows(raw: object) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def calendar_match_candidates(
    recording_id: str,
    *,
    settings: AppSettings,
) -> list[dict[str, Any]]:
    row = get_calendar_match(recording_id, settings=settings) or {}
    return _candidate_rows(row.get("candidates_json"))


def _candidate_attendee_details(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in value:
        if isinstance(row, dict):
            name = str(row.get("name") or "").strip()
            email = str(row.get("email") or "").strip()
            label = str(row.get("label") or name or email).strip()
        else:
            name = ""
            email = ""
            label = str(row or "").strip()
        if not label:
            continue
        dedupe_key = (email.lower(), label.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        payload = {"label": label}
        if name:
            payload["name"] = name
        if email:
            payload["email"] = email
        out.append(payload)
    return out


def _candidate_attendees(candidate: dict[str, Any]) -> list[str]:
    attendees = candidate.get("attendees")
    if isinstance(attendees, list):
        cleaned = [str(value).strip() for value in attendees if str(value).strip()]
        if cleaned:
            return cleaned
    return [row["label"] for row in _candidate_attendee_details(candidate.get("attendee_details"))]


def selected_calendar_candidate(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    row = get_calendar_match(recording_id, settings=settings) or {}
    selected_event_id = str(row.get("selected_event_id") or "").strip()
    if not selected_event_id:
        return {}
    for candidate in _candidate_rows(row.get("candidates_json")):
        if str(candidate.get("event_id") or "").strip() == selected_event_id:
            normalized = dict(candidate)
            normalized["attendees"] = _candidate_attendees(normalized)
            return normalized
    return {}


def calendar_summary_context(
    recording_id: str,
    *,
    settings: AppSettings,
) -> tuple[str | None, list[str]]:
    selected = selected_calendar_candidate(recording_id, settings=settings)
    title = str(selected.get("subject") or selected.get("summary") or "").strip() or None
    return title, _candidate_attendees(selected)


def _event_identifier(event: dict[str, Any]) -> str:
    source_id = str(event.get("source_id") or "").strip()
    uid = str(event.get("uid") or "").strip()
    starts_at = str(event.get("starts_at") or "").strip()
    return f"{source_id}:{uid}:{starts_at}"


def _event_attendee_details(event: dict[str, Any]) -> list[dict[str, str]]:
    return _candidate_attendee_details(event.get("attendees_json"))


def _event_attendees(event: dict[str, Any]) -> list[str]:
    return [row["label"] for row in _event_attendee_details(event)]


def _event_tokens(event: dict[str, Any]) -> set[str]:
    values: list[object] = [
        event.get("summary"),
        event.get("location"),
        event.get("organizer"),
    ]
    values.extend(_event_attendees(event))
    tokens: set[str] = set()
    for value in values:
        tokens.update(_tokenize(value))
    return tokens


def _score_event(
    *,
    event: dict[str, Any],
    captured_at: datetime,
    duration_sec: float | None,
    recording_tokens: set[str],
) -> dict[str, Any] | None:
    starts_at = _parse_datetime(event.get("starts_at"))
    ends_at = _parse_datetime(event.get("ends_at"))
    if starts_at is None or ends_at is None or ends_at <= starts_at:
        return None

    start_delta_minutes = abs((captured_at - starts_at).total_seconds()) / 60.0
    proximity_score = max(0.0, 1.0 - (start_delta_minutes / 180.0))
    capture_inside = starts_at <= captured_at <= ends_at
    presence_score = 1.0 if capture_inside else 0.0

    overlap_score = 0.0
    duration_score = 0.0
    if duration_sec is not None:
        recording_end = captured_at + timedelta(seconds=duration_sec)
        overlap_seconds = max(
            0.0,
            (min(recording_end, ends_at) - max(captured_at, starts_at)).total_seconds(),
        )
        overlap_score = min(1.0, overlap_seconds / max(duration_sec, 1.0))
        event_duration_seconds = max((ends_at - starts_at).total_seconds(), 60.0)
        duration_score = max(
            0.0,
            1.0
            - (
                abs(event_duration_seconds - duration_sec)
                / max(event_duration_seconds, duration_sec, 1.0)
            ),
        )

    event_tokens = _event_tokens(event)
    shared_tokens = sorted(recording_tokens & event_tokens)
    token_score = (
        min(1.0, len(shared_tokens) / max(len(recording_tokens), 1))
        if recording_tokens
        else 0.0
    )

    if duration_sec is None:
        score = (0.60 * proximity_score) + (0.25 * presence_score) + (0.15 * token_score)
    else:
        score = (
            (0.30 * proximity_score)
            + (0.30 * presence_score)
            + (0.25 * overlap_score)
            + (0.10 * duration_score)
            + (0.05 * token_score)
        )
    confidence = round(max(0.0, min(score, 1.0)), 4)

    rationale = [f"Start is {int(round(start_delta_minutes))}m from capture time."]
    if capture_inside:
        rationale.append("Capture time falls inside the event window.")
    if duration_sec is not None:
        rationale.append(
            f"Recording overlaps {int(round(overlap_score * 100))}% of the event window."
        )
        rationale.append(
            f"Duration similarity is {int(round(duration_score * 100))}%."
        )
    if shared_tokens:
        rationale.append(f"Shared filename/title tokens: {', '.join(shared_tokens)}.")
    rationale.append(f"Final confidence {confidence:.2f}.")

    return {
        "event_id": _event_identifier(event),
        "uid": str(event.get("uid") or "").strip(),
        "source_id": event.get("source_id"),
        "source_name": str(event.get("source_name") or "").strip() or None,
        "source_kind": str(event.get("source_kind") or "").strip() or None,
        "subject": str(event.get("summary") or "").strip() or None,
        "summary": str(event.get("summary") or "").strip() or None,
        "description": str(event.get("description") or "").strip() or None,
        "location": str(event.get("location") or "").strip() or None,
        "starts_at": _utc_iso(starts_at),
        "ends_at": _utc_iso(ends_at),
        "all_day": bool(event.get("all_day")),
        "organizer": str(event.get("organizer") or "").strip() or None,
        "organizer_name": str(event.get("organizer_name") or "").strip() or None,
        "organizer_email": str(event.get("organizer_email") or "").strip() or None,
        "attendees": _event_attendees(event),
        "attendee_details": _event_attendee_details(event),
        "score": confidence,
        "confidence": confidence,
        "rationale": rationale,
    }


def _auto_selection(candidates: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    if not candidates:
        return None, None
    top_score = float(candidates[0].get("score") or 0.0)
    if top_score < _AUTO_SELECT_MIN_SCORE:
        return None, None
    if len(candidates) > 1:
        next_score = float(candidates[1].get("score") or 0.0)
        if (top_score - next_score) < _AUTO_SELECT_MIN_MARGIN:
            return None, None
    return str(candidates[0].get("event_id") or "").strip() or None, top_score


def match_recording_to_calendar(
    recording: dict[str, Any],
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    recording_id = str(recording.get("id") or "").strip()
    captured_at = _parse_datetime(recording.get("captured_at"))
    if captured_at is None:
        return {
            "recording_id": recording_id,
            "candidates": [],
            "selected_event_id": None,
            "selected_confidence": None,
        }

    duration_sec = _safe_duration_seconds(recording.get("duration_sec"))
    window_end = captured_at + max(
        _SEARCH_WINDOW_AFTER,
        timedelta(seconds=duration_sec or 0.0) + timedelta(hours=2),
    )
    events = list_calendar_events(
        starts_from=_utc_iso(captured_at - _SEARCH_WINDOW_BEFORE),
        ends_to=_utc_iso(window_end),
        settings=settings,
    )
    recording_tokens = _recording_tokens(recording)
    candidates = [
        candidate
        for event in events
        if (candidate := _score_event(
            event=event,
            captured_at=captured_at,
            duration_sec=duration_sec,
            recording_tokens=recording_tokens,
        ))
        is not None
    ]
    candidates.sort(
        key=lambda candidate: (
            -float(candidate.get("score") or 0.0),
            str(candidate.get("starts_at") or ""),
            str(candidate.get("event_id") or ""),
        )
    )
    selected_event_id, selected_confidence = _auto_selection(candidates)
    return {
        "recording_id": recording_id,
        "candidates": candidates,
        "selected_event_id": selected_event_id,
        "selected_confidence": selected_confidence,
    }


def refresh_recording_calendar_match(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    recording = get_recording(recording_id, settings=settings)
    if recording is None:
        raise KeyError(recording_id)
    existing = get_calendar_match(recording_id, settings=settings) or {}
    match = match_recording_to_calendar(recording, settings=settings)
    if not match["candidates"] and _candidate_rows(existing.get("candidates_json")):
        return existing
    return upsert_calendar_match(
        recording_id=recording_id,
        candidates=match["candidates"],
        selected_event_id=match["selected_event_id"],
        selected_confidence=match["selected_confidence"],
        settings=settings,
    )


__all__ = [
    "calendar_match_candidates",
    "calendar_summary_context",
    "match_recording_to_calendar",
    "refresh_recording_calendar_match",
    "selected_calendar_candidate",
]
