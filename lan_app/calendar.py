"""Calendar matching for recordings using Microsoft Graph calendarView."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
import re
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import AppSettings
from .db import (
    get_calendar_match,
    get_recording,
    set_calendar_match_selection,
    upsert_calendar_match,
)
from .ms_graph import MicrosoftGraphClient

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ISO_FRACTION_RE = re.compile(
    r"^(?P<body>.+?)(?P<fraction>\.\d+)(?P<suffix>Z|[+-]\d{2}:\d{2})?$"
)
_MANUAL_NO_EVENT_CONFIDENCE = -1.0
_GRAPH_CANDIDATE_LIMIT = 25
_UI_CANDIDATE_LIMIT = 5
_WINDOWS_TZ_TO_IANA = {
    "UTC": "UTC",
    "GMT Standard Time": "Europe/London",
    "W. Europe Standard Time": "Europe/Berlin",
    "Central Europe Standard Time": "Europe/Budapest",
    "Romance Standard Time": "Europe/Paris",
    "E. Europe Standard Time": "Europe/Bucharest",
    "Russian Standard Time": "Europe/Moscow",
    "Turkey Standard Time": "Europe/Istanbul",
    "India Standard Time": "Asia/Kolkata",
    "China Standard Time": "Asia/Shanghai",
    "Tokyo Standard Time": "Asia/Tokyo",
    "Korea Standard Time": "Asia/Seoul",
    "Pacific Standard Time": "America/Los_Angeles",
    "Mountain Standard Time": "America/Denver",
    "Central Standard Time": "America/Chicago",
    "Eastern Standard Time": "America/New_York",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "New Zealand Standard Time": "Pacific/Auckland",
}


def refresh_calendar_context(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    recording = get_recording(recording_id, settings=cfg)
    if recording is None:
        raise KeyError(recording_id)

    capture_time = _parse_iso_datetime(recording.get("captured_at"))
    if capture_time is None:
        capture_time = datetime.now(tz=timezone.utc)

    window = timedelta(minutes=cfg.calendar_match_window_minutes)
    window_start = capture_time - window
    window_end = capture_time + window
    recording_start, recording_end = _recording_interval(recording, capture_time)
    recording_tokens = _tokenize(str(recording.get("source_filename") or ""))

    client = MicrosoftGraphClient(cfg)
    query = urlencode(
        {
            "startDateTime": _iso_z(window_start),
            "endDateTime": _iso_z(window_end),
            "$top": _GRAPH_CANDIDATE_LIMIT,
            "$orderby": "start/dateTime",
        }
    )
    response = client.graph_get(f"/me/calendarView?{query}")
    events = response.get("value")
    if not isinstance(events, list):
        events = []

    candidates: list[dict[str, Any]] = []
    window_seconds = max(int(window.total_seconds()), 1)
    for item in events:
        if not isinstance(item, dict):
            continue
        candidate = _build_candidate(
            item,
            recording_start=recording_start,
            recording_end=recording_end,
            window_seconds=window_seconds,
            recording_tokens=recording_tokens,
        )
        if candidate is None:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)

    existing = get_calendar_match(recording_id, settings=cfg)
    selected_event_id, selected_confidence = _resolve_selected_event(
        existing=existing,
        candidates=candidates,
        threshold=cfg.calendar_auto_match_threshold,
    )
    row = upsert_calendar_match(
        recording_id=recording_id,
        candidates=candidates,
        selected_event_id=selected_event_id,
        selected_confidence=selected_confidence,
        settings=cfg,
    )
    return _build_context(recording, row)


def load_calendar_context(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    recording = get_recording(recording_id, settings=cfg)
    if recording is None:
        raise KeyError(recording_id)
    row = get_calendar_match(recording_id, settings=cfg) or {
        "recording_id": recording_id,
        "selected_event_id": None,
        "selected_confidence": None,
        "candidates_json": [],
    }
    return _build_context(recording, row)


def select_calendar_event(
    recording_id: str,
    event_id: str | None,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    recording = get_recording(recording_id, settings=cfg)
    if recording is None:
        raise KeyError(recording_id)

    current = get_calendar_match(recording_id, settings=cfg) or {
        "recording_id": recording_id,
        "selected_event_id": None,
        "selected_confidence": None,
        "candidates_json": [],
    }
    candidates = _candidate_list(current)
    if event_id is None:
        selected_confidence = _MANUAL_NO_EVENT_CONFIDENCE
    else:
        selected = _candidate_by_id(candidates, event_id)
        if selected is None:
            raise ValueError("Unknown event_id for this recording")
        selected_confidence = float(selected.get("score") or 0.0)

    row = set_calendar_match_selection(
        recording_id=recording_id,
        event_id=event_id,
        selected_confidence=selected_confidence,
        settings=cfg,
    )
    return _build_context(recording, row)


def _build_candidate(
    event: dict[str, Any],
    *,
    recording_start: datetime,
    recording_end: datetime,
    window_seconds: int,
    recording_tokens: set[str],
) -> dict[str, Any] | None:
    event_id = str(event.get("id") or "").strip()
    if not event_id:
        return None

    subject = str(event.get("subject") or "").strip()
    start = _parse_event_datetime(event.get("start"))
    end = _parse_event_datetime(event.get("end")) or start
    if start is None:
        return None
    if end is None or end < start:
        end = start

    overlap_component = _overlap_component(
        recording_start=recording_start,
        recording_end=recording_end,
        event_start=start,
        event_end=end,
    )
    proximity_component = 0.0
    if overlap_component <= 0:
        proximity_component = _proximity_component(
            recording_start=recording_start,
            recording_end=recording_end,
            event_start=start,
            event_end=end,
            window_seconds=window_seconds,
        )
    keyword_component = _keyword_component(recording_tokens, _tokenize(subject))
    score = min(
        1.0,
        max(0.0, 0.7 * overlap_component + 0.2 * proximity_component + 0.1 * keyword_component),
    )

    title_tokens = sorted(_tokenize(subject))
    organizer = _extract_party(event.get("organizer"))
    attendees = _extract_attendees(event.get("attendees"))
    location = _extract_location(event.get("location"))
    rationale = (
        f"time_overlap={overlap_component:.2f}; "
        f"proximity={proximity_component:.2f}; "
        f"subject_match={keyword_component:.2f}"
    )
    return {
        "event_id": event_id,
        "subject": subject or "(no subject)",
        "start": _iso_z(start),
        "end": _iso_z(end),
        "organizer": organizer,
        "attendees": attendees,
        "location": location,
        "title_tokens": title_tokens,
        "score": round(score, 4),
        "rationale": rationale,
    }


def _build_context(
    recording: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    candidates = _candidate_list(row)
    selected_event_id = row.get("selected_event_id")
    selected_confidence = row.get("selected_confidence")
    selected = _candidate_by_id(candidates, selected_event_id)
    visible_candidates = _visible_candidates(candidates, selected)

    return {
        "recording_id": recording["id"],
        "captured_at": recording.get("captured_at"),
        "selected_event_id": selected_event_id,
        "selected_confidence": selected_confidence,
        "selected_event": selected,
        "signals": {
            "title_tokens": (selected or {}).get("title_tokens", []),
            "attendees": (selected or {}).get("attendees", []),
            "organizer": (selected or {}).get("organizer"),
        },
        "candidates": visible_candidates,
        "candidate_total": len(candidates),
        "manual_no_event": (
            selected_event_id is None and selected_confidence == _MANUAL_NO_EVENT_CONFIDENCE
        ),
    }


def _resolve_selected_event(
    *,
    existing: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    threshold: float,
) -> tuple[str | None, float | None]:
    if existing:
        selected_event_id = existing.get("selected_event_id")
        selected_confidence = existing.get("selected_confidence")
        if (
            selected_event_id is None
            and selected_confidence == _MANUAL_NO_EVENT_CONFIDENCE
        ):
            return None, _MANUAL_NO_EVENT_CONFIDENCE
        selected = _candidate_by_id(candidates, selected_event_id)
        if selected is not None:
            return str(selected["event_id"]), float(selected.get("score") or 0.0)

    if not candidates:
        return None, None
    best = candidates[0]
    best_score = float(best.get("score") or 0.0)
    if best_score >= threshold:
        return str(best["event_id"]), best_score
    return None, None


def _recording_interval(
    recording: dict[str, Any],
    capture_time: datetime,
) -> tuple[datetime, datetime]:
    duration_raw = recording.get("duration_sec")
    duration = int(duration_raw) if isinstance(duration_raw, int | float) else 0
    if duration <= 0:
        return capture_time, capture_time
    return capture_time, capture_time + timedelta(seconds=duration)


def _overlap_component(
    *,
    recording_start: datetime,
    recording_end: datetime,
    event_start: datetime,
    event_end: datetime,
) -> float:
    if recording_end <= recording_start:
        return 1.0 if event_start <= recording_start <= event_end else 0.0
    overlap_start = max(recording_start, event_start)
    overlap_end = min(recording_end, event_end)
    if overlap_end <= overlap_start:
        return 0.0
    overlap_seconds = (overlap_end - overlap_start).total_seconds()
    recording_seconds = (recording_end - recording_start).total_seconds()
    if recording_seconds <= 0:
        return 0.0
    return min(1.0, max(0.0, overlap_seconds / recording_seconds))


def _proximity_component(
    *,
    recording_start: datetime,
    recording_end: datetime,
    event_start: datetime,
    event_end: datetime,
    window_seconds: int,
) -> float:
    if window_seconds <= 0:
        return 0.0
    if recording_end <= recording_start:
        # Point-like recordings (missing/zero duration): score by nearest event edge.
        distance_to_start = abs((recording_start - event_start).total_seconds())
        distance_to_end = abs((recording_start - event_end).total_seconds())
        distance = min(distance_to_start, distance_to_end)
    else:
        if event_end <= recording_start:
            distance = (recording_start - event_end).total_seconds()
        elif event_start >= recording_end:
            distance = (event_start - recording_end).total_seconds()
        else:
            distance = 0.0
    normalized = 1.0 - min(max(distance, 0.0), float(window_seconds)) / float(window_seconds)
    return max(0.0, normalized)


def _keyword_component(
    recording_tokens: set[str],
    subject_tokens: set[str],
) -> float:
    if not recording_tokens or not subject_tokens:
        return 0.0
    overlap = len(recording_tokens.intersection(subject_tokens))
    return overlap / float(len(recording_tokens))


def _candidate_list(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    raw = row.get("candidates_json")
    if not isinstance(raw, list):
        return []
    out = [item for item in raw if isinstance(item, dict) and item.get("event_id")]
    out.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
    return out


def _candidate_by_id(
    candidates: list[dict[str, Any]],
    event_id: str | None,
) -> dict[str, Any] | None:
    if not event_id:
        return None
    for item in candidates:
        if str(item.get("event_id")) == event_id:
            return item
    return None


def _visible_candidates(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    visible = list(candidates[:_UI_CANDIDATE_LIMIT])
    if selected is None:
        return visible
    selected_id = str(selected.get("event_id") or "").strip()
    if not selected_id:
        return visible
    if any(str(item.get("event_id")) == selected_id for item in visible):
        return visible
    return [*visible, selected]


def _extract_party(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    email = value.get("emailAddress")
    if not isinstance(email, dict):
        return None
    name = str(email.get("name") or "").strip()
    address = str(email.get("address") or "").strip()
    return name or address or None


def _extract_attendees(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    attendees: list[str] = []
    for item in value:
        entry = _extract_party(item)
        if entry:
            attendees.append(entry)
    return attendees


def _extract_location(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    display_name = str(value.get("displayName") or "").strip()
    return display_name or None


def _parse_event_datetime(value: Any) -> datetime | None:
    if isinstance(value, dict):
        date_time = value.get("dateTime")
        time_zone = value.get("timeZone")
    else:
        date_time = value
        time_zone = None
    if not isinstance(date_time, str):
        return None
    tz_name = str(time_zone).strip() or None if isinstance(time_zone, str) else None
    return _parse_iso_datetime(date_time, default_timezone=tz_name)


def _parse_iso_datetime(
    value: Any,
    *,
    default_timezone: str | None = None,
) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = _normalize_iso_datetime_text(text)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        if default_timezone is None:
            tz = timezone.utc
        else:
            tz = _timezone_from_name(default_timezone)
            if tz is None:
                return None
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(timezone.utc)


def _normalize_iso_datetime_text(text: str) -> str:
    match = _ISO_FRACTION_RE.match(text)
    if match is None:
        return text
    fraction = match.group("fraction")
    digits = fraction[1:]
    if len(digits) <= 6:
        return text
    suffix = match.group("suffix") or ""
    return f"{match.group('body')}.{digits[:6]}{suffix}"


def _timezone_from_name(name: str | None) -> tzinfo | None:
    if name is None:
        return None
    cleaned = name.strip()
    if not cleaned:
        return None
    if cleaned.upper() == "UTC":
        return timezone.utc
    candidate = _WINDOWS_TZ_TO_IANA.get(cleaned, cleaned)
    try:
        return ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        return None


def _tokenize(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value)}


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


__all__ = [
    "load_calendar_context",
    "refresh_calendar_context",
    "select_calendar_event",
]
