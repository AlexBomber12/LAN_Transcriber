from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

from lan_app.config import AppSettings
from lan_app.db import (
    get_calendar_source,
    replace_calendar_events_for_window,
    update_calendar_source_sync_state,
)

from .ics import CalendarFetchError, CalendarParseError, fetch_ics_url, parse_ics_events, validate_ics_url


class CalendarSyncError(RuntimeError):
    """Raised when a source sync cannot complete."""


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def calendar_expansion_window(
    settings: AppSettings,
    *,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    anchor = (now or datetime.now(tz=timezone.utc)).astimezone(timezone.utc)
    past_days = max(0, int(settings.calendar_expand_past_days))
    future_days = max(1, int(settings.calendar_expand_future_days))
    start = (anchor - timedelta(days=past_days)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = (anchor + timedelta(days=future_days + 1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return start, end


def redacted_calendar_source(source: dict[str, Any]) -> dict[str, Any]:
    kind = str(source.get("kind") or "").strip().lower()
    url_raw = str(source.get("url") or "").strip()
    url_host = ""
    if url_raw:
        try:
            url_host = str(urlsplit(url_raw).hostname or "").strip()
        except ValueError:
            url_host = ""
    return {
        "id": source.get("id"),
        "name": source.get("name"),
        "kind": kind,
        "created_at": source.get("created_at"),
        "last_synced_at": source.get("last_synced_at"),
        "last_error": source.get("last_error"),
        "url_configured": bool(url_raw),
        "url_host": url_host or None,
        "file_configured": bool(str(source.get("file_ics") or "").strip()),
    }


def _source_payload_for_sync(source: dict[str, Any], settings: AppSettings) -> bytes:
    kind = str(source.get("kind") or "").strip().lower()
    if kind == "url":
        raw_url = str(source.get("url") or "").strip()
        if not raw_url:
            raise CalendarSyncError("Calendar source URL is not configured")
        valid_url = validate_ics_url(raw_url)
        try:
            return fetch_ics_url(
                valid_url,
                timeout_seconds=float(settings.calendar_fetch_timeout_seconds),
                max_bytes=int(settings.calendar_fetch_max_bytes),
                max_redirects=int(settings.calendar_fetch_max_redirects),
            )
        except CalendarFetchError as exc:
            raise CalendarSyncError(str(exc)) from exc
    if kind == "file":
        payload = str(source.get("file_ics") or "").strip()
        if not payload:
            raise CalendarSyncError("Calendar source file payload is empty")
        return payload.encode("utf-8")
    raise CalendarSyncError("Calendar source kind is unsupported")


def sync_calendar_source(
    source_id: int,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    source = get_calendar_source(source_id, settings=settings)
    if source is None:
        raise KeyError(source_id)

    now = datetime.now(tz=timezone.utc)
    window_start_dt, window_end_dt = calendar_expansion_window(settings, now=now)
    window_start = _utc_iso(window_start_dt)
    window_end = _utc_iso(window_end_dt)
    synced_at = _utc_iso(now)

    try:
        payload = _source_payload_for_sync(source, settings)
        try:
            events = parse_ics_events(
                payload,
                window_start=window_start_dt,
                window_end=window_end_dt,
                synced_at=now,
            )
        except CalendarParseError as exc:
            raise CalendarSyncError(str(exc)) from exc
        stored_count = replace_calendar_events_for_window(
            source_id=int(source_id),
            window_start=window_start,
            window_end=window_end,
            events=events,
            settings=settings,
        )
        update_calendar_source_sync_state(
            int(source_id),
            last_synced_at=synced_at,
            last_error=None,
            settings=settings,
        )
    except CalendarSyncError as exc:
        update_calendar_source_sync_state(
            int(source_id),
            last_error=str(exc)[:1000],
            settings=settings,
        )
        raise

    return {
        "source_id": int(source_id),
        "window_start": window_start,
        "window_end": window_end,
        "events_count": stored_count,
        "synced_at": synced_at,
    }


__all__ = [
    "CalendarSyncError",
    "calendar_expansion_window",
    "redacted_calendar_source",
    "sync_calendar_source",
]
