from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from icalendar import Calendar
import recurring_ical_events


class CalendarFetchError(RuntimeError):
    """Raised when calendar content cannot be fetched from a URL source."""


class CalendarParseError(ValueError):
    """Raised when ICS content cannot be parsed into event instances."""


_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _coerce_utc_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _is_loopback_hostname(hostname: str) -> bool:
    normalized = hostname.strip().lower().rstrip(".")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _resolves_only_loopback(hostname: str) -> bool:
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    seen: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _family, _socktype, _proto, _canonname, sockaddr in addr_info:
        raw_ip = str(sockaddr[0]).split("%", 1)[0]
        try:
            seen.append(ipaddress.ip_address(raw_ip))
        except ValueError:
            continue
    return any(item.is_loopback for item in seen)


def validate_ics_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        raise ValueError("ICS URL is required")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("ICS URL must use http or https")
    if parts.username or parts.password:
        raise ValueError("ICS URL must not include credentials")
    if not parts.hostname:
        raise ValueError("ICS URL must include a host")
    if _is_loopback_hostname(parts.hostname) or _resolves_only_loopback(parts.hostname):
        raise ValueError("ICS URL host must not resolve to localhost/loopback")
    return parts.geturl()


def fetch_ics_url(
    url: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_redirects: int,
) -> bytes:
    current_url = validate_ics_url(url)
    redirects = 0
    timeout = httpx.Timeout(timeout_seconds)
    headers = {"Accept": "text/calendar,text/plain;q=0.9,*/*;q=0.8"}
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        while True:
            try:
                with client.stream("GET", current_url, headers=headers) as response:
                    if response.status_code in _REDIRECT_STATUS_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise CalendarFetchError("Calendar redirect did not provide a location")
                        if redirects >= max_redirects:
                            raise CalendarFetchError("Calendar fetch exceeded redirect limit")
                        current_url = validate_ics_url(urljoin(current_url, location))
                        redirects += 1
                        continue
                    response.raise_for_status()
                    chunks: list[bytes] = []
                    total_bytes = 0
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise CalendarFetchError("Calendar feed exceeded maximum allowed size")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except httpx.TimeoutException as exc:
                raise CalendarFetchError("Calendar fetch timed out") from exc
            except httpx.HTTPStatusError as exc:
                status_code = int(exc.response.status_code)
                raise CalendarFetchError(f"Calendar fetch failed with HTTP {status_code}") from exc
            except httpx.HTTPError as exc:
                raise CalendarFetchError("Calendar fetch failed") from exc


def _property_text(component: Any, key: str) -> str | None:
    value = component.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _organizer_text(component: Any) -> str | None:
    organizer = component.get("ORGANIZER")
    if organizer is None:
        return None
    params = getattr(organizer, "params", {})
    cn = params.get("CN")
    if cn:
        value = str(cn).strip()
        if value:
            return value
    raw = str(organizer).strip()
    if raw.lower().startswith("mailto:"):
        raw = raw[7:]
    return raw or None


def _updated_at_text(component: Any, *, fallback: str) -> str:
    for key in ("LAST-MODIFIED", "DTSTAMP"):
        try:
            decoded = component.decoded(key)
        except KeyError:
            continue
        except Exception:
            continue
        if isinstance(decoded, (date, datetime)):
            return _utc_iso(_coerce_utc_datetime(decoded))
    return fallback


def _normalise_event(component: Any, *, fallback_updated_at: str) -> dict[str, Any] | None:
    uid = str(component.get("UID") or "").strip()
    if not uid:
        return None
    status = str(component.get("STATUS") or "").strip().upper()
    if status == "CANCELLED":
        return None

    try:
        dtstart_raw = component.decoded("DTSTART")
    except KeyError:
        return None
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise CalendarParseError("Unable to decode DTSTART") from exc

    all_day = isinstance(dtstart_raw, date) and not isinstance(dtstart_raw, datetime)
    starts_at = _coerce_utc_datetime(dtstart_raw)

    dtend_raw: date | datetime | None = None
    try:
        dtend_raw = component.decoded("DTEND")
    except KeyError:
        dtend_raw = None
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise CalendarParseError("Unable to decode DTEND") from exc

    if all_day:
        if dtend_raw is None:
            ends_at = starts_at + timedelta(days=1)
        else:
            ends_at = _coerce_utc_datetime(dtend_raw)
            if ends_at <= starts_at:
                ends_at = starts_at + timedelta(days=1)
    else:
        if dtend_raw is not None:
            ends_at = _coerce_utc_datetime(dtend_raw)
        else:
            duration_raw = None
            try:
                duration_raw = component.decoded("DURATION")
            except KeyError:
                duration_raw = None
            except Exception as exc:  # pragma: no cover - defensive parsing
                raise CalendarParseError("Unable to decode DURATION") from exc
            if isinstance(duration_raw, timedelta):
                ends_at = starts_at + duration_raw
            else:
                ends_at = starts_at + timedelta(hours=1)
            if ends_at <= starts_at:
                ends_at = starts_at + timedelta(minutes=1)

    return {
        "uid": uid,
        "starts_at": _utc_iso(starts_at),
        "ends_at": _utc_iso(ends_at),
        "all_day": all_day,
        "summary": _property_text(component, "SUMMARY"),
        "description": _property_text(component, "DESCRIPTION"),
        "location": _property_text(component, "LOCATION"),
        "organizer": _organizer_text(component),
        "updated_at": _updated_at_text(component, fallback=fallback_updated_at),
    }


def parse_ics_events(
    payload: str | bytes,
    *,
    window_start: datetime,
    window_end: datetime,
    synced_at: datetime | None = None,
) -> list[dict[str, Any]]:
    start_utc = _coerce_utc_datetime(window_start)
    end_utc = _coerce_utc_datetime(window_end)
    if end_utc <= start_utc:
        raise ValueError("window_end must be after window_start")

    try:
        calendar = Calendar.from_ical(payload)
    except Exception as exc:
        raise CalendarParseError("ICS payload could not be parsed") from exc

    try:
        expanded_events = recurring_ical_events.of(calendar).between(start_utc, end_utc)
    except Exception as exc:
        raise CalendarParseError("ICS recurrence expansion failed") from exc

    fallback_updated_at = _utc_iso(synced_at or datetime.now(tz=timezone.utc))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for component in expanded_events:
        if str(getattr(component, "name", "")).upper() != "VEVENT":
            continue
        row = _normalise_event(component, fallback_updated_at=fallback_updated_at)
        if row is None:
            continue
        key = (str(row["uid"]), str(row["starts_at"]))
        by_key[key] = row

    return sorted(
        by_key.values(),
        key=lambda row: (
            str(row.get("starts_at") or ""),
            str(row.get("uid") or ""),
        ),
    )


__all__ = [
    "CalendarFetchError",
    "CalendarParseError",
    "fetch_ics_url",
    "parse_ics_events",
    "validate_ics_url",
]
