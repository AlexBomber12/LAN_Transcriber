from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from lan_app.calendar import ics, service


class DummyOrganizer:
    def __init__(self, value: str, *, params: dict[str, str] | None = None) -> None:
        self.value = value
        self.params = params or {}

    def __str__(self) -> str:
        return self.value


class DummyComponent:
    def __init__(
        self,
        *,
        props: dict[str, object] | None = None,
        decoded_values: dict[str, object] | None = None,
        decoded_errors: dict[str, Exception] | None = None,
        name: str = "VEVENT",
    ) -> None:
        self._props = props or {}
        self._decoded_values = decoded_values or {}
        self._decoded_errors = decoded_errors or {}
        self.name = name

    def get(self, key: str) -> object | None:
        return self._props.get(key)

    def decoded(self, key: str) -> object:
        if key in self._decoded_errors:
            raise self._decoded_errors[key]
        if key not in self._decoded_values:
            raise KeyError(key)
        return self._decoded_values[key]


class DummyResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        raise_for_status_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []
        self._raise_for_status_error = raise_for_status_error

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self._raise_for_status_error is not None:
            raise self._raise_for_status_error

    def iter_bytes(self):
        return iter(self._chunks)


class DummyClient:
    def __init__(self, actions: list[object]) -> None:
        self._actions = list(actions)

    def __enter__(self) -> "DummyClient":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        return False

    def stream(self, _method: str, _url: str, headers: dict[str, str]):
        assert "Accept" in headers
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _patch_client(monkeypatch: pytest.MonkeyPatch, actions: list[object]) -> None:
    monkeypatch.setattr(ics.httpx, "Client", lambda *args, **kwargs: DummyClient(actions))


def _service_settings() -> SimpleNamespace:
    return SimpleNamespace(
        calendar_expand_past_days=2,
        calendar_expand_future_days=3,
        calendar_fetch_timeout_seconds=4.5,
        calendar_fetch_max_bytes=12345,
        calendar_fetch_max_redirects=4,
    )


def test_coerce_utc_datetime_for_naive_aware_and_date_values():
    berlin = ZoneInfo("Europe/Berlin")
    naive = datetime(2026, 2, 1, 9, 0, 0)
    aware = datetime(2026, 2, 1, 9, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    date_value = date(2026, 2, 1)

    assert ics._coerce_utc_datetime(naive) == datetime(2026, 2, 1, 9, 0, tzinfo=timezone.utc)
    assert ics._coerce_utc_datetime(naive, default_tz=berlin) == datetime(
        2026, 2, 1, 8, 0, tzinfo=timezone.utc
    )
    assert ics._coerce_utc_datetime(aware) == datetime(2026, 2, 1, 7, 0, tzinfo=timezone.utc)
    assert ics._coerce_utc_datetime(date_value) == datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)


def test_is_loopback_hostname_handles_localhost_ip_and_invalid():
    assert ics._is_loopback_hostname("localhost")
    assert ics._is_loopback_hostname("api.localhost")
    assert ics._is_loopback_hostname("127.0.0.1")
    assert not ics._is_loopback_hostname("example.com")
    assert not ics._is_loopback_hostname("not-an-ip")


def test_resolves_only_loopback_handles_dns_error(monkeypatch: pytest.MonkeyPatch):
    def _raise_gaierror(*_args, **_kwargs):
        raise ics.socket.gaierror()

    monkeypatch.setattr(ics.socket, "getaddrinfo", _raise_gaierror)
    assert not ics._resolves_only_loopback("calendar.example")


def test_resolves_only_loopback_parses_addresses(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ics.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (0, 0, 0, "", ("not-an-ip", 443)),
            (0, 0, 0, "", ("203.0.113.10", 443)),
            (0, 0, 0, "", ("127.0.0.1", 443)),
        ],
    )
    assert ics._resolves_only_loopback("calendar.example")

    monkeypatch.setattr(
        ics.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(0, 0, 0, "", ("203.0.113.10", 443))],
    )
    assert not ics._resolves_only_loopback("calendar.example")


def test_validate_ics_url_rejects_invalid_inputs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ics, "_resolves_only_loopback", lambda _hostname: False)

    with pytest.raises(ValueError, match="required"):
        ics.validate_ics_url("")
    with pytest.raises(ValueError, match="http or https"):
        ics.validate_ics_url("ftp://example.com/calendar.ics")
    with pytest.raises(ValueError, match="credentials"):
        ics.validate_ics_url("https://user:pass@example.com/calendar.ics")
    with pytest.raises(ValueError, match="include a host"):
        ics.validate_ics_url("https:///calendar.ics")
    with pytest.raises(ValueError, match="localhost/loopback"):
        ics.validate_ics_url("https://localhost/calendar.ics")


def test_validate_ics_url_rejects_resolved_loopback_and_accepts_safe_host(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ics, "_is_loopback_hostname", lambda _hostname: False)
    monkeypatch.setattr(ics, "_resolves_only_loopback", lambda _hostname: True)
    with pytest.raises(ValueError, match="localhost/loopback"):
        ics.validate_ics_url("https://calendar.example.com/team.ics")

    monkeypatch.setattr(ics, "_resolves_only_loopback", lambda _hostname: False)
    assert (
        ics.validate_ics_url(" https://calendar.example.com/team.ics ")
        == "https://calendar.example.com/team.ics"
    )


def test_fetch_ics_url_returns_payload_and_handles_redirect(monkeypatch: pytest.MonkeyPatch):
    validated_urls: list[str] = []

    def _validate(url: str) -> str:
        validated_urls.append(url)
        return url

    monkeypatch.setattr(ics, "validate_ics_url", _validate)
    _patch_client(
        monkeypatch,
        [
            DummyResponse(status_code=302, headers={"location": "/next.ics"}),
            DummyResponse(status_code=200, chunks=[b"aa", b"", b"bb"]),
        ],
    )

    payload = ics.fetch_ics_url(
        "https://calendar.example.com/root.ics",
        timeout_seconds=1,
        max_bytes=10,
        max_redirects=3,
    )
    assert payload == b"aabb"
    assert validated_urls == [
        "https://calendar.example.com/root.ics",
        "https://calendar.example.com/next.ics",
    ]


def test_fetch_ics_url_redirect_without_location_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ics, "validate_ics_url", lambda url: url)
    _patch_client(monkeypatch, [DummyResponse(status_code=302, headers={})])

    with pytest.raises(ics.CalendarFetchError, match="did not provide a location"):
        ics.fetch_ics_url(
            "https://calendar.example.com/root.ics",
            timeout_seconds=1,
            max_bytes=10,
            max_redirects=2,
        )


def test_fetch_ics_url_redirect_limit_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ics, "validate_ics_url", lambda url: url)
    _patch_client(monkeypatch, [DummyResponse(status_code=302, headers={"location": "/next.ics"})])

    with pytest.raises(ics.CalendarFetchError, match="redirect limit"):
        ics.fetch_ics_url(
            "https://calendar.example.com/root.ics",
            timeout_seconds=1,
            max_bytes=10,
            max_redirects=0,
        )


def test_fetch_ics_url_timeout_http_status_and_http_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ics, "validate_ics_url", lambda url: url)
    _patch_client(monkeypatch, [httpx.TimeoutException("slow")])
    with pytest.raises(ics.CalendarFetchError, match="timed out"):
        ics.fetch_ics_url("https://calendar.example.com/root.ics", timeout_seconds=1, max_bytes=10, max_redirects=1)

    request = httpx.Request("GET", "https://calendar.example.com/root.ics")
    status_error = httpx.HTTPStatusError(
        "bad status",
        request=request,
        response=httpx.Response(status_code=503, request=request),
    )
    _patch_client(
        monkeypatch,
        [DummyResponse(status_code=503, raise_for_status_error=status_error)],
    )
    with pytest.raises(ics.CalendarFetchError, match="HTTP 503"):
        ics.fetch_ics_url("https://calendar.example.com/root.ics", timeout_seconds=1, max_bytes=10, max_redirects=1)

    _patch_client(monkeypatch, [httpx.ConnectError("network down", request=request)])
    with pytest.raises(ics.CalendarFetchError, match="fetch failed"):
        ics.fetch_ics_url("https://calendar.example.com/root.ics", timeout_seconds=1, max_bytes=10, max_redirects=1)


def test_fetch_ics_url_enforces_max_bytes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ics, "validate_ics_url", lambda url: url)
    _patch_client(monkeypatch, [DummyResponse(status_code=200, chunks=[b"abcd", b"ef"])])

    with pytest.raises(ics.CalendarFetchError, match="maximum allowed size"):
        ics.fetch_ics_url(
            "https://calendar.example.com/root.ics",
            timeout_seconds=1,
            max_bytes=5,
            max_redirects=1,
        )


def test_property_and_organizer_helpers():
    organizer_with_cn = DummyOrganizer("mailto:owner@example.com", params={"CN": " Owner Name "})
    organizer_mailto_only = DummyOrganizer("mailto:owner@example.com", params={"CN": "   "})
    organizer_plain = DummyOrganizer("owner@example.com", params={})
    component = DummyComponent(
        props={
            "SUMMARY": "  Team Sync  ",
            "DESCRIPTION": "   ",
            "ORGANIZER_A": organizer_with_cn,
            "ORGANIZER_B": organizer_mailto_only,
            "ORGANIZER_C": organizer_plain,
        }
    )

    assert ics._property_text(component, "SUMMARY") == "Team Sync"
    assert ics._property_text(component, "DESCRIPTION") is None
    assert ics._property_text(component, "MISSING") is None

    component._props["ORGANIZER"] = organizer_with_cn
    assert ics._organizer_text(component) == "Owner Name"
    component._props["ORGANIZER"] = organizer_mailto_only
    assert ics._organizer_text(component) == "owner@example.com"
    component._props["ORGANIZER"] = organizer_plain
    assert ics._organizer_text(component) == "owner@example.com"
    component._props["ORGANIZER"] = None
    assert ics._organizer_text(component) is None


def test_updated_at_text_uses_modified_stamp_or_fallback():
    fallback = "2026-01-01T00:00:00Z"
    with_last_modified = DummyComponent(
        decoded_values={"LAST-MODIFIED": datetime(2026, 2, 2, 10, tzinfo=timezone.utc)}
    )
    assert ics._updated_at_text(with_last_modified, fallback=fallback) == "2026-02-02T10:00:00Z"

    with_dtstamp = DummyComponent(
        decoded_values={"DTSTAMP": date(2026, 2, 3)},
        decoded_errors={"LAST-MODIFIED": RuntimeError("bad last-modified")},
    )
    assert ics._updated_at_text(with_dtstamp, fallback=fallback) == "2026-02-03T00:00:00Z"

    with_non_datetime_modified = DummyComponent(
        decoded_values={"LAST-MODIFIED": "not-a-datetime", "DTSTAMP": date(2026, 2, 4)}
    )
    assert ics._updated_at_text(with_non_datetime_modified, fallback=fallback) == "2026-02-04T00:00:00Z"

    without_any_timestamp = DummyComponent(decoded_values={})
    assert ics._updated_at_text(without_any_timestamp, fallback=fallback) == fallback


def test_normalise_event_rejects_missing_or_cancelled_and_missing_start():
    fallback = "2026-01-01T00:00:00Z"
    missing_uid = DummyComponent(decoded_values={"DTSTART": datetime(2026, 2, 1, tzinfo=timezone.utc)})
    cancelled = DummyComponent(
        props={"UID": "event-1", "STATUS": "cancelled"},
        decoded_values={"DTSTART": datetime(2026, 2, 1, tzinfo=timezone.utc)},
    )
    missing_start = DummyComponent(props={"UID": "event-2"}, decoded_values={})

    assert ics._normalise_event(missing_uid, fallback_updated_at=fallback) is None
    assert ics._normalise_event(cancelled, fallback_updated_at=fallback) is None
    assert ics._normalise_event(missing_start, fallback_updated_at=fallback) is None


def test_normalise_event_all_day_paths():
    fallback = "2026-01-01T00:00:00Z"
    all_day_no_end = DummyComponent(
        props={"UID": "all-day-1", "SUMMARY": "All day no end"},
        decoded_values={"DTSTART": date(2026, 2, 10)},
    )
    row = ics._normalise_event(all_day_no_end, fallback_updated_at=fallback)
    assert row is not None
    assert row["all_day"] is True
    assert row["starts_at"] == "2026-02-10T00:00:00Z"
    assert row["ends_at"] == "2026-02-11T00:00:00Z"

    all_day_end_before_start = DummyComponent(
        props={"UID": "all-day-2"},
        decoded_values={"DTSTART": date(2026, 2, 10), "DTEND": date(2026, 2, 9)},
    )
    row = ics._normalise_event(all_day_end_before_start, fallback_updated_at=fallback)
    assert row is not None
    assert row["ends_at"] == "2026-02-11T00:00:00Z"

    all_day_end_after_start = DummyComponent(
        props={"UID": "all-day-3"},
        decoded_values={"DTSTART": date(2026, 2, 10), "DTEND": date(2026, 2, 12)},
    )
    row = ics._normalise_event(all_day_end_after_start, fallback_updated_at=fallback)
    assert row is not None
    assert row["ends_at"] == "2026-02-12T00:00:00Z"


def test_normalise_event_timed_paths_with_dtend_and_duration():
    fallback = "2026-01-01T00:00:00Z"
    start = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)
    timed_with_dtend = DummyComponent(
        props={"UID": "timed-dtend"},
        decoded_values={"DTSTART": start, "DTEND": start + timedelta(minutes=30)},
    )
    row = ics._normalise_event(timed_with_dtend, fallback_updated_at=fallback)
    assert row is not None
    assert row["ends_at"] == "2026-02-10T12:30:00Z"

    timed_with_duration = DummyComponent(
        props={"UID": "timed-duration"},
        decoded_values={"DTSTART": start, "DURATION": timedelta(minutes=45)},
    )
    row = ics._normalise_event(timed_with_duration, fallback_updated_at=fallback)
    assert row is not None
    assert row["ends_at"] == "2026-02-10T12:45:00Z"

    timed_negative_duration = DummyComponent(
        props={"UID": "timed-negative"},
        decoded_values={"DTSTART": start, "DURATION": timedelta(minutes=-30)},
    )
    row = ics._normalise_event(timed_negative_duration, fallback_updated_at=fallback)
    assert row is not None
    assert row["ends_at"] == "2026-02-10T12:01:00Z"

    timed_default_duration = DummyComponent(
        props={"UID": "timed-default"},
        decoded_values={"DTSTART": start},
    )
    row = ics._normalise_event(timed_default_duration, fallback_updated_at=fallback)
    assert row is not None
    assert row["ends_at"] == "2026-02-10T13:00:00Z"


def test_parse_ics_events_rejects_bad_window():
    with pytest.raises(ValueError, match="window_end must be after window_start"):
        ics.parse_ics_events(
            "BEGIN:VCALENDAR\nEND:VCALENDAR",
            window_start=datetime(2026, 2, 10, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 10, tzinfo=timezone.utc),
        )


def test_parse_ics_events_wraps_calendar_and_expansion_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ics.Calendar, "from_ical", lambda _payload: (_ for _ in ()).throw(ValueError("bad ics")))
    with pytest.raises(ics.CalendarParseError, match="could not be parsed"):
        ics.parse_ics_events(
            "broken",
            window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 2, tzinfo=timezone.utc),
        )

    calendar = SimpleNamespace(get=lambda _key: "Etc/UTC")
    monkeypatch.setattr(ics.Calendar, "from_ical", lambda _payload: calendar)

    class _BrokenExpander:
        def between(self, _start, _end):
            raise RuntimeError("expand failed")

    monkeypatch.setattr(ics.recurring_ical_events, "of", lambda _calendar: _BrokenExpander())
    with pytest.raises(ics.CalendarParseError, match="recurrence expansion failed"):
        ics.parse_ics_events(
            "ok",
            window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 2, tzinfo=timezone.utc),
        )


def test_parse_ics_events_filters_deduplicates_and_sorts(monkeypatch: pytest.MonkeyPatch):
    class _Calendar:
        @staticmethod
        def get(key: str):
            if key == "X-WR-TIMEZONE":
                return "Bad/Timezone"
            return None

    class _Expander:
        def between(self, _start, _end):
            return [
                SimpleNamespace(name="VTODO", token="skip-non-event"),
                SimpleNamespace(name="VEVENT", token="drop-none"),
                SimpleNamespace(name="VEVENT", token="late"),
                SimpleNamespace(name="VEVENT", token="early-old"),
                SimpleNamespace(name="VEVENT", token="early-new"),
            ]

    monkeypatch.setattr(ics.Calendar, "from_ical", lambda _payload: _Calendar())
    monkeypatch.setattr(ics.recurring_ical_events, "of", lambda _calendar: _Expander())

    captured_tzs: list[object] = []

    def _normalise(component, *, fallback_updated_at: str, default_tz):  # noqa: ANN001
        assert fallback_updated_at == "2026-02-01T12:00:00Z"
        captured_tzs.append(default_tz)
        token = getattr(component, "token")
        if token == "drop-none":
            return None
        if token == "late":
            return {"uid": "2", "starts_at": "2026-02-03T00:00:00Z", "summary": "late"}
        if token == "early-old":
            return {"uid": "1", "starts_at": "2026-02-02T00:00:00Z", "summary": "old"}
        if token == "early-new":
            return {"uid": "1", "starts_at": "2026-02-02T00:00:00Z", "summary": "new"}
        raise AssertionError(f"unexpected token: {token}")

    monkeypatch.setattr(ics, "_normalise_event", _normalise)

    rows = ics.parse_ics_events(
        "payload",
        window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 2, 10, tzinfo=timezone.utc),
        synced_at=datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert rows == [
        {"uid": "1", "starts_at": "2026-02-02T00:00:00Z", "summary": "new"},
        {"uid": "2", "starts_at": "2026-02-03T00:00:00Z", "summary": "late"},
    ]
    assert captured_tzs == [None, None, None, None]


def test_parse_ics_events_without_timezone_header(monkeypatch: pytest.MonkeyPatch):
    calendar = SimpleNamespace(get=lambda _key: None)
    monkeypatch.setattr(ics.Calendar, "from_ical", lambda _payload: calendar)

    class _EmptyExpander:
        def between(self, _start, _end):
            return []

    monkeypatch.setattr(ics.recurring_ical_events, "of", lambda _calendar: _EmptyExpander())
    assert (
        ics.parse_ics_events(
            "payload",
            window_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 2, 10, tzinfo=timezone.utc),
        )
        == []
    )


def test_calendar_expansion_window_clamps_values():
    settings = SimpleNamespace(calendar_expand_past_days=-5, calendar_expand_future_days=0)
    start, end = service.calendar_expansion_window(
        settings,
        now=datetime(2026, 2, 10, 15, 45, tzinfo=timezone.utc),
    )
    assert start == datetime(2026, 2, 10, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 2, 12, 0, 0, tzinfo=timezone.utc)


def test_redacted_calendar_source_variants():
    source_with_url = {
        "id": 1,
        "name": "Team",
        "kind": "URL",
        "url": "https://calendar.example.com/team.ics",
        "file_ics": "",
        "created_at": "created",
        "last_synced_at": "synced",
        "last_error": "err",
    }
    redacted = service.redacted_calendar_source(source_with_url)
    assert redacted["kind"] == "url"
    assert redacted["url_configured"] is True
    assert redacted["url_host"] == "calendar.example.com"
    assert redacted["file_configured"] is False

    invalid_url_source = {"kind": "url", "url": "https://[not-valid", "file_ics": "ics"}
    redacted = service.redacted_calendar_source(invalid_url_source)
    assert redacted["url_host"] is None
    assert redacted["file_configured"] is True

    empty_url_source = {"kind": "", "url": "", "file_ics": ""}
    redacted = service.redacted_calendar_source(empty_url_source)
    assert redacted["url_configured"] is False
    assert redacted["url_host"] is None


def test_source_payload_for_sync_url_file_and_errors(monkeypatch: pytest.MonkeyPatch):
    settings = _service_settings()

    with pytest.raises(service.CalendarSyncError, match="not configured"):
        service._source_payload_for_sync({"kind": "url", "url": ""}, settings)

    monkeypatch.setattr(service, "validate_ics_url", lambda raw: f"validated:{raw}")
    fetch_calls: list[tuple[str, float, int, int]] = []

    def _fetch(url: str, *, timeout_seconds: float, max_bytes: int, max_redirects: int) -> bytes:
        fetch_calls.append((url, timeout_seconds, max_bytes, max_redirects))
        return b"calendar"

    monkeypatch.setattr(service, "fetch_ics_url", _fetch)
    payload = service._source_payload_for_sync(
        {"kind": "url", "url": "https://calendar.example.com/a.ics"},
        settings,
    )
    assert payload == b"calendar"
    assert fetch_calls == [("validated:https://calendar.example.com/a.ics", 4.5, 12345, 4)]

    monkeypatch.setattr(
        service,
        "fetch_ics_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ics.CalendarFetchError("fetch-failed")),
    )
    with pytest.raises(service.CalendarSyncError, match="fetch-failed"):
        service._source_payload_for_sync({"kind": "url", "url": "https://calendar.example.com/a.ics"}, settings)

    with pytest.raises(service.CalendarSyncError, match="file payload is empty"):
        service._source_payload_for_sync({"kind": "file", "file_ics": "   "}, settings)
    assert service._source_payload_for_sync({"kind": "file", "file_ics": "BEGIN:VCALENDAR"}, settings) == b"BEGIN:VCALENDAR"

    with pytest.raises(service.CalendarSyncError, match="unsupported"):
        service._source_payload_for_sync({"kind": "other"}, settings)


def test_sync_calendar_source_raises_key_error_when_source_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service, "get_calendar_source", lambda *_args, **_kwargs: None)
    with pytest.raises(KeyError):
        service.sync_calendar_source(9, settings=_service_settings())


def test_sync_calendar_source_success(monkeypatch: pytest.MonkeyPatch):
    fixed_now = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(service, "datetime", _FixedDateTime)
    monkeypatch.setattr(service, "get_calendar_source", lambda *_args, **_kwargs: {"id": 5, "kind": "file", "file_ics": "x"})
    monkeypatch.setattr(
        service,
        "calendar_expansion_window",
        lambda _settings, *, now: (datetime(2026, 2, 1, tzinfo=timezone.utc), datetime(2026, 2, 20, tzinfo=timezone.utc)),
    )
    monkeypatch.setattr(service, "_source_payload_for_sync", lambda *_args, **_kwargs: b"payload")
    monkeypatch.setattr(service, "parse_ics_events", lambda *_args, **_kwargs: [{"uid": "1"}])
    monkeypatch.setattr(service, "replace_calendar_events_for_window", lambda **_kwargs: 7)

    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        service,
        "update_calendar_source_sync_state",
        lambda source_id, **kwargs: updates.append({"source_id": source_id, **kwargs}),
    )

    result = service.sync_calendar_source(5, settings=_service_settings())
    assert result == {
        "source_id": 5,
        "window_start": "2026-02-01T00:00:00Z",
        "window_end": "2026-02-20T00:00:00Z",
        "events_count": 7,
        "synced_at": "2026-02-10T12:00:00Z",
    }
    assert updates == [
        {
            "source_id": 5,
            "last_synced_at": "2026-02-10T12:00:00Z",
            "last_error": None,
            "settings": _service_settings(),
        }
    ]


def test_sync_calendar_source_updates_error_state_for_parse_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service, "get_calendar_source", lambda *_args, **_kwargs: {"id": 7, "kind": "file"})
    monkeypatch.setattr(
        service,
        "calendar_expansion_window",
        lambda _settings, *, now: (datetime(2026, 2, 1, tzinfo=timezone.utc), datetime(2026, 2, 20, tzinfo=timezone.utc)),
    )
    monkeypatch.setattr(service, "_source_payload_for_sync", lambda *_args, **_kwargs: b"payload")
    monkeypatch.setattr(
        service,
        "parse_ics_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ics.CalendarParseError("parse-failed")),
    )

    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        service,
        "update_calendar_source_sync_state",
        lambda source_id, **kwargs: updates.append({"source_id": source_id, **kwargs}),
    )

    with pytest.raises(service.CalendarSyncError, match="parse-failed"):
        service.sync_calendar_source(7, settings=_service_settings())
    assert len(updates) == 1
    assert updates[0]["source_id"] == 7
    assert updates[0]["last_error"] == "parse-failed"


def test_sync_calendar_source_updates_error_state_for_payload_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service, "get_calendar_source", lambda *_args, **_kwargs: {"id": 8, "kind": "file"})
    monkeypatch.setattr(
        service,
        "calendar_expansion_window",
        lambda _settings, *, now: (datetime(2026, 2, 1, tzinfo=timezone.utc), datetime(2026, 2, 20, tzinfo=timezone.utc)),
    )
    long_error = "x" * 1205
    monkeypatch.setattr(
        service,
        "_source_payload_for_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(service.CalendarSyncError(long_error)),
    )

    updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        service,
        "update_calendar_source_sync_state",
        lambda source_id, **kwargs: updates.append({"source_id": source_id, **kwargs}),
    )

    with pytest.raises(service.CalendarSyncError, match="x{10,}"):
        service.sync_calendar_source(8, settings=_service_settings())
    assert len(updates) == 1
    assert updates[0]["source_id"] == 8
    assert len(str(updates[0]["last_error"])) == 1000
