from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from lan_app import api
from lan_app.calendar.service import CalendarSyncError
from lan_app.config import AppSettings
from lan_app.jobs import RecordingNotFoundError
from lan_app.ops import RecordingDeleteError


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    cfg.reaper_interval_seconds = 1
    return cfg


def _request(method: str, path: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
    )


class _StopLoop(RuntimeError):
    pass


def test_validate_recording_status_rejects_unknown() -> None:
    assert api._validate_recording_status(None) is None
    assert api._validate_recording_status(api.RECORDING_STATUSES[0]) in api.RECORDING_STATUSES
    with pytest.raises(HTTPException, match="Unsupported recording status"):
        api._validate_recording_status("not-a-status")


def test_validate_job_status_rejects_unknown() -> None:
    assert api._validate_job_status(None) is None
    assert api._validate_job_status(api.JOB_STATUSES[0]) in api.JOB_STATUSES
    with pytest.raises(HTTPException, match="Unsupported job status"):
        api._validate_job_status("not-a-status")


def test_is_public_auth_exempt_cases() -> None:
    assert api._is_public_auth_exempt(_request("POST", "/healthz")) is False
    assert api._is_public_auth_exempt(_request("GET", "/healthz")) is True
    assert api._is_public_auth_exempt(_request("GET", "/healthz/db")) is True
    assert api._is_public_auth_exempt(_request("GET", "/metrics")) is True
    assert api._is_public_auth_exempt(_request("GET", "/openapi.json")) is True
    assert api._is_public_auth_exempt(_request("GET", "/api/recordings")) is False


def test_parse_iso_datetime_validation() -> None:
    with pytest.raises(ValueError, match="from is required"):
        api._parse_iso_datetime("  ", field_name="from")
    with pytest.raises(ValueError, match="from must be ISO-8601 datetime"):
        api._parse_iso_datetime("not-iso", field_name="from")
    parsed = api._parse_iso_datetime("2026-02-01T12:34:56", field_name="from")
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


@pytest.mark.asyncio
async def test_auth_middleware_allows_public_get_when_auth_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(api, "auth_enabled", lambda _settings: True)
    monkeypatch.setattr(api, "request_requires_auth", lambda _request: True)
    monkeypatch.setattr(api, "request_is_authenticated", lambda _request, _settings: False)

    seen: list[str] = []

    async def _call_next(_request: Request) -> Response:
        seen.append("called")
        return Response(status_code=204)

    response = await api._enforce_optional_bearer_auth(
        _request("GET", "/healthz/app"),
        _call_next,
    )
    assert response.status_code == 204
    assert seen == ["called"]


def test_health_check_by_component_dispatch_and_keyerror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "check_db_health", lambda _settings: {"ok": True, "detail": "db"})
    monkeypatch.setattr(api, "check_redis_health", lambda _settings: {"ok": True, "detail": "redis"})
    monkeypatch.setattr(api, "check_worker_health", lambda _settings: {"ok": True, "detail": "worker"})

    assert api._health_check_by_component("db", cfg)["detail"] == "db"
    assert api._health_check_by_component("redis", cfg)["detail"] == "redis"
    assert api._health_check_by_component("worker", cfg)["detail"] == "worker"
    with pytest.raises(KeyError):
        api._health_check_by_component("unknown", cfg)


@pytest.mark.asyncio
async def test_healthz_component_maps_unknown_component_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_keyerror(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyError("unknown")

    monkeypatch.setattr(api, "run_in_threadpool", _raise_keyerror)
    with pytest.raises(HTTPException) as exc_info:
        await api.healthz_component("unknown")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_retention_cleanup_loop_logs_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_runtime(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("cleanup failed")

    async def _stop(_seconds: float) -> None:
        raise _StopLoop()

    logged: list[str] = []
    monkeypatch.setattr(api, "run_in_threadpool", _raise_runtime)
    monkeypatch.setattr(api.asyncio, "sleep", _stop)
    monkeypatch.setattr(api._logger, "exception", lambda message: logged.append(str(message)))

    with pytest.raises(_StopLoop):
        await api._retention_cleanup_loop()
    assert logged == ["Retention cleanup job failed"]


@pytest.mark.asyncio
async def test_stuck_job_reaper_loop_warns_on_recovered_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)

    async def _summary(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {"recovered_jobs": 2, "recovered_recordings": 5}

    async def _stop(_seconds: float) -> None:
        raise _StopLoop()

    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(api, "run_in_threadpool", _summary)
    monkeypatch.setattr(api.asyncio, "sleep", _stop)
    monkeypatch.setattr(api._logger, "warning", lambda *args: warnings.append(args))

    with pytest.raises(_StopLoop):
        await api._stuck_job_reaper_loop()
    assert warnings


@pytest.mark.asyncio
async def test_stuck_job_reaper_loop_skips_warning_when_no_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)

    async def _summary(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {"recovered_jobs": 0, "recovered_recordings": 0}

    async def _stop(_seconds: float) -> None:
        raise _StopLoop()

    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(api, "run_in_threadpool", _summary)
    monkeypatch.setattr(api.asyncio, "sleep", _stop)
    monkeypatch.setattr(api._logger, "warning", lambda *args: warnings.append(args))

    with pytest.raises(_StopLoop):
        await api._stuck_job_reaper_loop()
    assert warnings == []


@pytest.mark.asyncio
async def test_stuck_job_reaper_loop_logs_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)

    async def _raise_runtime(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("reaper failed")

    async def _stop(_seconds: float) -> None:
        raise _StopLoop()

    logged: list[str] = []
    monkeypatch.setattr(api, "run_in_threadpool", _raise_runtime)
    monkeypatch.setattr(api.asyncio, "sleep", _stop)
    monkeypatch.setattr(api._logger, "exception", lambda message: logged.append(str(message)))

    with pytest.raises(_StopLoop):
        await api._stuck_job_reaper_loop()
    assert logged == ["Stuck-job reaper failed"]


@pytest.mark.asyncio
async def test_lifespan_initializes_and_cancels_background_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    init_calls: list[AppSettings] = []
    cancelled: set[str] = set()

    async def _wait_forever(name: str) -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.add(name)
            raise

    monkeypatch.setattr(api, "init_db", lambda settings: init_calls.append(settings))
    monkeypatch.setattr(api, "write_metrics_snapshot", lambda _path: _wait_forever("metrics"))
    monkeypatch.setattr(api, "_retention_cleanup_loop", lambda: _wait_forever("cleanup"))
    monkeypatch.setattr(api, "_stuck_job_reaper_loop", lambda: _wait_forever("reaper"))

    async with api._lifespan(api.app):
        await asyncio.sleep(0)

    assert init_calls == [cfg]
    assert cancelled == {"metrics", "cleanup", "reaper"}
    assert cfg.metrics_snapshot_path.parent.exists()


@pytest.mark.asyncio
async def test_update_alias_refreshes_result_and_notifies_subscribers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias_path = tmp_path / "aliases.yaml"
    subscriber: asyncio.Queue[str] = asyncio.Queue()
    current_result = object()
    refresh_calls: list[tuple[object, Path]] = []

    monkeypatch.setattr(api.aliases, "ALIAS_PATH", alias_path)
    monkeypatch.setattr(api, "_subscribers", [subscriber])
    monkeypatch.setattr(
        api,
        "refresh_aliases",
        lambda result, path: refresh_calls.append((result, path)),
    )

    api.set_current_result(current_result)
    payload = await api.update_alias("S-1", api.AliasUpdate(alias="Alice"))
    api.set_current_result(None)

    assert payload == {"speaker": "S-1", "alias": "Alice"}
    assert refresh_calls == [(current_result, alias_path)]
    assert subscriber.get_nowait() == "updated"
    assert api.aliases.load_aliases(alias_path) == {"S-1": "Alice"}


def test_metrics_endpoint_returns_prometheus_payload() -> None:
    client = TestClient(api.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(api.CONTENT_TYPE_LATEST)


@pytest.mark.asyncio
async def test_events_stream_yields_and_unsubscribes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "_subscribers", [])
    response = await api.events()
    assert len(api._subscribers) == 1
    api._subscribers[0].put_nowait("updated")

    chunk = await asyncio.wait_for(anext(response.body_iterator), timeout=1)
    text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
    assert "speaker_alias_updated" in text

    await response.body_iterator.aclose()
    assert api._subscribers == []


def test_api_requeue_missing_recording_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: None)
    client = TestClient(api.app)

    response = client.post("/api/recordings/missing/actions/requeue", json={"job_type": "precheck"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Recording not found"


@pytest.mark.parametrize(
    ("raised", "status_code", "detail_text"),
    [
        (RecordingNotFoundError("not found"), 404, "Recording not found"),
        (ValueError("bad requeue"), 422, "bad requeue"),
        (RuntimeError("rq down"), 503, "Queue unavailable: rq down"),
    ],
)
def test_api_requeue_maps_enqueue_errors(
    raised: Exception,
    status_code: int,
    detail_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: {"id": "rec-1"})

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise raised

    monkeypatch.setattr(api, "enqueue_recording_job", _raise)
    client = TestClient(api.app)

    response = client.post("/api/recordings/rec-1/actions/requeue", json={"job_type": "precheck"})
    assert response.status_code == status_code
    assert response.json()["detail"] == detail_text


def test_api_force_reprocess_missing_recording_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: None)
    client = TestClient(api.app)

    response = client.post("/api/recordings/missing/actions/force-reprocess")
    assert response.status_code == 404
    assert response.json()["detail"] == "Recording not found"


def test_api_force_reprocess_returns_409_when_job_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lan_app.jobs import DuplicateRecordingJobError

    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: {"id": "rec-1"})

    def _raise_dupe(*_args: Any, **kwargs: Any) -> Any:
        # enqueue_recording_job atomically detects existing active jobs and
        # raises before any destructive work happens.
        assert kwargs.get("force_reprocess") is True
        raise DuplicateRecordingJobError(
            recording_id="rec-1",
            job_id="existing-job-xyz",
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _raise_dupe)
    client = TestClient(api.app)

    response = client.post("/api/recordings/rec-1/actions/force-reprocess")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["existing_job_id"] == "existing-job-xyz"
    assert "already queued or started" in detail["message"].lower()


@pytest.mark.parametrize(
    ("raised", "status_code", "detail_text"),
    [
        (RecordingNotFoundError("not found"), 404, "Recording not found"),
        (ValueError("bad requeue"), 422, "bad requeue"),
        (RuntimeError("rq down"), 503, "Queue unavailable: rq down"),
    ],
)
def test_api_force_reprocess_maps_enqueue_errors(
    raised: Exception,
    status_code: int,
    detail_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: {"id": "rec-1"})

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise raised

    monkeypatch.setattr(api, "enqueue_recording_job", _raise)
    client = TestClient(api.app)

    response = client.post("/api/recordings/rec-1/actions/force-reprocess")
    assert response.status_code == status_code
    assert response.json()["detail"] == detail_text


def test_api_quarantine_missing_recording_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: None)
    client = TestClient(api.app)

    response = client.post("/api/recordings/missing/actions/quarantine", json={"reason": "manual"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Recording not found"


def test_api_quarantine_returns_404_when_recording_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _get_recording(*_args: Any, **_kwargs: Any) -> dict[str, str] | None:
        calls["count"] += 1
        if calls["count"] == 1:
            return {"id": "rec-1"}
        return None

    monkeypatch.setattr(api, "get_recording", _get_recording)
    monkeypatch.setattr(api, "set_recording_status", lambda *_args, **_kwargs: None)
    client = TestClient(api.app)

    response = client.post("/api/recordings/rec-1/actions/quarantine", json={"reason": "manual"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Recording not found"


def test_api_delete_missing_recording_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: None)
    client = TestClient(api.app)

    response = client.post("/api/recordings/missing/actions/delete")
    assert response.status_code == 404
    assert response.json()["detail"] == "Recording not found"


def test_api_delete_returns_503_when_queue_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: {"id": "rec-1"})

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("redis down")

    monkeypatch.setattr(api, "purge_pending_recording_jobs", _raise)
    client = TestClient(api.app)

    response = client.post("/api/recordings/rec-1/actions/delete")
    assert response.status_code == 503
    assert response.json()["detail"] == "Queue unavailable: redis down"


def test_api_delete_returns_404_when_db_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: {"id": "rec-1"})
    monkeypatch.setattr(api, "purge_pending_recording_jobs", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        api,
        "delete_recording_with_artifacts",
        lambda *_args, **_kwargs: False,
    )
    client = TestClient(api.app)

    response = client.post("/api/recordings/rec-1/actions/delete")
    assert response.status_code == 404
    assert response.json()["detail"] == "Recording not found"


def test_api_delete_returns_500_when_disk_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "get_recording", lambda *_args, **_kwargs: {"id": "rec-1"})
    monkeypatch.setattr(api, "purge_pending_recording_jobs", lambda *_args, **_kwargs: 0)

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RecordingDeleteError("disk busy")

    monkeypatch.setattr(api, "delete_recording_with_artifacts", _raise)
    client = TestClient(api.app)

    response = client.post("/api/recordings/rec-1/actions/delete")
    assert response.status_code == 500
    assert response.json()["detail"] == "disk busy"


def test_api_create_calendar_source_validates_required_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "validate_ics_url", lambda url: url)
    client = TestClient(api.app)

    missing_name = client.post(
        "/api/calendar/sources",
        json={"name": "  ", "kind": "url", "url": "https://calendar.example.com/a.ics"},
    )
    assert missing_name.status_code == 422
    assert missing_name.json()["detail"] == "name is required"

    missing_url = client.post(
        "/api/calendar/sources",
        json={"name": "Team", "kind": "url", "url": "   "},
    )
    assert missing_url.status_code == 422
    assert missing_url.json()["detail"] == "url is required for kind=url"

    missing_file = client.post(
        "/api/calendar/sources",
        json={"name": "Team", "kind": "file", "file": "  "},
    )
    assert missing_file.status_code == 422
    assert missing_file.json()["detail"] == "file is required for kind=file"


def test_api_create_calendar_source_maps_create_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "validate_ics_url", lambda url: url)

    def _raise_value_error(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("duplicate")

    monkeypatch.setattr(api, "create_calendar_source", _raise_value_error)
    client = TestClient(api.app)

    response = client.post(
        "/api/calendar/sources",
        json={"name": "Team", "kind": "url", "url": "https://calendar.example.com/a.ics"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "duplicate"


def test_api_create_calendar_source_file_kind_success_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_create_calendar_source(*, name: str, kind: str, url: str | None, file_ics: str | None, settings: AppSettings) -> dict[str, Any]:
        captured.update(
            {"name": name, "kind": kind, "url": url, "file_ics": file_ics, "settings": settings}
        )
        return {"id": 123, "kind": kind, "name": name}

    monkeypatch.setattr(api, "create_calendar_source", _fake_create_calendar_source)
    monkeypatch.setattr(api, "redacted_calendar_source", lambda source: {"id": int(source["id"])})
    client = TestClient(api.app)

    response = client.post(
        "/api/calendar/sources",
        json={"name": "Inbox", "kind": "file", "file": "BEGIN:VCALENDAR\nEND:VCALENDAR"},
    )
    assert response.status_code == 200
    assert response.json() == {"id": 123}
    assert captured["url"] is None
    assert captured["kind"] == "file"


@pytest.mark.parametrize(
    ("raised", "status_code", "detail_text"),
    [
        (KeyError("missing"), 404, "Calendar source not found"),
        (CalendarSyncError("parse failed"), 422, "parse failed"),
    ],
)
def test_api_sync_calendar_source_maps_errors(
    raised: Exception,
    status_code: int,
    detail_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_sync(*_args: Any, **_kwargs: Any) -> Any:
        raise raised

    monkeypatch.setattr(api, "sync_calendar_source", _raise_sync)
    client = TestClient(api.app)

    response = client.post("/api/calendar/sources/42/sync")
    assert response.status_code == status_code
    assert response.json()["detail"] == detail_text


def test_api_list_calendar_events_validates_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "list_calendar_events", lambda **_kwargs: [])
    client = TestClient(api.app)

    invalid_iso = client.get(
        "/api/calendar/events",
        params={"from": "not-iso", "to": "2026-02-01T10:00:00Z"},
    )
    assert invalid_iso.status_code == 422
    assert invalid_iso.json()["detail"] == "from must be ISO-8601 datetime"

    reversed_window = client.get(
        "/api/calendar/events",
        params={"from": "2026-02-01T10:00:00Z", "to": "2026-02-01T10:00:00Z"},
    )
    assert reversed_window.status_code == 422
    assert reversed_window.json()["detail"] == "to must be after from"


def test_api_upload_maps_value_error_to_422(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "write_upload_to_path", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad upload")))
    client = TestClient(api.app)

    response = client.post(
        "/api/uploads",
        files={"file": ("audio.mp3", b"abc", "audio/mpeg")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "bad upload"


def test_api_upload_maps_generic_error_to_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "write_upload_to_path", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk error")))
    client = TestClient(api.app)

    response = client.post(
        "/api/uploads",
        files={"file": ("audio.mp3", b"abc", "audio/mpeg")},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Upload failed: disk error"
