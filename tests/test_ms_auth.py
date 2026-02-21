from __future__ import annotations

import threading
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from lan_app import api, ms_graph, ui_routes
from lan_app.config import AppSettings
from lan_app.db import init_db
from lan_app.ms_graph import GraphDeviceFlowLimitError, GraphNotConfiguredError


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_api_ms_verify_ok(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    monkeypatch.setattr(
        api,
        "ms_connection_state",
        lambda _settings: {
            "status": "connected",
            "account_display_name": "Alex Worker",
            "tenant_id": "tenant-1",
            "granted_scopes": ["offline_access", "Notes.ReadWrite"],
        },
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/api/connections/ms/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["account_display_name"] == "Alex Worker"


def test_api_ms_verify_uses_threadpool(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    def _fail_direct(_settings):
        raise AssertionError("ms_connection_state should run via threadpool")

    calls: dict[str, object] = {}

    async def _fake_threadpool(fn, *args, **kwargs):
        calls["fn"] = fn
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "status": "connected",
            "account_display_name": "Alex Worker",
            "tenant_id": "tenant-1",
            "granted_scopes": ["offline_access", "User.Read"],
        }

    monkeypatch.setattr(api, "ms_connection_state", _fail_direct)
    monkeypatch.setattr(api, "run_in_threadpool", _fake_threadpool)

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/api/connections/ms/verify")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert calls["fn"] == _fail_direct


def test_api_ms_verify_error(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    monkeypatch.setattr(
        api,
        "ms_connection_state",
        lambda _settings: {
            "status": "expired",
            "error": "Reconnect required.",
            "account_display_name": "Alex Worker",
            "tenant_id": "tenant-1",
            "granted_scopes": [],
        },
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/api/connections/ms/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "expired"


def test_api_ms_start_connect(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    monkeypatch.setattr(
        api,
        "start_device_flow_session",
        lambda _settings, reconnect=False: {
            "session_id": "sess-1",
            "status": "pending",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://microsoft.com/devicelogin",
            "message": "Use code ABCD-EFGH",
            "expires_in": 900,
            "reconnect": reconnect,
        },
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.post("/api/connections/ms/connect?reconnect=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess-1"
    assert body["reconnect"] is True


def test_api_ms_start_connect_not_configured(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    def _raise_not_configured(_settings, reconnect=False):
        raise GraphNotConfiguredError("missing env")

    monkeypatch.setattr(api, "start_device_flow_session", _raise_not_configured)

    client = TestClient(api.app, follow_redirects=True)
    resp = client.post("/api/connections/ms/connect")
    assert resp.status_code == 422


def test_api_ms_start_connect_too_many_pending(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    def _raise_too_many(_settings, reconnect=False):
        raise GraphDeviceFlowLimitError("too many")

    monkeypatch.setattr(api, "start_device_flow_session", _raise_too_many)

    client = TestClient(api.app, follow_redirects=True)
    resp = client.post("/api/connections/ms/connect")
    assert resp.status_code == 429


def test_api_ms_connect_poll(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    monkeypatch.setattr(
        api,
        "get_device_flow_session",
        lambda _session_id: {"session_id": "sess-1", "status": "pending"},
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/api/connections/ms/connect/sess-1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


def test_api_ms_connect_poll_not_found(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    def _missing(_session_id: str):
        raise KeyError(_session_id)

    monkeypatch.setattr(api, "get_device_flow_session", _missing)

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/api/connections/ms/connect/missing")
    assert resp.status_code == 404


def test_connections_page_shows_ms_state(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    monkeypatch.setattr(
        ui_routes,
        "ms_connection_state",
        lambda _settings: {
            "configured": True,
            "status": "connected",
            "state": "Connected",
            "account_display_name": "Alex Worker",
            "tenant_id": "tenant-1",
            "requested_scopes": [
                "offline_access",
                "User.Read",
                "Notes.ReadWrite",
                "Calendars.Read",
            ],
            "granted_scopes": ["offline_access", "User.Read", "Notes.ReadWrite"],
            "error": None,
        },
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/connections")
    assert resp.status_code == 200
    assert "Connected" in resp.text
    assert "Alex Worker" in resp.text
    assert "Reconnect" in resp.text


def test_connections_page_uses_threadpool(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    def _fail_direct(_settings):
        raise AssertionError("ms_connection_state should run via threadpool")

    calls: dict[str, object] = {}

    async def _fake_threadpool(fn, *args, **kwargs):
        calls["fn"] = fn
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "configured": True,
            "status": "connected",
            "state": "Connected",
            "account_display_name": "Alex Worker",
            "tenant_id": "tenant-1",
            "requested_scopes": [
                "offline_access",
                "User.Read",
                "Notes.ReadWrite",
                "Calendars.Read",
            ],
            "granted_scopes": ["offline_access", "User.Read", "Notes.ReadWrite"],
            "error": None,
        }

    monkeypatch.setattr(ui_routes, "ms_connection_state", _fail_direct)
    monkeypatch.setattr(ui_routes, "run_in_threadpool", _fake_threadpool)

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/connections")
    assert resp.status_code == 200
    assert "Alex Worker" in resp.text
    assert calls["fn"] == _fail_direct


def test_start_device_flow_session_reuses_pending(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ms_tenant_id = "tenant-1"
    cfg.ms_client_id = "client-1"
    cfg.ms_scopes = "offline_access User.Read Notes.ReadWrite Calendars.Read"

    with ms_graph._DEVICE_FLOW_LOCK:
        ms_graph._DEVICE_FLOW_SESSIONS.clear()

    initiated: list[int] = []
    started_session_ids: list[str] = []

    class _FakeClient:
        def __init__(self, settings=None):
            self.settings = settings

        def clear_cache(self):
            return None

        def initiate_device_flow(self):
            initiated.append(1)
            return {
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://microsoft.com/devicelogin",
                "expires_in": 900,
            }

    class _FakeThread:
        def __init__(self, target=None, kwargs=None, daemon=None):
            self._kwargs = kwargs or {}

        def start(self):
            started_session_ids.append(self._kwargs["session_id"])

    monkeypatch.setattr(ms_graph, "MicrosoftGraphClient", _FakeClient)
    monkeypatch.setattr(ms_graph.threading, "Thread", _FakeThread)

    first = ms_graph.start_device_flow_session(cfg)
    second = ms_graph.start_device_flow_session(cfg)

    assert first["status"] == "pending"
    assert second["status"] == "pending"
    assert first["session_id"] == second["session_id"]
    assert first["reused"] is False
    assert second["reused"] is True
    assert len(initiated) == 1
    assert started_session_ids == [first["session_id"]]


def test_start_device_flow_session_serializes_initiation(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ms_tenant_id = "tenant-1"
    cfg.ms_client_id = "client-1"
    cfg.ms_scopes = "offline_access User.Read Notes.ReadWrite Calendars.Read"

    with ms_graph._DEVICE_FLOW_LOCK:
        ms_graph._DEVICE_FLOW_SESSIONS.clear()

    initiated: list[int] = []
    start_gate = threading.Event()
    release_gate = threading.Event()

    class _FakeClient:
        def __init__(self, settings=None):
            self.settings = settings

        def clear_cache(self):
            return None

        def initiate_device_flow(self):
            initiated.append(1)
            start_gate.set()
            release_gate.wait(timeout=2)
            return {
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://microsoft.com/devicelogin",
                "expires_in": 900,
            }

    monkeypatch.setattr(ms_graph, "MicrosoftGraphClient", _FakeClient)
    monkeypatch.setattr(ms_graph, "_complete_device_flow_in_background", lambda **_k: None)

    results: list[dict[str, object]] = []
    errors: list[Exception] = []

    def _call_start():
        try:
            results.append(ms_graph.start_device_flow_session(cfg))
        except Exception as exc:  # pragma: no cover - test helper
            errors.append(exc)

    t1 = threading.Thread(target=_call_start)
    t2 = threading.Thread(target=_call_start)

    t1.start()
    assert start_gate.wait(timeout=1), "first initiation did not start in time"
    t2.start()
    release_gate.set()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert not errors
    assert len(results) == 2
    assert len(initiated) == 1
    assert results[0]["session_id"] == results[1]["session_id"]
    assert sorted([results[0]["reused"], results[1]["reused"]]) == [False, True]


def test_graph_post_html_does_not_retry_non_idempotent_page_creation(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.ms_tenant_id = "tenant-1"
    cfg.ms_client_id = "client-1"
    cfg.ms_scopes = "offline_access User.Read Notes.ReadWrite Calendars.Read"

    class _FakePublicClientApplication:
        def __init__(self, *args, **kwargs):
            pass

        def get_accounts(self):
            return []

    monkeypatch.setattr(
        ms_graph.msal,
        "PublicClientApplication",
        _FakePublicClientApplication,
    )
    client = ms_graph.MicrosoftGraphClient(cfg)

    monkeypatch.setattr(
        client,
        "acquire_token_silent",
        lambda: {"access_token": "token-1"},
    )

    calls: list[tuple[str, str]] = []

    class _FakeResponse:
        status_code = 503
        content = b""
        headers: dict[str, str] = {}

        def json(self):  # pragma: no cover - not used in this test
            return {}

    class _FakeHttpClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method: str, url: str, **kwargs):
            calls.append((method, url))
            return _FakeResponse()

    monkeypatch.setattr(ms_graph.httpx, "Client", _FakeHttpClient)

    with pytest.raises(ms_graph.GraphRequestError):
        client.graph_post_html("/me/onenote/sections/sec-1/pages", "<html></html>")

    assert len(calls) == 1
