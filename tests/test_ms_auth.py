from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lan_app import api, ui_routes
from lan_app.config import AppSettings
from lan_app.db import init_db
from lan_app.ms_graph import GraphNotConfiguredError


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
            "requested_scopes": ["offline_access", "Notes.ReadWrite", "Calendars.Read"],
            "granted_scopes": ["offline_access", "Notes.ReadWrite"],
            "error": None,
        },
    )

    client = TestClient(api.app, follow_redirects=True)
    resp = client.get("/connections")
    assert resp.status_code == 200
    assert "Connected" in resp.text
    assert "Alex Worker" in resp.text
    assert "Reconnect" in resp.text
