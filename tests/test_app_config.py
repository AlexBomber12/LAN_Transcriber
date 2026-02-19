from pathlib import Path

from lan_app.config import AppSettings


def test_metrics_snapshot_path_from_documented_env(monkeypatch, tmp_path: Path):
    expected = tmp_path / "metrics.snap"
    monkeypatch.setenv("LAN_PROM_SNAPSHOT_PATH", str(expected))
    monkeypatch.delenv("LAN_METRICS_SNAPSHOT_PATH", raising=False)
    monkeypatch.delenv("PROM_SNAPSHOT_PATH", raising=False)

    cfg = AppSettings()
    assert cfg.metrics_snapshot_path == expected


def test_metrics_snapshot_path_legacy_fallback(monkeypatch, tmp_path: Path):
    expected = tmp_path / "legacy.snap"
    monkeypatch.delenv("LAN_PROM_SNAPSHOT_PATH", raising=False)
    monkeypatch.delenv("LAN_METRICS_SNAPSHOT_PATH", raising=False)
    monkeypatch.setenv("PROM_SNAPSHOT_PATH", str(expected))

    cfg = AppSettings()
    assert cfg.metrics_snapshot_path == expected


def test_redis_url_alias(monkeypatch):
    monkeypatch.setenv("LAN_REDIS_URL", "redis://localhost:6380/0")
    cfg = AppSettings()
    assert cfg.redis_url == "redis://localhost:6380/0"


def test_ms_auth_settings_from_env(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "auth" / "msal_cache.bin"
    monkeypatch.setenv("MS_TENANT_ID", "tenant-id")
    monkeypatch.setenv("MS_CLIENT_ID", "client-id")
    monkeypatch.setenv("MS_SCOPES", "offline_access Notes.ReadWrite Calendars.Read")
    monkeypatch.setenv("MSAL_CACHE_PATH", str(cache_path))

    cfg = AppSettings()
    assert cfg.ms_tenant_id == "tenant-id"
    assert cfg.ms_client_id == "client-id"
    assert cfg.ms_scopes_list == [
        "offline_access",
        "Notes.ReadWrite",
        "Calendars.Read",
    ]
    assert cfg.msal_cache_path == cache_path
