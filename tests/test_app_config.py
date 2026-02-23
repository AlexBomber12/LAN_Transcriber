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


def test_sqlite_busy_timeout_from_env(monkeypatch):
    monkeypatch.setenv("LAN_SQLITE_BUSY_TIMEOUT_MS", "12345")
    cfg = AppSettings()
    assert cfg.sqlite_busy_timeout_ms == 12345


def test_ms_auth_settings_from_env(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "auth" / "msal_cache.bin"
    monkeypatch.setenv("MS_TENANT_ID", "tenant-id")
    monkeypatch.setenv("MS_CLIENT_ID", "client-id")
    monkeypatch.setenv(
        "MS_SCOPES",
        "offline_access User.Read Notes.ReadWrite Calendars.Read",
    )
    monkeypatch.setenv("MSAL_CACHE_PATH", str(cache_path))

    cfg = AppSettings()
    assert cfg.ms_tenant_id == "tenant-id"
    assert cfg.ms_client_id == "client-id"
    assert cfg.ms_scopes_list == [
        "offline_access",
        "User.Read",
        "Notes.ReadWrite",
        "Calendars.Read",
    ]
    assert cfg.msal_cache_path == cache_path


def test_calendar_match_settings_from_env(monkeypatch):
    monkeypatch.setenv("CALENDAR_MATCH_WINDOW_MINUTES", "30")
    monkeypatch.setenv("CALENDAR_AUTO_MATCH_THRESHOLD", "0.7")

    cfg = AppSettings()
    assert cfg.calendar_match_window_minutes == 30
    assert cfg.calendar_auto_match_threshold == 0.7


def test_routing_threshold_from_env(monkeypatch):
    monkeypatch.setenv("ROUTING_AUTO_SELECT_THRESHOLD", "0.73")

    cfg = AppSettings()
    assert cfg.routing_auto_select_threshold == 0.73


def test_security_settings_from_env(monkeypatch):
    monkeypatch.setenv("LAN_API_BEARER_TOKEN", "secret-token")
    monkeypatch.setenv("LAN_INGEST_LOCK_TTL_SECONDS", "123")

    cfg = AppSettings()
    assert cfg.api_bearer_token == "secret-token"
    assert cfg.ingest_lock_ttl_seconds == 123


def test_worker_and_reaper_settings_from_env(monkeypatch):
    monkeypatch.setenv("LAN_RQ_JOB_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("LAN_MAX_JOB_ATTEMPTS", "5")
    monkeypatch.setenv("LAN_STUCK_JOB_SECONDS", "900")
    monkeypatch.setenv("LAN_REAPER_INTERVAL_SECONDS", "60")

    cfg = AppSettings()
    assert cfg.rq_job_timeout_seconds == 1800
    assert cfg.max_job_attempts == 5
    assert cfg.stuck_job_seconds == 900
    assert cfg.reaper_interval_seconds == 60
