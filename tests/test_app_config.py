import os
from pathlib import Path
import subprocess
import sys

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


def test_routing_threshold_from_env(monkeypatch):
    monkeypatch.setenv("ROUTING_AUTO_SELECT_THRESHOLD", "0.73")

    cfg = AppSettings()
    assert cfg.routing_auto_select_threshold == 0.73


def test_security_settings_from_env(monkeypatch):
    monkeypatch.setenv("LAN_API_BEARER_TOKEN", "secret-token")

    cfg = AppSettings()
    assert cfg.api_bearer_token == "secret-token"


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


def test_lan_env_dev_defaults_with_warnings(monkeypatch, caplog):
    monkeypatch.setenv("LAN_ENV", "dev")
    monkeypatch.delenv("LAN_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    with caplog.at_level("WARNING"):
        cfg = AppSettings()

    assert cfg.redis_url == "redis://127.0.0.1:6379/0"
    assert cfg.llm_base_url == "http://127.0.0.1:8000"
    assert "LAN_REDIS_URL is not set in LAN_ENV=dev" in caplog.text
    assert "LLM_BASE_URL is not set in LAN_ENV=dev" in caplog.text


def test_staging_missing_redis_url_fails_import():
    env = os.environ.copy()
    env["LAN_ENV"] = "staging"
    env["LLM_BASE_URL"] = "http://127.0.0.1:8000"
    env.pop("LAN_REDIS_URL", None)
    env.pop("REDIS_URL", None)

    result = subprocess.run(
        [sys.executable, "-c", "import lan_app.api"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "LAN_REDIS_URL" in f"{result.stdout}\n{result.stderr}"


def test_dev_missing_urls_allows_import():
    env = os.environ.copy()
    env["LAN_ENV"] = "dev"
    env.pop("LAN_REDIS_URL", None)
    env.pop("REDIS_URL", None)
    env.pop("LLM_BASE_URL", None)

    result = subprocess.run(
        [sys.executable, "-c", "import lan_app.api"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
