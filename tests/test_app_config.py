import os
from pathlib import Path
import subprocess
import sys

import pytest

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


def test_llm_model_env_required_and_trimmed(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "  gpt-oss:120b  ")
    cfg = AppSettings()
    assert cfg.llm_model == "gpt-oss:120b"


def test_llm_model_prefixed_alias_is_accepted(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LAN_LLM_MODEL", "  gpt-oss:prefixed  ")
    cfg = AppSettings()
    assert cfg.llm_model == "gpt-oss:prefixed"


def test_llm_model_missing_fails_with_clear_message(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LAN_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL is required\\. Set it in \\.env"):
        AppSettings()


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


def test_upload_capture_timezone_defaults_and_aliases(monkeypatch):
    monkeypatch.delenv("LAN_UPLOAD_CAPTURE_TIMEZONE", raising=False)
    monkeypatch.delenv("UPLOAD_CAPTURE_TIMEZONE", raising=False)
    assert AppSettings().upload_capture_timezone == "Europe/Rome"

    monkeypatch.setenv("UPLOAD_CAPTURE_TIMEZONE", " Europe/Berlin ")
    assert AppSettings().upload_capture_timezone == "Europe/Berlin"

    monkeypatch.setenv("LAN_UPLOAD_CAPTURE_TIMEZONE", "  ")
    assert AppSettings().upload_capture_timezone == "Europe/Rome"

    monkeypatch.setenv("LAN_UPLOAD_CAPTURE_TIMEZONE", "America/New_York")
    assert AppSettings().upload_capture_timezone == "America/New_York"


def test_upload_capture_timezone_invalid_fails_clearly(monkeypatch):
    monkeypatch.setenv("UPLOAD_CAPTURE_TIMEZONE", "Mars/Olympus")
    with pytest.raises(
        ValueError,
        match="UPLOAD_CAPTURE_TIMEZONE must be a valid IANA timezone",
    ):
        AppSettings()


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
    env["LLM_MODEL"] = "test-llm-model"
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


def test_dev_missing_llm_model_fails_import():
    env = os.environ.copy()
    env["LAN_ENV"] = "dev"
    env.pop("LLM_MODEL", None)
    env.pop("LAN_LLM_MODEL", None)

    result = subprocess.run(
        [sys.executable, "-c", "import lan_app.api"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "LLM_MODEL is required" in f"{result.stdout}\n{result.stderr}"


def test_non_dev_requires_both_runtime_urls(monkeypatch):
    monkeypatch.setenv("LAN_ENV", "prod")
    monkeypatch.delenv("LAN_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="LAN_REDIS_URL, LLM_BASE_URL"):
        AppSettings()


def test_non_dev_accepts_when_runtime_urls_are_present(monkeypatch):
    monkeypatch.setenv("LAN_ENV", "staging")
    monkeypatch.setenv("LAN_REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000")

    cfg = AppSettings()
    assert cfg.redis_url == "redis://127.0.0.1:6379/0"
    assert cfg.llm_base_url == "http://127.0.0.1:8000"


def test_dev_keeps_explicit_llm_base_url(monkeypatch):
    monkeypatch.setenv("LAN_ENV", "dev")
    monkeypatch.setenv("LAN_REDIS_URL", "redis://127.0.0.1:6379/3")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234")

    cfg = AppSettings()
    assert cfg.redis_url == "redis://127.0.0.1:6379/3"
    assert cfg.llm_base_url == "http://localhost:1234"


def test_llm_max_tokens_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS_RETRY", raising=False)
    defaults = AppSettings()
    assert defaults.llm_max_tokens == 1024
    assert defaults.llm_max_tokens_retry == 2048

    monkeypatch.setenv("LLM_MAX_TOKENS", "1536")
    monkeypatch.setenv("LLM_MAX_TOKENS_RETRY", "3072")
    overridden = AppSettings()
    assert overridden.llm_max_tokens == 1536
    assert overridden.llm_max_tokens_retry == 3072

    monkeypatch.setenv("LLM_MAX_TOKENS", "128")
    with pytest.raises(ValueError, match="LLM_MAX_TOKENS|llm_max_tokens"):
        AppSettings()


def test_llm_chunking_settings_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("LLM_CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("LLM_CHUNK_OVERLAP_CHARS", raising=False)
    monkeypatch.delenv("LLM_CHUNK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_CHUNK_SPLIT_MIN_CHARS", raising=False)
    monkeypatch.delenv("LLM_CHUNK_SPLIT_MAX_DEPTH", raising=False)
    monkeypatch.delenv("LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS", raising=False)
    monkeypatch.delenv("LLM_MERGE_MAX_TOKENS", raising=False)

    defaults = AppSettings()
    assert defaults.llm_chunk_max_chars == 4500
    assert defaults.llm_chunk_overlap_chars == 300
    assert defaults.llm_chunk_timeout_seconds == 120.0
    assert defaults.llm_chunk_split_min_chars == 1200
    assert defaults.llm_chunk_split_max_depth == 2
    assert defaults.llm_long_transcript_threshold_chars == 4500
    assert defaults.llm_merge_max_tokens is None

    monkeypatch.setenv("LLM_CHUNK_MAX_CHARS", "4096")
    monkeypatch.setenv("LLM_CHUNK_OVERLAP_CHARS", "256")
    monkeypatch.setenv("LLM_CHUNK_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("LLM_CHUNK_SPLIT_MIN_CHARS", "900")
    monkeypatch.setenv("LLM_CHUNK_SPLIT_MAX_DEPTH", "3")
    monkeypatch.setenv("LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS", "8192")
    monkeypatch.setenv("LLM_MERGE_MAX_TOKENS", "3072")

    overridden = AppSettings()
    assert overridden.llm_chunk_max_chars == 4096
    assert overridden.llm_chunk_overlap_chars == 256
    assert overridden.llm_chunk_timeout_seconds == 45.0
    assert overridden.llm_chunk_split_min_chars == 900
    assert overridden.llm_chunk_split_max_depth == 3
    assert overridden.llm_long_transcript_threshold_chars == 8192
    assert overridden.llm_merge_max_tokens == 3072

    monkeypatch.setenv("LLM_CHUNK_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="LLM_CHUNK_TIMEOUT_SECONDS|llm_chunk_timeout_seconds"):
        AppSettings()


def test_dev_missing_urls_allows_import():
    env = os.environ.copy()
    env["LAN_ENV"] = "dev"
    env["LLM_MODEL"] = "test-llm-model"
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


def test_diarization_model_id_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("LAN_DIARIZATION_MODEL_ID", raising=False)
    assert AppSettings().diarization_model_id == "pyannote/speaker-diarization-3.1"

    monkeypatch.setenv("LAN_DIARIZATION_MODEL_ID", "custom-org/custom-diar@v2")
    assert AppSettings().diarization_model_id == "custom-org/custom-diar@v2"

    monkeypatch.setenv("LAN_DIARIZATION_MODEL_ID", "   ")
    assert AppSettings().diarization_model_id == "pyannote/speaker-diarization-3.1"


def test_vad_method_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("LAN_VAD_METHOD", raising=False)
    assert AppSettings().vad_method == "silero"

    monkeypatch.setenv("LAN_VAD_METHOD", "pyannote")
    assert AppSettings().vad_method == "pyannote"

    monkeypatch.setenv("LAN_VAD_METHOD", "invalid")
    with pytest.raises(ValueError, match="LAN_VAD_METHOD|vad_method"):
        AppSettings()


def test_diarization_quality_settings_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("LAN_DIARIZATION_PROFILE", raising=False)
    monkeypatch.delenv("LAN_DIARIZATION_MIN_SPEAKERS", raising=False)
    monkeypatch.delenv("LAN_DIARIZATION_MAX_SPEAKERS", raising=False)
    monkeypatch.delenv(
        "LAN_DIARIZATION_DIALOG_RETRY_MIN_DURATION_SECONDS",
        raising=False,
    )
    monkeypatch.delenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_TURNS", raising=False)
    monkeypatch.delenv("LAN_DIARIZATION_MERGE_GAP_SECONDS", raising=False)
    monkeypatch.delenv("LAN_DIARIZATION_MIN_TURN_SECONDS", raising=False)

    cfg = AppSettings()
    assert cfg.diarization_profile == "auto"
    assert cfg.diarization_min_speakers is None
    assert cfg.diarization_max_speakers is None
    assert cfg.diarization_dialog_retry_min_duration_seconds == 20.0
    assert cfg.diarization_dialog_retry_min_turns == 6
    assert cfg.diarization_merge_gap_seconds == 0.5
    assert cfg.diarization_min_turn_seconds == 0.5

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "dialog")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_SPEAKERS", "3")
    monkeypatch.setenv("LAN_DIARIZATION_MAX_SPEAKERS", "4")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_DURATION_SECONDS", "9.5")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_TURNS", "5")
    monkeypatch.setenv("LAN_DIARIZATION_MERGE_GAP_SECONDS", "0.6")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_TURN_SECONDS", "0.4")

    cfg = AppSettings()
    assert cfg.diarization_profile == "dialog"
    assert cfg.diarization_min_speakers == 3
    assert cfg.diarization_max_speakers == 4
    assert cfg.diarization_dialog_retry_min_duration_seconds == 9.5
    assert cfg.diarization_dialog_retry_min_turns == 5
    assert cfg.diarization_merge_gap_seconds == 0.6
    assert cfg.diarization_min_turn_seconds == 0.4
