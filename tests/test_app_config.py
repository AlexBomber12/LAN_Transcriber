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
