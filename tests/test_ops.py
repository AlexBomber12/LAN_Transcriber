from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

from fastapi.testclient import TestClient

from lan_app import api, ui_routes
from lan_app.config import AppSettings
from lan_app.constants import RECORDING_STATUS_QUARANTINE
from lan_app.db import connect, create_recording, get_recording, init_db
from lan_app.ops import run_retention_cleanup


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def test_run_retention_cleanup_deletes_old_quarantine_and_tmp_entries(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-quarantine-old-1",
        source="drive",
        source_filename="old.wav",
        status=RECORDING_STATUS_QUARANTINE,
        settings=cfg,
    )
    create_recording(
        "rec-quarantine-fresh-1",
        source="drive",
        source_filename="fresh.wav",
        status=RECORDING_STATUS_QUARANTINE,
        settings=cfg,
    )

    now = datetime.now(tz=timezone.utc)
    old_ts = _iso_utc(now - timedelta(days=10))
    fresh_ts = _iso_utc(now - timedelta(days=1))
    with connect(cfg) as conn:
        conn.execute(
            "UPDATE recordings SET updated_at = ? WHERE id = ?",
            (old_ts, "rec-quarantine-old-1"),
        )
        conn.execute(
            "UPDATE recordings SET updated_at = ? WHERE id = ?",
            (fresh_ts, "rec-quarantine-fresh-1"),
        )
        conn.commit()

    old_rec_path = cfg.recordings_root / "rec-quarantine-old-1" / "raw"
    old_rec_path.mkdir(parents=True, exist_ok=True)
    (old_rec_path / "audio.wav").write_bytes(b"\x00")
    fresh_rec_path = cfg.recordings_root / "rec-quarantine-fresh-1" / "raw"
    fresh_rec_path.mkdir(parents=True, exist_ok=True)
    (fresh_rec_path / "audio.wav").write_bytes(b"\x00")

    tmp_root = cfg.data_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stale_tmp = tmp_root / "stale.tmp"
    stale_tmp.write_text("stale", encoding="utf-8")
    fresh_tmp = tmp_root / "fresh.tmp"
    fresh_tmp.write_text("fresh", encoding="utf-8")

    stale_mtime = (now - timedelta(days=9)).timestamp()
    fresh_mtime = (now - timedelta(hours=6)).timestamp()
    os.utime(stale_tmp, (stale_mtime, stale_mtime))
    os.utime(fresh_tmp, (fresh_mtime, fresh_mtime))

    summary = run_retention_cleanup(settings=cfg)
    assert summary["quarantine_recordings_deleted"] == 1
    assert summary["quarantine_directories_deleted"] == 1
    assert summary["tmp_entries_deleted"] == 1

    assert get_recording("rec-quarantine-old-1", settings=cfg) is None
    assert get_recording("rec-quarantine-fresh-1", settings=cfg) is not None
    assert not (cfg.recordings_root / "rec-quarantine-old-1").exists()
    assert (cfg.recordings_root / "rec-quarantine-fresh-1").exists()
    assert not stale_tmp.exists()
    assert fresh_tmp.exists()


def test_healthz_component_endpoints(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    monkeypatch.setattr(api, "check_db_health", lambda _settings: {"component": "db", "ok": True, "detail": "ok"})
    monkeypatch.setattr(
        api,
        "check_redis_health",
        lambda _settings: {"component": "redis", "ok": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        api,
        "check_worker_health",
        lambda _settings: {"component": "worker", "ok": False, "detail": "stale"},
    )
    monkeypatch.setattr(
        api,
        "collect_health_checks",
        lambda _settings: {
            "app": {"component": "app", "ok": True, "detail": "ok"},
            "db": {"component": "db", "ok": True, "detail": "ok"},
            "redis": {"component": "redis", "ok": True, "detail": "ok"},
            "worker": {"component": "worker", "ok": False, "detail": "stale"},
        },
    )

    client = TestClient(api.app)
    all_health = client.get("/healthz")
    assert all_health.status_code == 200
    payload = all_health.json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["worker"]["ok"] is False

    app_health = client.get("/healthz/app")
    assert app_health.status_code == 200
    assert app_health.json()["component"] == "app"

    worker_health = client.get("/healthz/worker")
    assert worker_health.status_code == 503
