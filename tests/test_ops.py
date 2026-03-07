from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import shutil

from fastapi.testclient import TestClient
import pytest

from lan_app import api, ui_routes
from lan_app.config import AppSettings
from lan_app.constants import RECORDING_STATUS_QUARANTINE
from lan_app.db import connect, create_recording, get_recording, init_db
from lan_app import healthchecks
from lan_app.ops import (
    RecordingDeleteError,
    delete_recording_with_artifacts,
    run_retention_cleanup,
)


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


def test_delete_recording_with_artifacts_removes_db_and_directory(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-delete-all-1",
        source="upload",
        source_filename="delete.wav",
        settings=cfg,
    )
    recording_root = cfg.recordings_root / "rec-delete-all-1"
    (recording_root / "raw").mkdir(parents=True, exist_ok=True)
    (recording_root / "derived").mkdir(parents=True, exist_ok=True)
    (recording_root / "logs").mkdir(parents=True, exist_ok=True)
    (recording_root / "raw" / "audio.wav").write_bytes(b"\x00")
    (recording_root / "derived" / "summary.json").write_text("{}", encoding="utf-8")
    (recording_root / "logs" / "step-precheck.log").write_text("log", encoding="utf-8")
    (recording_root / "temp.tmp").write_text("temp", encoding="utf-8")

    assert delete_recording_with_artifacts("rec-delete-all-1", settings=cfg) is True
    assert get_recording("rec-delete-all-1", settings=cfg) is None
    assert not recording_root.exists()


def test_delete_recording_with_artifacts_rejects_invalid_id(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(RecordingDeleteError, match="recording id is required"):
        delete_recording_with_artifacts("", settings=cfg)
    with pytest.raises(RecordingDeleteError, match="invalid recording id"):
        delete_recording_with_artifacts("../escape", settings=cfg)


def test_delete_recording_with_artifacts_rejects_symlink_escape(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.recordings_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    (cfg.recordings_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RecordingDeleteError, match="invalid recording path"):
        delete_recording_with_artifacts("escape", settings=cfg)


def test_delete_recording_with_artifacts_rejects_symlink_to_another_recording(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-delete-link-1",
        source="upload",
        source_filename="link.wav",
        settings=cfg,
    )
    create_recording(
        "rec-delete-target-1",
        source="upload",
        source_filename="target.wav",
        settings=cfg,
    )
    target_root = cfg.recordings_root / "rec-delete-target-1"
    (target_root / "raw").mkdir(parents=True, exist_ok=True)
    target_audio = target_root / "raw" / "audio.wav"
    target_audio.write_bytes(b"\x00")
    alias_root = cfg.recordings_root / "rec-delete-link-1"
    alias_root.parent.mkdir(parents=True, exist_ok=True)
    alias_root.symlink_to(target_root, target_is_directory=True)

    with pytest.raises(RecordingDeleteError, match="invalid recording path"):
        delete_recording_with_artifacts("rec-delete-link-1", settings=cfg)

    assert get_recording("rec-delete-link-1", settings=cfg) is not None
    assert get_recording("rec-delete-target-1", settings=cfg) is not None
    assert target_audio.exists()
    assert alias_root.is_symlink()


def test_delete_recording_with_artifacts_handles_file_root(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-delete-file-1",
        source="upload",
        source_filename="file-root.wav",
        settings=cfg,
    )
    recording_root = cfg.recordings_root / "rec-delete-file-1"
    recording_root.parent.mkdir(parents=True, exist_ok=True)
    recording_root.write_text("unexpected-file-root", encoding="utf-8")

    assert delete_recording_with_artifacts("rec-delete-file-1", settings=cfg) is True
    assert not recording_root.exists()


def test_delete_recording_with_artifacts_returns_false_before_disk_cleanup_when_row_missing(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    recording_root = cfg.recordings_root / "rec-delete-missing-1"
    (recording_root / "raw").mkdir(parents=True, exist_ok=True)
    (recording_root / "raw" / "audio.wav").write_bytes(b"\x00")

    assert delete_recording_with_artifacts("rec-delete-missing-1", settings=cfg) is False
    assert recording_root.exists()


def test_delete_recording_with_artifacts_raises_on_database_delete_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_root = cfg.recordings_root / "rec-delete-db-fail-1"
    (recording_root / "raw").mkdir(parents=True, exist_ok=True)
    audio_path = recording_root / "raw" / "audio.wav"
    audio_path.write_bytes(b"\x00")

    monkeypatch.setattr(
        "lan_app.ops.delete_recording",
        lambda *_a, **_k: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )

    with pytest.raises(RecordingDeleteError, match="database is locked"):
        delete_recording_with_artifacts("rec-delete-db-fail-1", settings=cfg)
    assert audio_path.exists()


def test_delete_recording_with_artifacts_raises_on_disk_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-delete-fail-1",
        source="upload",
        source_filename="fail.wav",
        settings=cfg,
    )
    raw_dir = cfg.recordings_root / "rec-delete-fail-1" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "audio.wav").write_bytes(b"\x00")

    real_rmtree = shutil.rmtree

    def _fail_once(path: Path | str, *args, **kwargs) -> None:
        if Path(path) == raw_dir:
            raise OSError("disk busy")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("lan_app.ops.shutil.rmtree", _fail_once)

    with pytest.raises(RecordingDeleteError, match="disk busy"):
        delete_recording_with_artifacts("rec-delete-fail-1", settings=cfg)
    assert get_recording("rec-delete-fail-1", settings=cfg) is None
    assert raw_dir.exists()


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


def test_healthchecks_main_runs_only_requested_component(monkeypatch):
    monkeypatch.setattr(
        healthchecks,
        "collect_health_checks",
        lambda settings=None: (_ for _ in ()).throw(AssertionError("unexpected collect")),
    )
    monkeypatch.setattr(
        healthchecks,
        "check_db_health",
        lambda settings=None: {"component": "db", "ok": True, "detail": "ok"},
    )

    exit_code = healthchecks.main(["db"])
    assert exit_code == 0
