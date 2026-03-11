from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from lan_app import db as db_module
from lan_app import ui_routes, uploads
from lan_app.config import AppSettings
from lan_app.db import create_recording, get_recording, init_db, list_recordings


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _init_db_to_version(cfg: AppSettings, target_version: int) -> None:
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cfg.db_path) as conn:
        for version, migration_path in db_module._migration_files():  # noqa: SLF001
            if version > target_version:
                break
            conn.executescript(migration_path.read_text(encoding="utf-8"))
            conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()


def test_upload_timestamp_helpers_cover_winter_and_dst_offsets() -> None:
    upload_capture_timezone = ZoneInfo("Europe/Rome")

    assert uploads.parse_plaud_captured_local_datetime("2026-03-06_12_02_16.mp3") == datetime(
        2026,
        3,
        6,
        12,
        2,
        16,
    )
    assert (
        uploads.infer_captured_at(
            "2026-03-06_12_02_16.mp3",
            upload_capture_timezone=upload_capture_timezone,
        )
        == "2026-03-06T11:02:16Z"
    )
    assert (
        uploads.infer_captured_at(
            "2026-07-06_12_02_16.mp3",
            upload_capture_timezone=upload_capture_timezone,
        )
        == "2026-07-06T10:02:16Z"
    )


def test_recording_capture_time_provenance_round_trips_in_db(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)

    created = create_recording(
        "rec-capture-provenance-1",
        source="upload",
        source_filename="2026-03-06_12_02_16.mp3",
        captured_at="2026-03-06T11:02:16Z",
        captured_at_source="2026-03-06T12:02:16",
        captured_at_timezone="Europe/Rome",
        captured_at_inferred_from_filename=True,
        settings=cfg,
    )

    assert created["captured_at"] == "2026-03-06T11:02:16Z"
    assert created["captured_at_source"] == "2026-03-06T12:02:16"
    assert created["captured_at_timezone"] == "Europe/Rome"
    assert created["captured_at_inferred_from_filename"] == 1

    fetched = get_recording("rec-capture-provenance-1", settings=cfg)
    assert fetched is not None
    assert fetched["captured_at_source"] == "2026-03-06T12:02:16"
    assert fetched["captured_at_timezone"] == "Europe/Rome"
    assert fetched["captured_at_inferred_from_filename"] == 1

    listed, total = list_recordings(settings=cfg)
    assert total == 1
    assert listed[0]["id"] == "rec-capture-provenance-1"
    assert listed[0]["captured_at_source"] == "2026-03-06T12:02:16"
    assert listed[0]["captured_at_timezone"] == "Europe/Rome"
    assert listed[0]["captured_at_inferred_from_filename"] == 1


def test_backfill_returns_zero_before_provenance_columns_exist(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _init_db_to_version(cfg, 18)

    assert db_module._backfill_legacy_upload_capture_times(cfg) == 0  # noqa: SLF001


def test_steady_state_db_calls_do_not_rerun_capture_time_backfill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)

    monkeypatch.setattr(
        db_module,
        "_backfill_legacy_upload_capture_times",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("capture-time backfill should not rerun")
        ),
    )

    listed, total = list_recordings(settings=cfg)
    assert listed == []
    assert total == 0


def test_init_db_backfills_only_eligible_legacy_upload_rows_and_is_idempotent(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    _init_db_to_version(cfg, 18)

    with sqlite3.connect(cfg.db_path) as conn:
        rows = [
            (
                "rec-legacy-fix",
                "upload",
                "2026-03-06_12_02_16.mp3",
                "2026-03-06T12:02:16Z",
            ),
            (
                "rec-legacy-non-upload",
                "calendar",
                "2026-03-06_12_02_16.mp3",
                "2026-03-06T12:02:16Z",
            ),
            (
                "rec-legacy-no-pattern",
                "upload",
                "meeting.mp3",
                "2026-03-06T12:02:16Z",
            ),
            (
                "rec-legacy-invalid-date",
                "upload",
                "2026-99-06_12_02_16.mp3",
                "2026-99-06T12:02:16Z",
            ),
            (
                "rec-legacy-manual-time",
                "upload",
                "2026-03-06_12_02_16.mp3",
                "2026-03-06T11:02:16Z",
            ),
        ]
        for recording_id, source, source_filename, captured_at in rows:
            conn.execute(
                """
                INSERT INTO recordings (
                    id,
                    source,
                    source_filename,
                    captured_at,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 'Queued', '2026-03-07T00:00:00Z', '2026-03-07T00:00:00Z')
                """,
                (
                    recording_id,
                    source,
                    source_filename,
                    captured_at,
                ),
            )
        conn.commit()

    init_db(cfg)

    fixed = get_recording("rec-legacy-fix", settings=cfg)
    assert fixed is not None
    assert fixed["captured_at"] == "2026-03-06T11:02:16Z"
    assert fixed["captured_at_source"] == "2026-03-06T12:02:16"
    assert fixed["captured_at_timezone"] == "Europe/Rome"
    assert fixed["captured_at_inferred_from_filename"] == 1

    non_upload = get_recording("rec-legacy-non-upload", settings=cfg)
    assert non_upload is not None
    assert non_upload["captured_at"] == "2026-03-06T12:02:16Z"
    assert non_upload["captured_at_source"] is None
    assert non_upload["captured_at_timezone"] is None
    assert non_upload["captured_at_inferred_from_filename"] == 0

    no_pattern = get_recording("rec-legacy-no-pattern", settings=cfg)
    assert no_pattern is not None
    assert no_pattern["captured_at"] == "2026-03-06T12:02:16Z"
    assert no_pattern["captured_at_source"] is None
    assert no_pattern["captured_at_timezone"] is None
    assert no_pattern["captured_at_inferred_from_filename"] == 0

    invalid_date = get_recording("rec-legacy-invalid-date", settings=cfg)
    assert invalid_date is not None
    assert invalid_date["captured_at"] == "2026-99-06T12:02:16Z"
    assert invalid_date["captured_at_source"] is None
    assert invalid_date["captured_at_timezone"] is None
    assert invalid_date["captured_at_inferred_from_filename"] == 0

    manual = get_recording("rec-legacy-manual-time", settings=cfg)
    assert manual is not None
    assert manual["captured_at"] == "2026-03-06T11:02:16Z"
    assert manual["captured_at_source"] is None
    assert manual["captured_at_timezone"] is None
    assert manual["captured_at_inferred_from_filename"] == 0

    assert db_module._backfill_legacy_upload_capture_times(cfg) == 0  # noqa: SLF001
    fixed_again = get_recording("rec-legacy-fix", settings=cfg)
    assert fixed_again == fixed


def test_local_timestamp_display_uses_fixed_utc_value() -> None:
    assert (
        ui_routes._format_local_timestamp("2026-03-06T11:02:16Z")  # noqa: SLF001
        == "2026-03-06 12:02:16 CET"
    )
