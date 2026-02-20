from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from .config import AppSettings
from .constants import (
    JOB_STATUSES,
    JOB_STATUS_FAILED,
    JOB_STATUS_FINISHED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_STARTED,
    JOB_TYPES,
    RECORDING_STATUSES,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_QUARANTINE,
)


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


_RECORDING_STATUSES_SQL = _quoted(RECORDING_STATUSES)
_JOB_STATUSES_SQL = _quoted(JOB_STATUSES)
_JOB_TYPES_SQL = _quoted(JOB_TYPES)

_MIGRATIONS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS recordings (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        source_filename TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        duration_sec INTEGER,
        status TEXT NOT NULL CHECK(status IN ({_RECORDING_STATUSES_SQL})),
        quarantine_reason TEXT,
        language_auto TEXT,
        language_override TEXT,
        project_id INTEGER,
        onenote_page_id TEXT,
        drive_file_id TEXT,
        drive_md5 TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        recording_id TEXT NOT NULL,
        type TEXT NOT NULL CHECK(type IN ({_JOB_TYPES_SQL})),
        status TEXT NOT NULL CHECK(status IN ({_JOB_STATUSES_SQL})),
        attempt INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        started_at TEXT,
        finished_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        onenote_section_id TEXT,
        onenote_notebook_id TEXT,
        auto_publish INTEGER NOT NULL DEFAULT 0 CHECK(auto_publish IN (0, 1))
    );

    CREATE TABLE IF NOT EXISTS voice_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        display_name TEXT NOT NULL,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS speaker_assignments (
        recording_id TEXT NOT NULL,
        diar_speaker_label TEXT NOT NULL,
        voice_profile_id INTEGER NOT NULL,
        confidence REAL NOT NULL,
        PRIMARY KEY(recording_id, diar_speaker_label),
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
        FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS calendar_matches (
        recording_id TEXT PRIMARY KEY,
        selected_event_id TEXT,
        selected_confidence REAL,
        candidates_json TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS meeting_metrics (
        recording_id TEXT PRIMARY KEY,
        json TEXT NOT NULL DEFAULT '{{}}',
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS participant_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id TEXT NOT NULL,
        voice_profile_id INTEGER,
        diar_speaker_label TEXT NOT NULL,
        json TEXT NOT NULL DEFAULT '{{}}',
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
        FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL
    );

    CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status);
    CREATE INDEX IF NOT EXISTS idx_recordings_created_at ON recordings(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_jobs_recording_id ON jobs(recording_id);
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
    """,
    """
    ALTER TABLE recordings ADD COLUMN target_summary_language TEXT;
    """,
)

_UNSET = object()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key, value in list(out.items()):
        if value is None:
            continue
        if key.endswith("_json") or key == "json":
            try:
                out[key] = json.loads(value)
            except (TypeError, ValueError):
                pass
    return out


def _validate_recording_status(status: str) -> None:
    if status not in RECORDING_STATUSES:
        raise ValueError(f"Unsupported recording status: {status}")


def _validate_job_status(status: str) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"Unsupported job status: {status}")


def _validate_job_type(job_type: str) -> None:
    if job_type not in JOB_TYPES:
        raise ValueError(f"Unsupported job type: {job_type}")


def db_path(settings: AppSettings | None = None) -> Path:
    cfg = settings or AppSettings()
    return cfg.db_path


def connect(settings: AppSettings | None = None) -> sqlite3.Connection:
    cfg = settings or AppSettings()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(settings: AppSettings | None = None) -> Path:
    cfg = settings or AppSettings()
    with connect(cfg) as conn:
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        for target_version, sql in enumerate(_MIGRATIONS, start=1):
            if target_version <= current_version:
                continue
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {target_version}")
        conn.commit()
    return cfg.db_path


def create_recording(
    recording_id: str,
    source: str,
    source_filename: str,
    *,
    settings: AppSettings | None = None,
    captured_at: str | None = None,
    duration_sec: int | None = None,
    status: str = RECORDING_STATUS_QUEUED,
    quarantine_reason: str | None = None,
    language_auto: str | None = None,
    language_override: str | None = None,
    target_summary_language: str | None = None,
    project_id: int | None = None,
    onenote_page_id: str | None = None,
    drive_file_id: str | None = None,
    drive_md5: str | None = None,
) -> dict[str, Any]:
    init_db(settings)
    _validate_recording_status(status)
    now = _utc_now()
    captured = captured_at or now
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO recordings (
                id, source, source_filename, captured_at, duration_sec, status,
                quarantine_reason, language_auto, language_override, target_summary_language, project_id,
                onenote_page_id, drive_file_id, drive_md5, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recording_id,
                source,
                source_filename,
                captured,
                duration_sec,
                status,
                quarantine_reason,
                language_auto,
                language_override,
                target_summary_language,
                project_id,
                onenote_page_id,
                drive_file_id,
                drive_md5,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def get_recording(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
    return _as_dict(row)


def list_recordings(
    *,
    settings: AppSettings | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    init_db(settings)
    filters: list[str] = []
    params: list[Any] = []
    if status is not None:
        _validate_recording_status(status)
        filters.append("status = ?")
        params.append(status)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(offset, 0)

    with connect(settings) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM recordings {where_sql}",
                params,
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT *
            FROM recordings
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            [*params, safe_limit, safe_offset],
        ).fetchall()
    return [_as_dict(row) or {} for row in rows], total


def set_recording_status(
    recording_id: str,
    status: str,
    *,
    settings: AppSettings | None = None,
    quarantine_reason: str | None = None,
) -> bool:
    init_db(settings)
    _validate_recording_status(status)
    now = _utc_now()
    with connect(settings) as conn:
        updated = conn.execute(
            """
            UPDATE recordings
            SET status = ?, quarantine_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                quarantine_reason if status == RECORDING_STATUS_QUARANTINE else None,
                now,
                recording_id,
            ),
        )
        conn.commit()
    return updated.rowcount > 0


def set_recording_language_settings(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
    language_auto: str | None | object = _UNSET,
    transcript_language_override: str | None | object = _UNSET,
    target_summary_language: str | None | object = _UNSET,
) -> bool:
    init_db(settings)
    now = _utc_now()
    updates: list[str] = []
    params: list[Any] = []

    if language_auto is not _UNSET:
        updates.append("language_auto = ?")
        params.append(language_auto)
    if transcript_language_override is not _UNSET:
        updates.append("language_override = ?")
        params.append(transcript_language_override)
    if target_summary_language is not _UNSET:
        updates.append("target_summary_language = ?")
        params.append(target_summary_language)

    if not updates:
        return False

    updates.append("updated_at = ?")
    params.append(now)
    params.append(recording_id)

    with connect(settings) as conn:
        updated = conn.execute(
            f"""
            UPDATE recordings
            SET {", ".join(updates)}
            WHERE id = ?
            """,
            params,
        )
        conn.commit()
    return updated.rowcount > 0


def delete_recording(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    with connect(settings) as conn:
        deleted = conn.execute(
            "DELETE FROM recordings WHERE id = ?",
            (recording_id,),
        )
        conn.commit()
    return deleted.rowcount > 0


def create_job(
    job_id: str,
    recording_id: str,
    job_type: str,
    *,
    settings: AppSettings | None = None,
    status: str = JOB_STATUS_QUEUED,
    attempt: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    init_db(settings)
    _validate_job_type(job_type)
    _validate_job_status(status)
    now = _utc_now()
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, recording_id, type, status, attempt, error,
                started_at, finished_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                recording_id,
                job_type,
                status,
                attempt,
                error,
                None,
                None,
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def get_job(
    job_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _as_dict(row)


def list_jobs(
    *,
    settings: AppSettings | None = None,
    status: str | None = None,
    recording_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    init_db(settings)
    filters: list[str] = []
    params: list[Any] = []
    if status is not None:
        _validate_job_status(status)
        filters.append("status = ?")
        params.append(status)
    if recording_id is not None:
        filters.append("recording_id = ?")
        params.append(recording_id)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(offset, 0)

    with connect(settings) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM jobs {where_sql}",
                params,
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT *
            FROM jobs
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            [*params, safe_limit, safe_offset],
        ).fetchall()
    return [_as_dict(row) or {} for row in rows], total


def start_job(
    job_id: str,
    *,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    now = _utc_now()
    with connect(settings) as conn:
        row = conn.execute("SELECT attempt FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False
        next_attempt = int(row["attempt"]) + 1
        updated = conn.execute(
            """
            UPDATE jobs
            SET status = ?, attempt = ?, error = NULL, started_at = ?, finished_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (JOB_STATUS_STARTED, next_attempt, now, now, job_id),
        )
        conn.commit()
    return updated.rowcount > 0


def finish_job(
    job_id: str,
    *,
    settings: AppSettings | None = None,
    error: str | None = None,
) -> bool:
    return _set_job_terminal_state(
        job_id=job_id,
        status=JOB_STATUS_FINISHED,
        error=error,
        settings=settings,
    )


def fail_job(
    job_id: str,
    error: str,
    *,
    settings: AppSettings | None = None,
) -> bool:
    return _set_job_terminal_state(
        job_id=job_id,
        status=JOB_STATUS_FAILED,
        error=error,
        settings=settings,
    )


def list_projects(
    *,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
    return [_as_dict(row) or {} for row in rows]


def create_project(
    name: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    init_db(settings)
    with connect(settings) as conn:
        cursor = conn.execute("INSERT INTO projects (name) VALUES (?)", (name,))
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def delete_project(
    project_id: int,
    *,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    with connect(settings) as conn:
        deleted = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
    return deleted.rowcount > 0


def list_voice_profiles(
    *,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute(
            "SELECT * FROM voice_profiles ORDER BY display_name"
        ).fetchall()
    return [_as_dict(row) or {} for row in rows]


def create_voice_profile(
    display_name: str,
    notes: str | None = None,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    init_db(settings)
    with connect(settings) as conn:
        cursor = conn.execute(
            "INSERT INTO voice_profiles (display_name, notes) VALUES (?, ?)",
            (display_name, notes),
        )
        row = conn.execute(
            "SELECT * FROM voice_profiles WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def delete_voice_profile(
    profile_id: int,
    *,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    with connect(settings) as conn:
        deleted = conn.execute(
            "DELETE FROM voice_profiles WHERE id = ?", (profile_id,)
        )
        conn.commit()
    return deleted.rowcount > 0


def get_calendar_match(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM calendar_matches WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
    return _as_dict(row)


def upsert_calendar_match(
    *,
    recording_id: str,
    candidates: list[dict[str, Any]],
    selected_event_id: str | None,
    selected_confidence: float | None,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    init_db(settings)
    candidates_json = json.dumps(candidates, ensure_ascii=True)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO calendar_matches (
                recording_id,
                selected_event_id,
                selected_confidence,
                candidates_json
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                selected_event_id = excluded.selected_event_id,
                selected_confidence = excluded.selected_confidence,
                candidates_json = excluded.candidates_json
            """,
            (
                recording_id,
                selected_event_id,
                selected_confidence,
                candidates_json,
            ),
        )
        row = conn.execute(
            "SELECT * FROM calendar_matches WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def set_calendar_match_selection(
    *,
    recording_id: str,
    event_id: str | None,
    selected_confidence: float | None,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    init_db(settings)
    with connect(settings) as conn:
        existing = conn.execute(
            "SELECT candidates_json FROM calendar_matches WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO calendar_matches (
                    recording_id,
                    selected_event_id,
                    selected_confidence,
                    candidates_json
                )
                VALUES (?, ?, ?, '[]')
                """,
                (
                    recording_id,
                    event_id,
                    selected_confidence,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE calendar_matches
                SET selected_event_id = ?, selected_confidence = ?
                WHERE recording_id = ?
                """,
                (
                    event_id,
                    selected_confidence,
                    recording_id,
                ),
            )
        row = conn.execute(
            "SELECT * FROM calendar_matches WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def _set_job_terminal_state(
    *,
    job_id: str,
    status: str,
    error: str | None,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    if status not in (JOB_STATUS_FINISHED, JOB_STATUS_FAILED):
        raise ValueError(f"Unsupported terminal state: {status}")
    now = _utc_now()
    with connect(settings) as conn:
        updated = conn.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, finished_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error, now, now, job_id),
        )
        conn.commit()
    return updated.rowcount > 0


__all__ = [
    "db_path",
    "connect",
    "init_db",
    "create_recording",
    "get_recording",
    "list_recordings",
    "set_recording_status",
    "set_recording_language_settings",
    "delete_recording",
    "create_job",
    "get_job",
    "list_jobs",
    "start_job",
    "finish_job",
    "fail_job",
    "list_projects",
    "create_project",
    "delete_project",
    "list_voice_profiles",
    "create_voice_profile",
    "delete_voice_profile",
    "get_calendar_match",
    "upsert_calendar_match",
    "set_calendar_match_selection",
]
