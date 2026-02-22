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
    RECORDING_STATUS_PUBLISHED,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_QUARANTINE,
)


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


_RECORDING_STATUSES_SQL = _quoted(RECORDING_STATUSES)
_JOB_STATUSES_SQL = _quoted(JOB_STATUSES)
_JOB_TYPES_SQL = _quoted(JOB_TYPES)
_PROJECT_ASSIGNMENT_SOURCES = {"manual", "auto"}

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
    """
    CREATE TABLE IF NOT EXISTS voice_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        voice_profile_id INTEGER NOT NULL,
        recording_id TEXT,
        diar_speaker_label TEXT,
        snippet_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE,
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_speaker_assignments_recording_id
        ON speaker_assignments(recording_id);
    CREATE INDEX IF NOT EXISTS idx_voice_samples_profile_id ON voice_samples(voice_profile_id);
    CREATE INDEX IF NOT EXISTS idx_voice_samples_recording_id ON voice_samples(recording_id);
    """,
    """
    ALTER TABLE voice_samples RENAME TO voice_samples_old;

    CREATE TABLE voice_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        voice_profile_id INTEGER NOT NULL,
        recording_id TEXT,
        diar_speaker_label TEXT,
        snippet_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE,
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
    );

    INSERT INTO voice_samples (
        id,
        voice_profile_id,
        recording_id,
        diar_speaker_label,
        snippet_path,
        created_at
    )
    SELECT
        id,
        voice_profile_id,
        recording_id,
        diar_speaker_label,
        snippet_path,
        created_at
    FROM voice_samples_old;

    DROP TABLE voice_samples_old;

    CREATE INDEX IF NOT EXISTS idx_voice_samples_profile_id ON voice_samples(voice_profile_id);
    CREATE INDEX IF NOT EXISTS idx_voice_samples_recording_id ON voice_samples(recording_id);
    """,
    """
    ALTER TABLE recordings ADD COLUMN onenote_page_url TEXT;
    """,
    """
    ALTER TABLE recordings ADD COLUMN suggested_project_id INTEGER;
    """,
    """
    ALTER TABLE recordings ADD COLUMN routing_confidence REAL;
    """,
    """
    ALTER TABLE recordings ADD COLUMN routing_rationale_json TEXT NOT NULL DEFAULT '[]';
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_recordings_suggested_project_id
        ON recordings(suggested_project_id);

    CREATE TABLE IF NOT EXISTS routing_training_examples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id TEXT NOT NULL,
        project_id INTEGER NOT NULL,
        calendar_subject_tokens_json TEXT NOT NULL DEFAULT '[]',
        tags_json TEXT NOT NULL DEFAULT '[]',
        voice_profile_ids_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_routing_training_examples_project_id
        ON routing_training_examples(project_id);
    CREATE INDEX IF NOT EXISTS idx_routing_training_examples_recording_id
        ON routing_training_examples(recording_id);

    CREATE TABLE IF NOT EXISTS routing_project_keyword_weights (
        project_id INTEGER NOT NULL,
        keyword TEXT NOT NULL,
        weight REAL NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(project_id, keyword),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_routing_project_keyword_weights_project_id
        ON routing_project_keyword_weights(project_id);
    """,
    """
    ALTER TABLE recordings ADD COLUMN project_assignment_source TEXT;

    UPDATE recordings
    SET project_assignment_source = 'manual'
    WHERE project_id IS NOT NULL AND project_assignment_source IS NULL;
    """,
    """
    DELETE FROM jobs
    WHERE type IN ('stt', 'diarize', 'align', 'language', 'llm', 'metrics')
      AND status = 'queued'
      AND started_at IS NULL
      AND finished_at IS NULL;
    """,
)

_UNSET = object()

_RECORDING_SELECT_SQL = """
SELECT
    r.*,
    p.name AS project_name,
    sp.name AS suggested_project_name
FROM recordings AS r
LEFT JOIN projects AS p ON p.id = r.project_id
LEFT JOIN projects AS sp ON sp.id = r.suggested_project_id
"""


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


def _normalise_project_assignment_source(source: str | None) -> str | None:
    if source is None:
        return None
    value = str(source).strip().lower()
    if not value:
        return None
    if value not in _PROJECT_ASSIGNMENT_SOURCES:
        options = ", ".join(sorted(_PROJECT_ASSIGNMENT_SOURCES))
        raise ValueError(f"Unsupported project assignment source: {value} ({options})")
    return value


def _normalise_keyword(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


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
    project_assignment_source: str | None = None,
    onenote_page_id: str | None = None,
    drive_file_id: str | None = None,
    drive_md5: str | None = None,
) -> dict[str, Any]:
    init_db(settings)
    _validate_recording_status(status)
    now = _utc_now()
    captured = captured_at or now
    resolved_assignment_source = _normalise_project_assignment_source(
        project_assignment_source
    )
    if project_id is None:
        resolved_assignment_source = None
    elif resolved_assignment_source is None:
        resolved_assignment_source = "manual"
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO recordings (
                id, source, source_filename, captured_at, duration_sec, status,
                quarantine_reason, language_auto, language_override, target_summary_language, project_id,
                project_assignment_source,
                onenote_page_id, drive_file_id, drive_md5, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                resolved_assignment_source,
                onenote_page_id,
                drive_file_id,
                drive_md5,
                now,
                now,
            ),
        )
        row = conn.execute(
            f"{_RECORDING_SELECT_SQL} WHERE r.id = ?",
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
            f"{_RECORDING_SELECT_SQL} WHERE r.id = ?",
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
        filters.append("r.status = ?")
        params.append(status)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(offset, 0)

    with connect(settings) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM recordings AS r {where_sql}",
                params,
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            {_RECORDING_SELECT_SQL}
            {where_sql}
            ORDER BY r.created_at DESC
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


def set_recording_project(
    recording_id: str,
    project_id: int | None,
    *,
    settings: AppSettings | None = None,
    assignment_source: str | None = None,
) -> bool:
    init_db(settings)
    now = _utc_now()
    resolved_project_id = None if project_id is None else int(project_id)
    resolved_assignment_source = _normalise_project_assignment_source(assignment_source)
    if resolved_project_id is None:
        resolved_assignment_source = None
    elif resolved_assignment_source is None:
        resolved_assignment_source = "manual"
    with connect(settings) as conn:
        updated = conn.execute(
            """
            UPDATE recordings
            SET project_id = ?, project_assignment_source = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                resolved_project_id,
                resolved_assignment_source,
                now,
                recording_id,
            ),
        )
        conn.commit()
    return updated.rowcount > 0


def set_recording_routing_suggestion(
    recording_id: str,
    *,
    suggested_project_id: int | None,
    routing_confidence: float | None,
    routing_rationale: list[str] | None,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    now = _utc_now()
    resolved_project_id = (
        None if suggested_project_id is None else int(suggested_project_id)
    )
    if routing_confidence is None:
        confidence_value = None
    else:
        confidence_value = max(0.0, min(float(routing_confidence), 1.0))
    rationale_rows = [str(item).strip() for item in (routing_rationale or [])]
    rationale = [item for item in rationale_rows if item]
    with connect(settings) as conn:
        updated = conn.execute(
            """
            UPDATE recordings
            SET suggested_project_id = ?, routing_confidence = ?, routing_rationale_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                resolved_project_id,
                confidence_value,
                json.dumps(rationale, ensure_ascii=True),
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


def set_recording_publish_result(
    recording_id: str,
    *,
    onenote_page_id: str,
    onenote_page_url: str | None = None,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    page_id = str(onenote_page_id).strip()
    if not page_id:
        raise ValueError("onenote_page_id is required")
    page_url = str(onenote_page_url).strip() if onenote_page_url is not None else ""
    now = _utc_now()
    with connect(settings) as conn:
        updated = conn.execute(
            """
            UPDATE recordings
            SET status = ?, quarantine_reason = NULL, onenote_page_id = ?, onenote_page_url = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                RECORDING_STATUS_PUBLISHED,
                page_id,
                page_url or None,
                now,
                recording_id,
            ),
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


def requeue_job(
    job_id: str,
    *,
    error: str | None = None,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    now = _utc_now()
    with connect(settings) as conn:
        updated = conn.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, started_at = NULL, finished_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (JOB_STATUS_QUEUED, error, now, job_id),
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


def get_project(
    project_id: int,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    return _as_dict(row)


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


def update_project_onenote_mapping(
    project_id: int,
    *,
    onenote_notebook_id: str | None = None,
    onenote_section_id: str | None = None,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    notebook_id = str(onenote_notebook_id or "").strip() or None
    section_id = str(onenote_section_id or "").strip() or None
    with connect(settings) as conn:
        updated = conn.execute(
            """
            UPDATE projects
            SET onenote_notebook_id = ?, onenote_section_id = ?
            WHERE id = ?
            """,
            (
                notebook_id,
                section_id,
                project_id,
            ),
        )
        if updated.rowcount < 1:
            conn.commit()
            return None
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def delete_project(
    project_id: int,
    *,
    settings: AppSettings | None = None,
) -> bool:
    init_db(settings)
    target_project_id = int(project_id)
    now = _utc_now()
    with connect(settings) as conn:
        conn.execute(
            """
            UPDATE recordings
            SET
                suggested_project_id = NULL,
                routing_confidence = NULL,
                routing_rationale_json = '[]',
                updated_at = ?
            WHERE suggested_project_id = ?
            """,
            (now, target_project_id),
        )
        deleted = conn.execute("DELETE FROM projects WHERE id = ?", (target_project_id,))
        conn.commit()
    return deleted.rowcount > 0


def create_routing_training_example(
    *,
    recording_id: str,
    project_id: int,
    calendar_subject_tokens: list[str],
    tags: list[str],
    voice_profile_ids: list[int],
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    init_db(settings)
    now = _utc_now()
    normalized_calendar = sorted(
        {_normalise_keyword(token) for token in calendar_subject_tokens if _normalise_keyword(token)}
    )
    normalized_tags = sorted({_normalise_keyword(token) for token in tags if _normalise_keyword(token)})
    normalized_voice_ids = sorted({int(value) for value in voice_profile_ids})
    with connect(settings) as conn:
        cursor = conn.execute(
            """
            INSERT INTO routing_training_examples (
                recording_id,
                project_id,
                calendar_subject_tokens_json,
                tags_json,
                voice_profile_ids_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                recording_id,
                int(project_id),
                json.dumps(normalized_calendar, ensure_ascii=True),
                json.dumps(normalized_tags, ensure_ascii=True),
                json.dumps(normalized_voice_ids, ensure_ascii=True),
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM routing_training_examples WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def count_routing_training_examples(
    *,
    project_id: int | None = None,
    settings: AppSettings | None = None,
) -> int:
    init_db(settings)
    clauses: list[str] = []
    params: list[Any] = []
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(int(project_id))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(settings) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM routing_training_examples {where_sql}",
            tuple(params),
        ).fetchone()
    return int(row[0]) if row is not None else 0


def increment_project_keyword_weights(
    *,
    project_id: int,
    keyword_deltas: dict[str, float],
    settings: AppSettings | None = None,
) -> int:
    init_db(settings)
    now = _utc_now()
    updates = 0
    with connect(settings) as conn:
        for raw_keyword, raw_delta in keyword_deltas.items():
            keyword = _normalise_keyword(raw_keyword)
            if not keyword:
                continue
            delta = float(raw_delta)
            if delta == 0.0:
                continue
            conn.execute(
                """
                INSERT INTO routing_project_keyword_weights (
                    project_id,
                    keyword,
                    weight,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id, keyword) DO UPDATE SET
                    weight = routing_project_keyword_weights.weight + excluded.weight,
                    updated_at = excluded.updated_at
                """,
                (
                    int(project_id),
                    keyword,
                    delta,
                    now,
                ),
            )
            updates += 1
        conn.commit()
    return updates


def list_project_keyword_weights(
    *,
    project_id: int | None = None,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    clauses: list[str] = []
    params: list[Any] = []
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(int(project_id))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM routing_project_keyword_weights
            {where_sql}
            ORDER BY project_id, keyword
            """,
            tuple(params),
        ).fetchall()
    return [_as_dict(row) or {} for row in rows]


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


def list_speaker_assignments(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute(
            """
            SELECT
                sa.recording_id,
                sa.diar_speaker_label,
                sa.voice_profile_id,
                sa.confidence,
                vp.display_name AS voice_profile_name
            FROM speaker_assignments AS sa
            LEFT JOIN voice_profiles AS vp ON vp.id = sa.voice_profile_id
            WHERE sa.recording_id = ?
            ORDER BY sa.diar_speaker_label
            """,
            (recording_id,),
        ).fetchall()
    return [_as_dict(row) or {} for row in rows]


def set_speaker_assignment(
    *,
    recording_id: str,
    diar_speaker_label: str,
    voice_profile_id: int | None,
    confidence: float = 1.0,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    diar_label = str(diar_speaker_label).strip()
    if not diar_label:
        raise ValueError("diar_speaker_label is required")
    with connect(settings) as conn:
        if voice_profile_id is None:
            conn.execute(
                """
                DELETE FROM speaker_assignments
                WHERE recording_id = ? AND diar_speaker_label = ?
                """,
                (recording_id, diar_label),
            )
            conn.commit()
            return None
        profile_id = int(voice_profile_id)
        score = max(0.0, min(float(confidence), 1.0))
        conn.execute(
            """
            INSERT INTO speaker_assignments (
                recording_id,
                diar_speaker_label,
                voice_profile_id,
                confidence
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(recording_id, diar_speaker_label) DO UPDATE SET
                voice_profile_id = excluded.voice_profile_id,
                confidence = excluded.confidence
            """,
            (recording_id, diar_label, profile_id, score),
        )
        row = conn.execute(
            """
            SELECT
                sa.recording_id,
                sa.diar_speaker_label,
                sa.voice_profile_id,
                sa.confidence,
                vp.display_name AS voice_profile_name
            FROM speaker_assignments AS sa
            LEFT JOIN voice_profiles AS vp ON vp.id = sa.voice_profile_id
            WHERE sa.recording_id = ? AND sa.diar_speaker_label = ?
            """,
            (recording_id, diar_label),
        ).fetchone()
        conn.commit()
    return _as_dict(row)


def list_voice_samples(
    *,
    voice_profile_id: int | None = None,
    recording_id: str | None = None,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    clauses: list[str] = []
    params: list[Any] = []
    if voice_profile_id is not None:
        clauses.append("vs.voice_profile_id = ?")
        params.append(int(voice_profile_id))
    if recording_id is not None:
        clauses.append("vs.recording_id = ?")
        params.append(recording_id)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT
            vs.*,
            vp.display_name AS voice_profile_name
        FROM voice_samples AS vs
        LEFT JOIN voice_profiles AS vp ON vp.id = vs.voice_profile_id
        {where_sql}
        ORDER BY vs.created_at DESC, vs.id DESC
    """
    with connect(settings) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_as_dict(row) or {} for row in rows]


def create_voice_sample(
    *,
    voice_profile_id: int,
    snippet_path: str,
    recording_id: str | None = None,
    diar_speaker_label: str | None = None,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    init_db(settings)
    snippet = str(snippet_path).strip()
    if not snippet:
        raise ValueError("snippet_path is required")
    clean_recording = str(recording_id).strip() if recording_id is not None else None
    clean_label = str(diar_speaker_label).strip() if diar_speaker_label is not None else None
    now = _utc_now()
    with connect(settings) as conn:
        cursor = conn.execute(
            """
            INSERT INTO voice_samples (
                voice_profile_id,
                recording_id,
                diar_speaker_label,
                snippet_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                int(voice_profile_id),
                clean_recording or None,
                clean_label or None,
                snippet,
                now,
            ),
        )
        row = conn.execute(
            """
            SELECT
                vs.*,
                vp.display_name AS voice_profile_name
            FROM voice_samples AS vs
            LEFT JOIN voice_profiles AS vp ON vp.id = vs.voice_profile_id
            WHERE vs.id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def get_voice_sample(
    sample_id: int,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute(
            """
            SELECT
                vs.*,
                vp.display_name AS voice_profile_name
            FROM voice_samples AS vs
            LEFT JOIN voice_profiles AS vp ON vp.id = vs.voice_profile_id
            WHERE vs.id = ?
            """,
            (sample_id,),
        ).fetchone()
    return _as_dict(row)


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


def get_meeting_metrics(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT * FROM meeting_metrics WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
    return _as_dict(row)


def upsert_meeting_metrics(
    *,
    recording_id: str,
    payload: dict[str, Any],
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    init_db(settings)
    encoded = json.dumps(payload, ensure_ascii=True)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT INTO meeting_metrics (recording_id, json)
            VALUES (?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                json = excluded.json
            """,
            (recording_id, encoded),
        )
        row = conn.execute(
            "SELECT * FROM meeting_metrics WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
        conn.commit()
    return _as_dict(row) or {}


def list_participant_metrics(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM participant_metrics
            WHERE recording_id = ?
            ORDER BY diar_speaker_label, id
            """,
            (recording_id,),
        ).fetchall()
    return [_as_dict(row) or {} for row in rows]


def replace_participant_metrics(
    *,
    recording_id: str,
    rows: list[dict[str, Any]],
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    with connect(settings) as conn:
        conn.execute(
            "DELETE FROM participant_metrics WHERE recording_id = ?",
            (recording_id,),
        )
        for row in rows:
            diar_label = str(row.get("diar_speaker_label") or "").strip()
            if not diar_label:
                continue
            voice_profile_id_raw = row.get("voice_profile_id")
            if voice_profile_id_raw is None:
                voice_profile_id = None
            else:
                try:
                    voice_profile_id = int(voice_profile_id_raw)
                except (TypeError, ValueError):
                    voice_profile_id = None
            payload = row.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            conn.execute(
                """
                INSERT INTO participant_metrics (
                    recording_id,
                    voice_profile_id,
                    diar_speaker_label,
                    json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    recording_id,
                    voice_profile_id,
                    diar_label,
                    json.dumps(payload, ensure_ascii=True),
                ),
            )
        out_rows = conn.execute(
            """
            SELECT *
            FROM participant_metrics
            WHERE recording_id = ?
            ORDER BY diar_speaker_label, id
            """,
            (recording_id,),
        ).fetchall()
        conn.commit()
    return [_as_dict(row) or {} for row in out_rows]


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
    "set_recording_project",
    "set_recording_routing_suggestion",
    "set_recording_language_settings",
    "set_recording_publish_result",
    "delete_recording",
    "create_job",
    "get_job",
    "list_jobs",
    "start_job",
    "requeue_job",
    "finish_job",
    "fail_job",
    "list_projects",
    "get_project",
    "create_project",
    "update_project_onenote_mapping",
    "delete_project",
    "create_routing_training_example",
    "count_routing_training_examples",
    "increment_project_keyword_weights",
    "list_project_keyword_weights",
    "list_voice_profiles",
    "create_voice_profile",
    "delete_voice_profile",
    "list_speaker_assignments",
    "set_speaker_assignment",
    "list_voice_samples",
    "create_voice_sample",
    "get_voice_sample",
    "get_calendar_match",
    "upsert_calendar_match",
    "set_calendar_match_selection",
    "get_meeting_metrics",
    "upsert_meeting_metrics",
    "list_participant_metrics",
    "replace_participant_metrics",
]
