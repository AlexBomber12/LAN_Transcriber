CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    duration_sec INTEGER,
    status TEXT NOT NULL CHECK(status IN ('Queued', 'Processing', 'NeedsReview', 'Ready', 'Published', 'Quarantine', 'Failed')),
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
    type TEXT NOT NULL CHECK(type IN ('ingest', 'precheck', 'stt', 'diarize', 'align', 'language', 'llm', 'metrics', 'publish', 'cleanup')),
    status TEXT NOT NULL CHECK(status IN ('queued', 'started', 'finished', 'failed')),
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
    json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS participant_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL,
    voice_profile_id INTEGER,
    diar_speaker_label TEXT NOT NULL,
    json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
    FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status);
CREATE INDEX IF NOT EXISTS idx_recordings_created_at ON recordings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_recording_id ON jobs(recording_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
