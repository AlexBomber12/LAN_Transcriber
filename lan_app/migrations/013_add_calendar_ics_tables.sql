CREATE TABLE IF NOT EXISTS calendar_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('url', 'file')),
    url TEXT,
    file_ics TEXT,
    created_at TEXT NOT NULL,
    last_synced_at TEXT,
    last_error TEXT,
    CHECK(
        (kind = 'url' AND url IS NOT NULL AND LENGTH(TRIM(url)) > 0)
        OR (kind = 'file' AND file_ics IS NOT NULL AND LENGTH(file_ics) > 0)
    )
);

CREATE TABLE IF NOT EXISTS calendar_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    uid TEXT NOT NULL,
    starts_at TEXT NOT NULL,
    ends_at TEXT NOT NULL,
    all_day INTEGER NOT NULL CHECK(all_day IN (0, 1)),
    summary TEXT,
    description TEXT,
    location TEXT,
    organizer TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES calendar_sources(id) ON DELETE CASCADE,
    UNIQUE(source_id, uid, starts_at)
);

CREATE INDEX IF NOT EXISTS idx_calendar_events_source_starts
    ON calendar_events(source_id, starts_at);
CREATE INDEX IF NOT EXISTS idx_calendar_events_starts
    ON calendar_events(starts_at);
