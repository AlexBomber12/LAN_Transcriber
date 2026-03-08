ALTER TABLE voice_profiles ADD COLUMN created_at TEXT;
ALTER TABLE voice_profiles ADD COLUMN updated_at TEXT;
ALTER TABLE voice_profiles ADD COLUMN merged_into_voice_profile_id INTEGER;
ALTER TABLE voice_profiles ADD COLUMN merged_at TEXT;

UPDATE voice_profiles
SET created_at = COALESCE(
    created_at,
    (
        SELECT MIN(vs.created_at)
        FROM voice_samples AS vs
        WHERE vs.voice_profile_id = voice_profiles.id
    ),
    '1970-01-01T00:00:00Z'
);

UPDATE voice_profiles
SET updated_at = COALESCE(updated_at, created_at, '1970-01-01T00:00:00Z');

CREATE INDEX IF NOT EXISTS idx_voice_profiles_merged_into_profile_id
    ON voice_profiles(merged_into_voice_profile_id);

ALTER TABLE speaker_assignments RENAME TO speaker_assignments_old;

CREATE TABLE speaker_assignments (
    recording_id TEXT NOT NULL,
    diar_speaker_label TEXT NOT NULL,
    voice_profile_id INTEGER,
    confidence REAL NOT NULL,
    candidate_matches_json TEXT NOT NULL DEFAULT '[]',
    low_confidence INTEGER NOT NULL DEFAULT 0 CHECK(low_confidence IN (0, 1)),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(recording_id, diar_speaker_label),
    FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
    FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL
);

INSERT INTO speaker_assignments (
    recording_id,
    diar_speaker_label,
    voice_profile_id,
    confidence,
    candidate_matches_json,
    low_confidence,
    updated_at
)
SELECT
    recording_id,
    diar_speaker_label,
    voice_profile_id,
    confidence,
    '[]',
    0,
    COALESCE(
        (
            SELECT r.updated_at
            FROM recordings AS r
            WHERE r.id = speaker_assignments_old.recording_id
        ),
        '1970-01-01T00:00:00Z'
    )
FROM speaker_assignments_old;

DROP TABLE speaker_assignments_old;

CREATE INDEX IF NOT EXISTS idx_speaker_assignments_recording_id
    ON speaker_assignments(recording_id);

ALTER TABLE voice_samples RENAME TO voice_samples_old;

CREATE TABLE voice_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    voice_profile_id INTEGER,
    recording_id TEXT,
    diar_speaker_label TEXT,
    snippet_path TEXT NOT NULL,
    sample_source TEXT NOT NULL DEFAULT 'manual',
    sample_start_sec REAL,
    sample_end_sec REAL,
    embedding_json TEXT,
    candidate_matches_json TEXT NOT NULL DEFAULT '[]',
    needs_review INTEGER NOT NULL DEFAULT 0 CHECK(needs_review IN (0, 1)),
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL,
    FOREIGN KEY(recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

INSERT INTO voice_samples (
    id,
    voice_profile_id,
    recording_id,
    diar_speaker_label,
    snippet_path,
    sample_source,
    sample_start_sec,
    sample_end_sec,
    embedding_json,
    candidate_matches_json,
    needs_review,
    confidence,
    created_at
)
SELECT
    id,
    voice_profile_id,
    recording_id,
    diar_speaker_label,
    snippet_path,
    'manual',
    NULL,
    NULL,
    NULL,
    '[]',
    0,
    CASE
        WHEN voice_profile_id IS NULL THEN NULL
        ELSE 1.0
    END,
    created_at
FROM voice_samples_old;

DROP TABLE voice_samples_old;

CREATE INDEX IF NOT EXISTS idx_voice_samples_profile_id ON voice_samples(voice_profile_id);
CREATE INDEX IF NOT EXISTS idx_voice_samples_recording_id ON voice_samples(recording_id);
