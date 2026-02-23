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
