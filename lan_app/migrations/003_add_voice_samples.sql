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
