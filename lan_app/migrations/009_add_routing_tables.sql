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
