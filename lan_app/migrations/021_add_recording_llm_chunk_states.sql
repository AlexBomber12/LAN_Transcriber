CREATE TABLE IF NOT EXISTS recording_llm_chunk_states (
    recording_id TEXT NOT NULL,
    chunk_group TEXT NOT NULL,
    chunk_index TEXT NOT NULL,
    chunk_total INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('planned', 'running', 'completed', 'failed', 'cancelled', 'split')),
    attempt INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    duration_ms INTEGER,
    error_code TEXT,
    error_text TEXT,
    parent_chunk_index TEXT,
    metadata_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (recording_id, chunk_group, chunk_index),
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_recording_llm_chunk_states_recording_group
ON recording_llm_chunk_states(recording_id, chunk_group);
