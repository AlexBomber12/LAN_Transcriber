CREATE TABLE IF NOT EXISTS recording_pipeline_stages (
    recording_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    stage_order INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'cancelled')),
    attempt INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    duration_ms INTEGER,
    error_code TEXT,
    error_text TEXT,
    metadata_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (recording_id, stage_name),
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_recording_pipeline_stages_recording_order
ON recording_pipeline_stages(recording_id, stage_order);
