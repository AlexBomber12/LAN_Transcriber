ALTER TABLE recordings ADD COLUMN pipeline_stage TEXT;
ALTER TABLE recordings ADD COLUMN pipeline_progress REAL;
ALTER TABLE recordings ADD COLUMN pipeline_updated_at TEXT;
ALTER TABLE recordings ADD COLUMN last_warning TEXT;
