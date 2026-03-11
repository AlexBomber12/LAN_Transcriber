ALTER TABLE recordings
    ADD COLUMN captured_at_source TEXT;

ALTER TABLE recordings
    ADD COLUMN captured_at_timezone TEXT;

ALTER TABLE recordings
    ADD COLUMN captured_at_inferred_from_filename INTEGER NOT NULL DEFAULT 0 CHECK(captured_at_inferred_from_filename IN (0, 1));
