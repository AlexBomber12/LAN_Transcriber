ALTER TABLE speaker_assignments ADD COLUMN review_state TEXT;
ALTER TABLE speaker_assignments ADD COLUMN local_display_name TEXT;

UPDATE speaker_assignments
SET review_state = CASE
    WHEN voice_profile_id IS NOT NULL THEN 'confirmed_canonical'
    WHEN low_confidence = 1 THEN 'system_suggested'
    WHEN TRIM(COALESCE(candidate_matches_json, '')) NOT IN ('', '[]') THEN 'system_suggested'
    ELSE 'kept_unknown'
END
WHERE review_state IS NULL OR TRIM(review_state) = '';

UPDATE speaker_assignments
SET local_display_name = NULL
WHERE TRIM(COALESCE(local_display_name, '')) = '';

ALTER TABLE speaker_assignments RENAME TO speaker_assignments_old;

CREATE TABLE speaker_assignments (
    recording_id TEXT NOT NULL,
    diar_speaker_label TEXT NOT NULL,
    voice_profile_id INTEGER,
    confidence REAL NOT NULL,
    candidate_matches_json TEXT NOT NULL DEFAULT '[]',
    low_confidence INTEGER NOT NULL DEFAULT 0 CHECK(low_confidence IN (0, 1)),
    review_state TEXT NOT NULL DEFAULT 'kept_unknown' CHECK(
        review_state IN (
            'system_suggested',
            'confirmed_canonical',
            'kept_unknown',
            'local_label'
        )
    ),
    local_display_name TEXT,
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
    review_state,
    local_display_name,
    updated_at
)
SELECT
    recording_id,
    diar_speaker_label,
    voice_profile_id,
    confidence,
    candidate_matches_json,
    low_confidence,
    COALESCE(
        NULLIF(TRIM(review_state), ''),
        CASE
            WHEN voice_profile_id IS NOT NULL THEN 'confirmed_canonical'
            WHEN low_confidence = 1 THEN 'system_suggested'
            WHEN TRIM(COALESCE(candidate_matches_json, '')) NOT IN ('', '[]') THEN 'system_suggested'
            ELSE 'kept_unknown'
        END
    ),
    NULLIF(TRIM(local_display_name), ''),
    updated_at
FROM speaker_assignments_old;

DROP TABLE speaker_assignments_old;

CREATE INDEX IF NOT EXISTS idx_speaker_assignments_recording_id
    ON speaker_assignments(recording_id);
