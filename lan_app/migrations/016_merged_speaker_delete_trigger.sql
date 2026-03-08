CREATE TRIGGER IF NOT EXISTS trg_voice_profiles_clear_merged_reference_on_delete
AFTER DELETE ON voice_profiles
BEGIN
    UPDATE voice_profiles
    SET
        merged_into_voice_profile_id = NULL,
        merged_at = NULL,
        updated_at = COALESCE(updated_at, OLD.updated_at, '1970-01-01T00:00:00Z')
    WHERE merged_into_voice_profile_id = OLD.id;
END;
