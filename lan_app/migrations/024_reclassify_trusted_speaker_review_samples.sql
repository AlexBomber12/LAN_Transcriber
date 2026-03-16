UPDATE voice_samples
SET sample_source = 'trusted_sample'
WHERE sample_source = 'manual'
  AND TRIM(COALESCE(recording_id, '')) <> ''
  AND TRIM(COALESCE(diar_speaker_label, '')) <> ''
  AND snippet_path LIKE 'recordings/%/derived/snippets/%';
