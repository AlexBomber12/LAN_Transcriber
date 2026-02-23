DELETE FROM jobs
WHERE type IN ('stt', 'diarize', 'align', 'language', 'llm', 'metrics')
  AND status = 'queued'
  AND started_at IS NULL
  AND finished_at IS NULL
  AND (
      SELECT (julianday(jobs.created_at) - julianday(recordings.created_at)) * 86400.0
      FROM recordings
      WHERE recordings.id = jobs.recording_id
  ) BETWEEN 0.0 AND 30.0;
