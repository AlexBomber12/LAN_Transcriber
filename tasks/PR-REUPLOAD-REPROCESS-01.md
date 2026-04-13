Run PLANNED PR

PR_ID: PR-REUPLOAD-REPROCESS-01
Branch: pr-reupload-reprocess-01
Title: Force full reprocess when re-uploading an existing audio file

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
When a user uploads an audio file that matches an existing recording (by hash/fingerprint), the system reuses the existing recording and its old derived artifacts. The checkpoint/resume logic sees valid artifact files and skips stages. This means new pipeline features (e.g. task=transcribe, speaker merge) never apply to re-uploaded recordings. Users expect re-upload to mean "process again with latest pipeline".

Phase 1 - Inspect
Read:
- lan_app/uploads.py: how uploads detect duplicate files and map to existing recordings
- lan_app/api.py: upload endpoint
- lan_app/worker_tasks.py: _run_precheck_pipeline, how stage validation skips stages with existing artifacts

Phase 2 - Implement

CHANGE 1: Clear derived artifacts on re-upload match
When upload detects that a file matches an existing recording:
- Clear all derived artifacts EXCEPT audio_sanitized.wav and audio_sanitize.json (same logic as Force Reprocess)
- Reset pipeline stage rows in DB
- Set status to Queued
- Enqueue processing job

This ensures the recording is fully reprocessed with current pipeline code.

CHANGE 2: Log re-upload event
Log at INFO level: "re-upload detected for {recording_id}, clearing derived artifacts for full reprocess"

Phase 3 - Test and verify
- Run full CI.
- Upload a file, wait for completion. Upload the same file again. Verify all stages run from scratch with new artifacts.

Success criteria:
- Re-uploading a file triggers full reprocessing, not just reuse of old artifacts.
- audio_sanitized.wav is preserved (no redundant ffmpeg).
- No existing tests break.
