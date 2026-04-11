Run PLANNED PR

PR_ID: PR-FORCE-REPROCESS-01
Branch: pr-force-reprocess-01
Title: Add Force Full Reprocess button that clears derived artifacts and re-runs the entire pipeline from scratch

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

Context
When new pipeline features are deployed (e.g. speaker merge, new LLM fields), old recordings retain stale artifacts. The checkpoint/resume system validates artifact file existence, not content, so new fields never appear unless derived files are manually deleted and the recording is requeued. Currently this requires SSH access and manual Python commands. Operators need a UI button.

The existing requeue API at /api/recordings/{id}/actions/requeue does reset_pipeline_state=True (clears stage DB rows) but does NOT delete derived artifact files. If old files exist and pass validation, stages may still be skipped.

Phase 1 - Inspect and map
Read these files:
- lan_app/api.py: existing api_requeue_recording endpoint (line ~307)
- lan_app/jobs.py: enqueue_recording_job, reset_pipeline_state logic
- lan_app/pipeline_stages.py: validate_stage_artifacts, stage_artifact_paths
- lan_app/worker_tasks.py: _run_precheck_pipeline (line ~3909), how stage_rows drive resume logic (lines ~3930-3959)
- lan_app/templates/ (find the recording detail page where action buttons live, e.g. requeue button)
- lan_app/ui_routes.py (find the route that handles requeue from UI)

Phase 2 - Implement

CHANGE 1: Add API endpoint for force reprocess
In lan_app/api.py, add a new endpoint:

@app.post("/api/recordings/{recording_id}/actions/force-reprocess")
async def api_force_reprocess_recording(recording_id: str) -> dict:

Logic:
- Validate recording exists.
- Delete all files in derived/ EXCEPT audio_sanitized.wav and audio_sanitize.json (sanitized audio is expensive to regenerate and doesn't change).
- Delete subdirectories in derived/ (e.g. snippets/).
- Call enqueue_recording_job with reset_pipeline_state=True.
- Return {"recording_id": ..., "job_id": ..., "reprocessed": True}.
- If a job is already running for this recording, return 409 Conflict.

CHANGE 2: Add helper function for clearing derived artifacts
In lan_app/ops.py (or a suitable location), add:

def clear_derived_artifacts(recording_id: str, *, settings: AppSettings, keep: tuple[str, ...] = ("audio_sanitized.wav", "audio_sanitize.json")) -> list[str]:
    """Delete all derived artifacts except the ones in keep. Returns list of deleted filenames."""

This function encapsulates the logic we did manually via SSH. It handles both files and directories (like snippets/).

CHANGE 3: Add UI button
In the recording detail page (full-page inspector or control center), add a "Force Reprocess" button next to the existing Requeue button. Style it as a destructive action (red outline or warning color). Add a confirmation dialog: "This will delete all processed results and re-run the full pipeline. Continue?"

The button should POST to the new API endpoint via HTMX or fetch, then show a status indicator.

CHANGE 4: Add UI route if needed
If the existing requeue UI route goes through ui_routes.py rather than the API, add a corresponding route that calls clear_derived_artifacts + enqueue.

Phase 3 - Test and verify
- Add API test: POST /api/recordings/{id}/actions/force-reprocess returns 200 and clears derived files.
- Add API test: POST when job already running returns 409.
- Add API test: POST for nonexistent recording returns 404.
- Add test for clear_derived_artifacts: verify it deletes all files except audio_sanitized.wav and audio_sanitize.json, handles missing directories, handles subdirectories.
- Run full CI.

Success criteria:
- Operator can force a full reprocess from the UI without SSH.
- All derived artifacts are deleted before reprocessing.
- audio_sanitized.wav is preserved (no redundant ffmpeg conversion).
- Confirmation dialog prevents accidental reprocessing.
- Running jobs are not interrupted (409 if job active).
- No existing tests break.
