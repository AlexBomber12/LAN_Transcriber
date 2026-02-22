PR-JOB-MODEL-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/job-model-01-single-pipeline-job
PR title: PR-JOB-MODEL-01 Single job pipeline model
Base branch: main

Goal: Switch to the "1 job = full pipeline" model. Each recording must have exactly 1 queue job that runs the full pipeline and produces all derived artifacts. Stop creating DB-only placeholder jobs for stt, diarize, align, language, llm, metrics. Prevent accidental requeue of legacy job types from API/UI. If legacy job types are somehow executed, they must not change recording status and must be clearly marked as unsupported.

Non-goals:
- Do not refactor the pipeline internals.
- Do not remove legacy job type constants from DB schema or JOB_TYPES (keep compatibility).
- Do not introduce new queue chaining.

Changes required:

1) Google Drive ingest creates exactly 1 job per new recording
- File: lan_app/gdrive.py
- Update _PIPELINE_STEPS to contain only 1 element: JOB_TYPE_PRECHECK.
- Rewrite _enqueue_pipeline_jobs() to enqueue only 1 job via enqueue_recording_job(recording_id, job_type=JOB_TYPE_PRECHECK, settings=settings) and return [job_id].
- Remove DB placeholder creation for the remaining steps.
- Keep ingest_once() response field jobs_created accurate (must be 1).
- Keep existing behavior for captured_at parsing, download cleanup, and idempotency.

2) Worker must not apply status transitions for legacy job types
- File: lan_app/worker_tasks.py
- In process_job():
  - Keep validation that job_type is in JOB_TYPES.
  - If job_type is not JOB_TYPE_PRECHECK:
    - Start the DB job (start_job) so attempt/started_at are recorded.
    - Append a step log line stating this job type is unsupported under single-job mode and that the caller must use precheck requeue.
    - Mark the DB job failed (fail_job) with an explicit error like "unsupported legacy job type under single-job pipeline: <type>".
    - Do NOT call set_recording_status at all in this branch (recording status must remain unchanged).
    - Return a small dict payload such as status="ignored" and include job_id/recording_id/job_type.
  - Only for JOB_TYPE_PRECHECK follow the normal behavior: set recording Processing, run _run_precheck_pipeline (which already runs the full pipeline), then set final status (Ready, NeedsReview, Quarantine) and finish_job.

3) API requeue must only allow the pipeline job type
- File: lan_app/api.py
- Keep RequeueAction.job_type for backward compatibility with existing clients/tests, but enforce that only DEFAULT_REQUEUE_JOB_TYPE (precheck) is accepted.
- In api_requeue_recording():
  - If payload.job_type is not DEFAULT_REQUEUE_JOB_TYPE: return 422 with a clear message like "Only precheck is supported in single-job pipeline mode".
  - Enqueue only DEFAULT_REQUEUE_JOB_TYPE.
  - Response remains {recording_id, job_id, job_type}.

4) UI retry must requeue the pipeline job, not the failed job type
- File: lan_app/ui_routes.py
- Endpoint /ui/recordings/{recording_id}/jobs/{job_id}/retry currently requeues with job_type=failed_job.type.
- Change it to always enqueue DEFAULT_REQUEUE_JOB_TYPE.
- Optional: if failed job type is not DEFAULT_REQUEUE_JOB_TYPE, add an error flash-style HTMLResponse explaining that only pipeline requeue is supported, but still allow requeue to proceed.

5) Clean up existing placeholder jobs created by older ingest runs
- File: lan_app/db.py
- Append a new migration entry at the end of _MIGRATIONS that deletes stale DB-only placeholder jobs:
  - Delete rows from jobs where type in ('stt','diarize','align','language','llm','metrics') AND status='queued' AND started_at IS NULL AND finished_at IS NULL.
  - Keep the migration safe and idempotent.
- Do not delete precheck jobs.

6) Tests
- Ensure existing tests remain green. The gdrive tests import _PIPELINE_STEPS and compare counts to len(_PIPELINE_STEPS), so updating _PIPELINE_STEPS to only precheck should naturally update expectations.
- Add 1 new test in tests/test_db_queue.py that verifies requeue rejects non-precheck job_type with 422.
  - Use TestClient(lan_app.api.app) and POST /api/recordings/<id>/actions/requeue with job_type="stt" (or any non-precheck), expect 422.

Local verification commands:
- scripts/ci.sh
- pytest -q

Success criteria:
- ingest_once creates exactly 1 DB job per new recording, and jobs_created is 1.
- No new placeholder jobs are inserted for stt/diarize/align/language/llm/metrics.
- API requeue returns 422 for non-precheck job_type, and 200 for precheck.
- UI retry always requeues precheck.
- If a legacy job type is executed, the DB job becomes failed and a step log is written, but recording status does not change.
- scripts/ci.sh is green.
```