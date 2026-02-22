PR-WORKER-ROBUST-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/worker-robust-01
PR title: PR-WORKER-ROBUST-01 Worker robustness and stuck job recovery
Base branch: main

Goal:
Make job execution reliable under restarts and failures.
1) Add graceful shutdown so the worker stops taking new jobs and finishes current work when SIGTERM is received.
2) Add job timeouts and max attempts handling.
3) Add stuck job recovery so recordings do not remain "processing" forever.

A) Worker graceful shutdown
- File: lan_app/worker.py
- Add signal handlers for SIGTERM and SIGINT.
- On signal:
  - request the RQ worker to stop after current job finishes
  - exit cleanly
- Do not implement a custom infinite loop that breaks RQ internals.
- Log a clear message that shutdown was requested.

B) Job timeouts and attempts
- Add settings:
  - LAN_RQ_JOB_TIMEOUT_SECONDS (default 7200)
  - LAN_MAX_JOB_ATTEMPTS (default 3)
- Ensure enqueue_recording_job uses job_timeout for the RQ job.
- Ensure DB job attempt counter is incremented on start.
- If attempts exceed LAN_MAX_JOB_ATTEMPTS:
  - mark job as failed terminal with error "max attempts exceeded"
  - set recording status to NeedsReview
  - do not auto requeue

C) Stuck job recovery (reaper)
- Implement a recovery routine that can be run periodically by the API process.
- Add settings:
  - LAN_STUCK_JOB_SECONDS (default 7200)
  - LAN_REAPER_INTERVAL_SECONDS (default 300)
- Define "stuck":
  - DB job status is "started" and started_at is older than now - LAN_STUCK_JOB_SECONDS
  - OR recording status is Processing and there is no corresponding started job
- Recovery behavior:
  - Mark the job failed with error "stuck job recovered"
  - Set recording status to NeedsReview
  - Append a line to step log stating recovery occurred and timestamp
- Expose reaper as a function in lan_app/reaper.py (new) so it is testable.

D) Integrate reaper in the API loop
- If the app already uses a background loop, add the reaper there.
- Otherwise integrate it into startup/shutdown or lifespan depending on current architecture.
- Run reaper every LAN_REAPER_INTERVAL_SECONDS.
- Ensure the task stops on shutdown cleanly.

E) UI indicators
- Recording detail page should show a warning if a recording was recovered from a stuck job.
- Minimal approach: add a derived field in DB for last_warning on recording, or use job.error.

F) Tests
- Add unit tests for reaper:
  - Create a recording and a started job with started_at older than threshold.
  - Run reaper once.
  - Assert job is failed and recording status is NeedsReview.
- Add test for max attempts:
  - Simulate job with attempts >= max, then requeue should not proceed or should produce terminal failure.

Files to touch (expected):
- lan_app/settings.py
- lan_app/queueing.py (or wherever enqueue_recording_job is)
- lan_app/worker.py
- lan_app/reaper.py (new)
- lan_app/api.py (to start and stop background reaper task)
- lan_app/db.py (helpers to query stuck jobs, mark failure, set status)
- templates and ui_routes for warning display
- tests/

Local verification:
- scripts/ci.sh
- Manual: start worker, enqueue a long job, send SIGTERM, verify it stops after the current job.

Success criteria:
- Worker shuts down gracefully on SIGTERM.
- Jobs have enforced timeouts and max attempts.
- Stuck jobs are detected and recovered automatically, leaving recordings in NeedsReview, not Processing.
- scripts/ci.sh is green.
```
