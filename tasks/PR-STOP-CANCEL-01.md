PR-STOP-CANCEL-01
=================

Branch: pr-stop-cancel-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Add a user-visible Stop action that lets the operator stop processing for a recording without deleting it. The Stop control must appear next to Back on the recording detail page and support both queued and currently processing recordings. This PR is the soft-cancel layer: queued jobs should be removed immediately; running jobs should observe a cancel flag at safe checkpoints between stages and LLM chunks and transition to a clear terminal stopped/cancelled status.

Constraints
- This PR is soft cancel only. Do not yet implement child-process hard termination; that is the next PR.
- Build on the stage/chunk checkpoint work so soft stop interacts correctly with resume state.
- Preserve existing delete, requeue, and quarantine behavior.
- Keep UI changes minimal and server-rendered.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Add explicit stopped/cancelling recording states
- Extend lan_app/constants.py and the database status validation so recordings can represent soft cancellation clearly.
- Add at least:
  - Stopping or equivalent in-progress cancel-request state
  - Stopped or Cancelled as a terminal state
- Update the relevant DB CHECK constraints via a new migration using the next number.
- Update helper validators/selectors/UI badge styling to understand the new status values.
- Decide on the canonical names and use them consistently everywhere.

2) Persist cancel intent durably
- Add fields or a dedicated table to persist cancel requests, for example:
  - cancel_requested_at
  - cancel_requested_by
  - cancel_reason_code or cancel_reason_text
- The persistence must survive worker retries/restarts.
- Add DB helpers for setting, clearing, and reading cancellation state.
- Keep the contract simple: a recording can have an active cancel request that the worker checks cooperatively.

3) Add Stop control to the recording detail UI
- In lan_app/templates/recording_detail.html add a Stop button next to the Back button.
- Show it only when the recording is in a stop-eligible state, at minimum:
  - Queued
  - Processing
  - Stopping
- On click, show a lightweight confirmation prompt before submitting.
- After requesting stop, disable the button or show a clear Stopping state to avoid duplicate submissions.
- Keep the page server-rendered and production-safe; do not add a large JS framework.

4) Add backend endpoint for stop requests
- Add a POST endpoint under the UI router, for example /ui/recordings/{recording_id}/stop.
- For queued jobs:
  - remove the pending queue job(s) using the existing queue helpers where appropriate
  - mark the recording as Stopped/Cancelled directly
  - clear pipeline progress
- For running jobs:
  - persist cancel_requested_at / requested_by / reason=user_stop
  - move status to Stopping
  - do not pretend the job is already fully stopped until the worker acknowledges it
- Return back to the recording detail page with updated status.

5) Make worker_tasks observe soft cancel checkpoints
- In lan_app/worker_tasks.py add cooperative cancellation checks at safe points:
  - before each coarse stage starts
  - after each coarse stage completes
  - before each LLM chunk
  - after each LLM chunk
  - before merge/routing/export finalization
- If a cancel request is present:
  - stop progressing further
  - persist the current stage/chunk state appropriately as cancelled or incomplete
  - mark the recording terminal status as Stopped/Cancelled
  - finish/fail the job in a controlled way without surfacing a fake processing error
- Keep logs explicit: cancelled_by_user should be visible.

6) Integrate with checkpoint state
- Soft stop must not corrupt the new stage/chunk checkpoint tables.
- Mark in-progress stages/chunks as cancelled or leave them resumable depending on what is already safely complete.
- Explicit user requeue later should clear stop markers and restart cleanly.
- A stopped recording must not auto-resume just because the worker is restarted.

7) Preserve queued-job cancellation support
- Reuse the existing jobs.cancel_pending_queue_job(...) and purge helpers where possible.
- If a recording has a queued precheck job and Stop is requested before the worker starts it, the queue job should be removed and the recording should end in Stopped/Cancelled without ever entering Processing.

8) UI visibility and status presentation
- Add badge styling for the new status values.
- Show a readable reason such as Cancelled by user or Stop requested by user where appropriate.
- Keep the overview/progress display coherent when a stop happens mid-processing.

9) Tests with full coverage
Add deterministic offline tests for at least these cases:
- Stop on a queued recording removes pending queue work and sets a terminal stopped/cancelled state
- Stop on a processing recording sets a cancel request and transitions to Stopping until the worker acknowledges it
- worker soft-cancel check stops before the next stage and marks the recording terminal appropriately
- worker soft-cancel check during LLM chunk processing stops after the current safe checkpoint without continuing further chunks
- explicit requeue after stop clears cancellation markers and restarts correctly
- UI renders the Stop button only in eligible states
- new status values round-trip through validation/selectors/templates
- duplicate stop requests do not corrupt state

10) Documentation and operator notes
- Update README/runbook briefly to explain soft Stop behavior:
  - queued jobs stop immediately
  - running jobs stop at the next safe checkpoint
  - explicit requeue is needed to restart later

Verification steps (must be included in PR description)
- Request Stop on a queued recording and confirm the job is removed and the recording ends in the stopped/cancelled state immediately.
- Request Stop on a processing recording and confirm the worker stops at the next stage/chunk checkpoint without continuing to the end.
- Confirm the UI shows the Stop button next to Back only when appropriate.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Stop button in recording detail UI
- Durable cancel request persistence and new recording states
- Worker cooperative soft-cancel handling
- Updated tests and docs

Success criteria
- Operators can stop queued and running recordings without deleting them.
- Running recordings stop safely at the next cooperative checkpoint.
- Stopped recordings do not resume accidentally.
- CI remains green.
```
