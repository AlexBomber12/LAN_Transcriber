PR-TEST-PIPELINE-RESILIENCE-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/test-pipeline-resilience-01
PR title: PR-TEST-PIPELINE-RESILIENCE-01 Add worker-level resilience tests for external dependency failures
Base branch: main

Goal
Add fast unit tests that validate the worker layer behaves correctly when pipeline components fail due to external dependency issues.

Scope
- Tests only.
- No Docker builds.
- No real whisperx/pyannote/torch execution.

Implementation

A) Add a new test file tests/test_worker_resilience.py
- Use the existing pattern from tests/test_ui_routes.py to build AppSettings and init_db.
- Seed a recording and a precheck job.
- Create a small raw audio file under recordings_root/<id>/raw/audio.wav.

Test 1: run_pipeline exception marks job failed and recording status updates
- Monkeypatch lan_app.worker_tasks.run_precheck to return a non-quarantine PrecheckResult.
- Monkeypatch lan_app.worker_tasks.run_pipeline to raise RuntimeError("dep failure").
- Call process_job(job_id, recording_id, JOB_TYPE_PRECHECK) and assert:
  - job status becomes failed
  - job error contains "dep failure"
  - recording status becomes failed or needs_review based on current policy

Test 2: progress update failures do not hide the original error
- Monkeypatch set_recording_progress to raise on first call.
- Ensure process_job still fails with the original run_pipeline error and job error is preserved.

B) Keep tests aligned with existing status policy
- Do not hardcode statuses that are not stable.
- Use existing constants:
  - JOB_STATUS_FAILED
  - RECORDING_STATUS_FAILED
  - RECORDING_STATUS_NEEDS_REVIEW

Local verification
- scripts/ci.sh

Success criteria
- New worker resilience tests are green.
- These tests fail if worker stops updating statuses correctly when pipeline fails.
- scripts/ci.sh is green.
```
