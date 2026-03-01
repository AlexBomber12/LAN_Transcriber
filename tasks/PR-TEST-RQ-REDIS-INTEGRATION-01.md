    PR-TEST-RQ-REDIS-INTEGRATION-01
    ===============================

    Prompt (copy as-is into Codex)
    ------------------------------

    ```text
    You are Codex Agent working on the LAN-Transcriber repository.

Goal
Add a real Redis+RQ integration test that validates enqueue_recording_job -> RQ execution -> DB status transitions, without running heavy ML pipeline code.

Context
The app uses Redis RQ for job execution (lan_app.jobs.enqueue_recording_job) and lan_app.worker_tasks.process_job for lifecycle transitions.
Most tests call functions directly or mock queue behavior. We want at least one CI test that exercises real Redis and real RQ execution wiring.

Non-goals
- Do not download or run WhisperX, pyannote, or LLM models.
- The test must be fast and deterministic.
- Do not require docker-in-docker; use a Redis service in GitHub Actions.

Implementation requirements

1) Add Redis service to the main CI workflow
- Update .github/workflows/ci.yml for the existing unit test job to include a Redis service container.
- Expose port 6379 on localhost.
- Use a non-zero DB index (e.g., /15) for tests to avoid collisions.

2) Add an integration test using rq.SimpleWorker
- Create tests/test_rq_redis_integration.py.
- The test should:
  - set env vars so AppSettings uses tmp_path directories:
    - LAN_DATA_ROOT=tmp_path
    - LAN_RECORDINGS_ROOT=tmp_path/recordings
    - LAN_DB_PATH=tmp_path/db/app.db
    - LAN_REDIS_URL=redis://127.0.0.1:6379/15
  - init the sqlite DB
  - create a recording row in DB (using existing db helpers)
  - monkeypatch the heavy pipeline execution path in worker_tasks to a fast stub that marks the recording as READY and returns (status, quarantine_reason)
    - patch the highest-level function so the test does not accidentally import or run ML code
  - enqueue the job via lan_app.jobs.enqueue_recording_job
  - run the worker in-process via rq.SimpleWorker(...).work(burst=True)
  - assert terminal success:
    - job moved to finished/terminal success state
    - recording status == READY (or your repo's equivalent)
    - quarantine_reason is None

3) Isolation and cleanup
- Flush only redis DB 15 before and after the test.
- Ensure the test does not require network.

4) Coverage
- Prefer test-only changes. If minimal production glue is needed, it must be fully covered (100% statement and branch).

Deliverables
- Redis service in CI.
- New integration test module using SimpleWorker burst.

Success criteria
- CI runs the new integration test and it passes.
- The test fails if Redis config, enqueue logic, or worker lifecycle transitions break.
    ```
