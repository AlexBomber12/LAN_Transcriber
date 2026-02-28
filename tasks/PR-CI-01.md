PR-CI-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/ci-01
PR title: PR-CI-01 Fix GitHub Actions failures (docker smoke pytest, staging deploy secrets) and remove duplicate CI workflow
Base branch: main

Context
GitHub Actions currently fail due to:
1) Docker smoke step runs pytest but pytest is not available on PATH (venv is not activated).
2) staging-deploy workflow fails on every push because required secrets are not configured.
3) There is a duplicate lint-and-test workflow that runs the same scripts/ci.sh, doubling CI noise.

Goals
1) Make CI green on PR and push without requiring any staging secrets.
2) Keep docker smoke test running when a smoke image exists, but execute it inside the same venv created by scripts/ci.sh.
3) Ensure staging deploy is opt-in:
   - manual run via workflow_dispatch, or
   - automatic run only when all required secrets are present.
4) Remove or disable the duplicate lint-and-test workflow.

Non-goals
- Do not change application runtime behavior.
- Do not change docker-build-and-push workflow.

Implementation

A) Fix Docker smoke step in .github/workflows/ci.yml
- In the job unit-and-ui-tests, step "Docker smoke":
  - Activate the CI venv created by scripts/ci.sh:
      . .venv-ci/bin/activate
  - Run the smoke test via python module invocation:
      python -m pytest -q tests/test_docker_smoke.py
  - Do not rely on the bare pytest executable.
  - Keep the step guarded by SMOKE_IMAGE != ''.

B) Make staging deploy non-blocking
File: .github/workflows/staging-deploy.yml
- Add workflow_dispatch trigger.
- Add a job-level if condition so the deploy job is skipped when secrets are missing:
  - Required secrets: STAGING_HOST, STAGING_USER, STAGING_SSH_KEY
- Remove the "Validate secrets" step that exits 1.
- Ensure the workflow no longer fails when secrets are not set.

C) Remove duplicate CI workflow
File: .github/workflows/lint-and-test.yml
Choose one:
Option 1 (preferred): convert to manual-only
- Change triggers to:
    on: workflow_dispatch
- Rename the workflow name to indicate it is manual.
- Keep job name, but it will no longer run on push or pull_request.

Option 2: delete lint-and-test.yml
- If deleted, ensure ci.yml remains the only CI test workflow.

Local verification
- scripts/ci.sh

Success criteria
- On a push to main without any staging secrets, Actions do not fail due to staging-deploy.
- Docker smoke step no longer fails with "pytest: command not found".
- Only 1 CI workflow runs tests on PR and push (no duplicate lint-and-test).
```
