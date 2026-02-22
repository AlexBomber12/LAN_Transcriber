PR-STAGING-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/staging-01-fix-deploy
PR title: PR-STAGING-01 Fix staging deploy workflow and add infra/staging
Base branch: main

Goal: Make .github/workflows/staging-deploy.yml actually deploy to the VPS on pushes to main, using a real infra/staging/ docker compose stack, and validate the deployment with the updated smoke test (healthz + openapi). Remove the logic that incorrectly treats GitHub Actions as "CI mock mode" and skips the remote steps.

Changes required:

1) Add infra/staging files used by the workflow
- Create folder: infra/staging/
- Add infra/staging/docker-compose.yml
  - Must NOT use build.
  - Must use an image reference, defaulting to ghcr.io/alexbomber12/lan-transcriber:latest (overrideable via TRANSCRIBER_IMAGE env var).
  - Define services: redis, db (oneshot init), api, worker similar to root docker-compose.yml.
  - Use volumes so data persists inside ~/lan-staging/data (bind ./data:/data).
  - Expose api on 7860 (bind 0.0.0.0:7860:7860 or configurable).
  - Set pull_policy to always for the api/worker image.
- Add infra/staging/.env.staging.example
  - Include placeholders for all needed LAN_* and LLM_* and MS_* and GDRIVE_* env vars, but no secrets.
  - Include TRANSCRIBER_IMAGE and LAN_API_BIND_HOST.
- Add infra/staging/README.md with VPS steps:
  - copy .env.staging.example to .env
  - docker compose pull
  - docker compose up -d
  - run smoke test endpoint checks

2) Fix staging deploy GitHub workflow
- File: .github/workflows/staging-deploy.yml
- Remove "Detect CI" and "Mock deploy (CI)" logic. This workflow triggers on push to main only, so it should do the real deploy.
- Keep a strict "Validate secrets" step that fails if STAGING_HOST, STAGING_USER, STAGING_SSH_KEY are missing.
- Ensure the workflow copies infra/staging/* to ~/lan-staging on the VPS (scp action).
- Remote script changes:
  - cd ~/lan-staging
  - docker compose pull
  - docker compose up -d --remove-orphans
  - Avoid --build
  - Ensure data folder exists (mkdir -p data)
  - Optional: attempt ghcr login if a secret token is present:
    - Support secrets STAGING_GHCR_USERNAME (default alexbomber12) and STAGING_GHCR_TOKEN, and do docker login ghcr.io before pull.
    - If token is absent, proceed (works for public images).
- Replace the smoke test step to use the updated scripts/smoke_test.py:
  - python scripts/smoke_test.py --base-url http://$STAGING_HOST:7860
  - Do not pass --file.

3) Ensure workflow uses correct Python version
- If needed, add actions/setup-python to the workflow so smoke_test has requests installed and runs reliably.
- Install minimal dependency requests in the workflow before calling smoke_test, or make smoke_test depend only on stdlib urllib (either approach is fine, but be consistent).

Local verification commands:
- scripts/ci.sh
- yamllint is not present; do not add new tooling.

Success criteria:
- infra/staging/ exists with docker-compose.yml, .env example, README.
- staging-deploy workflow no longer skips remote steps on GitHub Actions.
- Workflow copies compose files to VPS and runs docker compose pull + up without --build.
- Workflow runs smoke_test against the deployed host and passes.
- scripts/ci.sh remains green.
```