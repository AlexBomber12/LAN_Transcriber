    PR-CI-DOCKER-SMOKE-PR-01
    ========================

    Prompt (copy as-is into Codex)
    ------------------------------

    ```text
    You are Codex Agent working on the LAN-Transcriber repository.

Goal
Add a pull_request CI job that builds the Docker runtime-lite image and runs the existing docker smoke test on it, so Docker-only and runtime dependency regressions are caught before merge.

Context
Current .github/workflows/ci.yml runs Python lint/tests with ci-requirements, which does not install the full runtime ML stack.
Docker smoke (tests/test_docker_smoke.py) currently runs only in the docker-build-and-push workflow on main/tags, which is too late for catching regressions on PRs.
We want to detect issues like missing native libs or missing runtime deps (e.g., matplotlib import chain through whisperx/pyannote) as early as possible.

Non-goals
- Do not push images on PR.
- Do not require secrets.
- Keep the job reasonably fast and stable.

Implementation requirements

1) Add a new job to .github/workflows/ci.yml
- Job name: docker-smoke-pr
- Trigger: same events as ci.yml (pull_request and push)
- Steps:
  - checkout
  - setup docker buildx
  - build Dockerfile target runtime-lite
  - load the image into the local docker daemon (tag it as lan-transcriber:pr)
  - run pytest for only the docker smoke test:
    - SMOKE_IMAGE=lan-transcriber:pr pytest -q tests/test_docker_smoke.py

2) Cache docker layers
- Use buildx cache with the GitHub Actions cache backend:
  - cache-from: type=gha
  - cache-to: type=gha,mode=max

3) Keep runner deps minimal
- For the smoke pytest invocation, install only what is needed on the runner (not the repo):
  - python -m pip install -U pip
  - python -m pip install pytest requests

4) Make failures actionable
- Ensure logs show the image tag used (lan-transcriber:pr).
- If tests/test_docker_smoke.py already prints container logs on failure, keep it.
- If not, enhance the test to dump logs on failure.

Deliverables
- Updated .github/workflows/ci.yml with docker-smoke-pr job.

Success criteria
- On PR, CI builds runtime-lite and runs tests/test_docker_smoke.py.
- Docker-only regressions are caught before merge.
- No secrets are required and the job is stable.
    ```
