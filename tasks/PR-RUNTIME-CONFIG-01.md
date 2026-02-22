PR-RUNTIME-CONFIG-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/runtime-config-01
PR title: PR-RUNTIME-CONFIG-01 Fail-fast config for staging/prod and FastAPI lifespan
Base branch: main

Goal:
1) Remove docker-specific silent defaults that mask misconfiguration.
2) Add explicit environment mode and fail-fast validation for staging/prod.
3) Replace deprecated FastAPI startup/shutdown events with lifespan.
4) Align docs and compose files with the new behavior.

A) LAN_ENV and validation
- Add setting: LAN_ENV with allowed values dev, staging, prod (default dev).
- Validation:
  - dev:
    - allow defaults for LAN_REDIS_URL and LLM_BASE_URL if not set
    - log a warning when defaults are used
  - staging/prod:
    - require LAN_REDIS_URL to be explicitly set
    - require LLM_BASE_URL to be explicitly set
    - if missing raise a clear startup error and exit
- Replace docker host defaults:
  - no default redis://redis:6379/0
  - no default http://llm:8000
- Update docker-compose.yml and infra/staging/docker-compose.yml to set explicit env vars so staging works.

B) FastAPI lifespan
- File: lan_app/api.py
- Replace startup and shutdown events with lifespan context manager.
- Move into lifespan:
  - init_db()
  - background tasks:
    - retention cleanup loop
    - reaper loop
- Cancel and await background tasks on shutdown.

C) Docs
- Update README and infra/staging/README.md:
  - LAN_ENV behavior
  - required env vars for staging/prod
  - auth token var if enabled

D) Tests
- LAN_ENV=staging with missing LAN_REDIS_URL should fail.
- LAN_ENV=dev with missing vars should allow importing app.

Local verification:
- scripts/ci.sh
- docker compose up works with explicit env vars.

Success criteria:
- staging and prod fail fast on missing critical env vars.
- dev works with warnings.
- lifespan replaces deprecated patterns.
- scripts/ci.sh is green.
```
