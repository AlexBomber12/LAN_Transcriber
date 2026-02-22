PR-ENTRYPOINT-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/entrypoint-01-unify-fastapi
PR title: PR-ENTRYPOINT-01 Unify entrypoint to lan_app.api + worker
Base branch: main

Goal: Make lan_app.api:app the single official HTTP entrypoint across Dockerfile defaults, systemd services, and smoke tests. Align repo tests that currently import web_transcribe.app to use lan_app.api.app. Keep web_transcribe module importable for compatibility, but stop treating it as the primary app.

Changes required:

1) Dockerfile default command must run lan_app.api:app
- File: Dockerfile
- For runtime-lite stage:
  - Change CMD to: uvicorn lan_app.api:app --host 0.0.0.0 --port 7860
- For runtime-full stage:
  - Change CMD to: uvicorn lan_app.api:app --host 0.0.0.0 --port 7860
- Do not change the stage order (keep runtime-lite as the final stage) to avoid changing the default target used by existing workflows.
- If runtime-full fails to import lan_app.api due to missing runtime dependencies, fix requirements.txt by adding missing runtime deps used by lan_app (at minimum: jinja2, pydantic-settings, python-multipart, prometheus_client, pyyaml, google-auth, google-api-python-client, uvicorn). Keep versions flexible, similar to current style.

2) systemd units must represent the real architecture: api + worker
- Folder: systemd/
- Replace the legacy single unit with 2 units:
  - systemd/lan-transcriber-api.service
    - ExecStart uses uvicorn lan_app.api:app --host 0.0.0.0 --port 7860
    - Include Environment=LAN_DATA_ROOT=... etc only if the file already documents it, otherwise keep it minimal and point to README/runbook.
    - Optionally run DB init as ExecStartPre: python -m lan_app.db_init
  - systemd/lan-transcriber-worker.service
    - ExecStart: python -m lan_app.worker
- Keep the existing systemd/recording-transcriber.service file but change its Description to "DEPRECATED" and make it a thin wrapper running lan-transcriber-api only, or update README to instruct users to switch. Do not leave it pointing to web_transcribe.py.

3) Update scripts/smoke_test.py to match the new API
- File: scripts/smoke_test.py
- Remove the upload and /api/job polling logic (those endpoints do not exist in lan_app.api).
- New smoke behavior:
  - wait for GET {base_url}/healthz to return 200
  - verify GET {base_url}/healthz/app, /healthz/db, /healthz/redis return 200
  - verify GET {base_url}/openapi.json returns 200
- CLI:
  - Keep --base-url required.
  - Remove --file argument or make it optional but unused (prefer removing to avoid false confidence).
- The script must exit 0 on success and non-zero on failure.

4) Update unit tests that import web_transcribe.app
- Files:
  - tests/test_smoke.py
  - tests/test_ui.py
- Change imports to: from lan_app.api import app
- Ensure these tests still assert root endpoints and openapi status codes correctly.

Local verification commands:
- scripts/ci.sh
- python scripts/smoke_test.py --base-url http://127.0.0.1:7860  (run with the app already started via docker compose)

Success criteria:
- Docker runtime-lite and runtime-full containers start lan_app.api:app by default.
- Updated smoke_test passes against a running lan_app.api instance.
- Updated tests pass (scripts/ci.sh green).
- systemd folder contains clear api and worker units; legacy unit no longer points to web_transcribe.py.
```