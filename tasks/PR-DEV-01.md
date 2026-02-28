PR-DEV-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/dev-01
PR title: PR-DEV-01 Add docker-compose.dev.yml for fast iteration without rebuilding images
Base branch: main

Goal
Provide a development compose overlay that allows changing Python code without rebuilding Docker images.

Background
Rebuilding images is slow. The runtime image already contains dependencies; we can bind-mount the repository into /app to replace code.

Scope
- Add a dev-only compose overlay file.
- Do not change production docker-compose.yml behavior.

Implementation

A) Add docker-compose.dev.yml
- Create docker-compose.dev.yml that overlays docker-compose.yml and does:
  - For services api, worker, db (optional): mount repo into /app:
      - ./:/app
  - Optionally enable uvicorn reload for api in dev:
      command: ["uvicorn", "lan_app.api:app", "--host", "0.0.0.0", "--port", "7860", "--reload", "--reload-dir", "/app/lan_app", "--reload-dir", "/app/lan_transcriber"]
  - Keep ports and /data volume as in base compose.

B) Documentation
- Update README.md with a short "Dev mode" section:
  - Build once:
      docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
  - After code changes:
      docker compose restart api worker
  - Explain that rebuild is needed only when requirements change.

Local verification
- Bring the stack up with the dev overlay.
- Change a trivial UI string and confirm it appears after restart (no rebuild).

Success criteria
- docker-compose.dev.yml exists and provides bind-mount code hot iteration.
- README includes clear dev commands.
- scripts/ci.sh is green.
```
