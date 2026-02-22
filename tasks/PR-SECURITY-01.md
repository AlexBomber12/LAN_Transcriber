PR-SECURITY-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/security-01-auth-abuse-guards
PR title: PR-SECURITY-01 Optional bearer auth and abuse guards
Base branch: main

Goal:
1) Add optional bearer token authentication for the FastAPI app.
2) Add abuse guards for high-impact endpoints: ingest, requeue, delete.
The app must remain usable without auth for localhost-only deployments, but staging/prod must be safe when exposed.

Key requirements:

A) Optional bearer authentication
- Add new setting: LAN_API_BEARER_TOKEN (string, optional).
- If LAN_API_BEARER_TOKEN is not set:
  - Current behavior remains (no auth required).
- If LAN_API_BEARER_TOKEN is set:
  - Protect all state-changing endpoints under /api and all UI POST actions.
  - Health endpoints and metrics must remain accessible without auth:
    - GET /healthz
    - GET /healthz/{component}
    - GET /metrics
    - GET /openapi.json
  - Accepted auth methods:
    1) Authorization header: "Bearer <token>"
    2) HttpOnly cookie set by a simple UI login form.
- Implement UI login:
  - GET /ui/login renders a small form with a single token input.
  - POST /ui/login validates token equals LAN_API_BEARER_TOKEN and sets HttpOnly cookie, then redirects to /ui.
  - GET /ui/logout clears the cookie.
- Middleware or dependency must enforce auth consistently for:
  - POST /api/actions/ingest
  - POST /api/recordings/{id}/actions/requeue
  - POST /api/recordings/{id}/actions/delete
  - Any other POST /api/* endpoints that mutate state
  - UI POST routes that trigger actions (requeue, retry, delete, publish)

B) Abuse guards: rate limiting and dedupe
1) Ingest lock
- Prevent concurrent ingest runs.
- Use Redis (same connection as RQ) to implement a lock key with TTL.
- Behavior:
  - If lock acquired: proceed.
  - If lock exists: return 409 with JSON body indicating ingest already running and include retry_after_seconds.
- Suggested keys:
  - lan:ingest:lock
- TTL:
  - 300 seconds default (configurable by LAN_INGEST_LOCK_TTL_SECONDS).

2) Requeue dedupe
- Prevent duplicate queued/started pipeline jobs for the same recording.
- Rule:
  - If there is an existing job for recording_id with status in ("queued","started") and type is DEFAULT_REQUEUE_JOB_TYPE:
    - Do not create a new one.
    - Return 409 with existing job_id and a clear message.
  - Otherwise enqueue a new job as today.
- Implement this check in DB layer (single query helper), so it can be reused by UI retry too.

3) Delete protection
- Add a lightweight confirmation for UI delete form:
  - UI should require a typed string "DELETE" or a checkbox confirmation.
- API delete remains protected by bearer auth when enabled.

C) Tests
- Add tests that run in both modes:
1) When LAN_API_BEARER_TOKEN is set:
  - Unauthenticated POST /api/actions/ingest returns 401.
  - Authenticated POST /api/actions/ingest returns 200 or 409 depending on lock.
2) Requeue dedupe:
  - First requeue returns 200 with job_id.
  - Second requeue returns 409 and returns the same job_id.
3) UI login:
  - POST /ui/login with correct token sets cookie and allows a protected UI POST action.
- Keep tests hermetic:
  - For Redis lock and dedupe tests, stub lock functions behind an interface and unit test them without network if possible.

D) Docs
- Update README and infra/staging/.env.staging.example to include LAN_API_BEARER_TOKEN and abuse-guard env vars.

Files to touch (expected):
- lan_app/settings.py (new settings)
- lan_app/api.py (auth enforcement for API routes)
- lan_app/ui_routes.py and templates (login/logout, delete confirm)
- lan_app/auth.py (new) or lan_app/security.py (new)
- lan_app/locks.py (new) for Redis lock helpers
- lan_app/db.py (pending-job query helper)
- tests/ (new tests for auth, dedupe, login)
- docs/ or README

Local verification:
- scripts/ci.sh
- Run app with LAN_API_BEARER_TOKEN set and verify:
  - /ui redirects to /ui/login until logged in
  - POST actions require auth
  - /healthz is still open

Success criteria:
- With LAN_API_BEARER_TOKEN unset: behavior unchanged, UI works as before.
- With LAN_API_BEARER_TOKEN set: unauthenticated state-changing requests are rejected.
- Ingest cannot run concurrently (2nd call returns 409).
- Requeue does not create duplicate pending jobs (2nd call returns 409 with existing job_id).
- scripts/ci.sh is green.
```
