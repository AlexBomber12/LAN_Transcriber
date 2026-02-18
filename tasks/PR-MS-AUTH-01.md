PR ID: PR-MS-AUTH-01
Branch: pr/ms-auth-01

Goal
Microsoft Graph delegated auth (work) via Device Code Flow + token cache.

Hard constraints
- Follow AGENTS.md PLANNED PR runbook.
- No secrets in the repo. Any credentials, keys, tokens, or config files must live under /data and be mounted via docker-compose or provided via env vars.
- Implement only what is required for this PR. Do not bundle extra refactors, dependency upgrades, or feature additions.
- Keep changes incremental and keep the application runnable at the end of the PR.
- Preserve Linux-first behavior. Do not add Windows-only steps.
- Maintain backwards compatibility for already committed developer workflows where possible.

Context
We are building a LAN application with a simple DB-like UI to manage meeting recordings. Ingest comes from Google Drive (Service Account + shared folder), processing runs locally, summaries are generated via Spark LLM (OpenAI-compatible API), and publishing goes to OneNote via Microsoft Graph (work account).

Depends on
PR-UI-SHELL-01

Work plan
1) Add Microsoft Graph delegated auth via Device Code Flow (work account)
   - Add dependency: msal
   - Configuration env vars:
     - MS_TENANT_ID
     - MS_CLIENT_ID
     - MS_SCOPES (default: "offline_access Notes.ReadWrite Calendars.Read")
   - Token cache stored at /data/secrets/msal_cache.bin (or /data/auth/msal_cache.bin), not in repo.

2) Implement Connections page workflow
   - /connections shows:
     - Connected / Expired state
     - Granted scopes
     - Tenant/account display name
   - Button "Connect" triggers device code flow:
     - show user_code + verification_uri
     - poll until authenticated
   - Button "Reconnect" clears cache and repeats.

3) Implement backend Graph client
   - Use msal to acquire token silently; if fails, return "needs reconnect".
   - Provide a minimal wrapper for Graph GET/POST with retries.

4) Add a health check endpoint for Graph
   - GET /api/connections/ms/verify:
     - call /me and return ok/error.

Local verification
- With valid env vars, connect once and verify:
  - /api/connections/ms/verify returns ok
  - token cache persists across container restart
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- The app can connect to Microsoft Graph without storing passwords.
- Re-auth is only required when tenant policy forces it.
- Token cache is persisted under /data and never committed.
