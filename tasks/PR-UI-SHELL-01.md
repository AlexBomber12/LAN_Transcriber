PR ID: PR-UI-SHELL-01
Branch: pr/ui-shell-01

Goal
Web UI skeleton: dashboard + DB-table recordings list + recording detail shell + connections shell.

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
PR-DB-QUEUE-01

Work plan
1) Choose UI approach for MVP: server-rendered HTML + HTMX
   - Add FastAPI routes that render HTML templates (Jinja2).
   - Use HTMX for table filtering and inline actions.
   - Keep JS minimal. No new frontend framework.

2) Implement pages (shells first, minimal but usable)
   - / (Dashboard)
   - /recordings (table view)
   - /recordings/<id> (detail view with tabs placeholders: Overview, Calendar, Project, Speakers, Language, Metrics, Log)
   - /projects (list + create/edit minimal)
   - /voices (list + create/edit minimal)
   - /queue (jobs list)
   - /connections (Graph + Drive status placeholders)

3) Implement API-backed actions
   - Buttons and forms should call backend endpoints created in PR-DB-QUEUE-01.

4) UI conventions
   - Table-first, “DB window” look.
   - Make statuses very obvious.
   - Provide filters and pagination (even simple).
   - Provide a single action dropdown per row: open, requeue, quarantine, delete.

Local verification
- docker-compose up
- Navigate pages and ensure lists render with empty DB.
- Add 1 recording via sqlite/seed and verify rendering.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- UI exists and is usable as a control panel even before full processing is implemented.
- Recordings table and detail pages render correctly.
- No heavy frontend dependencies were added.
