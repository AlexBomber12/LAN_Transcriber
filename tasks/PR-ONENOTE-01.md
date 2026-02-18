PR ID: PR-ONENOTE-01
Branch: pr/onenote-01

Goal
Projects mapping to OneNote sections + Publish to OneNote (work).

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
PR-MS-AUTH-01, PR-LLM-01, PR-METRICS-01

Work plan
1) Projects mapping to OneNote locations
   - Projects page:
     - list projects
     - set OneNote notebook_id and section_id per project
   - Provide helper UI to browse:
     - notebooks
     - sections in a selected notebook
   - Store mapping in projects table.

2) Implement Publish to OneNote
   - Endpoint:
     - POST /api/recordings/<id>/publish
   - Behavior:
     - require recording status Ready or NeedsReview
     - generate OneNote HTML page:
       - title: "YYYY-MM-DD HH:MM | <topic> | <participants> | <duration>"
       - sections: Summary, Decisions, Action items, Metrics, Calendar context, Links
     - include links to Drive artifact folder (if available) and to raw audio path (if accessible)
   - Store onenote_page_id and mark Published.

3) UI
   - Recording Detail Overview tab:
     - Publish button
     - Published indicator + link to OneNote page (if Graph returns it)

Local verification
- With Graph connected, publish a test page into a test section.
- Verify page content formatting and that action items and metrics appear.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- Project-to-OneNote mapping is configurable in UI.
- A recording can be published to OneNote reliably with structured content and links.
