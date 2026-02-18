PR ID: PR-ROUTING-01
Branch: pr/routing-01

Goal
Project suggestion (calendar + voices + text) with confidence + NeedsReview workflow.

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
PR-CALENDAR-01, PR-VOICE-01, PR-ONENOTE-01

Work plan
1) Implement project suggestion with confidence
   - Signals (MVP):
     - calendar subject keyword match
     - organizer and attendee match (if voice profiles are linked to calendar attendees later, keep optional)
     - LLM tags/keywords
     - voice profile presence (if assignments exist)
   - Output:
     - suggested_project_id and confidence
     - rationale (human-readable list)

2) Workflow rules
   - If confidence >= threshold:
     - auto-select project (but still allow override)
   - If confidence < threshold:
     - force NeedsReview and route to Unsorted-like view.

3) Learning from manual corrections
   - When user manually selects a project:
     - store a training example:
       - calendar subject tokens
       - tags
       - voice profile ids present
     - update per-project keyword weights (simple) stored in DB.

4) UI
   - Recordings list: show suggested project and confidence.
   - Recording detail -> Project tab: show rationale + "Train routing" toggle.

Local verification
- Create 2 projects, assign 2 recordings, then ingest a 3rd similar one and confirm suggestion works.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- The app suggests projects and becomes more accurate after manual corrections.
- Misclassifications default to NeedsReview instead of silent auto-publish.
