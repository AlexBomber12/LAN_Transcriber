PR ID: PR-OPS-01
Branch: pr/ops-01

Goal
Retention, quarantine cleanup, retries, runbook, and production hardening for LAN deployment.

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
PR-ONENOTE-01, PR-ROUTING-01

Work plan
1) Retention and quarantine cleanup
   - Config:
     - QUARANTINE_RETENTION_DAYS default 7
   - Scheduled cleanup job that:
     - deletes quarantine files older than retention
     - prunes stale intermediate artifacts if needed

2) Robust retries and failure handling
   - Retry strategy per job type (max attempts, backoff).
   - Separate "Failed" vs "NeedsReview" clearly.
   - Provide a UI for retrying failed steps.

3) Operational runbook
   - docs/runbook.md:
     - initial setup (Drive SA, Graph app, Spark LLM)
     - common failures and fixes
     - backup/restore of /data
     - upgrade steps
   - Include safety notes: secrets handling and token cache behavior (re-auth required by policy).

4) Production hardening for LAN
   - Add basic auth or LAN-only binding guidance.
   - Ensure docker-compose exposes only needed ports.
   - Add /healthz checks for app, worker, db, redis.

Local verification
- Simulate a failed job and verify retry flow.
- Simulate quarantine cleanup on a test file.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- The app is maintainable over time: logs, retries, cleanup, and documentation exist.
- LAN deployment is safe by default.
