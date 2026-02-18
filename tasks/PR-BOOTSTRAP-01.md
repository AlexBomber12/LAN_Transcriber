PR ID: PR-BOOTSTRAP-01
Branch: pr/bootstrap-01

Goal
Repo bootstrap: tasks system, CI scripts, artifacts, runtime data layout.

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
none

Work plan
1) Add repo-level task system and conventions
   - Create tasks/ directory.
   - Add tasks/QUEUE.md (single source of truth for execution order and PR statuses).
   - Add tasks/PR-*.md files (this repo will receive them over time; this PR adds structure and at least one example task file).
   - Add or update AGENTS.md to match the workflow used in the reference repo:
     - PLANNED PR runbook (queue-driven)
     - Required gates: scripts/ci.sh exit 0, artifacts generation, minimal final report template
     - Branch naming rules and constraints

2) Standardize local gates and artifacts
   - Add scripts/ci.sh that runs the same checks as GitHub Actions, locally and in CI:
     - python -m ruff check .
     - pytest with coverage target (match existing CI thresholds)
   - Add scripts/make-review-artifacts.sh that always produces:
     - artifacts/ci.log
     - artifacts/pr.patch
   - Ensure scripts are executable and documented.

3) Runtime data layout
   - Define /data as the single runtime state root for:
     - artifacts (transcripts, summaries, snippets)
     - token caches (msal)
     - voice samples
     - db
     - logs
   - Ensure docker-compose mounts ./data (repo-local) to /data inside containers.

4) PR template alignment
   - Add .github/pull_request_template.md mirroring the report requirements from AGENTS.md.

Local verification
- scripts/ci.sh exits 0.
- docker-compose up starts without errors (even if the app is still minimal).

Artifacts
- Run scripts/make-review-artifacts.sh and commit artifacts/ci.log and artifacts/pr.patch (or attach them to the PR, depending on repo convention).

Success criteria
- tasks/QUEUE.md exists and documents the execution order and status legend.
- AGENTS.md exists and enforces the runbook used in the reference repo.
- scripts/ci.sh and scripts/make-review-artifacts.sh exist, are executable, and work on a clean checkout.
- docker-compose mounts /data and the repo remains free of secrets.
