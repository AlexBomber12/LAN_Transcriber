# AGENTS

These rules apply to every PR and every task in this repo.

Quick rules
- Always choose a work mode using an exact trigger phrase from the Work Modes section.
- PLANNED PRs must follow `tasks/QUEUE.md` and the corresponding `tasks/PR-*.md` file exactly.
- Never commit secrets. Runtime secret files belong under `/data/secrets` (mounted) or injected via env vars. Developer-local tool secrets (Codex/MCP) must be provided via OS env vars or user-level Codex config, never committed to the repo.
- Always run the local gate `scripts/ci.sh` until it exits with code 0.
- Always generate review artifacts: `artifacts/ci.log` and `artifacts/pr.patch`.
- Codex Review is "green" only when the Codex bot reacts with 👍 (`+1`) on the PR's review anchor comment (see Codex Review gate). Do not use screenshots.
- Do not drift from the plan and do not add “nice-to-have” work.

## Work Modes
Exact trigger phrases:
- `Run PLANNED PR`
- `Run MICRO PR: <one sentence description>`
- `Fix code review comment`

Meaning:
- `Run PLANNED PR`: the default mode. Work strictly from `tasks/QUEUE.md`.
- `Run MICRO PR: ...`: a tiny change. Do not touch `tasks/QUEUE.md` and do not create `tasks/PR-*.md`.
- `Fix code review comment`: fix feedback on an existing PR branch.


## Codex Review gate (GitHub PR)

This repo treats Codex Review as a single pass/fail signal.

Pass signal
- 👍 (`+1`) reaction from the Codex bot on the PR's review anchor comment.

In-progress signal
- 👀 (`eyes`) reaction from the Codex bot.

Feedback
- If there is no 👍, assume the PR is not approved yet.
- Codex may post a comment with findings (often labeled `P1` or `P2`). Fix all `P1` and `P2` by default to obtain 👍.

Definitions
- Codex bot: any GitHub user or app whose login contains `codex` (case-insensitive), for example `chatgpt-codex-conn`.
- Review anchor comment: the most recent PR conversation comment authored by the PR author that matches at least one:
  - contains `@codex review`
  - contains `Artifacts` and at least one `artifacts/` path
  - contains any `artifacts/` path
  If none exist, use the first PR author comment in the conversation.

Fix loop (used in `Fix code review comment` mode)
1. Fetch PR comments, reviews, and reactions via GitHub MCP (preferred). No screenshots.
2. Locate the review anchor comment and check Codex reactions.
3. If 👍 exists, stop. The PR is green.
4. Otherwise, collect the latest Codex feedback after the anchor comment:
   - review comments
   - top-level PR comments
5. Extract actionable items and fix them. Treat `P1` as mandatory; treat `P2` as mandatory unless the user explicitly waives them.
6. Run `scripts/ci.sh` until exit code 0 and generate required review artifacts.
7. Commit and push to the same PR branch.
8. Poll the PR for up to 15 minutes:
   - if 👍 appears on the anchor comment, stop
   - if a new Codex feedback comment appears, repeat the loop

Safety limits
- Max 5 fix cycles per request. If still not green, stop and report what remains.

## MCP servers and tool usage

Allowed MCP servers (by name)
- github (required)
- playwright (optional)
- shadcn (optional)
- stitch (optional)
- context7 (optional)

Rules
- Use GitHub MCP to read PR conversation, reviews, and reactions. This is the source of truth for the Codex Review gate.
- Use Playwright MCP only to reproduce UI issues or run E2E checks when a UI change is part of the task.
- Use shadcn MCP only to look up component usage or generate small component snippets. Do not mass-edit styling or tokens.
- Use Stitch MCP only when the task explicitly involves UI layout or design assets. Never write API keys into any repo file.
- Use Context7 MCP only to confirm library behavior from primary docs. Keep outputs short and actionable.

If an MCP server is not listed above, do not use it.

## Repo invariants
These are non-negotiable contracts:
- `/data` is the single runtime state root.
  - DB: `/data/db/app.db`
  - Artifacts: `/data/recordings/<recording_id>/...`
  - Microsoft (MSAL) cache: `/data/auth/msal_cache.bin`
  - Google SA key JSON: `/data/secrets/gdrive_sa.json` (example)
  - Step logs: `/data/recordings/<id>/logs/step-*.log`
- No passwords. Microsoft Graph uses delegated OAuth via Device Code Flow only.
- Google Drive ingest uses Service Account + shared folder (Inbox) only.
- Spark LLM is called via an OpenAI-compatible API (base URL via env).
- Recording statuses (MVP): `Queued`, `Processing`, `NeedsReview`, `Ready`, `Published`, `Quarantine`, `Failed`.
- Any OneNote auto-publish is OFF by default unless explicitly enabled per project.

## Local gates
Single entrypoint: `scripts/ci.sh`.

If `scripts/ci.sh` does not exist yet (temporarily, before PR-BOOTSTRAP-01), use the fallback:
- `python -m ruff check .`
- `pytest -q`

Do not claim “green” unless the exit code is 0.

## Required review artifacts
Single entrypoint: `scripts/make-review-artifacts.sh`.

If the script does not exist yet (temporarily), manual fallback:
- Save CI output to `artifacts/ci.log`
- Save a patch to `artifacts/pr.patch` (diff from `origin/main` to HEAD)

Artifacts are required for every PR.

## Branch naming
- PLANNED: use `Branch:` from the active `tasks/PR-*.md` as the source of truth.
- If `Branch:` is missing, use `pr-<sanitized-pr-id>`:
  - lowercase
  - replace `.` with `-`
  - allow only `[a-z0-9-]`
- MICRO: `micro-YYYYMMDD-<short-slug>`

## PLANNED PR runbook (queue-driven)

### Rules
- Preflight: `git status --porcelain` must be empty. If not, stop and list dirty files.
- Task selection:
  - if any item is `DOING`, take the earliest `DOING`
  - otherwise take the earliest `TODO` whose dependencies are all `DONE`
  - `PR_ID` must match `tasks/QUEUE.md` exactly
  - `TASK_FILE` must come from the `Tasks file:` line, do not guess
- Read `TASK_FILE` fully before coding.
- Create the branch from `origin/main`.
- Implement only the scope defined in `TASK_FILE`. No extra refactors, upgrades, or bundled features.
- During the PR, do not edit `tasks/PR-*.md` unless the user explicitly requests it.
- CI and artifacts are mandatory.
- Queue update rules:
  - when starting: set the current PR `- Status: DOING`
  - when CI is green and before push: set `- Status: DONE`
  - only change the `- Status:` line for the current PR

### Checklist
- [ ] Preflight clean
- [ ] Selected PR from `tasks/QUEUE.md`; recorded `PR_ID` and `TASK_FILE`
- [ ] Read `TASK_FILE`
- [ ] `git fetch origin main` and created branch from `origin/main`
- [ ] Implemented only `TASK_FILE` scope
- [ ] Ran `scripts/ci.sh` to exit 0
- [ ] Generated `artifacts/ci.log` and `artifacts/pr.patch`
- [ ] Updated `tasks/QUEUE.md` for current PR: `DOING` -> `DONE`
- [ ] Commit message: `<PR_ID>: <short summary>`
- [ ] Pushed branch
- [ ] Created PR via GitHub CLI (`gh`) or provided manual PR steps
- [ ] Final report prepared (see below)

### Final report (PR description or final message)
- PR_ID
- TASK_FILE
- Branch
- What changed (1-5 bullets)
- How verified (exact command)
- Artifacts: `artifacts/ci.log`, `artifacts/pr.patch`
- Manual test steps (if applicable)
- MCP usage (if applicable)

## MICRO PR runbook

### Eligibility (all must be true)
- <= 3 files changed
- <= 100 lines changed (excluding lockfile noise)
- no DB migrations/schema changes
- no dependency upgrades
- no auth/permissions/publish changes
- no large refactors or sweeping formatting

If any condition fails, MICRO is not allowed. Use PLANNED PR.

### Rules
- Do not create `tasks/PR-*.md`
- Do not edit `tasks/QUEUE.md`

### Checklist
- [ ] Preflight clean
- [ ] `git fetch origin main`
- [ ] Branch `micro-YYYYMMDD-<short-slug>` from `origin/main`
- [ ] Only the requested change
- [ ] Ran `scripts/ci.sh` to exit 0
- [ ] Generated review artifacts
- [ ] Commit: `MICRO: <short summary>`
- [ ] Pushed branch and opened PR

## REVIEW FIX runbook (existing PR branch)
- Do not select a new task from `tasks/QUEUE.md`
- Do not create a new branch
- Stay on the existing PR branch
- Do not edit `tasks/QUEUE.md` or `tasks/PR-*.md`
- Fix only the review comments
- Use the Codex Review gate above and stop only when the Codex 👍 pass signal is present
- Run `scripts/ci.sh` to exit 0
- Generate review artifacts
- Commit and push to the same PR branch

## Cursor, Codex, GitHub workflow (practical)
- For PLANNED PR: paste the full `tasks/PR-*.md` content into the agent as the sole instruction source.
- Make meaningful commits, but do not split into cosmetic micro-commits.
- Never commit `/data`, keys, caches, or any tokens.
- Default config via env vars and `.env.example` (no secrets).
- If you need to confirm an API/SDK behavior, prefer primary documentation sources over guessing.

## Secrets and logging rules
- Do not print secrets in logs, diffs, commit messages, or PR descriptions.
- Do not hardcode MCP API keys in config files. Use `env_http_headers` and environment variables.
- For Stitch MCP, never use `[mcp_servers.stitch.http_headers]` with a literal key. Use only `[mcp_servers.stitch.env_http_headers]` and the `STITCH_API_KEY` env var.
- Do not modify user-level Codex config (`~/.codex/config.toml`) as part of repo PRs unless the user explicitly requests it.
- Do not add secrets to `.env`, `docker-compose.yml`, or README.
- Keys and token caches must be stored under `/data` and mounted.

## Queue stability rules (PLANNED PR only)
- `tasks/` is the source of truth.
- Do not rewrite tasks retroactively during a PR.
- If the user updates `tasks/` while you are working, stop and ask for explicit direction: continue as-is, incorporate changes, or revert.

