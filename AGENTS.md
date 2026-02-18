# AGENTS

## PLANNED PR Runbook

This repository uses a queue-driven PR workflow.

### Source of truth
- `tasks/QUEUE.md` is the only execution order source.
- Every queue item must have a matching `tasks/PR-<ID>.md` file.
- Do not skip queue order. A PR can start only after all dependencies are `DONE`.

### Execution rules
- Implement one planned PR at a time.
- Read the PR task file and implement only that scope.
- Keep changes incremental and keep the app runnable.
- Preserve Linux-first behavior.
- Keep existing developer workflows working whenever possible.

### Branch naming
- Use the exact `Branch:` value declared in the PR task file.
- Branch names must follow `pr/<topic>-<nn>` (example: `pr/bootstrap-01`).
- Do not mix multiple planned PR scopes into one branch.

### Security and data location constraints
- Never commit secrets.
- Credentials, token caches, keys, and runtime config must live under `/data` (mounted from `./data`) or be passed via environment variables.
- Runtime mutable state must be under `/data` (for example: artifacts, `msal` token cache, voices, DB, logs).

### Required gates (must pass before handoff)
1. `scripts/ci.sh` exits with code `0`.
2. `scripts/make-review-artifacts.sh` runs and produces:
   - `artifacts/ci.log`
   - `artifacts/pr.patch`
3. Submit a minimal final report using the template below.

### Minimal final report template
- PR ID:
- Branch:
- Scope delivered:
- Queue update:
- Validation:
  - `scripts/ci.sh`: PASS/FAIL
- Artifacts:
  - `artifacts/ci.log`
  - `artifacts/pr.patch`
- Secrets check:
  - Confirmed no secrets added to repo.
  - Confirmed runtime mutable data path is `/data`.
- Out of scope / deferred:
