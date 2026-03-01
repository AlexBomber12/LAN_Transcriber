PR-COV-LAN_APP-HEALTH-WORKER-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/cov-lan-app-health-worker-01
PR title: PR-COV-LAN_APP-HEALTH-WORKER-01 Raise small lan_app modules to 100% coverage
Base branch: main

Targets
- lan_app/healthchecks.py
- lan_app/auth.py
- lan_app/worker.py
- lan_app/workers.py
- lan_app/db_init.py
- lan_app/uploads.py
- lan_app/hf_repo.py

Approach
Add offline unit tests using monkeypatch and fakes. No Redis, no GPU.

Success criteria
- Each target module reaches 100% statement and branch coverage.
- scripts/ci.sh is green.
```
