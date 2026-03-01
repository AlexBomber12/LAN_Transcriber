PR-COV-LAN_APP-EXPORT-OPS-JOBS-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/cov-lan-app-export-ops-jobs-01
PR title: PR-COV-LAN_APP-EXPORT-OPS-JOBS-01 Raise exporter/jobs/ops/reaper to 100% coverage
Base branch: main

Targets
- lan_app/exporter.py
- lan_app/jobs.py
- lan_app/ops.py
- lan_app/reaper.py

Approach
Offline tests, stub RQ/Redis, use tmp dirs for files.

Success criteria
- All targets reach 100% statement and branch coverage.
- scripts/ci.sh is green.
```
