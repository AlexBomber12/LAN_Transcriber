PR-COV-LAN_APP-DB-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/cov-lan-app-db-01
PR title: PR-COV-LAN_APP-DB-01 Raise lan_app.db to 100% statement and branch coverage
Base branch: main

Approach
sqlite tmp db, systematic CRUD coverage, simulate locked-db retry once.

Success criteria
- lan_app/db.py reaches 100% statement and branch coverage.
- scripts/ci.sh is green.
```
