PR-COVERAGE-ENFORCE-100-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/coverage-enforce-100-01
PR title: PR-COVERAGE-ENFORCE-100-01 Enforce 100% statement and branch coverage in CI
Base branch: main

Change
- Update scripts/ci.sh to add:
  --cov-fail-under=100
while keeping:
  --cov=lan_transcriber --cov=lan_app --cov-branch

Success criteria
- CI fails unless coverage is exactly 100% statement and branch coverage.
```
