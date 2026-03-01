PR-COV-LAN_APP-CALENDAR-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/cov-lan-app-calendar-01
PR title: PR-COV-LAN_APP-CALENDAR-01 Raise calendar ICS parsing + service matching to 100% coverage
Base branch: main

Targets
- lan_app/calendar/ics.py
- lan_app/calendar/service.py

Approach
Static ICS strings, deterministic time, cover all parse/match branches.

Success criteria
- Both modules reach 100% statement and branch coverage.
- scripts/ci.sh is green.
```
