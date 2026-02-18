PR ID: PR-CALENDAR-01
Branch: pr/calendar-01

Goal
Read M365 calendar via Graph and match events to recordings; manual override in UI.

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
PR-MS-AUTH-01, PR-GDRIVE-INGEST-01

Work plan
1) Implement calendar event fetch for a time window around captured_at
   - Use Graph endpoint:
     - /me/calendarView?startDateTime=...&endDateTime=...
   - Default window: captured_at - 45 minutes to + 45 minutes (configurable).
   - Store candidates in DB (calendar_matches) with:
     - event_id, subject, start/end, organizer, attendees, location
     - computed score and rationale

2) Event matching and scoring
   - Score components (MVP):
     - time overlap score (highest weight)
     - proximity score (if no overlap)
     - subject keyword match score (optional)
   - Pick best candidate automatically if score >= threshold, else leave unmatched.

3) UI: Calendar tab in Recording Detail
   - Show selected event (if any)
   - Show candidate list (top 5) with radio selection and "Save selection"
   - Show extracted signals:
     - title tokens
     - attendees list
     - organizer
   - Provide a "No event" option.

4) API
   - GET /api/recordings/<id>/calendar
   - POST /api/recordings/<id>/calendar/select {event_id|null}

Local verification
- With Graph connected, open a recording and fetch candidates.
- Select an event manually and verify DB persisted selection.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- Recordings can be matched to calendar events automatically and manually.
- Calendar context (title + attendees) is stored and available for routing and display.
