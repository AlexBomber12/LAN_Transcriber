PR: PR-CALENDAR-ICS-01
Title: Calendar import via ICS (URL subscription and optional file upload)

Context
Microsoft Graph calendar integration has been removed. We still need calendar support for meeting context, but via vendor-neutral ICS. Implement importing events from an ICS feed URL (subscription). Optionally support importing by uploading an .ics file as a source type.

Scope
- Backend:
  - Store calendar sources and normalized events in SQLite
  - Provide endpoints to create sources, sync sources, list sources, and query events
  - Implement server-side fetch + parse for ICS URLs (avoid browser CORS issues)
  - Parse recurring events and timezones correctly for common cases
- UI:
  - A simple Calendars page to add a source and trigger Sync Now
  - A basic events list view for a date range (enough to validate the pipeline end-to-end)
- Tests:
  - Unit tests for ICS parsing (timezone + recurrence)
  - API tests for source creation and sync

Design decisions (use these defaults unless the codebase already has a preferred pattern)
- Data model:
  - calendar_sources(id, name, kind, url, created_at, last_synced_at, last_error)
  - calendar_events(id, source_id, uid, starts_at, ends_at, all_day, summary, description, location, organizer, updated_at)
- Recurrence expansion window:
  - Expand recurring events for a bounded window (example: from today - 30 days to today + 180 days)
  - Make this window configurable via env or settings if the app already supports runtime config
- Privacy:
  - Treat the ICS URL as sensitive. Do not return it in normal list APIs (return a redacted/boolean instead).
  - Do not log the full URL or the fetched calendar body.

Security and reliability requirements (minimum viable hardening)
- Only allow http and https URLs.
- Enforce timeouts and a maximum download size for ICS fetches.
- Limit redirects (for example: max 5) and do not follow redirects to non-http(s) schemes.
- If feasible without breaking LAN usage, add basic SSRF protection:
  - At minimum, block file:// and localhost-only schemes
  - Do not allow credentials in the URL (user:pass@host)

Backend tasks
1) Dependencies
- Add the required Python dependencies for ICS parsing and recurrence expansion.
- Prefer widely used libraries with good timezone + RRULE support.
- Add them to the main dependency files and ensure they are included in Docker images and CI.

2) Migrations
- Add migration files to create calendar_sources and calendar_events.
- Add any indexes needed for range queries (starts_at) and joins (source_id).

3) ICS ingestion module
- Create a module (for example: lan_app/calendar/ics.py) that:
  - fetches ICS content for a given source (URL or stored file content)
  - parses VEVENT entries into normalized events
  - expands recurrence rules into concrete instances inside the expansion window
  - returns a list of normalized event records ready for upsert

4) Upsert logic
- Implement deterministic upsert so repeated syncs are idempotent.
- Suggested natural keys:
  - (source_id, uid, starts_at) for expanded instances
- Handle deletions gracefully:
  - Simplest MVP: on each sync, delete all events for that source inside the expansion window and re-insert
  - Or: compute diff and upsert, but keep it simple if the dataset is small

5) API endpoints
- POST /api/calendar/sources
  - body: name, kind (url|file), url or file
  - response: source id and metadata (do not echo the raw URL if you treat it as secret)
- GET /api/calendar/sources
  - list sources with last_synced_at and last_error, plus a redacted url indicator
- POST /api/calendar/sources/{id}/sync
  - triggers fetch + parse + upsert, updates last_synced_at and last_error
- GET /api/calendar/events?from=...&to=...&source_id=...
  - returns events for the time range
- If the API layer has an auth option, keep it consistent with existing patterns.

UI tasks
1) Add a Calendars page (or Settings -> Calendars)
- List existing sources with:
  - name
  - last sync time
  - last error (if any)
  - Sync Now button
- Add source form:
  - name
  - source type: URL (MVP) and optionally file
  - URL input with basic validation
- Redact URL in UI after saving (show host only or show "Configured" boolean).

2) Add an Events view
- Minimal UI that shows events for:
  - a date range picker (or simple Today/This week presets)
  - a table with start, end, summary, location
- This is to validate the ingestion end-to-end.

Tests
1) Unit tests
- Add fixture ICS files:
  - timezone-aware DTSTART/DTEND
  - recurring event with RRULE
- Validate that parsing returns the correct number of expanded instances inside a window and correct timestamps.

2) API tests
- Create a source using a local test server or mocked http client for fetching ICS.
- Trigger sync and assert events are stored and returned by the events endpoint.

Validation checklist
- Manual:
  - Add ICS source URL, click Sync Now, see events in Events list
  - Re-sync is idempotent (no duplicates)
  - Bad URL shows last_error and does not crash the worker/api
- Automated:
  - Unit tests green
  - API tests green
  - Docker smoke and CI green

Success criteria
- User can add an ICS calendar source (URL) and manually sync it.
- Events appear in the UI for a selected date range.
- Recurring events expand correctly within a bounded window.
- ICS fetch uses server-side networking with timeouts and size limits.
- Sensitive URL details are not leaked to logs or normal list APIs.
- All relevant tests and CI checks are green.
