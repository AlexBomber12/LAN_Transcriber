PR-CALENDAR-MATCHING-ICS-01
===========================

Branch: pr-calendar-matching-ics-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make ICS calendar support reliable and actually connected to recordings. Right now the app can sync ICS sources, but lan_app/calendar/ics.py does not persist ATTENDEE data, lan_app/db.py replace_calendar_events_for_window() can leave stale overlapping events on resync, and calendar_matches exists without a complete runtime path that consistently populates/uses it. This PR must harden ICS parsing/sync, implement real recording-to-calendar matching, and expose manual override in the recording UI.

Constraints
- Build on top of the existing ICS calendar feature. Do not reintroduce Microsoft Graph.
- Keep the implementation vendor-neutral and SQLite-backed.
- Do not redesign the whole calendar UI. Add the missing reliability and recording integration only.
- Maintain 100% statement and branch coverage for every changed/new project module.
- Keep matching deterministic and explainable.

Implementation requirements

1) Extend the ICS event model with attendee-aware data
- Parse ATTENDEE entries from ICS payloads in lan_app/calendar/ics.py, preserving practical user-facing fields such as display name and email where available.
- Normalize organizer information similarly if the current organizer string is insufficient.
- Persist attendee data in calendar_events via a safe migration, for example attendees_json plus any organizer normalization columns you need.
- Preserve recurrence expansion, all-day handling, and timezone normalization.

2) Fix sync-window replacement semantics
- In lan_app/db.py replace_calendar_events_for_window(), stop deleting only rows whose starts_at falls inside the sync window.
- Remove/replace stale events by overlap with the synced window so events that start before the window but extend into it are handled correctly.
- Keep sync idempotent and deterministic.
- Preserve safe error handling and source last_error behavior.

3) Implement a dedicated recording-to-calendar matching service
- Add a focused matching module, for example lan_app/calendar/matching.py.
- Score candidate events for a recording using deterministic signals such as:
  - captured_at proximity to event start
  - time overlap with the recording interval when duration is known
  - recording duration versus event duration when useful
  - optional token overlap using recording source filename/title if cheap and reliable
- Return ordered candidates with explicit score/rationale payloads.
- Auto-select only when the match is strong enough or unambiguous. Otherwise persist candidates and leave selection empty for human review.
- Reuse existing calendar_matches persistence instead of inventing a second parallel mechanism.

4) Wire matching into the real runtime flow
- Ensure calendar matching runs for actual recordings, not only in tests.
- A practical implementation is:
  - initial candidate generation when a recording is created or first processed using captured_at
  - refresh/re-score after sanitized duration is known if that improves accuracy
  - store/update candidates via upsert_calendar_match
- Never block transcription if no calendar match exists.
- Ensure selected calendar context is available before summary/glossary context is finalized whenever possible.

5) Make selected calendar context usable by downstream logic
- Ensure the selected calendar candidate payload contains the fields downstream code needs, including title/summary, attendees, organizer, timing, source name, and rationale/confidence.
- Make worker_tasks summary context and the glossary pipeline able to consume the selected event consistently.
- Keep the payload JSON-storable and backwards compatible where reasonable.

6) Add manual override UI on the recording detail page
- Add a calendar section or tab on the recording detail page that shows:
  - current selected event or None
  - ordered candidate events with local time, source, confidence, and rationale
  - attendees preview
  - action to choose a candidate or clear the selection
- Keep the UI lightweight and server-rendered. No frontend framework changes.
- If no candidates exist, show a clear empty state.

7) Improve calendar-page observability
- Render calendar event times in local/Europe-Rome time rather than raw UTC-looking strings.
- Show enough source/sync information to understand whether sync worked, for example last synced time, event count, and last error if present.
- Keep the current source model (URL/file). No large source-management redesign is needed in this PR.

8) Tests (100% coverage)
Add deterministic offline tests for at least these cases:
- ICS parsing captures attendees and organizer details correctly
- recurring events retain attendee data through expansion
- resync deletes/replaces stale overlapping events at the window boundary correctly
- matcher scoring orders candidates predictably and only auto-selects strong matches
- real runtime flow populates calendar_matches for recordings
- manual selection/clear routes persist correctly
- recording detail UI renders selected event, candidates, rationale, and attendees
- downstream summary/glossary context can load attendees from the selected match

9) Documentation
- Update README and/or docs/runbook.md briefly to explain:
  - ICS attendees are now parsed and stored
  - recordings are automatically matched to calendar events when possible
  - ambiguous matches remain reviewable in the recording UI
  - calendar times in the UI are shown in local/Europe-Rome time

Verification steps (must be included in PR description)
- Sync an ICS source with recurring meetings and confirm attendees are visible in stored/rendered events.
- Process a recording with a known captured_at near a meeting and confirm calendar_matches is populated with candidates and a sensible auto-selection when confidence is high.
- Override the match manually on the recording detail page and confirm summary/glossary context uses the selected event.
- Re-sync a calendar where an event spans the sync-window boundary and confirm stale rows are not left behind.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Attendee-aware ICS parsing and persistence
- Correct sync-window replacement logic
- Deterministic recording-to-calendar matcher with rationale/confidence
- Runtime population of calendar_matches
- Recording-detail manual override UI for calendar selection
- Better calendar-page timezone/observability behavior
- Tests with 100% statement and branch coverage for changed/new modules

Success criteria
- Calendar sync no longer feels unstable around recurring/boundary events.
- Recordings actually receive usable calendar candidates and selected-event context.
- Attendees from ICS are preserved and visible.
- Manual override exists when auto-match is ambiguous.
- Downstream summary/glossary logic can consume the selected calendar event reliably.
- CI remains green.
```
