PR-CAPTURE-TIME-SEMANTICS-01
=============================

Branch: pr-capture-time-semantics-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Implement a correct capture-time model for uploaded recordings so that filename timestamps from Plaud-style uploads are treated as local source time, then normalized to UTC before persistence. Today lan_app/uploads.py parses a filename like 2026-03-06_12_02_16.mp3 as 2026-03-06T12:02:16Z by attaching timezone.utc directly. The UI then converts that stored UTC value to Europe/Rome and shows 13:02:16 CET, which is wrong by 1 hour in winter and will also be wrong across DST boundaries. This breaks calendar matching because the recording is searched against the wrong event time. Fix the time semantics end-to-end, preserve local display behavior, and add safe metadata/backfill for already-uploaded rows.

Constraints
- Scope is upload/Plaud capture-time semantics, persistence metadata, idempotent backfill, and tests/docs.
- Do not redesign calendar matching scoring in this PR. Matching should simply start from correct captured_at values.
- Do not change calendar event display formatting in /calendars.
- Do not change recording-detail local display formatting logic beyond feeding it correct data.
- Do not alter non-upload ingestion semantics unless required for shared helper correctness.
- Keep 100% statement and branch coverage for every changed/new module.
- Keep deployment safe for existing SQLite databases and LAN Docker usage.

Implementation requirements

1) Add explicit upload capture timezone configuration
- Add AppSettings support for a new setting:
  - upload_capture_timezone: default Europe/Rome
  - accept env aliases LAN_UPLOAD_CAPTURE_TIMEZONE and UPLOAD_CAPTURE_TIMEZONE
- Validate the timezone via zoneinfo. If the configured timezone is invalid, fail fast during settings validation with a clear message.
- Do not reuse the UI display timezone setting implicitly. Upload source timezone and UI display timezone are different concerns.

2) Split filename parsing into local-time extraction and UTC normalization
- In lan_app/uploads.py stop attaching timezone.utc directly inside parse_plaud_captured_at(...).
- Introduce a helper that extracts a naive local datetime from Plaud-style filenames, for example:
  - parse_plaud_captured_local_datetime(filename) -> datetime | None
- Introduce a normalization helper that takes a naive local datetime plus a configured ZoneInfo and returns canonical UTC ISO text.
- Make infer_captured_at(...) use the configured upload_capture_timezone when the filename contains an embedded timestamp.
- Preserve the fallback behavior for filenames with no embedded timestamp: use current UTC time.
- Preserve safe_filename/suffix/write helpers.

3) Persist capture-time provenance metadata on recordings
- Add explicit recording columns via a new migration file using the next migration number. Add at least:
  - captured_at_source TEXT NULL
  - captured_at_timezone TEXT NULL
  - captured_at_inferred_from_filename INTEGER NOT NULL DEFAULT 0
- Update recording creation/select/update helpers so these fields round-trip through lan_app.db and are available in get_recording/list_recordings selectors.
- Update create_recording(...) to accept optional values for these new fields.
- For Plaud-style inferred filenames on upload, persist:
  - captured_at as canonical UTC ISO text
  - captured_at_source as the original local timestamp string or canonical local ISO form
  - captured_at_timezone as the configured source timezone key, for example Europe/Rome
  - captured_at_inferred_from_filename = 1
- For uploads without an embedded timestamp, persist:
  - captured_at as current UTC
  - captured_at_source = NULL
  - captured_at_timezone = configured upload timezone or NULL if you deliberately choose that contract
  - captured_at_inferred_from_filename = 0
- Keep the schema backward compatible for old rows where these fields do not yet exist.

4) Apply the new semantics at upload entrypoints
- Update the upload API path in lan_app/api.py so uploaded recordings use the corrected capture-time inference and populate the new metadata columns.
- Audit any other upload/reingest entrypoint that calls infer_captured_at(...) or create_recording(...), and make behavior consistent.
- Ensure downstream calendar refresh uses the corrected captured_at immediately after recording creation.

5) Add a safe idempotent backfill for existing upload rows
- Existing upload rows created before this PR are already wrong if their captured_at came from the filename. Add an idempotent Python backfill that runs after migrations during init_db(...) or another deterministic DB-init path.
- Backfill scope must be narrow and safe:
  - source == upload
  - source_filename matches the Plaud-style timestamp pattern
  - existing rows are clearly legacy rows without the new provenance metadata populated
- Backfill algorithm:
  - parse local timestamp from source_filename
  - interpret it in upload_capture_timezone
  - recompute canonical UTC captured_at
  - persist provenance metadata
- The backfill must be idempotent and must not double-shift rows that were already fixed.
- Log a concise summary of how many rows were backfilled.
- Do not touch rows that have manual/non-inferred timestamps or rows that cannot be parsed safely.

6) Keep UI semantics stable but show correct data automatically
- Do not redesign recording detail layout.
- Ensure the existing local timestamp display helper in lan_app/ui_routes.py continues to show the correct local time once the stored UTC value is fixed.
- If useful, expose captured_at_source/captured_at_timezone in the overview tab, but keep any UI addition small and production-safe.
- Do not change the already-fixed HH:MM:SS duration behavior from PR-GPU-SCHEDULER-01.

7) Tests with full coverage
Add or update deterministic offline tests for at least these cases:
- parse Plaud timestamp from filename as a naive local datetime
- infer_captured_at(...) with upload_capture_timezone=Europe/Rome on a winter date yields UTC one hour earlier than local input
- infer_captured_at(...) on a summer date yields UTC two hours earlier than local input when DST applies
- filenames without embedded timestamps still fall back to current UTC
- invalid upload_capture_timezone fails settings validation clearly
- create_recording/get_recording/list_recordings round-trip the new capture-time provenance fields
- backfill updates only eligible legacy upload rows and is idempotent on repeated runs
- backfill does not modify rows from non-upload sources or rows without Plaud-style filenames
- calendar refresh after upload receives corrected captured_at values
- existing display helpers still render the fixed captured_at correctly in local time
Update existing tests that currently assert the old wrong UTC semantics, especially uploads-related tests and any coverage tests that hardcode the old expectation.

8) Documentation and operator notes
- Update README and/or runbook docs with a short section that explains:
  - upload filename timestamps are interpreted in UPLOAD_CAPTURE_TIMEZONE
  - values are stored in UTC in the database
  - Europe/Rome is the default for this deployment
  - existing legacy upload rows are backfilled automatically once after upgrade
- Keep the operator note practical and brief.

Verification steps (must be included in PR description)
- Upload a Plaud-style file named 2026-03-06_12_02_16.mp3 with UPLOAD_CAPTURE_TIMEZONE=Europe/Rome and confirm the database stores 2026-03-06T11:02:16Z while the UI shows 2026-03-06 12:02:16 CET.
- Confirm a DST-season example also converts correctly.
- Confirm calendar matching now searches around the corrected capture time without any scoring changes.
- Confirm legacy upload rows are backfilled once and not shifted again on the next app start.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Correct local-source-to-UTC capture-time handling for uploads
- New persisted capture-time provenance fields on recordings
- Safe idempotent legacy backfill
- Updated tests and docs

Success criteria
- Uploaded Plaud-style filenames no longer shift by +1h or +2h when shown in the local UI.
- Calendar matching starts from the correct captured_at value for uploads.
- Old upload rows are repaired safely and only once.
- No unrelated time display behavior regresses.
- CI remains green.
```
