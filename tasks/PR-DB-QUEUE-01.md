PR ID: PR-DB-QUEUE-01
Branch: pr/db-queue-01

Goal
Introduce SQLite DB, migrations, job queue, and unified recording/job status model.

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
PR-REFRACTOR-CORE-01

Work plan
1) Add SQLite DB (MVP) with migrations
   - Choose a migration approach that is simple and stable:
     - Alembic + SQLAlchemy, or
     - sqlite schema versioning with explicit migration scripts
   - Store DB at /data/db/app.db.

2) Define core tables (MVP schema)
   - recordings:
     - id, source, source_filename, captured_at, duration_sec
     - status (Queued/Processing/NeedsReview/Ready/Published/Quarantine/Failed)
     - quarantine_reason, language_auto, language_override
     - project_id nullable, onenote_page_id nullable
     - drive_file_id nullable, drive_md5 nullable
     - created_at, updated_at
   - jobs:
     - id, recording_id, type, status, attempt, error, started_at, finished_at
   - projects:
     - id, name, onenote_section_id, onenote_notebook_id, auto_publish bool default false
   - voice_profiles:
     - id, display_name, notes optional
   - speaker_assignments:
     - recording_id, diar_speaker_label, voice_profile_id, confidence
   - calendar_matches:
     - recording_id, selected_event_id, selected_confidence
     - candidates_json (or separate table) storing top candidates
   - metrics tables (placeholders for PR-METRICS-01):
     - meeting_metrics (recording_id, json)
     - participant_metrics (recording_id, voice_profile_id nullable, diar_speaker_label, json)

3) Add a job queue and worker
   - Use Redis + RQ for MVP.
   - Implement job types as explicit constants:
     - ingest, precheck, stt, diarize, align, language, llm, metrics, publish, cleanup
   - Implement a worker process that:
     - picks jobs
     - updates job status + recording status
     - writes step logs under /data/recordings/<id>/logs/

4) Add API endpoints (backend only)
   - GET /api/recordings (filters, pagination)
   - GET /api/recordings/<id>
   - POST /api/recordings/<id>/actions/requeue
   - POST /api/recordings/<id>/actions/quarantine
   - POST /api/recordings/<id>/actions/delete
   - GET /api/jobs (basic)

Local verification
- docker-compose up starts db + redis + worker + api.
- Create a fake recording row and enqueue a no-op job to validate the pipeline.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- DB schema is in place with migrations and stored under /data.
- Jobs can be enqueued and processed by a worker with status transitions recorded in DB.
- Basic API endpoints exist for UI to consume.
