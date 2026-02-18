PR ID: PR-GDRIVE-INGEST-01
Branch: pr/gdrive-ingest-01

Goal
Google Drive API ingest using Service Account + shared folder (Inbox).

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
PR-DB-QUEUE-01

Work plan
1) Implement Google Drive API ingest using Service Account + shared folder
   - Add dependencies:
     - google-auth
     - google-api-python-client
   - Configuration (env vars):
     - GDRIVE_SA_JSON_PATH (mounted under /data/secrets/)
     - GDRIVE_INBOX_FOLDER_ID
     - GDRIVE_POLL_INTERVAL_SECONDS (default 60)
   - Document setup:
     - enable Drive API in GCP
     - create service account and key JSON
     - share the Inbox folder with service account email (viewer is enough if downloads allowed)

2) Poll for new files in Inbox
   - For MVP, use list with query:
     - "'<folderId>' in parents and trashed=false"
   - Detect new by drive fileId not yet present in recordings.drive_file_id.
   - Store drive md5Checksum when available to prevent reprocessing duplicates.
   - Download file to /data/recordings/<id>/raw/audio.<ext>
   - Move or label processed files in Drive:
     - either move to a Processed folder (optional in MVP), or
     - add an appProperties flag "lan_ingested=true" (preferred if allowed), or
     - keep DB as source of truth and leave file in Inbox (least invasive)

3) Parse captured_at from Plaud filename
   - Implement robust parsing for formats like:
     - "2026-02-18 16_01_43.mp3"
     - "2026-02-18 16-01-43.mp3"
   - If parse fails, fall back to Drive createdTime and mark a warning flag.

4) Enqueue processing jobs
   - After ingest:
     - create jobs: precheck -> stt -> diarize -> align -> language -> llm -> metrics -> ready
   - Keep steps lightweight if later PRs are not merged; create placeholders but do not break.

Local verification
- Provide a small script or admin UI action to run a one-shot ingest cycle.
- With a shared test folder, ingest at least 1 file and verify recording row + raw file on disk.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- New files appearing in the shared Drive Inbox are ingested into DB and stored under /data.
- Ingest is idempotent and does not duplicate recordings.
- captured_at is derived from filename when possible.
