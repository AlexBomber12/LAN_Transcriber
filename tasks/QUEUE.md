QUEUE

Purpose
- Execute planned PRs in order, without skipping.
- Each PR should be implemented exactly as described in its corresponding tasks file.

Status legend
- TODO: not started
- DOING: in progress
- DONE: merged to main

Queue (in order)

1) PR-BOOTSTRAP-01: Repo bootstrap: tasks system, CI scripts, artifacts, runtime data layout
- Status: DONE
- Tasks file: tasks/PR-BOOTSTRAP-01.md
- Depends on: none

2) PR-REFRACTOR-CORE-01: Refactor existing code into a stable core pipeline library and /data-backed state
- Status: DONE
- Tasks file: tasks/PR-REFRACTOR-CORE-01.md
- Depends on: PR-BOOTSTRAP-01

3) PR-DB-QUEUE-01: Introduce SQLite DB, migrations, job queue, and unified recording/job status model
- Status: DONE
- Tasks file: tasks/PR-DB-QUEUE-01.md
- Depends on: PR-REFRACTOR-CORE-01

4) PR-UI-SHELL-01: Web UI skeleton: dashboard + DB-table recordings list + recording detail shell + connections shell
- Status: DONE
- Tasks file: tasks/PR-UI-SHELL-01.md
- Depends on: PR-DB-QUEUE-01

5) PR-GDRIVE-INGEST-01: Google Drive API ingest using Service Account + shared folder (Inbox)
- Status: DONE
- Tasks file: tasks/PR-GDRIVE-INGEST-01.md
- Depends on: PR-DB-QUEUE-01

6) PR-MS-AUTH-01: Microsoft Graph delegated auth (work) via Device Code Flow + token cache
- Status: DONE
- Tasks file: tasks/PR-MS-AUTH-01.md
- Depends on: PR-UI-SHELL-01

7) PR-CALENDAR-01: Read M365 calendar via Graph and match events to recordings; manual override in UI
- Status: DONE
- Tasks file: tasks/PR-CALENDAR-01.md
- Depends on: PR-MS-AUTH-01 and PR-GDRIVE-INGEST-01

8) PR-PIPELINE-01: STT + diarization + word timestamps + speaker-attributed transcript + speaker snippets
- Status: DONE
- Tasks file: tasks/PR-PIPELINE-01.md
- Depends on: PR-GDRIVE-INGEST-01 and PR-REFRACTOR-CORE-01 and PR-DB-QUEUE-01

9) PR-LANG-01: Multi-language support (2 languages): language spans, dominant language, UI override + reprocess hooks
- Status: DONE
- Tasks file: tasks/PR-LANG-01.md
- Depends on: PR-PIPELINE-01

10) PR-LLM-01: Spark LLM: topic, summary, decisions, actions (owner/deadline), emotional summary, question typing
- Status: DONE
- Tasks file: tasks/PR-LLM-01.md
- Depends on: PR-PIPELINE-01 and PR-LANG-01

11) PR-METRICS-01: Conversation analytics metrics per participant and per meeting + UI rendering + exports
- Status: DONE
- Tasks file: tasks/PR-METRICS-01.md
- Depends on: PR-PIPELINE-01 and PR-LLM-01

12) PR-VOICE-01: Voice profiles + mapping diarization speakers to known people (human-in-the-loop UI)
- Status: DONE
- Tasks file: tasks/PR-VOICE-01.md
- Depends on: PR-PIPELINE-01 and PR-UI-SHELL-01

13) PR-ONENOTE-01: Projects mapping to OneNote sections + Publish to OneNote (work)
- Status: DONE
- Tasks file: tasks/PR-ONENOTE-01.md
- Depends on: PR-MS-AUTH-01 and PR-LLM-01 and PR-METRICS-01

14) PR-ROUTING-01: Project suggestion (calendar + voices + text) with confidence + NeedsReview workflow
- Status: DONE
- Tasks file: tasks/PR-ROUTING-01.md
- Depends on: PR-CALENDAR-01 and PR-VOICE-01 and PR-ONENOTE-01

15) PR-OPS-01: Retention, quarantine cleanup, retries, runbook, and production hardening for LAN deployment
- Status: DONE
- Tasks file: tasks/PR-OPS-01.md
- Depends on: PR-ONENOTE-01 and PR-ROUTING-01

16) PR-JOB-MODEL-01: Single job pipeline model (remove placeholder jobs; restrict requeue/retry)
- Status: DONE
- Tasks file: tasks/PR-JOB-MODEL-01.md
- Depends on: PR-OPS-01

17) PR-ENTRYPOINT-01: Unify entrypoint to lan_app.api:app + worker (Dockerfile CMD, systemd, smoke_test, tests)
- Status: DONE
- Tasks file: tasks/PR-ENTRYPOINT-01.md
- Depends on: PR-JOB-MODEL-01

18) PR-STAGING-01: Fix staging deploy workflow and add infra/staging (compose + env template + real smoke)
- Status: DONE
- Tasks file: tasks/PR-STAGING-01.md
- Depends on: PR-ENTRYPOINT-01

19) PR-SECURITY-01: Optional bearer auth + abuse guards (rate limit, dedupe) for ingest/requeue/delete
- Status: DONE
- Tasks file: tasks/PR-SECURITY-01.md
- Depends on: PR-STAGING-01

20) PR-WORKER-ROBUST-01: Worker robustness: graceful shutdown, timeouts, stuck job recovery, terminal failures
- Status: DONE
- Tasks file: tasks/PR-WORKER-ROBUST-01.md
- Depends on: PR-SECURITY-01

21) PR-PIPELINE-MODULAR-01: Split pipeline.py into testable modules + consolidate utils + robust LLM parsing with schema and raw artifacts
- Status: DONE
- Tasks file: tasks/PR-PIPELINE-MODULAR-01.md
- Depends on: PR-WORKER-ROBUST-01

22) PR-DB-RESILIENCE-01: SQLite resilience: busy timeout, retry-on-locked, migration files, safer connection management
- Status: DONE
- Tasks file: tasks/PR-DB-RESILIENCE-01.md
- Depends on: PR-PIPELINE-MODULAR-01

23) PR-UI-PROGRESS-01: UI feedback: pipeline progress/stage + Connections page real status and Run ingest button
- Status: DONE
- Tasks file: tasks/PR-UI-PROGRESS-01.md
- Depends on: PR-DB-RESILIENCE-01

24) PR-RUNTIME-CONFIG-01: Runtime hardening: fail-fast config for staging/prod + FastAPI lifespan + docs alignment
- Status: DONE
- Tasks file: tasks/PR-RUNTIME-CONFIG-01.md
- Depends on: PR-UI-PROGRESS-01

25) PR-UI-UPLOAD-01: Upload ingest API (multipart) create recordings from UI uploads
- Status: DONE
- Tasks file: tasks/PR-UI-UPLOAD-01.md
- Depends on: PR-RUNTIME-CONFIG-01

26) PR-UI-UPLOAD-02: Upload page UI: multi-file upload + per-file upload progress + processing polling
- Status: DONE
- Tasks file: tasks/PR-UI-UPLOAD-02.md
- Depends on: PR-UI-UPLOAD-01

27) PR-EXPORT-01: Export-only output: OneNote-ready markdown + Download ZIP per recording
- Status: DONE
- Tasks file: tasks/PR-EXPORT-01.md
- Depends on: PR-UI-UPLOAD-02

28) PR-UI-PROGRESS-02: Show pipeline progress on Recordings list table
- Status: DONE
- Tasks file: tasks/PR-UI-PROGRESS-02.md
- Depends on: PR-EXPORT-01

29) PR-REMOVE-MS-01: Remove Microsoft Graph, calendar matching UI, OneNote publish UI and msal dependency
- Status: DONE
- Tasks file: tasks/PR-REMOVE-MS-01.md
- Depends on: PR-UI-PROGRESS-02

30) PR-REMOVE-GDRIVE-01: Remove Google Drive ingest, Connections page, ingest lock, and Google API deps
- Status: DONE
- Tasks file: tasks/PR-REMOVE-GDRIVE-01.md
- Depends on: PR-REMOVE-MS-01

31) PR-DOCS-EXPORT-ONLY-01: Docs update for upload + export-only mode (README, runbook, env example, nginx notes)
- Status: DONE
- Tasks file: tasks/PR-DOCS-EXPORT-ONLY-01.md
- Depends on: PR-REMOVE-GDRIVE-01

32) PR-FIX-DIARIZATION-REVISION-01: Fix pyannote diarization revision handling and fallback (avoid failed recordings)
- Status: DONE
- Tasks file: tasks/PR-FIX-DIARIZATION-REVISION-01.md
- Depends on: PR-DOCS-EXPORT-ONLY-01

33) PR-FIX-WHISPERX-API-01: Fix WhisperX API usage (no whisperx.transcribe) and add modern-path unit test
- Status: DONE
- Tasks file: tasks/PR-FIX-WHISPERX-API-01.md
- Depends on: PR-FIX-DIARIZATION-REVISION-01
