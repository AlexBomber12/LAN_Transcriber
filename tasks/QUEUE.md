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
- Status: TODO
- Tasks file: tasks/PR-VOICE-01.md
- Depends on: PR-PIPELINE-01 and PR-UI-SHELL-01

13) PR-ONENOTE-01: Projects mapping to OneNote sections + Publish to OneNote (work)
- Status: TODO
- Tasks file: tasks/PR-ONENOTE-01.md
- Depends on: PR-MS-AUTH-01 and PR-LLM-01 and PR-METRICS-01

14) PR-ROUTING-01: Project suggestion (calendar + voices + text) with confidence + NeedsReview workflow
- Status: TODO
- Tasks file: tasks/PR-ROUTING-01.md
- Depends on: PR-CALENDAR-01 and PR-VOICE-01 and PR-ONENOTE-01

15) PR-OPS-01: Retention, quarantine cleanup, retries, runbook, and production hardening for LAN deployment
- Status: TODO
- Tasks file: tasks/PR-OPS-01.md
- Depends on: PR-ONENOTE-01 and PR-ROUTING-01
