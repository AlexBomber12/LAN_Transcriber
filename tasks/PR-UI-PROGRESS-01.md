PR-UI-PROGRESS-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/ui-progress-01
PR title: PR-UI-PROGRESS-01 UI progress feedback and Connections real status
Base branch: main

Goal:
1) Add pipeline stage and progress tracking stored in DB.
2) Show progress in the recording detail UI with HTMX polling.
3) Make Connections page reflect real Google Drive status and provide Run ingest button.

A) DB fields for progress
- Add migration:
  - recordings.pipeline_stage TEXT
  - recordings.pipeline_progress REAL
  - recordings.pipeline_updated_at TEXT
  - recordings.last_warning TEXT
- Add helpers:
  - set_recording_progress(recording_id, stage, progress)
  - clear_recording_progress(recording_id)

B) Pipeline progress callback
- In lan_transcriber/pipeline.py orchestrator:
  - Add progress callback parameter.
  - Emit stage updates:
    - precheck 0.05
    - stt 0.30
    - diarize 0.50
    - align 0.60
    - language 0.70
    - llm 0.85
    - metrics 0.95
    - done 1.00
- In lan_app/worker_tasks.py:
  - Provide a callback that writes stage and progress into DB.
  - Clear progress on finish or failure.

C) HTMX progress polling
- Add GET /ui/recordings/{id}/progress returning an HTML partial.
- Update recording detail template:
  - When status is Processing, poll progress endpoint every 2 seconds and render progress bar and stage.

D) Connections page improvements
- In /ui/connections:
  - Show Google Drive configured when required settings exist.
  - Add "Test connection" action that lists 1 item from the folder and shows result.
  - Add "Run ingest now" button calling POST /api/actions/ingest.
  - Display 409 errors (ingest already running) clearly.

E) Tests
- Test progress endpoint returns expected HTML content after setting progress in DB.
- Keep scripts/ci.sh green.

Local verification:
- scripts/ci.sh

Success criteria:
- Processing recordings show live stage and progress in UI.
- Connections page shows real GDrive config state and can trigger ingest.
- scripts/ci.sh is green.
```
