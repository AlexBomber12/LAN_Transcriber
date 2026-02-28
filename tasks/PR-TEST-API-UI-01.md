PR-TEST-API-UI-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/test-api-ui-01
PR title: PR-TEST-API-UI-01 Add end-to-end API and UI route tests for Upload and Export workflow
Base branch: main

Prerequisites
This PR assumes the following are implemented:
- POST /api/uploads (PR-UI-UPLOAD-01)
- /upload UI page (PR-UI-UPLOAD-02)
- /ui/recordings/{id}/export.zip and export preview (PR-EXPORT-01)

Goal
Increase coverage of the primary user workflow without Docker:
Upload -> recording exists -> UI pages render -> Export ZIP downloads.

Implementation

A) Add tests/test_upload_export_flow.py
- Create a TestClient against lan_app.api:app.
- Use the same config setup as tests/test_ui_routes.py:
  - monkeypatch api._settings and ui_routes._settings
  - init_db(cfg)

- Stub queueing so tests do not require Redis:
  - monkeypatch lan_app.api.enqueue_recording_job to a stub that creates a DB job row and returns a job_id.

Test 1: upload creates recording and file
- POST /api/uploads with a small fake mp3.
- Assert 200 and response includes recording_id.
- Assert raw audio exists under recordings_root/<id>/raw/audio.mp3.

Test 2: UI pages render for new recording
- GET /upload returns 200 and contains the file picker.
- GET /recordings returns 200 and includes the new recording id or filename.
- GET /recordings/<id> returns 200.

Test 3: Export ZIP downloads
- GET /ui/recordings/<id>/export.zip returns 200.
- Parse ZIP bytes and assert it contains onenote.md and manifest.json.

B) Keep tests stable and fast
- Do not depend on any external network.
- Do not import whisperx.

Local verification
- scripts/ci.sh

Success criteria
- The upload and export flow is covered by unit tests.
- Tests run quickly and do not require Docker.
- scripts/ci.sh is green.
```
