PR-UI-UPLOAD-02

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/ui-upload-02
PR title: PR-UI-UPLOAD-02 Upload page UI (multi-file) with per-file progress and processing polling
Base branch: main

Goal:
1) Add a dedicated Upload page for adding 1 or more recordings directly through the web UI.
2) Show upload progress (0-100%) per file while the browser sends data to the server.
3) After upload, show processing progress and stage per file by polling the existing recordings API.

Scope:
- UI only. Use existing POST /api/uploads (PR-UI-UPLOAD-01).
- Do not introduce SSE/WebSocket yet. Use polling for processing progress.

A) Navigation
- Update lan_app/templates/base.html:
  - Add a new nav item: Upload -> /upload
  - Keep Connections for now.

B) UI route
- In lan_app/ui_routes.py add:
  - GET /upload -> templates.TemplateResponse("upload.html", {"active": "upload"})
- Ensure it works with optional bearer auth like other pages.

C) Upload page template
- Add lan_app/templates/upload.html (extends base.html), using the same minimal DB-window style.
- Layout:
  1) Header "Upload"
  2) Dropzone area and file picker:
     - <input type="file" id="file-input" multiple>
     - A visible drop area that forwards dropped files to the same handler
  3) A table/list showing one row per file with columns:
     - Filename
     - Size
     - Upload progress
     - Processing progress
     - Status
     - Link (Open recording)

D) Browser upload logic (plain JS)
- Implement upload using XMLHttpRequest so progress events are available:
  - POST /api/uploads
  - FormData with field name "file"
  - xhr.upload.onprogress => update upload percent
- After a file is uploaded successfully:
  - Read recording_id from JSON response
  - Show link to /recordings/<recording_id>
  - Start polling GET /api/recordings/<recording_id> every 2 seconds:
    - status = recording.status
    - stage = recording.pipeline_stage
    - progress = recording.pipeline_progress (0.0-1.0)
  - Render:
    - processing_percent = round(progress * 100)
    - stage label: stage or "Waiting"
  - Stop polling when status is Ready, NeedsReview, Published, Quarantine, Failed.

E) Concurrency and robustness
- Upload files sequentially by default to avoid saturating the server.
- Provide a simple queue in JS:
  - user can select 50 files without the browser spawning 50 parallel uploads.
- Handle errors:
  - If upload returns non-200, show error in that row.
  - If polling fails, show "polling failed" but keep the Open link.

F) Tests
- Update tests/test_ui_routes.py:
  - Add a test that GET /upload returns 200 and contains "Upload" and id="file-input".
  - No JS execution is required in tests.

Local verification:
- scripts/ci.sh
- Manual:
  1) Open /upload
  2) Drop 2 files
  3) Verify upload progress bars move
  4) Verify the list switches to processing polling and shows stage and percent
  5) Click Open and see the recording detail page

Success criteria:
- /upload exists, is linked in the nav, and accepts multi-file selection and drag-drop.
- Each file row shows upload percent during transfer and then shows pipeline stage and percent during processing.
- scripts/ci.sh is green.
```
