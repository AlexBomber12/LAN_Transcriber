PR-UI-CONTROL-CENTER-03
========================

Branch: pr-ui-control-center-03

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make the left side of the new Control Center genuinely useful by combining the operator actions that are used most often: Upload, queue/recording counters, filters, and the recordings list. The operator should be able to upload files and watch them appear in the same working surface without navigating away from /.

Relevant code to inspect first
- lan_app/ui_routes.py
- lan_app/uploads.py
- lan_app/templates/upload.html
- lan_app/templates/recordings.html
- lan_app/templates/dashboard.html
- any new Control Center partials/templates from PR-UI-CONTROL-CENTER-01 and PR-UI-CONTROL-CENTER-02
- tests/test_ui_routes.py
- tests/test_upload.py
- tests/test_upload_export_flow.py
- tests_playwright/test_ui_smoke_playwright.py

Constraints
- Build on the new / Control Center shell. Do not rework the shell layout again.
- Reuse the existing upload API and upload JS behavior instead of inventing a second upload stack.
- Stay with Jinja2 + HTMX + minimal inline JS.
- Do not build a client-side global state store.
- Keep /upload and /recordings as fallback pages.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Embed the upload workflow into the Control Center left pane
Reuse the existing multi-file upload UI and progress logic from upload.html inside the left pane of /.
The operator must be able to:
- choose files or drag-and-drop
- see upload progress
- see processing progress after upload
- open the new recording
Do not fork the upload logic into two unrelated implementations.

2) Add compact queue and recording status controls
Bring the most useful counters into the left pane or top strip with click-to-filter behavior. At minimum support filtering by status from the Control Center without leaving the page.
Use the existing recording statuses and queue statuses. Keep labels and colors consistent with the rest of the app.

3) Add a practical recordings list for Control Center
Render the recordings list inside the left pane with the operator-relevant columns. Reuse the current recordings table behavior where possible, including:
- status
- progress
- suggested project
- confidence
- captured time
- duration
- source
- quick actions
Do not force the user back to /recordings for the normal daily workflow.

4) Add a conservative search/filter model
Support a simple search query parameter q from the Control Center. If the repository does not yet have a safe list_recordings search helper, add a conservative implementation that matches only stable fields such as recording id and source filename. Do not introduce a heavy full-text subsystem in this PR.

5) Make the left pane live
After upload or during processing, the left pane should refresh the relevant parts without a full page reload. A valid implementation can use HTMX polling or targeted refreshes. The operator should see:
- newly uploaded recordings appear in the list
- processing progress update
- counters update
Keep the refresh scope narrow. Do not constantly redraw the full page.

6) Preserve row actions
Keep the existing row actions functional from the Control Center list:
- open
- requeue
- quarantine
- delete
If full embedded inspector selection is not complete until the next PR, keep the open fallback stable.

7) Keep direct pages aligned
Refactor /upload and /recordings to reuse the same partials where practical so there is only 1 source of truth for:
- upload panel
- recordings list
- recordings filters

8) Tests with full coverage
Add or update deterministic tests for at least these cases:
- / renders the embedded upload controls
- a newly uploaded file appears in the Control Center list flow
- status filters update the left pane correctly
- q search filters the list conservatively and deterministically
- control-center row actions still work
- /upload and /recordings still render from the shared building blocks
Extend Playwright smoke only if needed, and keep it focused.

9) Documentation
Update the operator docs briefly to explain that upload and recording monitoring can now happen from / without switching pages.

Verification steps (must be included in PR description)
- Upload a file from / and confirm it appears in the left-pane list without leaving the page.
- Filter the list by status and confirm the counters and rows remain consistent.
- Confirm /upload and /recordings still work as direct pages.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Embedded upload workflow on the Control Center page
- Live counters/filters/list in the left pane
- Conservative q search support if needed
- Shared partials across /, /upload, and /recordings
- Tests with 100% statement and branch coverage

Success criteria
- Operators can upload and monitor recordings from 1 place.
- The left pane becomes a live work queue instead of a static list.
- No duplicate upload implementation is introduced.
- CI remains green.
```