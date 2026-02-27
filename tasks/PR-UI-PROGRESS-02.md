PR-UI-PROGRESS-02

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/ui-progress-02
PR title: PR-UI-PROGRESS-02 Show pipeline progress on Recordings list table
Base branch: main

Goal:
1) Show processing progress percent and stage directly in the Recordings list table.
2) Keep UI consistent with existing progress logic on the recording detail page.

A) Compute progress fields for list rows
- In lan_app/ui_routes.py ui_recordings:
  - After list_recordings returns items:
    - For each item:
      - progress_ratio = _safe_pipeline_progress(item.get("pipeline_progress"))
      - item["progress_percent"] = int(round(progress_ratio * 100))
      - item["progress_stage_label"] = _pipeline_stage_label(item.get("pipeline_stage"))
- Do not change DB schema.

B) Update recordings list template
- In lan_app/templates/recordings.html:
  - Add a new column "Progress" after Status.
  - For each row:
    - If r.status == "Processing":
      - Render "<percent>% <stage>"
    - If r.status == "Queued":
      - Render "0% Waiting" (or percent based on pipeline_progress)
    - Else:
      - Render "100% Done" when status is Ready, NeedsReview, Published
      - Render "—" for Quarantine / Failed unless progress exists
  - Keep width small and consistent with the DB-window style.

C) Tests
- Update tests/test_ui_routes.py:
  - Seed a Processing recording with set_recording_progress(stage="diarize", progress=0.5).
  - GET /recordings and assert:
    - "Progress" column exists
    - "50%" appears in the row

Local verification:
- scripts/ci.sh

Success criteria:
- Recordings list shows pipeline progress percent and stage for Processing items.
- scripts/ci.sh is green.
```
