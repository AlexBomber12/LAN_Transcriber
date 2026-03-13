PR-UI-CONTROL-CENTER-01
========================

Branch: pr-ui-control-center-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Prepare the current server-rendered UI for a real 1-page operator workflow by extracting reusable Control Center building blocks from the existing pages. Today the relevant operator flow is fragmented across dashboard.html, upload.html, recordings.html, recording_detail.html, glossary.html, and voices.html. This PR must create reusable Jinja partials and route-level context helpers without changing product semantics yet.

Relevant code to inspect first
- lan_app/ui_routes.py
- lan_app/templates/base.html
- lan_app/templates/dashboard.html
- lan_app/templates/upload.html
- lan_app/templates/recordings.html
- lan_app/templates/recording_detail.html
- lan_app/templates/glossary.html
- lan_app/templates/voices.html
- lan_app/templates/partials/recording_progress.html
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui.py
- tests/test_cov_lan_app_ui_routes.py

Constraints
- Stay with the current architecture: FastAPI + server-rendered Jinja2 + HTMX + minimal inline JS. Do not introduce React, Alpine, Vue, or SPA routing.
- This PR is foundation only. Do not redesign the product workflow yet.
- Do not change database schema in this PR.
- Existing full-page routes must keep working: /, /upload, /recordings, /recordings/{id}, /glossary, /voices.
- Avoid visual churn. Existing pages should look the same or extremely close after the refactor.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Extract reusable UI partials
Create a small partial system under lan_app/templates/partials/ with focused fragments for the pieces that will later compose the Control Center. At minimum extract reusable fragments for:
- status counters / summary strip
- upload panel shell
- recordings filters bar
- recordings table
- empty inspector shell / placeholder
- inspector action bar shell
- reusable page notices / helper messages where duplication already exists
You may add a partial subfolder such as templates/partials/control_center/ if that keeps the structure clean.

2) Extract route-level context builders
In lan_app/ui_routes.py introduce small, explicit helpers that prepare data for the reusable UI blocks. Do not keep page assembly logic monolithic. The helpers should cover at least:
- dashboard / status counts context
- recordings list state and pagination context
- upload shell context
- selected recording summary shell context
- compact glossary summary shell context if needed later
The goal is to make Control Center assembly possible without copying existing page logic.

3) Add partial-friendly endpoints or rendering paths
Add focused HTMX-friendly render paths for the reusable blocks. A valid implementation can use either:
- dedicated endpoints like /ui/control-center/recordings-table
- or existing endpoints with a clear partial mode
Choose 1 approach and keep it consistent.
Do not make partial rendering depend on brittle template conditionals scattered everywhere.

4) Refactor existing full pages to consume the new partials
After extraction, make the current full pages render via the partials instead of carrying duplicated markup. This applies especially to:
- dashboard counters
- recordings table and filters
- upload panel shell
- recording detail action shell where reuse makes sense
Keep page behavior unchanged in this PR.

5) Keep contracts stable
Do not change route URLs, form names, action names, or existing operator semantics in this PR.
Do not change speaker-review meaning, glossary meaning, or export behavior yet.
The output of existing pages must stay stable enough that downstream PRs can build on it safely.

6) Tests with full coverage
Add or update deterministic tests for at least these cases:
- dashboard still renders from extracted summary partials
- recordings page still renders filters, pagination, and progress from extracted partials
- upload page still renders its queue table and upload controls
- any new partial endpoints or partial modes render the expected HTML fragments
- no existing route regresses because of the refactor
Keep tests offline and do not require real model execution.

7) Documentation
Update the relevant developer docs briefly so future PRs know the new partial structure and the intended Control Center composition path. Keep this concise. README or AGENTS-adjacent developer notes are enough if they already mention UI structure.

Verification steps (must be included in PR description)
- Open /, /upload, /recordings, /recordings/{id}, /glossary, and /voices and confirm they still render successfully after the refactor.
- Render the new partial-friendly paths directly and confirm they return only the intended fragment HTML.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Reusable partial templates for current operator UI blocks
- Route-level context helpers in ui_routes.py
- Partial-friendly render paths for future Control Center composition
- Existing full pages refactored to consume the new partials
- Tests with 100% statement and branch coverage

Success criteria
- The current UI is decomposed into reusable building blocks without changing operator behavior.
- Future Control Center PRs can assemble 1-page workflow pieces without copying templates.
- Existing routes stay stable and CI remains green.
```