PR-UI-CONTROL-CENTER-04
========================

Branch: pr-ui-control-center-04

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Embed the existing recording detail workflow into the right pane of the Control Center so operators can select a recording from / and review it without leaving the page. The full-page route /recordings/{id} must remain as a fallback, but it should share the same inspector building blocks.

Relevant code to inspect first
- lan_app/ui_routes.py
- lan_app/templates/recording_detail.html
- lan_app/templates/partials/recording_progress.html
- lan_app/exporter.py
- any new Control Center partials/templates from earlier PRs
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py

Constraints
- Reuse the existing recording detail behavior. Do not rewrite the product logic behind Overview, Speakers, Language, Project, Calendar, Metrics, or Log.
- Stay server-rendered with HTMX. No SPA routing.
- Keep /recordings/{id} available as a full-page fallback.
- The embedded inspector must work with the current stop/progress/export actions.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Split recording detail into reusable inspector building blocks
Refactor recording_detail.html so the inspector can render both:
- as a full page for /recordings/{id}
- as an embedded right-pane inspector on /
Extract reusable partials for:
- inspector header / action bar
- tab navigation
- tab body content or tab-specific fragments where that improves reuse
Avoid copying the same markup into two templates.

2) Add an embedded inspector render path
Create a focused render path for the right pane, such as:
- /ui/recordings/{id}/inspector
- or an existing route with a clear embedded mode
Choose 1 clean approach and keep it consistent.
The embedded inspector must support tab switching without forcing the operator to leave /.

3) Sync inspector state with Control Center URL state
When a recording is selected from the left pane, the Control Center URL must carry:
- selected=<recording_id>
- tab=<tab_name>
The right pane should reload from that URL state on refresh.

4) Keep inspector tabs functional inside the right pane
Support at least these tabs in embedded mode:
- overview
- speakers
- language
- project
- calendar
- metrics
- log
Do not regress existing forms and actions inside those tabs.

5) Adapt progress polling for embedded mode
The current progress partial redirects to the full-page route when a recording reaches a terminal state. Make that behavior safe for embedded mode so the right pane stays in the Control Center when appropriate.
You may add an explicit embedded flag or target-aware logic, but keep the behavior deterministic.

6) Add an inspector action bar suitable for 1-page workflow
In the embedded inspector include clear actions for:
- requeue
- quarantine
- delete
- stop when eligible
- download ZIP
- open full page
Make sure these actions do not unexpectedly dump the user onto a different page unless that action truly requires it.

7) Keep the full-page route healthy
/recordings/{id} must continue to render correctly and should use the same underlying inspector partials to avoid divergence.

8) Tests with full coverage
Add or update deterministic tests for at least these cases:
- selecting a recording on / loads the embedded inspector
- embedded tab switches work and keep URL state
- embedded progress polling does not break when the recording reaches a terminal state
- action bar contains the expected actions
- /recordings/{id} still renders correctly from the shared inspector blocks
Extend Playwright smoke if needed so the new 1-page flow is covered.

9) Documentation
Update the docs briefly to explain that the main review flow now happens on / and the full-page recording route is still available as a fallback.

Verification steps (must be included in PR description)
- Open /, select a recording, and switch across at least 3 tabs without leaving the page.
- Download the export ZIP from the embedded inspector.
- Confirm /recordings/{id} still works as a standalone page.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Reusable embedded/full-page inspector building blocks
- Right-pane inspector render path
- URL-synced tab state for / + selected recording
- Embedded-safe progress behavior
- Tests with 100% statement and branch coverage

Success criteria
- The operator can inspect and review a recording from / without losing list context.
- Full-page recording detail remains available.
- Control Center becomes the real primary workflow surface.
- CI remains green.
```