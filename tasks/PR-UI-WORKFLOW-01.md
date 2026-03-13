PR-UI-WORKFLOW-01
==================

Branch: pr-ui-workflow-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Finish the 1-page operator workflow by cleaning up navigation, reducing page-switching, and adding smoke coverage for the new Control Center flow. After the earlier PRs, / should already support upload, recordings list, embedded inspector, speaker decisions, and corrections. This PR must make that workflow feel intentional rather than like several pages stitched together.

Relevant code to inspect first
- lan_app/templates/base.html
- lan_app/ui_routes.py
- all Control Center templates/partials added in earlier PRs
- lan_app/templates/recording_detail.html
- lan_app/templates/glossary.html
- lan_app/templates/voices.html
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py

Constraints
- This is a workflow polish PR, not a redesign from scratch.
- Keep all direct routes alive: /upload, /recordings, /recordings/{id}, /glossary, /voices.
- Do not introduce a JS framework.
- Keep the UI simple and utilitarian.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Make / the obvious primary workflow
Adjust navigation and entry points so the main daily operator path is clearly centered on /.
A good implementation may include:
- making Dashboard explicitly behave like the Control Center
- reducing redundant primary-nav emphasis for pages that are now legacy fallbacks
- adding clear quick actions in the Control Center for opening advanced areas when needed
Keep direct links available for users who still want them.

2) Reduce unnecessary page switching
Where the app still forces the operator to leave / for common daily actions, tighten that flow.
Examples:
- opening corrections from the main workflow
- opening voice management from the main workflow
- returning to the previously selected recording after an action
You do not need to eliminate all navigation, but the main daily loop should no longer feel fragmented.

3) Improve empty states and helper copy
Polish the operator-facing microcopy so the main workflow is self-explanatory. Focus on:
- no recordings yet
- no selected recording
- no speaker decision yet
- no corrections yet
- advanced pages now available as fallbacks
Keep wording direct and practical.

4) Keep advanced admin pages available but secondary
Voices and Corrections should still be reachable directly, but the main workflow should not require living on those pages. Add lightweight in-app entry points from the Control Center where helpful, while keeping the dedicated pages intact.

5) Add smoke coverage for the new 1-page workflow
Extend the existing Playwright smoke test or add a focused second smoke test to cover the new path on /. At minimum validate:
- load /
- upload a file from /
- select the recording without leaving the page
- open at least 1 embedded tab
- reach the export action
If the app state needed for full speaker/correction interaction is too synthetic for Playwright smoke, keep those deeper branches in deterministic unit tests and explain the split clearly.

6) Keep route/test maintainability high
Do not create brittle UI tests tied to tiny layout details. Prefer stable ids, clear anchors, and focused assertions that reflect real operator workflows.

7) Documentation
Update README or docs/runbook.md so the new primary workflow is described in order:
- open /
- upload or pick a recording
- review in the embedded inspector
- use speaker decisions and corrections when needed
- use direct pages only as fallbacks or admin screens

Verification steps (must be included in PR description)
- Use / as the main workflow from upload to export without leaving the page unless intentionally opening a fallback page.
- Confirm direct routes still work.
- Run scripts/ci.sh and keep CI green.
- Run the relevant Playwright smoke test and keep it green.

Deliverables
- Navigation and workflow cleanup centered on /
- Better empty states and helper copy
- Lightweight in-app entry points to advanced areas
- Playwright smoke coverage for the 1-page workflow
- Tests with 100% statement and branch coverage for changed/new modules

Success criteria
- The operator naturally uses / as the main workspace.
- Daily actions no longer require bouncing across multiple top-level pages.
- Direct pages remain available as safe fallbacks.
- CI and smoke tests remain green.
```