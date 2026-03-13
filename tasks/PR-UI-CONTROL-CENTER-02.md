PR-UI-CONTROL-CENTER-02
========================

Branch: pr-ui-control-center-02

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Replace the passive dashboard landing page with a real Control Center shell on /. The new / page must become the main operator workspace scaffold: a top summary/action strip, a left work pane, and a right inspector pane. This PR creates the shell and URL-state model, but it does not yet need to embed the full recordings workflow from the later PRs.

Relevant code to inspect first
- lan_app/ui_routes.py
- lan_app/templates/base.html
- lan_app/templates/dashboard.html
- any new partials introduced by PR-UI-CONTROL-CENTER-01
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui.py
- tests/test_cov_lan_app_ui_routes.py

Constraints
- Build on top of the partial foundation introduced in PR-UI-CONTROL-CENTER-01.
- Keep the app server-rendered. Use HTMX for incremental loading only.
- Do not turn this into a full SPA.
- Existing direct pages /upload and /recordings must remain available.
- Keep the visual language close to the current plain utilitarian UI. Do not do a styling redesign.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Turn / into a Control Center shell
Create a dedicated dashboard template or Control Center template for / that contains:
- a compact top strip with status counters and quick context
- a left pane reserved for the live operator list/work area
- a right pane reserved for the selected recording inspector
- empty states that explain what will appear in each pane
The page should feel like an operator console, not a static report page.

2) Define Control Center URL state
Add explicit query-param state handling for the new page. Support at minimum:
- selected for the selected recording id
- status for list filtering state
- q for simple search state
- tab for the inspector tab state
The shell should preserve URL state across refreshes so the operator can reload and remain in context.

3) Add stable pane containers and HTMX hooks
Give the left pane and right pane stable ids and clear render boundaries so later PRs can swap content into them without re-rendering the entire page.
Do not yet overbuild the behavior, but the shell must be ready for:
- left pane list refresh
- right pane inspector refresh
- top-strip counter refresh

4) Provide useful empty states
When there is no selected recording, the right pane must show a meaningful placeholder such as:
- select a recording from the left pane
- upload a file to begin
- open a full-page recording view if preferred
Avoid generic blank boxes.

5) Preserve summary visibility
Do not lose the current dashboard value completely. Keep the most useful counters or summaries visible in the new Control Center top strip so the operator still sees queue/recording health at a glance.

6) Keep old routes alive
Do not delete or break:
- /upload
- /recordings
- /recordings/{id}
This PR changes the landing page and shell only.

7) Tests with full coverage
Add or update deterministic tests for at least these cases:
- GET / renders the new Control Center shell
- selected/status/q/tab query params are accepted and reflected in shell state
- right-pane empty state renders when no recording is selected
- existing direct routes still work
Keep tests offline.

8) Documentation
Update README or docs/runbook.md briefly so operators understand that / is now the primary working surface and the old pages remain available as direct fallbacks.

Verification steps (must be included in PR description)
- Open / with no query params and confirm the new 2-pane shell renders.
- Open /?selected=<id>&tab=speakers and confirm the shell keeps that state in the UI even before the full embedded inspector arrives.
- Confirm /upload and /recordings still work unchanged.
- Run scripts/ci.sh and keep CI green.

Deliverables
- New Control Center shell on /
- URL-state handling for selected/status/q/tab
- Stable pane containers for later HTMX loading
- Useful empty states
- Tests with 100% statement and branch coverage

Success criteria
- The landing page becomes a real operator workspace shell instead of a passive dashboard.
- State is URL-driven and ready for 1-page workflow PRs.
- Existing routes remain available and CI stays green.
```