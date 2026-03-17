Run PLANNED PR

PR_ID: PR-UI-STITCH-WORKLIST-02
Branch: pr-ui-stitch-worklist-02
Title: Flatten the Control Center worklist into a row-click meeting inbox with derived titles, dot status, and no filter chrome

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict. Do not drift into system-bar work, upload redesign, inspector redesign, speaker UI, or broader Stitch restyling.

The connected Google Stitch MCP project named "Control Center" may be used only as a visual reference for the already approved list language: calm spacing, flat row-based inbox layout, colored status dots, compact progress bars, and a 3-dots actions affordance. Do not use Stitch to invent new filters, new hero sections, new cards, or a new page structure. Do not paste raw Stitch output into the repo. Keep the implementation Jinja and HTMX based.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Inspect the current Control Center worklist implementation and confirm the exact files involved before coding. Expected files include:
- lan_app/templates/partials/control_center/recordings_panel.html
- lan_app/templates/partials/control_center/recordings_filters.html
- lan_app/templates/partials/control_center/recordings_status_cards.html
- lan_app/templates/partials/control_center/recordings_table.html
- lan_app/templates/partials/control_center/work_pane.html
- lan_app/ui_routes.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py only if current UI assertions break
Use this inspection only to map the current worklist chrome, row rendering, title data flow, row selection behavior, and progress calculation. Do not broaden the task.

Phase 2 - Implement
Implement only the Control Center worklist simplification and row redesign described below.

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Current problem
The Control Center worklist still behaves like a mini dashboard instead of a compact operator inbox. It contains too much chrome before the actual rows, uses a button-based selection flow, and prioritizes filename/status text over the meeting identity that the operator actually needs.

The operator wants the worklist to behave like this:
- no explanatory worklist text
- no filter strip by status/search/rows in the Control Center
- no triage/status pill strip in the Control Center
- no Select button inside each row
- the whole row should open/select the recording
- status should be communicated by a colored dot, not a large badge column
- the worklist identity should show recording ID plus a human-readable meeting title resolved from calendar or summary
- the title should update naturally as calendar matches or summary artifacts appear
- the duration must stay in HH:MM:SS form
- the progress bar should treat upload as the first small part of overall processing progress
- actions should live under a 3-dots affordance

Required product contract for the Control Center worklist
The Control Center worklist must become a compact meeting inbox with flat rows.

It must show only the operator-facing essentials:
- status dot
- meeting identity block
- captured date/time
- duration (HH:MM:SS)
- recognition progress
- actions under 3 dots

It must not show these Control Center worklist chrome elements anymore:
- the heading "Operator inbox"
- the description "Keep uploads, triage, and selection in one compact daily inbox."
- the filter strip with Status / Search / Rows / Apply / Clear
- the label "Triage"
- the entire status-pill strip used as a status filter
- the Select link/button inside each row
- the filename-first identity layout
- the separate Source column in the Control Center worklist
- the large Status badge column in the Control Center worklist

Exact required changes

1. Remove worklist title copy
In the Control Center worklist panel, remove these texts completely:
- "Operator inbox"
- "Keep uploads, triage, and selection in one compact daily inbox."
Keep the surrounding panel and the remaining top-right visible count if it is still needed by the existing layout.
Do not add replacement copy.

2. Remove the Control Center filter strip
Remove the Control Center worklist filter UI that currently contains:
- Status
- Search
- Rows
- Apply
- Clear
This removal is for the Control Center operator inbox only.
Do not break or redesign the standalone /recordings page if it still needs its own filter UI.
If the backend still supports query params for direct links or legacy routes, that is fine, but the Control Center worklist must not render this filter strip.

3. Remove the Control Center triage/status-pill strip
Remove the entire Control Center status-pill filter strip, including:
- the label "Triage"
- All / Queued / Processing / Stopping / Stopped / NeedsReview / Ready / Published / Quarantine / Failed pills
This removal is for the Control Center operator inbox only.
Do not move these pills elsewhere in this PR.

4. Redesign the Control Center rows into a row-click meeting inbox
Change the Control Center worklist table/list presentation so it matches this operator contract:
- the row itself becomes the selection/open target in Control Center mode
- clicking the row selects the recording and updates the right inspector
- the 3-dots actions control remains interactive and must not accidentally trigger row selection
- remove the separate Select button/link from the row

Implement the row structure for Control Center mode like this:
- leading status dot
- identity block
- captured date/time
- duration
- progress
- 3-dots actions

Do not keep the current Control Center columns as-is.
In particular, remove the visible Control Center columns for:
- Status text/badge column
- Source column

5. Replace status badge with status dot
For the Control Center worklist only, replace the current large text badge presentation with a compact colored status dot.
Use operator-friendly tone mapping consistent with the approved Stitch-inspired list style.
The dot should communicate state at a glance without requiring a separate status column.
Keep any necessary machine-readable status text for accessibility using aria-label/title if needed, but do not keep the large badge UI in the Control Center worklist rows.

6. Replace filename-first identity with derived meeting identity
In the Control Center worklist rows, stop leading with source filename.
Render a 2-line identity block using this priority logic:
- line 1: recording ID
- line 2: human-readable meeting title resolved dynamically from current recording context

Required title resolution order:
1. selected/manual or best available calendar match title
2. summary topic from the current summary artifact
3. source filename fallback

Important behavior rule:
The displayed meeting title must update naturally when the necessary recording artifacts appear or change. Do not require a separate manual sync. Reuse the existing HTMX/refresh flow where possible.
Do not add a new persistent DB field for this if the title can be derived truthfully at render time from existing artifacts/state.

7. Keep captured date/time and duration, but make duration explicitly HH:MM:SS
Keep the captured timestamp column in the Control Center worklist.
Keep the duration column, but ensure the visible format is always HH:MM:SS in Control Center mode, including short recordings.
Do not regress the existing duration backfill behavior.

8. Make upload count as the first part of total recognition progress
For the Control Center worklist progress display, treat upload completion as the first small part of the total end-to-end progress.
Use a simple display model such as:
- upload-complete baseline = 5%
- remaining pipeline progress fills the remaining 95%

Implementation rule:
- do not rewrite the stored pipeline_progress contract across the whole application
- apply this as a display-layer progress mapping for the Control Center worklist
- terminal states must still show 100%
- newly uploaded/queued rows that already exist in the inbox but have not started processing should show the small baseline instead of 0%

9. Replace the visible Actions select with a 3-dots affordance
The Actions cell in the Control Center worklist must become a compact 3-dots action affordance.
It may still use the existing action logic under the hood, but the visible operator control must be a 3-dots trigger, not a large select box labeled "Actions".
Keep the available actions equivalent to the current Control Center row actions unless a tiny compatibility adjustment is required by the new trigger.

Hard constraints
- Do not change the top navigation
- Do not change the upload card in this PR
- Do not change the bottom system bar in this PR
- Do not change the compact inspector layout in this PR
- Do not change speaker UI in this PR
- Do not redesign the whole Control Center
- Do not introduce a new global search/filter pattern elsewhere as a replacement
- Do not add new helper text
- Do not add new summary cards
- Do not move diagnostics into the worklist
- Do not break standalone /recordings behavior unless the task explicitly says Control Center only

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- update tests that assert the old worklist headings, filters, status pills, Select action, or old row structure
- add or update tests for:
  - row-click selection in Control Center mode
  - derived meeting title precedence and fallback
  - progress display baseline including upload contribution in Control Center mode
  - no Control Center rendering of the removed filter and triage sections
- keep coverage green

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py only if current UI expectations break

Success criteria
- the Control Center worklist no longer shows "Operator inbox"
- the Control Center worklist no longer shows "Keep uploads, triage, and selection in one compact daily inbox."
- the Control Center worklist no longer shows the Status/Search/Rows filter strip
- the Control Center worklist no longer shows the Triage status-pill strip
- the Control Center worklist rows no longer show a Select button/link
- clicking a Control Center row selects/opens the recording
- the Control Center worklist rows use a colored status dot instead of a large status badge column
- the Control Center worklist identity shows recording ID plus a derived meeting title from calendar/summary with filename fallback
- the derived title updates when artifacts appear through the normal refresh path
- the duration is shown as HH:MM:SS
- the Control Center progress display includes a small upload baseline such as 5%
- the visible Actions control is a compact 3-dots affordance
- no unrelated Control Center sections were redesigned
- CI passes

Final output requirements
Provide a concise changelog with these sections only:
- Removed
- Row redesign
- Title derivation
- Progress mapping
- Actions trigger
- Tests updated
Do not include unrelated cleanup.
