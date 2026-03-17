Run PLANNED PR

PR_ID: PR-UI-STITCH-UPLOAD-01
Branch: pr-ui-stitch-upload-01
Title: Simplify the Control Center upload area into a compact UPLOAD block with queue cards

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict. Do not drift into worklist redesign, inspector redesign, system-bar work, speaker UI, or broader Control Center restyling.

Use the connected Google Stitch MCP project "Control Center" only as a visual reference for spacing, typography, rounded corners, and upload-card composition. Do not invent a new layout beyond the exact requirements below. Do not paste raw Stitch output into the repo. Keep the implementation Jinja and HTMX based.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Inspect the current implementation of the Control Center upload section and confirm the exact files involved before coding.
Expected files are likely in:
- lan_app/templates/partials/control_center/
- the page-level Control Center template or wrapper partial
- related CSS blocks in base.html or the Control Center partials
- lan_app/ui_routes.py if upload queue context is prepared there
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py only if UI assertions break

Locate and confirm:
- the large body title "Control Center"
- the "Live intake" pill
- the current upload dropzone block
- the current explanatory block that starts with "Only in-flight uploads stay here"
- the current rendering path for active uploads / queued uploads, if any already exists elsewhere
- any CSS controlling upload-card spacing and queue-card presentation

Use this inspection only to map the current upload area structure, live queue data flow, and markup/CSS that must change. Do not broaden the task beyond the upload/workspace area.

Phase 2 - Implement
Implement only the Control Center upload-area simplification and queue-card layout described below.

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Current problem
The Control Center upload area still behaves like an explanatory hero card instead of a compact working intake surface. It shows too much narration, a large page title, and a right-side explanatory text panel instead of a simple queue-at-a-glance view.

The operator wants this upload area to behave like this:
- no large body title "Control Center"
- no "Live intake" pill
- the left section title must be a compact uppercase "UPLOAD" label styled in the same family as "WORKLIST"
- the dropzone stays functional and compact
- the right side stops explaining upload behavior in text
- the right side instead shows a compact active-jobs counter and up to 2 queue cards
- queued items show "waiting in queue..."
- when there are 0 jobs, the counter remains visible and no cards are shown

Required product contract for the Control Center upload area
The Control Center upload section must become a compact intake block.

It must show only:
- a small "UPLOAD" section title with a calm upload icon
- the existing compact dropzone with choose-files and drag/drop behavior
- a compact active-jobs counter
- up to 2 upload queue cards on the right

It must not show anymore:
- the large body heading "Control Center"
- the pill "Live intake"
- the explanatory block title "Only in-flight uploads stay here"
- the explanatory paragraph about finished recordings moving into the worklist
- the explanatory fallback text "No active uploads. New files appear here until they enter the main inbox."

Exact required changes

1. Remove the large body title
Remove the large "Control Center" heading from the page body.
Important:
- keep the top navigation item "Control Center"
- remove only the large body heading in the page content

2. Remove the "Live intake" pill
Remove the pill/badge labeled "Live intake" from the upload area.
Do not replace it with another badge.

3. Rename the left upload section to "UPLOAD"
The upload section title must become:
UPLOAD
Style requirement:
- match the visual style family of "WORKLIST"
- uppercase
- same small section-heading feel
- same blue accent treatment
Add a small upload icon next to the title.
Choose a calm, simple upload icon that fits the current design language. No decorative icon cluster.

4. Keep the left dropzone functional and compact
Keep the left upload dropzone functional.
Do not remove:
- choose files button
- drag/drop behavior
- accepted formats text if it already belongs to the dropzone
Keep the dropzone compact and consistent with the already reduced height.
Do not enlarge it or turn it into a hero block.

5. Remove the full explanatory block on the right
Completely remove this block and all its text:
- "Only in-flight uploads stay here"
- "Finished recordings move into the main worklist below, so this panel only tracks files that are still entering the system."
- "No active uploads. New files appear here until they enter the main inbox."
Do not replace it with another explanatory text block.

6. Replace the explanatory block with upload queue cards
In the right side of the UPLOAD area, render a compact upload queue card region inspired by the approved reference.
This region must contain:
- a small active-jobs counter label, for example "3 ACTIVE JOBS"
- up to 2 cards only:
  - the currently active upload
  - the next queued upload
If more than 2 jobs exist, show only the first 2 and let the counter reflect the real total.

7. Upload queue card content
For each card:
- primary line: recording filename / job label
- progress bar
- state text or state styling

The active upload card:
- shows current progress, such as 65%
- shows a filled progress bar
- should read as the current in-flight upload

The queued card:
- shows the secondary state text:
  waiting in queue...
- does not pretend to have real upload progress
- keeps its progress treatment subdued / inactive compared to the active card

8. Empty-state behavior
If there are no active uploads and no queued uploads:
- show the active-jobs counter as 0 ACTIVE JOBS
- show no cards
- do not show explanatory fallback text
- do not insert a "No active uploads" paragraph

Hard constraints
- Do not touch the worklist in this PR
- Do not touch the right recording-details pane in this PR
- Do not change the bottom system bar
- Do not change speaker UI
- Do not redesign the whole Control Center
- Do not add new explanatory copy
- Do not restore hero/dashboard elements
- Do not alter unrelated spacing or typography outside the upload section
- Do not introduce tabs, helper cards, or admin shortcuts
- Keep the current visual palette, border-radius language, and clean Stitch-inspired spacing

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- update tests only where needed because of removed text, removed title, removed pill, and changed upload queue markup
- add or update tests for:
  - no large body "Control Center" heading in page content
  - no "Live intake" pill in the upload area
  - "UPLOAD" section title rendering
  - right-side upload queue cards rendering
  - empty state with 0 ACTIVE JOBS and no cards
- keep coverage green

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py only if current UI expectations break

Success criteria
- the large body "Control Center" title is gone
- the "Live intake" pill is gone
- the upload block title is "UPLOAD" in the same visual style family as WORKLIST
- the upload block has a small upload icon
- the right explanatory block is gone
- the right side now shows a compact active-jobs counter plus up to 2 queue cards
- queued items display "waiting in queue..."
- when there are 0 jobs, the counter remains visible and no cards are rendered
- the left dropzone remains functional and compact
- no unrelated UI changed
- CI passes

Final output requirements
Provide a concise changelog with these sections only:
- Removed
- Upload title
- Queue cards
- Empty state
- Tests updated
Do not include unrelated cleanup.
