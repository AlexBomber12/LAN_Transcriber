Run PLANNED PR

Branch: pr-ui-stitch-recording-details-01

You are working in the LAN_Transcriber repository. This PR is a precise compact-inspector replacement for the Control Center right pane. Do not treat it as a redesign. Do not add new tabs, new dashboard sections, or new helper copy. Keep the current Stitch-inspired color palette, rounded corners, spacing rhythm, and overall visual language, but change the meaning of the compact right pane so it becomes a single Recording Details card whose only job is to help the operator understand what the selected recording is.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Inspect the current implementation and confirm the exact files involved in the compact right pane. The expected files are likely:
- lan_app/ui_routes.py
- lan_app/templates/partials/control_center/inspector_pane.html
- lan_app/templates/partials/control_center/inspector_empty.html
- lan_app/templates/partials/recording_inspector.html
- lan_app/templates/partials/recording_inspector_embedded_body.html
- lan_app/templates/partials/control_center/selected_recording_shell.html
- the template or stylesheet location that currently defines compact inspector spacing and card styles
Use inspection only to locate the existing embedded inspector data flow and markup. Do not broaden the task. After inspection, replace the current embedded-tab approach with the compact Recording Details contract below.

Phase 2 - Implement the compact Recording Details contract

A. No-selection state
Keep the no-selection state as a short placeholder only.
When no recording is selected, the right pane must show only:
- No recording selected
Do not show tabs.
Do not show pills.
Do not show summary counters.
Do not show helper paragraphs.
Do not add any new empty-state narrative.

B. Remove compact tabs entirely
For the embedded Control Center right pane only, remove the compact tab set completely.
Remove:
- Overview
- Speakers
- Language
- Project
and any other compact tab navigation currently rendered in the right pane.
Important:
- this applies only to the compact Control Center inspector
- do not remove or redesign the full-page recording view tabs
- do not break the dedicated full-page recording page

C. Replace the compact embedded inspector with a single card titled Recording Details
When a recording is selected, the right pane must render one compact card with this structure:

1. Card title
- Recording Details

2. Header row
- primary title: use the best available human-readable title in this order:
  a) an already available confirmed human-readable title if one exists in current repo data
  b) matched calendar title
  c) summary topic/title
  d) job ID
- secondary line: always include job ID
- also include the original filename as smaller secondary text
- add a Download ZIP button in the header area
Do not add Play.
Do not add audio controls.
Do not add tab navigation.

3. KEY METADATA section
Render a section titled:
- KEY METADATA

Inside it show compact rows for:
- Status
- Captured at
- Duration
- Language
- Review blocker / Next action
- Matched meeting only if it is available and not already used as the primary title
Do not show Source because it is always upload in this workflow.
Do not show broad diagnostics tables.
Do not show project routing tables.
Do not show pipeline stages.
Do not show engine/source-device fields.

4. SPEAKERS section
Render a section titled:
- SPEAKERS

Show all diarized speakers as rows, even if they are unnamed.
For each row:
- show the diarized label or resolved display name
- if a resolved display name exists, show it
- if there is no resolved name, show the diarized label and Unknown
- if a confidence / probability exists and is meaningful, show it
- if there is no reliable score, do not invent one
Examples of acceptable output:
- Andrea · 98%
- Speaker 02 · Unknown
Do not show speaker action buttons in this compact panel.
Do not show snippet cards here.
Do not show speaker management forms here.
This section is for identification only, not editing.

5. TONE section
Render a section titled:
- TONE

Show exactly one short phrase from the existing summary/emotional artifact.
If the artifact is not available yet, show:
- Not available yet

6. SUMMARY section
Render a section titled:
- SUMMARY

Show a short preview only, 3 to 6 lines maximum, derived from the existing summary artifact.
Prefer the existing summary text or bullets already produced by the pipeline.
If summary is not available yet, show:
- Not available yet
Do not render a full transcript here.
Do not render long decision/action lists here.

7. Bottom action
At the bottom of the card, render one full-width button:
- Open Recording Page
This must route to the existing dedicated full-page recording screen.
Do not add any other bottom buttons in this PR.

D. Behavioral rules
- selecting a recording from the left worklist must update this right pane
- the pane is for recognition and orientation only, not deep editing
- the pane must help the operator answer: what is this recording?
- do not reuse the full-page recording inspector inside the Control Center
- do not keep hidden compact tabs in the DOM
- do not create a mini copy of the full-page screen

E. Data rules
Use existing repo data only. Do not invent a new persistence model in this PR.
Use current available sources such as:
- recording fields
- matched calendar context if already available
- summary artifact topic/summary/emotional summary
- diarized speakers and current speaker assignments
If a value is not available yet, show:
- Not available yet
Do not add a new manual naming workflow in this PR.

Phase 3 - Verify and summarize
- run scripts/ci.sh until exit code 0
- update tests only as needed for the changed compact right pane contract
- generate required review artifacts
- provide a concise final changelog with Deleted / Kept / Refactored / Tests

Hard constraints
- Do not redesign the left worklist in this PR
- Do not touch the bottom system bar in this PR
- Do not change the full-page recording page contract
- Do not add tabs back in another form
- Do not add hero text, helper text, or instructional copy
- Do not add speaker editing controls to this compact pane
- Do not add audio players
- Do not add source/device/engine technical metadata
- Do not use Google Stitch to invent a different layout for this task; use the current approved visual language only
- Keep the solution Jinja/HTMX based

Success criteria
- when nothing is selected, the right pane shows only a short No recording selected placeholder
- when a recording is selected, the right pane shows a single Recording Details card
- compact tabs are gone from the right pane
- the header shows a human-readable title using the required fallback order
- the header includes job ID, filename, and Download ZIP
- KEY METADATA shows Status, Captured at, Duration, Language, and Review blocker / Next action
- Matched meeting is shown only when available and not duplicating the title
- SPEAKERS shows all diarized speakers as compact identification rows
- TONE shows one short phrase from the existing artifact or Not available yet
- SUMMARY shows a short preview of 3 to 6 lines or Not available yet
- the bottom of the card has a full-width Open Recording Page button
- no tabs remain in compact mode
- the compact pane is clearly lighter and more identification-focused than before
- full-page recording functionality remains intact
- CI passes
