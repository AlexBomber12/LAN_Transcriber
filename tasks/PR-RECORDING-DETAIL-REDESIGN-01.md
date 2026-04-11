Run PLANNED PR

Branch: pr-recording-detail-redesign-01

You are working in the LAN_Transcriber repository. This PR redesigns the full-page recording detail inspector to be more compact and operator-friendly. The current Speakers tab uses enormous per-speaker cards that require excessive scrolling. This PR replaces them with a compact speaker table and adds a transcript preview to the Overview tab. Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_app/templates/partials/recording_inspector_full_overview.html
- lan_app/templates/partials/recording_inspector_full_speakers.html
- lan_app/templates/partials/speaker_review_cards.html
- lan_app/templates/partials/recording_inspector_full_transcript.html
- lan_app/templates/partials/recording_inspector_body.html
- lan_app/templates/partials/recording_inspector_tabs.html
- lan_app/ui_routes.py (the route functions that serve the overview, speakers, and transcript tab data)
- lan_app/db.py (any queries that fetch speaker rows, snippets, transcript data)

Phase 2 - Implement

CHANGE 1: Redesign Speakers tab from cards to compact table

Replace the current speaker_review_cards.html card layout with a compact table. The current layout renders each speaker as a full card with 2 info boxes (Current Decision + Recognition Cue), a snippet candidates section with up to 3 audio players, and 4 action sections (Confirm match, Keep unknown, Local label only, Add trusted sample) all visible simultaneously. Each card is approximately 500px tall, making 6 speakers require 9+ screens of scrolling.

New layout for the Speakers tab (full-page mode only, do not change the compact Control Center inspector):

A. Speaker table
Render all speakers in a table with these columns:
- SPEAKER ID: avatar circle (S0, S1 format already in speaker_review_cards.html) plus the diarized label (SPEAKER_00)
- LABEL / CANONICAL: show the resolved display name if assigned, otherwise show the diarized label. If the speaker has a canonical match, show it. If unknown, show "Unknown Speaker" in muted text. Make this a dropdown select or inline element that allows quick reassignment from existing canonical speakers.
- QUALITY SNIPPET: show exactly 1 audio player for the best snippet (the one marked "Best match" or the first clean snippet). If no clean snippet exists, show "Low Sig" or "No clips" in muted amber/red text. Do not show purity percentage numbers. Do show a small play button inline.
- ACTIONS: a "..." (three-dot) overflow menu button that opens a dropdown/popover with these actions:
  - Confirm match (opens a small inline form or modal with canonical speaker select + confirm button)
  - Keep unknown (single-click action with confirmation)
  - Local label only (opens inline input + submit)
  - Add trusted sample (if eligible)
  - The current review state badge (Needs review / Confirmed / Unknown) should be visible as a small badge on the row itself, not inside the menu.

Each table row should be approximately 48-56px tall. The entire speaker table for 6 speakers should fit in roughly 1 screen height.

B. Keep the snippet warnings and repair banner
The "Regenerate snippets" banner and snippet warnings (N candidates rejected, N too short) should remain above the table as compact notice bars, same as current implementation.

C. Remove the Create Canonical Speaker form from the bottom of the Speakers tab
Move the "Create Canonical Speaker" functionality into the "..." actions menu. Add a "Create new speaker" option at the bottom of the canonical speaker dropdown inside the Confirm match action. When selected, show inline fields for Display name and Notes, then a "Create and confirm" button. This eliminates the disconnected form at the bottom of the page.

D. Preserve all existing POST endpoints
All form actions must still POST to the same URLs:
- /ui/recordings/{id}/speakers/assign
- /ui/recordings/{id}/speakers/keep-unknown
- /ui/recordings/{id}/speakers/local-label
- /ui/recordings/{id}/speakers/create-and-assign
- /ui/recordings/{id}/speakers/regenerate-snippets
Do not change any Python route logic for these endpoints. Only change the HTML templates.

CHANGE 2: Add transcript preview to Overview tab

Add a new section to the Overview tab called "Transcript Preview" below the Metadata Tags section.

A. Content
Show the first 10-15 speaker turns from the transcript, rendered as compact lines:
- [HH:MM:SS] Speaker Label: text...
- Use the same data source as the full Transcript tab
- If transcript is not available yet, show "Not available yet" in muted italic text
- Add a "View full transcript" link/button that switches to the Transcript tab

B. Purpose
This gives the operator immediate context about what the recording contains without switching tabs. The Stitch mockup shows this as "Quick Artifacts > Markdown Transcript" but since we only need the transcript (not audio chunks), render it as a simple preview section.

C. Data source
The transcript data is already loaded for the Transcript tab. Reuse the same data but limit to the first 10-15 turns. If the route currently only loads transcript data when the Transcript tab is active, adjust the overview route to also include a truncated transcript preview. Minimize the change: pass a transcript_preview list of the first 15 turns to the overview template context.

CHANGE 3: Two-column layout on Overview tab

Restructure the Overview tab to use a 2-column layout on large screens (lg: breakpoint and above):
- Left column: Pipeline Status (existing section, unchanged)
- Right column: Transcript Preview (new section from Change 2)
- Below both columns (full width): Metadata Tags (existing section, unchanged)

On small screens (below lg:), stack vertically: Pipeline Status, then Transcript Preview, then Metadata Tags.

CHANGE 4: Collapse Pipeline Status for completed recordings

When a recording status is in a terminal state (Ready, Published, NeedsReview, Failed, Quarantine), collapse the Pipeline Status section by default. Show a summary line like "All 13 stages completed in 34m 52s" with a disclosure triangle/chevron to expand the full stage list. When the recording is Processing or Stopping, keep Pipeline Status expanded with live updates as currently implemented.

Hard constraints
- Do not modify any Python API endpoint URLs or request/response contracts
- Do not change the compact Control Center inspector (recording_details_card.html)
- Do not remove any existing functionality, only reorganize the layout
- Do not change the Transcript, Summary, Diagnostics, or Export tabs
- Keep the solution Jinja/HTMX based
- Do not add JavaScript frameworks
- All existing speaker review POST actions must continue to work
- Do not change the speaker_review_mode == 'compact' path in speaker_review_cards.html
- Preserve the existing full-page recording_inspector_header.html (the header with title, status badge, action buttons)

Phase 3 - Verify and summarize
- Run scripts/ci.sh until exit code 0
- Update tests as needed for changed template structure
- Generate required review artifacts per AGENTS.md
- Provide a concise final changelog with Deleted / Kept / Refactored / Tests

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests/test_cov_lan_app_ui.py
- tests_playwright/test_ui_smoke_playwright.py

Success criteria
- Speakers tab renders all speakers in a compact table with 1 row per speaker (approximately 48-56px per row)
- Each speaker row shows: avatar, label/canonical, 1 audio snippet player, actions menu
- The "..." menu contains Confirm match, Keep unknown, Local label, Add trusted sample
- Create Canonical Speaker is accessible from within the Confirm match flow, not as a separate bottom form
- Overview tab shows a Transcript Preview section with the first 10-15 turns
- Overview tab uses 2-column layout on large screens (Pipeline Status left, Transcript Preview right)
- Pipeline Status is collapsed by default for terminal-status recordings
- All existing speaker review POST endpoints still work
- No regressions in the compact Control Center inspector
- CI passes
