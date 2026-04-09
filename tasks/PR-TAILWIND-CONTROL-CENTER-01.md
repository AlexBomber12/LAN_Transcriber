Run PLANNED PR

PR_ID: PR-TAILWIND-CONTROL-CENTER-01
Branch: pr-tailwind-control-center-01
Title: Migrate Control Center page and all its partials from legacy CSS to Stitch Tailwind design

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict. Do not touch recording_detail.html, voices.html, calendars.html, projects.html, queue.html, glossary.html, or login.html.

CRITICAL DESIGN RULE: The visual source of truth is docs/stitch-reference/control-center.html. Read this file before writing any markup. Copy Tailwind classes from this reference. Do not invent your own classes, do not use legacy CSS classes, do not freestyle the design. If a component exists in the Stitch reference, use the exact same Tailwind utility classes. If a component does not exist in the Stitch reference, use the same Tailwind design language (same color tokens, same spacing scale, same border-radius values, same font sizes).

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read docs/stitch-reference/control-center.html completely. This is the visual target.

Read every file in the current Control Center template set:
- lan_app/templates/control_center.html
- every file in lan_app/templates/partials/control_center/
- lan_app/templates/partials/recording_inspector.html
- lan_app/templates/partials/recording_inspector_body.html
- lan_app/templates/partials/recording_inspector_embedded_body.html
- lan_app/templates/partials/recording_inspector_header.html
- lan_app/templates/partials/recording_inspector_tabs.html
- lan_app/templates/partials/recording_progress.html
- lan_app/templates/partials/speaker_review_cards.html

Map each component in the current templates to its Stitch reference equivalent:
- Upload pipeline section (left: dropzone, right: active jobs + queue cards)
- Recordings worklist table (status dot, name, date, duration, confidence/progress, actions)
- Right inspector pane (recording details, metadata, speakers, language detection)
- System bar (bottom status bar)

Phase 2 - Implement
Rewrite every Control Center template to use Tailwind utility classes matching the Stitch reference.

Overall page layout from the Stitch reference:
- main area: flex with left column w-[55%] and right column w-[45%]
- left column: upload pipeline card + recordings worklist table
- right column: recording inspector/details panel
- both columns scroll independently

For each component, follow this exact process:
1. Find the matching section in docs/stitch-reference/control-center.html
2. Copy the Tailwind classes from that section
3. Replace hardcoded data with Jinja variables from the current template
4. Keep all existing hx-get, hx-post, hx-target, hx-trigger, hx-swap attributes
5. Keep all existing onclick handlers and data-* attributes
6. Keep all existing Jinja conditionals and loops

Specific components to migrate:

UPLOAD PIPELINE
Stitch reference section: "Upload Pipeline" with cloud_upload icon
- compact card with bg-white rounded-xl border border-slate-200
- left: dashed dropzone with drag/drop and browse
- right: active jobs counter pill + progress cards
- keep existing upload JavaScript and HTMX bindings

RECORDINGS WORKLIST
Stitch reference section: "Recordings Worklist" table
- table with thead bg-slate-50, text-[10px] uppercase headers
- status column: material icon check_circle (green) or pending (amber)
- name column: font-semibold, selected row highlighted with bg-primary/5 border-l-4 border-l-primary
- confidence column: progress bar (w-12 h-1.5 rounded-full) + percentage
- actions column: more_vert icon button
- keep existing row click handlers and HTMX selection logic

INSPECTOR PANE (right column)
Stitch reference section: right 45% column
- header: eyebrow "RECORDING DETAILS", title "Selected: {name}", Play + Download buttons
- tabs: Overview | Speakers | Language | Project with border-b-2 active indicator
- overview tab: Key Metadata grid (status, engine, created, source device)
- speakers section: speaker cards with S1/S2 avatars, match probability, action buttons
- language detection bar
- "Open Full Correction Editor" button at bottom
- keep all existing HTMX tab-switching logic

SYSTEM BAR
Current: dark floating bar at bottom
Stitch reference footer: h-8 bg-slate-100 border-t with LAN Node status, DB free, version, uptime, documentation link
- migrate to match Stitch footer styling
- keep dynamic system status data from existing Jinja variables

Important HTMX preservation rules:
- every hx-get URL must remain exactly as-is
- every hx-target must remain exactly as-is
- every hx-trigger must remain exactly as-is
- every hx-swap must remain exactly as-is
- every id attribute used by HTMX targeting must remain exactly as-is
- the polling intervals for system bar and upload status must remain as-is

Hard constraints
- Do not modify lan_app/ui_routes.py or any Python route logic
- Do not modify any Python file except test files
- Do not change any API endpoint URLs
- Do not add new JavaScript
- Do not modify base.html (already migrated in PR-TAILWIND-BASE-01)
- Do not touch templates for other pages (recording_detail, voices, calendars, etc.)
- Do not use any legacy CSS classes from base.html. Use only Tailwind utilities.
- Copy classes from docs/stitch-reference/control-center.html. Do not invent new designs.

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- update tests where markup assertions changed
- the Control Center page layout must match Stitch reference: 55/45 split, dark navbar, upload cards, worklist table, inspector pane
- all HTMX interactions must still work: row selection, tab switching, upload, delete, action menus
- no legacy CSS class names used in Control Center templates

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui.py
- tests/test_cov_lan_app_ui.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py

Success criteria
- All Control Center templates use only Tailwind utility classes
- Page layout matches Stitch reference (55/45 column split)
- Upload pipeline matches Stitch reference (dropzone + queue cards)
- Recordings worklist table matches Stitch reference (icons, progress bars, selected row highlight)
- Inspector pane matches Stitch reference (metadata grid, speaker cards, tabs)
- System bar matches Stitch footer style
- All HTMX bindings preserved and functional
- All Jinja logic preserved
- No legacy CSS class names used in any Control Center template
- CI passes
