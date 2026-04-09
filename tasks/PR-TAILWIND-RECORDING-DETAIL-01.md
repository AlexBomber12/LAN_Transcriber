Run PLANNED PR

PR_ID: PR-TAILWIND-RECORDING-DETAIL-01
Branch: pr-tailwind-recording-detail-01
Title: Migrate Recording Detail full-page inspector from legacy CSS to Stitch Tailwind design

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict. Do not touch Control Center templates (already migrated), voices.html, calendars.html, projects.html, queue.html, glossary.html, or login.html.

CRITICAL DESIGN RULE: The visual source of truth is docs/stitch-reference/recording-detail.html. Read this file before writing any markup. Copy Tailwind classes from this reference. Do not invent your own classes, do not freestyle the design.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read docs/stitch-reference/recording-detail.html completely. This is the visual target.

Read every file in the recording detail template set:
- lan_app/templates/recording_detail.html
- lan_app/templates/partials/recording_inspector_full_overview.html
- lan_app/templates/partials/recording_inspector_full_speakers.html
- lan_app/templates/partials/recording_inspector_full_transcript.html
- lan_app/templates/partials/recording_inspector_full_summary.html
- lan_app/templates/partials/recording_inspector_full_diagnostics.html
- lan_app/templates/partials/recording_inspector_full_export.html

Map each section in the current templates to its Stitch reference equivalent:
- Breadcrumb navigation (Control Center > Recording Name)
- Header with title, status badge, metadata chips, action buttons (Requeue, Quarantine, Export, ZIP, Delete)
- Tab navigation (Overview, Speakers, Language, Project, Metrics, Log, Artifacts)
- Pipeline Status card (left column: vertical stepper with check/spinner/pending icons)
- Speaker Management table (right column: speaker ID, label dropdown, quality snippet, actions)
- Metadata Tags section (department, priority, region pills + Add Tag)
- Quick Artifacts section (Markdown transcript preview, Audio chunks list)
- Bottom footer bar (Active Model, Compute Node, Auto-save, Publish Result button)

Phase 2 - Implement
Rewrite every recording detail template to use Tailwind utility classes matching the Stitch reference.

Overall page layout from the Stitch reference:
- max-w-[1440px] mx-auto centered content
- breadcrumb at top
- header with title + status badge on left, action buttons on right
- tab bar with border-b-2 active indicators
- content: 12-column grid, left col-span-4 (pipeline + metadata), right col-span-8 (speakers + artifacts)
- fixed bottom footer bar

For each component, follow this exact process:
1. Find the matching section in docs/stitch-reference/recording-detail.html
2. Copy the Tailwind classes from that section
3. Replace hardcoded data with Jinja variables from the current template
4. Keep all existing hx-get, hx-post, hx-target, hx-trigger, hx-swap attributes
5. Keep all existing onclick handlers and data-* attributes
6. Keep all existing Jinja conditionals and loops

Specific components to migrate:

BREADCRUMB
Stitch: flex items-center gap-2 text-sm with chevron_right icon separator
- "Control Center" links to /
- Current recording name as final breadcrumb item

HEADER
Stitch: flex flex-wrap justify-between items-start gap-4
- Left: h1 text-3xl font-bold + status badge (Processing/Ready/etc) + metadata row (ID, date, duration, file size)
- Right: action buttons row (Requeue, Quarantine, Export, ZIP) + delete button
- Keep all existing action button onclick handlers and HTMX bindings

TAB NAVIGATION
Stitch: border-b with gap-8, active tab has border-b-2 border-primary text-primary
- Keep existing tab hrefs and HTMX tab-switching logic
- Keep existing active tab Jinja conditional

PIPELINE STATUS (Overview tab, left column)
Stitch: vertical stepper with flex gap-3 layout
- Done steps: check_circle icon green + w-0.5 green connecting line
- Active step: spinning border-2 border-primary border-t-transparent + progress bar
- Pending steps: circle icon slate + italic "Queued after..." text
- Replace hardcoded steps with Jinja loop over pipeline stages

SPEAKER MANAGEMENT (Overview tab, right column)
Stitch: table with thead bg-slate-50, columns: Speaker ID, Label/Canonical, Quality Snippet, Actions
- Speaker ID: colored badge (S1/S2/S3) + "Speaker NN" text
- Label: dropdown/select for canonical assignment
- Quality: play button + waveform visualization placeholder + confidence percentage
- Keep existing speaker review logic and HTMX bindings

METADATA TAGS
Stitch: flex flex-wrap gap-2 with px-2 py-1 bg-slate-100 rounded text-xs pills
- "+ Add Tag" dashed border button
- Keep existing tag Jinja variables

QUICK ARTIFACTS
Stitch: grid grid-cols-2 gap-4
- Markdown Transcript: monospace preview with copy button
- Audio Chunks: list of chunk files with sizes
- Keep existing artifact download links

BOTTOM FOOTER
Stitch: fixed bottom-0 with Active Model info, Compute Node info, Auto-save timestamp, Publish Result button
- Keep existing system status Jinja variables
- Keep existing publish button functionality

Hard constraints
- Do not modify lan_app/ui_routes.py or any Python route logic
- Do not modify any Python file except test files
- Do not change any API endpoint URLs
- Do not add new JavaScript
- Do not modify base.html
- Do not touch Control Center templates (already migrated)
- Do not touch other page templates (voices, calendars, etc.)
- Do not use any legacy CSS classes. Use only Tailwind utilities.
- Copy classes from docs/stitch-reference/recording-detail.html. Do not invent new designs.

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- update tests where markup assertions changed
- the recording detail page layout must match Stitch reference
- all HTMX interactions must still work: tab switching, speaker actions, export, publish
- no legacy CSS class names used in recording detail templates

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui.py
- tests/test_cov_lan_app_ui.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py

Success criteria
- All recording detail templates use only Tailwind utility classes
- Page layout matches Stitch reference (breadcrumb, header, tabs, 2-column grid, footer)
- Pipeline status stepper matches Stitch reference (vertical with icons and connecting lines)
- Speaker management table matches Stitch reference
- Quick artifacts section matches Stitch reference
- Bottom footer matches Stitch reference
- All HTMX bindings preserved and functional
- All Jinja logic preserved
- No legacy CSS class names used in any recording detail template
- CI passes
