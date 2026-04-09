Run PLANNED PR

PR_ID: PR-TAILWIND-BASE-01
Branch: pr-tailwind-base-01
Title: Add Tailwind CSS to base.html and migrate navbar, modal, and global elements

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict. Do not migrate page-level templates in this PR. Do not touch control_center.html, recording_detail.html, or any partials. Do not remove existing custom CSS yet.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read the Stitch reference files to understand the target design system:
- docs/stitch-reference/control-center.html (the tailwind config, navbar structure, footer structure)
- docs/stitch-reference/recording-detail.html (the navbar variant, footer variant)

Inspect the current base.html fully. Map:
- the Tailwind config block from the Stitch reference (colors, fonts, borderRadius)
- the current navbar markup and CSS classes
- the current modal markup and CSS classes
- the current global utility classes (badge, btn, filters, tabs, etc.)
- the full JavaScript section (must not be touched)

Phase 2 - Implement
Implement only the following changes to lan_app/templates/base.html:

1. Add Tailwind CDN and config
Add to the head section, before the existing style block:
- Tailwind CDN script: https://cdn.tailwindcss.com?plugins=forms,container-queries
- Google Fonts link for Inter (weights 400-900)
- Google Fonts link for Material Symbols Outlined
- Tailwind config script matching the Stitch reference exactly:
  - colors: primary #2463eb, background-light #f6f6f8, background-dark #111621, navy-nav #0f172a
  - fontFamily: display Inter
  - borderRadius: DEFAULT 0.25rem, lg 0.5rem, xl 0.75rem, full 9999px

2. Migrate the navbar
Replace the current .app-nav header block with Tailwind-based markup matching docs/stitch-reference/control-center.html exactly. Copy the Tailwind classes from the Stitch reference, do not invent new ones.

The navbar must preserve:
- all existing href values (/, /glossary, /voices, /calendars, /projects, /queue)
- the Jinja active-state logic: {% if active in ['dashboard', 'recordings', 'upload'] %}active{% endif %} etc.
- the hx-boost="true" attribute on body

The navbar must adopt from Stitch:
- bg-navy-nav dark background
- the logo area with material icon settings_input_component and "LAN Transcriber" text
- nav links styled as pills with active state using bg-primary/20 text-primary
- right side: search input, notifications button, user avatar circle
- the search input does not need to be functional, just present visually

3. Add a Stitch-style footer
Add a footer bar at the bottom of the page matching the Stitch control-center.html footer structure. Use the same Tailwind classes from the reference. The footer content will be static placeholder text for now (LAN Node status, version, uptime). The current system-bar partial will remain functional separately and will be migrated in a later PR.

4. Migrate the modal dialog
Replace the current .modal-backdrop and .modal CSS classes with Tailwind utility classes on the existing modal HTML elements. Keep all modal JavaScript functions untouched. Keep the same element IDs and aria attributes.

5. Do NOT remove existing CSS
Keep all existing CSS in the style block. Add a comment at the top of the style block:
/* LEGACY CSS - will be removed after full Tailwind migration. See PR-TAILWIND-REMAINING-01. */
This ensures all existing page templates continue to render correctly during migration.

6. Do NOT touch JavaScript
The entire script section at the bottom of base.html must remain exactly as-is. Do not modify, reformat, or move any JavaScript.

7. Body class
Add Tailwind body classes: font-display antialiased
Keep the existing hx-boost="true" attribute.

Hard constraints
- Do not modify any template file other than base.html
- Do not remove existing CSS from the style block
- Do not modify any JavaScript
- Do not change any Jinja block definitions ({% block title %}, {% block content %})
- Do not touch lan_app/ui_routes.py
- Do not touch any partial templates
- Do not modify any Python files except test files if assertions need updating
- Copy Tailwind classes from docs/stitch-reference/*.html, do not invent custom classes

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- update tests only where needed because of changed navbar markup
- the navbar must visually match the Stitch reference (dark background, pill-style links, search bar, avatar)
- all existing pages must still render without visual breakage (legacy CSS preserved)
- all modal functionality must work (open, close, delete confirm)

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui.py
- tests/test_cov_lan_app_ui.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py only if navbar assertions break

Success criteria
- Tailwind CDN and config are present in base.html head
- Tailwind config matches Stitch reference exactly (colors, fonts, radii)
- Navbar uses Tailwind classes copied from Stitch reference
- Navbar preserves all existing hrefs and Jinja active-state logic
- Footer bar is present with Stitch-style layout
- Modal dialog uses Tailwind utility classes
- All existing CSS is preserved with a LEGACY comment
- All JavaScript is untouched
- All existing pages render without breakage
- CI passes
