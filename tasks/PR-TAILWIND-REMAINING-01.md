Run PLANNED PR

PR_ID: PR-TAILWIND-REMAINING-01
Branch: pr-tailwind-remaining-01
Title: Migrate remaining pages to Tailwind and remove all legacy CSS from base.html

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

There is no Stitch reference file for these pages. Use the same Tailwind design language established in the already-migrated Control Center and Recording Detail pages. Match the same color tokens (primary, slate palette, navy-nav), spacing scale, border-radius values, font sizes, and component patterns (cards with bg-white rounded-xl border border-slate-200, buttons, badges, tables, tabs, forms).

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read the already-migrated templates to understand the established Tailwind patterns:
- lan_app/templates/base.html (navbar, footer, modal, Tailwind config)
- lan_app/templates/control_center.html (page layout patterns)
- lan_app/templates/partials/control_center/upload_panel.html (card and form patterns)

Then read every remaining page template that still uses legacy CSS classes:
- lan_app/templates/voices.html
- lan_app/templates/calendars.html
- lan_app/templates/projects.html
- lan_app/templates/queue.html
- lan_app/templates/glossary.html
- lan_app/templates/login.html
- lan_app/templates/dashboard.html
- lan_app/templates/recordings.html
- lan_app/templates/upload.html
- any remaining partials not yet migrated

For each file, list every legacy CSS class used (classes defined in the base.html style block, not Tailwind utilities).

Phase 2 - Implement

Part A: Migrate each remaining page template
For each page, replace all legacy CSS classes with Tailwind utility classes following the established design language.

General mapping rules:
- .control-center-pane, .control-center-panel -> bg-white rounded-xl border border-slate-200 p-5 shadow-sm
- .btn -> inline-flex items-center justify-center gap-2 rounded-lg h-10 px-4 bg-white border border-slate-200 text-sm font-semibold hover:bg-slate-50 transition-colors
- .btn-primary -> bg-primary text-white border-primary hover:bg-primary/90
- .btn-danger -> bg-red-50 text-red-600 border-red-100 hover:bg-red-100
- .badge -> inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border
- .tabs -> flex gap-8 border-b border-slate-200
- .tab -> border-b-2 border-transparent pb-3 text-sm font-bold text-slate-500
- .tab.active -> border-primary text-primary
- .info-grid -> grid grid-cols-[minmax(140px,180px)_1fr] gap-2 text-sm
- .stat-card -> bg-white border border-slate-200 rounded-xl p-4 shadow-sm
- .filters -> flex gap-3 items-center flex-wrap
- table -> w-full text-left text-sm with thead bg-slate-50 text-xs uppercase
- .placeholder -> text-slate-400 italic py-5
- .eyebrow -> text-[10px] font-bold text-primary uppercase tracking-[0.1em]
- .muted -> text-slate-500
- .conn-card -> bg-white border border-slate-200 rounded-xl p-4 shadow-sm
- form inputs -> rounded-lg border border-slate-200 px-3 py-2 text-sm focus:ring-1 focus:ring-primary focus:border-primary

For each page:
1. Replace every legacy class with its Tailwind equivalent
2. Keep all Jinja variables, conditionals, and loops
3. Keep all HTMX attributes
4. Keep all onclick handlers and data-* attributes
5. Keep all element IDs used by JavaScript

Part B: Remove legacy CSS from base.html
After all templates are migrated, remove the entire legacy CSS style block from base.html (everything marked with the LEGACY comment added in PR-TAILWIND-BASE-01).

Keep only:
- Tailwind CDN script
- Tailwind config script
- Google Fonts links
- Material Symbols link
- Any minimal CSS that Tailwind cannot express (if any)
- The body { font-family: 'Inter', sans-serif; } rule as a fallback
- The .material-symbols-outlined { font-size: 20px; } rule

Part C: Clean up status badge classes
The current CSS has status-specific classes (.s-Queued, .s-Processing, .s-Ready, etc.). These must be replaced with Jinja conditionals that output Tailwind classes directly. For example:
- Queued: bg-blue-100 border-blue-300 text-blue-800
- Processing: bg-amber-100 border-amber-300 text-amber-800
- Ready: bg-green-100 border-green-300 text-green-800
- Failed: bg-red-100 border-red-300 text-red-800
- Quarantine: bg-red-100 border-red-300 text-red-800
- Published: bg-violet-100 border-violet-300 text-violet-800
- NeedsReview: bg-yellow-100 border-yellow-300 text-yellow-800
- Stopped: bg-slate-100 border-slate-300 text-slate-800

Update every template that uses .s-* badge classes to use inline Tailwind conditionals or a Jinja macro.

Hard constraints
- Do not modify lan_app/ui_routes.py or any Python route logic
- Do not modify any Python file except test files
- Do not change any API endpoint URLs
- Do not add new JavaScript beyond what is strictly needed
- Do not modify already-migrated Control Center or Recording Detail templates unless removing a legacy class reference
- After this PR, no legacy CSS class should remain in any template
- After this PR, the base.html style block should contain only minimal non-Tailwind rules

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- grep -r for any remaining legacy CSS class names across all templates to confirm zero usage
- every page must render correctly with only Tailwind CSS (no legacy stylesheet)
- all HTMX interactions must work on every page
- status badges must display correct colors for each status

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui.py
- tests/test_cov_lan_app_ui.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py

Success criteria
- All page templates use only Tailwind utility classes
- The legacy CSS style block is removed from base.html
- base.html contains only Tailwind CDN, config, fonts, and minimal fallback CSS
- Status badge colors work via Jinja conditionals + Tailwind classes
- Every page renders correctly: voices, calendars, projects, queue, glossary, login
- All HTMX bindings preserved and functional on every page
- No legacy CSS class name appears in any template file
- CI passes
