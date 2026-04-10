Run PLANNED PR

PR_ID: PR-UI-POLISH-POST-TAILWIND-01
Branch: pr-ui-polish-post-tailwind-01
Title: Fix layout bugs, remove duplicate sections, and clean up post-Tailwind migration issues

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_app/templates/base.html (navbar, footer, main wrapper)
- lan_app/templates/recording_detail.html (nested main tag)
- lan_app/templates/control_center.html (layout shell)
- lan_app/templates/partials/control_center/inspector_pane.html (hx-swap causing scroll reset)
- lan_app/templates/partials/control_center/system_bar.html (dynamic footer)
- lan_app/templates/partials/recording_inspector_full_overview.html (duplicate info cards)
- lan_app/templates/partials/speaker_review_cards.html (speaker avatar overflow)
- lan_app/templates/partials/recording_inspector.html (progress polling)
- lan_app/templates/partials/recording_inspector_header.html (badge mapping)
- lan_app/templates/partials/control_center/recordings_table.html (badge mapping)
- lan_app/templates/queue.html (badge mapping, button classes)
- lan_app/templates/voices.html (button classes)
- lan_app/templates/glossary.html (button classes)
- lan_app/templates/projects.html (button classes)
- lan_app/templates/login.html (button classes)
- lan_app/templates/calendars.html (button classes)

Phase 2 - Implement
Fix exactly the following 9 issues. Do not add anything beyond these fixes.

FIX 1: Remove nested main tag from recording_detail.html
Problem: recording_detail.html wraps content in its own <main> tag, but base.html already wraps {% block content %} in a <main> tag. This creates nested <main><main> causing broken layout and a second navbar appearing in the middle of the page when navigating from Control Center to a recording detail page.
Fix: Remove the <main> wrapper from recording_detail.html. Keep only the inner content (nav breadcrumb, inspector include, system bar include). Use a <div> wrapper inside {% block content %} with the classes that were on the inner main: class="flex flex-col flex-1 max-w-[1440px] mx-auto w-full px-6 py-6". The base.html <main> provides the outer wrapper.

FIX 2: Merge double footer into single dynamic footer
Problem: Two footer bars appear at the bottom of the page. base.html has a static footer with placeholder text ("LAN Node: Online", "Local Database", "Version: LAN Transcriber"). control_center.html and recording_detail.html additionally include system_bar.html which has real dynamic data (node status, GPU, LLM info).
Fix:
- Remove the static placeholder footer from base.html entirely
- Make the system_bar.html the single footer for all pages
- Include system_bar.html in base.html after the main content block (before the delete modal), not inside individual page templates
- Remove the system_bar include from control_center.html and recording_detail.html since base.html will handle it
- The system bar must render at the bottom of the page with the same Stitch-style appearance (min-h-[2rem] bg-slate-100 border-t border-slate-200)

FIX 3: Remove search, notifications bell, and user avatar from navbar
Problem: The search input, notifications bell button, and user avatar circle in the navbar are non-functional decorative elements that add visual clutter.
Fix: Remove the entire right-side div in the navbar header that contains:
- the search input with "Search recordings..." placeholder
- the notifications bell button
- the user avatar circle ("OP")
Keep only the left side: logo + "LAN Transcriber" title + navigation links.

FIX 4: Fix inspector scroll position reset during Processing
Problem: When a recording is in Processing status, the right inspector pane in Control Center auto-refreshes every 2 seconds via hx-trigger="every 2s" with hx-swap="outerHTML". This replaces the entire inspector pane element including its scroll container, resetting the scroll position to the top each time the user is trying to read or interact with the details.
Fix: Change the inspector pane refresh strategy to preserve scroll position. Options (choose the simplest that works):
- Option A: Change hx-swap="outerHTML" to hx-swap="innerHTML" on the inspector pane. Move the hx-get, hx-trigger, and hx-target attributes to the outer container element that is not replaced. Adjust the system bar URL sync JavaScript in base.html if needed to work with innerHTML swap.
- Option B: Add hx-swap="outerHTML scroll:none" or use hx-preserve on the scroll container.
- Option C: Wrap the scrollable content inside the inspector pane in a separate element that preserves scroll, while the outer shell handles HTMX replacement.
Whichever option is chosen, verify that:
- auto-refresh still works during Processing
- scroll position is preserved when content updates
- tab switching still works
- system bar URL sync still works

FIX 5: Remove duplicate information cards from Recording Page Overview tab
Problem: The full-page recording inspector Overview tab shows 6 sections that duplicate information already visible in the page header and pipeline stepper. These verbose text cards clutter the Overview without adding value.
Remove these sections entirely from recording_inspector_full_overview.html:
- "State" card (right column, shows rec.status which is already in the header badge)
- "Blocker" card (right column, shows blocker info already visible in header status_reason)
- "Next Action" card (right column, shows action suggestion already implied by status)
- "Review Focus" section (right column, shows review flags, almost always empty)
- "Pipeline Signal" section (left column, shows raw pipeline data already visualized in Pipeline Status stepper)
- "Core Metadata" section (left column, shows captured/duration/source/language already in the page header metadata chips)
Keep only:
- "Pipeline Status" stepper (left column) - the primary visual pipeline display
- "Metadata Tags" section (left column, if it exists) - department/priority/region tags
After removing the 6 sections, adjust the layout. If the right column of the Overview tab is now empty, either:
- Make Pipeline Status full-width (remove the 2-column grid, let pipeline stepper use the full width)
- Or keep the 2-column layout and show a concise status summary in the right column (a single line with the current status badge and stage, not a verbose card)
The goal is a clean Overview tab where the Pipeline Status stepper is the hero element.

FIX 6: Fix speaker avatar text overflow in full-page Speakers tab
Problem: In speaker_review_cards.html line 19, the speaker avatar circle uses {{ row.speaker }} which outputs the full diarization label like "SPEAKER_00" inside a size-8 (32px) circle. The text overflows the circle and overlaps with the speaker name text next to it.
Fix: Generate a short 2-character avatar label. Use this logic in a Jinja expression or set block:
- If row.speaker starts with "SPEAKER_" then extract the number and display "S" + number, e.g. "SPEAKER_00" -> "S0", "SPEAKER_01" -> "S1", "SPEAKER_12" -> "S12"
- Otherwise (custom speaker names like "John Doe") display first 2 uppercase characters, e.g. "JO"
This matches the Stitch reference which uses "S1", "S2" format for speaker avatars.
The avatar circle class should remain size-8 with text-xs to fit 2-3 characters comfortably.

FIX 7: Consistent font rendering across all pages
Problem: The recording detail page renders with inconsistent font sizing and spacing compared to Control Center due to the nested main tag (FIX 1) and different padding/max-width values.
Fix: After fixing the nested main issue (FIX 1), verify that all pages use the same Inter font at consistent sizes. The base.html main wrapper provides the default layout. Individual pages that need different max-width (like recording_detail using 1440px vs default 1600px) should use a wrapper div inside {% block content %}, not a competing main element.

FIX 8: Extract duplicated status badge mapping into a shared Jinja macro
Problem: The same status-to-Tailwind-class mapping dict is copy-pasted in at least 3 templates:
- lan_app/templates/partials/speaker_review_cards.html (lines 3-11)
- lan_app/templates/queue.html (lines 5-17)
- lan_app/templates/partials/recording_inspector_header.html (badge mapping block)
Any future status color change requires editing all copies, which is error-prone.
Fix: Create a shared Jinja macro file at lan_app/templates/partials/macros.html (or a similar shared location). Define a macro or a dict that maps recording/job statuses to Tailwind badge classes. For example:
  {% macro status_badge_classes(status) %}
  ...returns the appropriate "bg-xxx border-xxx text-xxx" string...
  {% endmacro %}
Or define a single dict variable via {% set %} in an includable snippet.
Then replace all duplicated badge dicts across the 3+ templates with an import of this shared macro:
  {% from "partials/macros.html" import status_badge_classes %}
Make sure all existing statuses are covered: Queued, Processing, Stopping, Stopped, NeedsReview, Ready, Published, Quarantine, Failed, and the lowercase job variants (queued, started, finished, failed).

FIX 9: Extract duplicated button class strings into Jinja macros
Problem: Long identical Tailwind class strings for buttons are repeated 15+ times across templates. The primary offender is the standard button pattern:
  inline-flex items-center justify-center gap-2 min-h-[40px] px-3.5 py-2 border border-slate-300 rounded-full bg-white text-slate-900 cursor-pointer text-[13px] font-bold no-underline hover:bg-slate-50 transition-colors
And the danger button variant:
  inline-flex items-center justify-center gap-2 min-h-[40px] px-3.5 py-2 border border-red-200 rounded-full text-red-700 bg-red-50 cursor-pointer text-[13px] font-bold no-underline hover:bg-red-100 transition-colors
These appear in: base.html (modal), voices.html, glossary.html, projects.html, calendars.html, login.html, and possibly others.
Fix: Add button macros to the shared macro file created in FIX 8 (lan_app/templates/partials/macros.html). For example:
  {% macro btn_classes() %}inline-flex items-center justify-center gap-2 min-h-[40px] px-3.5 py-2 border border-slate-300 rounded-full bg-white text-slate-900 cursor-pointer text-[13px] font-bold no-underline hover:bg-slate-50 transition-colors{% endmacro %}
  {% macro btn_danger_classes() %}inline-flex items-center justify-center gap-2 min-h-[40px] px-3.5 py-2 border border-red-200 rounded-full text-red-700 bg-red-50 cursor-pointer text-[13px] font-bold no-underline hover:bg-red-100 transition-colors{% endmacro %}
Then replace all hardcoded button class strings with {{ btn_classes() }} or {{ btn_danger_classes() }} in every template that uses them.
Important: do not use @apply in a <style> block because Tailwind CDN/standalone JS does not support @apply. Jinja macros are the correct approach here.

Hard constraints
- Do not modify lan_app/ui_routes.py or any Python route logic
- Do not modify any Python file except test files
- Do not change any API endpoint URLs
- Do not add new features or redesign components
- Do not modify the Tailwind config
- Do not touch the control_center.html inline <style> block (the CSS-var for full-height layout is intentional and working)
- Do not change the upload panel, recordings worklist, or other sections not listed above
- Keep all existing HTMX functionality working (tab switching, uploads, actions, polling)
- Keep all existing JavaScript functions in base.html working
- Do not use @apply in CSS. Use only Jinja macros for class deduplication.

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- manually verify (or confirm via test assertions):
  - only one navbar visible on recording detail page, at the top
  - only one footer bar visible on all pages, showing dynamic system status data
  - no search input, bell icon, or user avatar in navbar
  - scrolling inspector pane during Processing does not reset scroll position on auto-refresh
  - Recording Page Overview tab shows only Pipeline Status stepper and Metadata Tags (no State/Blocker/Next Action/Review Focus/Pipeline Signal/Core Metadata cards)
  - speaker avatars show short labels like "S0", "S1" (not full "SPEAKER_00")
  - all pages use consistent Inter font rendering with no layout jumps
  - status badge classes are defined in exactly one place (partials/macros.html)
  - button class strings are defined in exactly one place (partials/macros.html)
  - grep for the old long button class string returns zero matches outside macros.html

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui.py
- tests/test_cov_lan_app_ui.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py

Success criteria
- No nested main tags on any page
- Single dynamic footer bar on all pages
- Navbar contains only logo + title + nav links (no search/bell/avatar)
- Inspector scroll position preserved during auto-refresh in Processing state
- Recording Page Overview shows only Pipeline Status + Metadata Tags
- Speaker avatars display "S0", "S1" etc. for diarized speakers, first 2 chars for named speakers
- Consistent font rendering across all pages
- Status badge mapping lives in one shared macro file, imported everywhere
- Button class strings live in one shared macro file, used everywhere
- No @apply used anywhere
- CI passes
