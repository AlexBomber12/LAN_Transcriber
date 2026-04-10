Run PLANNED PR

PR_ID: PR-UI-POLISH-02
Branch: pr-ui-polish-02
Title: Fix Cyrillic font, compact inspector avatar, navbar flicker, and inspector scroll reset

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_app/templates/base.html (font-face declarations, system_bar include)
- lan_app/templates/partials/control_center/inspector_pane.html (hx-swap outerHTML)
- lan_app/templates/partials/control_center/recording_details_card.html (speaker avatar [:2])
- lan_app/templates/partials/control_center/system_bar.html (hx-swap outerHTML, hx-trigger load)
- lan_app/static/fonts/ (current font files)

Phase 2 - Implement
Fix exactly these 4 issues. Do not add anything beyond these fixes.

FIX 1: Add Cyrillic support to Inter font
Problem: The Inter font is loaded only with latin unicode-range (U+0000-00FF). Russian/Cyrillic text renders in a fallback system font that looks visually different from Inter latin characters. This is clearly visible on the recording details panel where Russian summary text has different letter shapes and weight compared to English UI labels.
Fix: Download the Inter Cyrillic woff2 subset from Google Fonts (unicode-range U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116) and save it as lan_app/static/fonts/inter-cyrillic.woff2. Add a second @font-face block in base.html for the Cyrillic range:
  @font-face {
    font-family: 'Inter';
    font-style: normal;
    font-weight: 400 900;
    font-display: swap;
    src: url(/static/fonts/inter-cyrillic.woff2) format('woff2');
    unicode-range: U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116;
  }
To download the file, use this approach:
  curl -o lan_app/static/fonts/inter-cyrillic.woff2 "https://fonts.gstatic.com/s/inter/v18/UcCo3FwrK3iLTcviYwY.woff2"
If the download fails (network restrictions), use an alternative approach: change the body font-family in the CSS to include Cyrillic-capable fallbacks:
  body { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }
And update the Tailwind config fontFamily.display to the same stack:
  "display": ["Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"]
This ensures Cyrillic text falls back to Segoe UI which is visually similar to Inter.

FIX 2: Fix speaker avatar in compact inspector (recording_details_card.html)
Problem: In recording_details_card.html line 45, the speaker avatar uses {{ speaker.primary_label[:2] }} which produces "SP" for all speakers named "SPEAKER_00", "SPEAKER_01" etc. All speakers show the same "SP" avatar, making them indistinguishable. The full-page speaker_review_cards.html was already fixed to show "S0", "S1" format, but the compact inspector was missed.
Fix: Apply the same avatar logic that is already in speaker_review_cards.html. Before the speaker loop in recording_details_card.html, add the Jinja logic:
  {% set _raw = speaker.primary_label or '' %}
  {% if _raw[:8] == 'SPEAKER_' %}
    {% set _avatar = 'S' ~ (_raw[8:]|replace('0', '', 1) if _raw[8:] != '0' and _raw[8:] != '00' else '0') %}
  {% else %}
    {% set _avatar = _raw[:2]|upper %}
  {% endif %}
Then use {{ _avatar }} instead of {{ speaker.primary_label[:2] }} in the avatar div.
Use the same extraction logic as speaker_review_cards.html: strip leading zeros from the number part, prefix with "S". SPEAKER_00 -> S0, SPEAKER_01 -> S1, SPEAKER_12 -> S12.

FIX 3: Fix navbar flickering caused by system_bar outerHTML swap
Problem: The system_bar.html uses hx-swap="outerHTML" which replaces the entire footer element every 15 seconds and on page load. On non-Control-Center pages (like recording detail), the system_bar triggers with "load" in addition to "every 15s", causing an immediate re-fetch and DOM replacement right after page render. This outerHTML replacement causes the browser to reflow the entire page layout, making the navbar appear to flicker/jump.
Fix: Change system_bar.html to use hx-swap="innerHTML" instead of hx-swap="outerHTML". This requires restructuring the template:
- The outer <footer> element with id="control-center-system-bar" must remain static (not replaced)
- Move the hx-get, hx-trigger, and hx-target to an inner <div> inside the footer
- The inner div gets replaced via innerHTML, but the outer footer stays in the DOM
- This prevents full-element replacement and eliminates layout reflow
The system bar URL sync JavaScript in base.html may need adjustment:
- Currently it sets hx-get on the element with id="control-center-system-bar"
- After this change, it should set hx-get on the inner div instead
- Or keep setting it on the footer but use innerHTML swap
Test that the system bar still auto-refreshes every 15s and still responds to refresh-control-center-system-bar events.

FIX 4: Fix inspector scroll position reset during Processing auto-refresh
Problem: The inspector_pane.html still uses hx-swap="outerHTML" with hx-trigger="every 2s" during Processing. This replaces the entire inspector pane section including its scroll container, resetting scroll position to top every 2 seconds. Users cannot scroll down to read speakers/summary/tone while a recording is processing.
Fix: Change the inspector pane to preserve scroll position during auto-refresh. The recommended approach:
- Keep the outer <section id="control-center-inspector-pane"> as a static scroll container
- Add an inner wrapper div that carries the hx-get, hx-trigger, hx-target="this", hx-swap="innerHTML" attributes
- The outer section provides overflow-y-auto scrolling and is never replaced
- The inner wrapper content gets swapped without affecting the scroll container
Alternatively, use hx-swap="outerHTML scroll:none" if HTMX version supports it (check htmx.min.js version).
The system bar URL sync JavaScript in base.html uses target.id checks for 'control-center-inspector-pane'. If the inner wrapper gets a different id, update the JS accordingly. Verify:
- auto-refresh works during Processing
- scroll position is preserved
- tab switching still works
- selecting a different recording still works
- system bar URL sync still works

Hard constraints
- Do not modify lan_app/ui_routes.py or any Python route logic
- Do not modify any Python file except test files
- Do not change any API endpoint URLs
- Do not add new features
- Do not modify the Tailwind config colors or borderRadius
- Keep all existing HTMX functionality working
- Keep all existing JavaScript functions in base.html working
- Do not touch recording_inspector_full_overview.html (already cleaned)
- Do not touch speaker_review_cards.html (already fixed)

Phase 3 - Verify
Run scripts/ci.sh until exit code 0, generate required review artifacts, update only the current PR status line in tasks/QUEUE.md according to AGENTS.md, and provide a concise final changelog.

Verification requirements
- run scripts/ci.sh until exit code 0
- generate required review artifacts per AGENTS.md
- Cyrillic text renders in Inter font (or visually similar fallback) on all pages
- Compact inspector speaker avatars show "S0", "S1" (not "SP")
- System bar updates without causing visible page flicker or navbar jump
- Inspector pane scroll position is preserved during Processing auto-refresh

Expected test areas to update
At minimum inspect and update as needed:
- tests/test_ui.py
- tests/test_cov_lan_app_ui.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests_playwright/test_ui_smoke_playwright.py

Success criteria
- Cyrillic characters render in Inter or a visually matching fallback font
- Compact inspector avatars show "S0", "S1" format matching the full-page Speakers tab
- No visible page flicker or navbar jump during system bar refresh
- Inspector scroll position preserved during Processing auto-refresh
- CI passes
