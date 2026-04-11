Run PLANNED PR

Branch: pr-title-edit-01

You are working in the LAN_Transcriber repository. This PR adds inline recording title editing on both the full-page inspector header and the Control Center compact inspector. Currently, recordings have no user-editable title. The display title is derived from calendar match, summary topic, or source filename, but there is no way for the operator to set a custom name. Follow AGENTS.md exactly.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of:
- lan_app/db.py (recording schema, column definitions)
- lan_app/migrations/ (latest migration number)
- lan_app/ui_routes.py (the _control_center_meeting_title_context function at approx line 390, _embedded_recording_confirmed_title at approx line 443)
- lan_app/templates/partials/recording_inspector_header.html (the h1 title rendering)
- lan_app/templates/partials/control_center/recording_details_card.html (the compact title rendering)

Phase 2 - Implement

CHANGE 1: Add display_title column to recordings table
Add a new nullable TEXT column display_title to the recordings table via a migration file. Default is NULL. When NULL, the existing title derivation logic applies (calendar > summary > filename). When set, display_title takes absolute priority.

CHANGE 2: Add PATCH endpoint for title update
Add a new route: PATCH /ui/recordings/{recording_id}/title
Request body: form field display_title (text, max 200 chars, stripped)
If display_title is empty or whitespace-only after stripping, set the column to NULL (revert to auto-derived title).
Response: return the updated title partial via HTMX (hx-swap) so the page updates in place without full reload.
Require auth if auth is enabled (same guard as other /ui/recordings/ endpoints).

CHANGE 3: Update title derivation to respect display_title
In _control_center_meeting_title_context and _embedded_recording_confirmed_title, check the display_title column first. If it is not NULL and not empty, use it as the title with source "user". The existing fallback chain (calendar > summary > filename) only applies when display_title is NULL.

CHANGE 4: Inline editing on full-page inspector header
In recording_inspector_header.html, make the h1 title editable:
- Show the current title as an h1 with a small pencil icon button next to it
- Clicking the pencil icon transforms the h1 into an input field pre-filled with the current title
- The input field has a "Save" button and a "Cancel" button (or Enter to save, Escape to cancel)
- Save sends a PATCH to /ui/recordings/{id}/title via HTMX and swaps the title area
- If the user clears the input and saves, the title reverts to auto-derived
- Show a subtle "(auto)" or "(derived)" indicator next to the title when the title is not user-set, so the operator knows the title was auto-generated

CHANGE 5: Inline editing on Control Center compact inspector
In recording_details_card.html, make the title editable using the same mechanism. Since this is a compact view, use a simpler approach:
- Show the title with a small pencil icon
- Clicking transforms into an inline input with Enter to save
- Same PATCH endpoint, same behavior

Hard constraints
- Do not change any existing title derivation logic except to add the display_title priority check
- Do not modify the recordings table schema beyond adding the one column
- Keep the migration backward-compatible (ALTER TABLE ADD COLUMN with NULL default)
- Do not add JavaScript frameworks
- Keep the solution Jinja/HTMX based
- Do not change any other recording fields

Phase 3 - Verify and summarize
- Run scripts/ci.sh until exit code 0
- Update/add tests for the new endpoint and title derivation
- Generate required review artifacts per AGENTS.md

Success criteria
- A display_title column exists in the recordings table via migration
- PATCH /ui/recordings/{id}/title updates the display_title
- When display_title is set, it takes priority over all auto-derived titles
- When display_title is cleared, the title reverts to auto-derived (calendar > summary > filename)
- The full-page inspector header shows an editable title with pencil icon
- The compact inspector shows an editable title with pencil icon
- The worklist in Control Center shows the user-set title when available
- CI passes
