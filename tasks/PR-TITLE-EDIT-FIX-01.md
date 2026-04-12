Run PLANNED PR

PR_ID: PR-TITLE-EDIT-FIX-01
Branch: pr-title-edit-fix-01
Title: Fix title edit reset during processing and remove (auto) indicator

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
Two issues with the recording title:
1. When editing the title during processing, the inspector pane auto-refreshes (hx-trigger="every 2s") and overwrites the DOM, discarding the edit input mid-typing.
2. The "(auto)" label next to the pencil icon adds visual clutter without value. Users know they can click the pencil.

Phase 1 - Inspect
Read:
- lan_app/templates/partials/recording_title_edit.html: title edit partial with (auto) span
- lan_app/templates/partials/control_center/inspector_pane.html: auto-refresh trigger
- lan_app/templates/partials/control_center/recording_details_card.html: where title_edit is included

Phase 2 - Implement

FIX 1: Preserve title edit during auto-refresh
Option A (preferred): Add hx-preserve="true" attribute to the title edit div so HTMX skips it during swap.
Option B: Add id-based exclusion in the hx-select of the polling swap so the title container is not replaced during auto-refresh.
Option C: In the auto-refresh response, detect if the user is currently editing (form is open) and skip the title section.

Use whichever approach works cleanest with the existing HTMX swap pattern.

FIX 2: Remove (auto) indicator
In recording_title_edit.html, remove the entire block:
  {% if not _rec.title_is_user_set %}
  <span class="text-[11px] text-slate-400 italic" ...>(auto)</span>
  {% endif %}

Phase 3 - Test and verify
- Run full CI.
- Manually verify: start editing title while recording is processing, confirm edit is not reset by auto-refresh.
- Verify (auto) label no longer appears.

Success criteria:
- Title edit survives auto-refresh during processing.
- No (auto) label visible anywhere.
- No existing tests break.
