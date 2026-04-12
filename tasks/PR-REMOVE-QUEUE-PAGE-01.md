Run PLANNED PR

PR_ID: PR-REMOVE-QUEUE-PAGE-01
Branch: pr-remove-queue-page-01
Title: Remove Queue page from navigation and UI

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
The Queue page exposes internal job queue state (job IDs, precheck status, timestamps) that has no operational value for the user. Processing status is already visible in the Control Center worklist and recording detail. The Queue page adds confusion without utility.

Phase 1 - Inspect
Read:
- lan_app/templates/base.html: nav link to /queue (line ~94)
- lan_app/ui_routes.py: route handler for /queue
- lan_app/templates/ (find the queue page template)
- tests/ and tests_playwright/: any tests referencing /queue

Phase 2 - Implement

CHANGE 1: Remove nav link
In base.html, remove the Queue link from the navigation bar.

CHANGE 2: Remove or hide the route
Option A (preferred): Keep the route but remove it from nav. This way /queue still works as a hidden debug page if needed, but users never see it.
Option B: Remove the route entirely.

Use option A for safety. Add a comment: "Hidden debug page, not linked in nav."

CHANGE 3: Update tests
Remove or skip any tests that assert the Queue nav link is visible.

Phase 3 - Test and verify
- Run full CI.
- Verify Queue is not visible in navigation.
- Verify /queue still loads (hidden debug access).

Success criteria:
- Queue is not in the navigation bar.
- No existing tests break.
