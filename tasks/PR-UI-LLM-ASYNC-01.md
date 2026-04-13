Run PLANNED PR

PR_ID: PR-UI-LLM-ASYNC-01
Branch: pr-ui-llm-async-01
Title: Move LLM resummarize out of synchronous UI request into background job

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

Context
In lan_app/ui_routes.py:3134, _resummarize_recording makes a synchronous LLM call (asyncio.run(LLMClient().generate(...))) directly in the HTTP request handler. This blocks the request worker for the duration of the LLM call (potentially 10-30 seconds). The user sees the page hang until the response comes back.

This should be a background job: the UI shows "Resummarizing..." immediately, and the result appears when ready via polling or HTMX refresh.

Phase 1 - Inspect
Read:
- lan_app/ui_routes.py: _resummarize_recording (line ~3134), ui_resummarize_language (line ~7350)
- lan_app/jobs.py: existing job enqueue patterns
- lan_app/worker_tasks.py: how LLM stages are executed in background

Phase 2 - Implement

CHANGE 1: Convert resummarize to background job
When user clicks "Resummarize":
- Enqueue a lightweight background job (or use existing precheck job type with a "resummarize" flag)
- Return immediately with a "Resummarizing..." status indicator
- The worker picks up the job, runs LLM, updates summary.json
- UI auto-refresh (existing HTMX polling) picks up the new summary

CHANGE 2: Show progress in UI
While resummarize is running:
- Show a spinner or "Regenerating summary..." in the summary section
- When done, replace with the new summary via HTMX swap

CHANGE 3: Handle ui_resummarize_language similarly
Check if ui_resummarize_language (line ~7350) also blocks on LLM. If yes, apply the same pattern.

Phase 3 - Test and verify
- Run full CI.
- Click Resummarize in UI, verify page responds immediately.
- Verify summary updates after LLM completes.
- Verify stop/cancel works for resummarize jobs.

Success criteria:
- No LLM calls block UI request handlers.
- User sees immediate feedback ("Resummarizing...").
- Summary appears via auto-refresh when ready.
- No existing tests break.
