PR-OBSERVABILITY-ROOT-CAUSE-01
================================

Branch: pr-observability-root-cause-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Improve operator-facing observability so the UI and logs surface the real root cause of a recording failure or stop, rather than mostly generic retry-limit or opaque processing messages. The current system already has some progress and review-reason support, plus gpu_oom handling, but long-recording LLM failures and cancellation paths are still difficult to diagnose quickly. Add a root-cause-first error/diagnostics layer across worker status, recording detail UI, and progress partials.

Constraints
- Build on the previous resume/stop work rather than replacing it.
- Do not add an external observability stack. This PR is in-app diagnostics only.
- Keep UI changes compact and server-rendered.
- Preserve current functionality and artifact generation.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Define a normalized root-cause model
- Introduce a small, explicit set of reason codes and user-facing texts that cover the most important current cases, including at minimum:
  - gpu_oom
  - llm_chunk_timeout
  - llm_chunk_request_timeout
  - llm_chunk_parse_error
  - llm_chunk_connection_error
  - llm_merge_timeout or llm_merge_error if applicable
  - cancelled_by_user
  - force_stopped_by_user if the previous PR introduced that distinction
  - calendar_time_mismatch or suspicious_capture_time if such a detection already exists or is cheap to add
  - retry_limit_reached_only_as_wrapper, not as the primary displayed diagnosis
- Keep the mapping centralized and testable.

2) Preserve the primary cause when retries happen
- In lan_app/worker_tasks.py and related helper paths, do not let a late generic message like Processing hit the retry limit after repeated errors hide the primary reason.
- Persist both:
  - primary/root cause code + text
  - wrapper/retry context if you still want it for logs
- The recording detail page should show the primary reason first.

3) Surface current stage and chunk progress more explicitly
- Extend the existing progress model so the operator can see at least:
  - current stage label
  - chunk N/M for chunked LLM when applicable
  - stage/chunk elapsed time if available
  - whether the recording is resuming, stopping, cancelled, or force-stopped
- Update lan_app/templates/partials/recording_progress.html and relevant ui_routes helpers to display this cleanly.
- Keep the UI compact; do not build a large dashboard in this PR.

4) Show root-cause details on recording detail pages
- Update the overview and/or log/progress-related sections of lan_app/templates/recording_detail.html so operators can quickly see:
  - primary reason code/text
  - current or last stage
  - chunk index/total when relevant
  - whether the failure came from GPU, chunk timeout, parse error, connection problem, or user stop
- Use existing review_reason fields where appropriate, but do not overload them blindly if a cleaner dedicated display helper is needed.

5) Improve step-log clarity
- Standardize concise step-log lines for the major long-recording and stop scenarios.
- Include useful structured hints such as:
  - stage name
  - chunk index/total
  - elapsed time
  - attempt number
  - root cause code
- Avoid dumping huge prompt bodies or sensitive content.

6) Integrate with stage/chunk persistence
- Reuse the stage/chunk checkpoint metadata from prior PRs to drive UI diagnostics instead of inventing parallel state.
- When a chunk fails, the UI should be able to show chunk 3/10 timed out rather than only generic failure text.
- When a stop occurs, the UI should show whether it was acknowledged softly or required force termination.

7) Keep routing/review semantics sensible
- Do not break the existing NeedsReview flow.
- Where a recording lands in NeedsReview, its review reason should now be more specific and actionable.
- Where a recording lands in Stopped/Cancelled, do not mislabel it as a review failure.

8) Tests with full coverage
Add deterministic offline tests for at least these cases:
- retry-limit wrapper does not overwrite a more specific underlying reason
- gpu_oom remains specific and visible
- llm chunk timeout, parse error, and connection error map to distinct UI-visible reason texts/codes
- progress partial shows chunk N/M and relevant stage text when chunked processing is active
- cancelled_by_user and force-stopped-by-user display correctly
- detail page context helpers choose the primary root cause correctly
- step-log formatting helpers cover all main branches

9) Documentation and operator notes
- Update README/runbook briefly with a section on how to read root-cause diagnostics in the UI/logs.
- Keep this short and practical for the LAN operator.

Verification steps (must be included in PR description)
- Simulate a chunk timeout and confirm the UI shows the chunk-specific reason instead of only retry-limit text.
- Simulate a user stop and confirm the UI shows cancelled/stopped semantics rather than failure.
- Confirm progress partial displays current stage and chunk N/M while a long transcript is running.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Normalized root-cause codes/texts
- UI improvements for stage/chunk/root-cause visibility
- Clearer step-log diagnostics
- Updated tests and docs

Success criteria
- Operators can identify the true failure or stop reason quickly from the UI.
- Retry wrappers no longer hide the primary cause.
- Long-transcript progress is more transparent.
- CI remains green.
```
