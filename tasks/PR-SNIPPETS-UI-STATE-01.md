PR-SNIPPETS-UI-STATE-01
========================

Branch: pr-snippets-ui-state-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make the Speakers tab accurately explain snippet state throughout processing. Today lan_app/templates/recording_detail.html falls back to the generic placeholder “No snippet quality data found.” whenever there are no clean_snippets and no snippet_warnings. After PR-SNIPPETS-STAGE-REORDER-01, snippets should appear earlier, but the UI still needs explicit state handling so operators can tell the difference between “not generated yet”, “generating now”, “generation failed but processing continues”, and “generated but no clean snippets passed filters”. The result must make the Speakers tab useful and self-explanatory during Processing, not only after Completed.

Relevant code to inspect first
- lan_app/ui_routes.py
- lan_app/templates/recording_detail.html
- lan_app/pipeline_stages.py
- lan_app/worker_tasks.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- tests/test_pipeline_checkpoints_resume.py

Constraints
- Build on top of PR-SNIPPETS-STAGE-REORDER-01. Assume snippet_export is a real stage.
- Do not redesign snippet purity or file generation logic in this PR.
- Do not regress Add sample / Remap behavior when snippets are ready.
- Keep state messages deterministic and backed by real pipeline-stage/manifest data.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Add explicit snippet UI state modeling
- In lan_app/ui_routes.py introduce a focused helper that resolves snippet UI state for a recording/speaker, based on:
  - snippet_export stage row status and metadata
  - current recording status/current stage
  - existence and contents of derived/snippets_manifest.json
  - accepted/rejected manifest entries for the speaker
- Produce explicit states at minimum for:
  - not_started
  - running
  - ready_with_clean_snippets
  - ready_no_clean_snippets
  - failed_nonfatal
  - legacy_missing_manifest or unavailable
- Keep state calculation easy to test and avoid scattering template-only logic across multiple places.

2) Replace the generic placeholder with meaningful messages
- Update lan_app/templates/recording_detail.html so the Snippets column no longer falls back to “No snippet quality data found.” for all cases.
- Show distinct operator-facing messages such as:
  - snippets not generated yet because the pipeline has not reached snippet_export
  - snippets are currently being generated
  - snippet generation failed but the rest of processing continues
  - snippet generation completed but no clean snippets passed quality filters
- Reuse existing warning data where appropriate and avoid duplicating contradictory messages.

3) Make snippets usable during Processing
- If snippet_export is already completed and clean snippets exist while the overall recording is still Processing, the Speakers tab must immediately show the audio players and Add sample choices.
- Do not gate snippet display on terminal recording status.
- Keep Add sample enabled whenever clean snippets are present, even during llm_extract.

4) Integrate snippet stage visibility into the recording detail experience
- Surface snippet_export state naturally within the existing recording detail context.
- You may add a small status line, badge, or per-row helper text, but keep the UI minimal and consistent with the current style.
- Avoid visual clutter. The goal is clarity, not a redesign.

5) Improve no-clean-snippet messaging
- Reuse manifest-derived reasons from _snippet_warning_messages(...) and _no_clean_snippet_message(...), but make sure they interact correctly with stage state.
- If snippet_export is ready and every candidate was rejected, the UI should explain the actual reason instead of implying snippets were never generated.
- Preserve current degraded/overlap/failed_extract/short messaging where helpful.

6) Handle older recordings gracefully
- If a completed or older recording has no snippets manifest yet, do not mislabel it as currently generating snippets.
- Show a clear legacy/unavailable message that leaves room for PR-SNIPPETS-REPAIR-BACKFILL-01 to provide regeneration later.

7) Keep template and route contracts clean
- Prefer preparing structured UI state in ui_routes.py and keeping the template relatively dumb.
- Avoid embedding stage-resolution logic directly in Jinja.
- Keep response payloads backward-compatible where possible, but add explicit snippet_ui_state fields if that improves maintainability.

8) Tests with full coverage
Add or update deterministic tests for at least these cases:
- recording has not reached snippet_export yet -> UI shows a not-generated-yet message
- snippet_export is running -> UI shows a generating message
- snippet_export completed with accepted snippets while recording is still Processing -> audio controls and Add sample options are visible
- snippet_export completed but all candidates were rejected -> UI shows the correct no-clean-snippets reason
- snippet_export completed with non-fatal failure metadata or missing manifest -> UI shows a failure/unavailable message
- a legacy completed recording without manifest is handled clearly and does not pretend snippets are running
- Add sample remains disabled only when there are truly no clean snippets
Keep tests offline and do not depend on real audio files.

9) Documentation
- Update any brief operator-facing docs or runbook notes if they mention snippet visibility only after full completion.
- Keep the documentation concise.

Verification steps (must be included in PR description)
- Show screenshots or test assertions for the four core UI states: not_started, running, failed_nonfatal, ready.
- Confirm that clean snippets are visible and usable during llm_extract once snippet_export completed.
- Confirm that recordings without a manifest no longer collapse into the generic placeholder.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Explicit snippet UI state model in ui_routes.py
- Updated recording_detail template with state-aware messages
- Tests covering snippet UI states and Add sample availability
- Small docs note if needed

Success criteria
- Operators can immediately tell whether snippets are pending, generating, failed, or ready.
- Speakers tab becomes trustworthy during Processing, not only after completion.
- No generic ambiguous placeholder remains for all empty states.
- CI remains green with full coverage preserved.
```
