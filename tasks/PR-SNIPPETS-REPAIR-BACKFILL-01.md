PR-SNIPPETS-REPAIR-BACKFILL-01
================================

Branch: pr-snippets-repair-backfill-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Provide a safe way to regenerate missing snippet artifacts for older or partially processed recordings without rerunning the full pipeline. After PR-SNIPPETS-STAGE-REORDER-01 and PR-SNIPPETS-UI-STATE-01, new recordings should expose snippets earlier, but older recordings may still have no derived/snippets_manifest.json because they were processed before the stage reordering or because snippet export failed in a non-fatal way. Add a repair/backfill path so operators can regenerate snippets from existing sanitized audio and speaker-turn artifacts, both for one recording and for a batch of eligible recordings, without recomputing ASR/LLM.

Relevant code to inspect first
- lan_app/worker_tasks.py
- lan_app/ui_routes.py
- lan_app/pipeline_stages.py
- lan_transcriber/pipeline_steps/snippets.py
- tests/test_ui_routes.py
- tests/test_pipeline_checkpoints_resume.py
- README.md
- docs/runbook.md

Constraints
- Do not rerun the full processing job just to regenerate snippets.
- Reuse the existing snippet generation logic and manifest format from lan_transcriber.pipeline_steps.snippets.
- Keep repair safe: only run when prerequisites exist and paths are trustworthy.
- Do not create fake success for recordings that lack sanitized audio or speaker-turn prerequisites.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Add a focused snippet-regeneration backend path
- Implement a reusable helper/service that can regenerate snippets for a single recording using existing artifacts:
  - sanitized audio path from derived/audio_sanitized.wav when present, with a safe fallback to raw audio only if that is already an accepted current contract and fully validated
  - derived/diarization_segments.json
  - derived/speaker_turns.json
  - derived/diarization_metadata.json or equivalent degraded flag source
  - recording duration from precheck.json when needed
- The helper must call export_speaker_snippets(...) and write derived/snippets_manifest.json and derived/snippets/ exactly like the normal pipeline stage.
- Keep the helper separate enough that both UI actions and any batch command can reuse it.

2) Add single-recording repair from the UI
- Add a minimal operator action on the recording detail page, preferably within the Speakers tab or Actions menu, such as Regenerate snippets.
- This action should be available when:
  - recording is terminal or otherwise has the needed prerequisite artifacts
  - snippet manifest is missing, stale, or the operator explicitly wants to repair it
- On success, the page should reflect newly available snippets without requiring a full reprocess.
- On failure, show a clear non-destructive error message.

3) Add a batch/admin backfill path
- Add a CLI/admin command or focused utility for batch regeneration across eligible recordings.
- Keep the interface simple and deterministic, for example:
  - a command that targets one recording id
  - a command that scans for completed recordings missing snippets_manifest.json
- The batch path must skip ineligible recordings safely and report counts for regenerated, skipped, and failed items.
- Keep it offline and local; do not depend on external services.

4) Define clear eligibility rules
- A recording is eligible for snippet repair only when the required prerequisites exist and are parseable.
- If prerequisites are missing, do not create an empty success manifest that hides the real problem.
- Surface specific reasons such as missing speaker_turns, missing sanitized audio, or missing precheck duration.

5) Preserve checkpoint integrity and avoid accidental stage corruption
- Snippet repair/backfill should not mutate unrelated stage rows or clear later stages.
- It is a targeted artifact repair path, not a full pipeline rerun.
- If you store metadata about manual snippet regeneration, keep it additive and non-breaking.

6) Optional lazy repair integration for legacy UI state
- If a recording opens in the Speakers tab with no manifest and qualifies for repair, the UI may show a contextual button or call-to-action instead of a dead-end placeholder.
- Keep this lightweight and consistent with PR-SNIPPETS-UI-STATE-01.

7) Logging and observability
- Log snippet repair actions clearly, including recording id and outcome.
- For batch mode, print a concise summary of regenerated/skipped/failed counts.
- Do not leak large prompt/audio contents in logs.

8) Tests with full coverage
Add or update deterministic tests for at least these cases:
- single-recording regeneration succeeds when prerequisites exist and creates snippets_manifest.json
- regeneration skips or fails cleanly when prerequisites are missing
- UI action triggers regeneration and surfaces snippets afterward
- batch backfill only targets eligible recordings and reports counts correctly
- repeated regeneration is idempotent or replaces artifacts deterministically without corrupting state
- repair path does not rerun the full pipeline or mutate unrelated stage rows
Keep tests offline and do not require real long-running audio processing.

9) Documentation
- Update README.md and/or docs/runbook.md with a short operator note explaining how to regenerate snippets for older recordings.
- Document the batch/admin repair path and eligibility expectations.

Verification steps (must be included in PR description)
- Demonstrate regeneration for one legacy recording with missing snippets_manifest.json.
- Demonstrate that an ineligible recording fails safely with a clear reason.
- Demonstrate that batch backfill reports regenerated/skipped/failed counts.
- Confirm that no full pipeline rerun is required.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Reusable snippet regeneration helper/service
- Single-recording UI repair action
- Batch/admin backfill command for legacy recordings
- Tests and brief operator documentation

Success criteria
- Older recordings missing snippet artifacts can be repaired without rerunning ASR/LLM.
- Repair is safe, explicit, and deterministic.
- Operators have a clear path to recover snippet UX for legacy data.
- CI remains green with full coverage preserved.
```
