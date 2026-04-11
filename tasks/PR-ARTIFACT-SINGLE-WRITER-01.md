Run PLANNED PR

PR_ID: PR-ARTIFACT-SINGLE-WRITER-01
Branch: pr-artifact-single-writer-01
Title: Eliminate dual artifact writes: make orchestrator the single source of truth for all derived JSON artifacts

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

This task must be executed in 4 internal phases within a single run.

Context
diarization_metadata.json was written by both orchestrator.py (_write_diarization_metadata_artifact) and worker_tasks.py (_build_diarization_metadata_payload). When PR-140 added speaker_merge_diagnostics to orchestrator, worker_tasks silently overwrote the file without the new fields. This pattern exists for multiple artifacts: summary_json, speaker_turns_json, transcript_json, segments_json, transcript_txt. Any future feature that adds fields in orchestrator will hit the same bug.

The fix: worker_tasks.py should NEVER independently rebuild artifact payloads. It should either delegate to orchestrator or read-then-patch existing artifacts.

Phase 1 - Audit all dual write paths
For each derived artifact, identify every write in both files:

In orchestrator.py, find all atomic_write_json / atomic_write_text calls and document:
- Which artifact file
- What payload is written
- Under what conditions (success path, error path, no-speech path, quarantine path)

In worker_tasks.py, find all atomic_write_json / atomic_write_text calls to the SAME artifact files and document:
- Which artifact file
- What payload is written
- Whether it duplicates, extends, or conflicts with orchestrator's write
- Whether it runs AFTER orchestrator (overwrite risk) or BEFORE (no risk)

Produce a conflict matrix showing which artifacts have true dual-write conflicts.

Phase 2 - Classify each dual write
For each artifact with dual writes, classify:

A) DELEGATED: worker_tasks calls an orchestrator function that writes the artifact. No conflict. Keep as-is.
B) SEQUENTIAL: worker_tasks writes the artifact BEFORE orchestrator runs (e.g. initial empty state). No conflict. Keep as-is.
C) OVERWRITE: worker_tasks rebuilds and overwrites AFTER orchestrator has written the final version. THIS IS THE BUG PATTERN. Must fix.
D) INDEPENDENT: worker_tasks writes the artifact in a code path that orchestrator never reaches (e.g. error handling). No conflict. Keep as-is.

Phase 3 - Fix all OVERWRITE conflicts
For each type-C conflict:

Option A (preferred): Remove the worker_tasks write entirely if orchestrator already covers that code path.

Option B: If worker_tasks needs to add extra fields (e.g. stage timing, worker metadata), change it to READ the existing artifact, MERGE its fields, and WRITE back. Never rebuild from scratch.

Option C: If the payload construction is complex, extract it into a shared function in orchestrator (or a new shared module) and call it from both places with the same parameters.

For diarization_metadata.json specifically: verify that the hotfix from PR-141 (PR-SPEAKER-MERGE-DIAGNOSTICS-HOTFIX-01) is still correct after this refactor, or supersede it with the cleaner approach.

Phase 4 - Test and verify
- Run full CI.
- For each artifact that was changed, add or update a test that verifies:
  - The artifact is written exactly once in the normal success path
  - All expected fields are present after pipeline completion
  - Fields added by orchestrator are NOT stripped by worker_tasks
- Add a test that simulates the PR-140 scenario: a field written by orchestrator should survive the worker_tasks post-processing.

Success criteria:
- No derived JSON artifact is independently rebuilt by worker_tasks after orchestrator has written it.
- All existing fields in all artifacts are preserved.
- Future additions to orchestrator artifact payloads will not be silently dropped.
- No existing tests break.
- The conflict matrix is documented in a code comment or in AGENTS.md for future reference.
