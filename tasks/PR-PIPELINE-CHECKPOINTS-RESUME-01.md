PR-PIPELINE-CHECKPOINTS-RESUME-01
=================================

Branch: pr-pipeline-checkpoints-resume-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Introduce explicit pipeline checkpoints and resumable execution so a retry does not restart the whole recording from the beginning after a late-stage failure. Today lan_app/worker_tasks.py still treats the single precheck job as a wrapper around the entire processing pipeline. If LLM chunking fails near the end, process_job(...) retries the same job and reruns sanitize/precheck/ASR/diarization again. This wastes GPU time and makes long recordings painful. Keep the single-job model, but make execution stage-aware and resumable from the first incomplete stage.

Constraints
- Keep the single RQ job model. Do not reintroduce multiple queue job types for each pipeline stage.
- Do not implement chunk-level resume in this PR; that belongs to the next PR. This PR is stage-level only.
- Do not redesign UI pages beyond the small additions needed to surface stage state.
- Preserve current recording outputs, artifacts, and business behavior when a job completes successfully.
- Preserve current retry policy semantics except where stage resume requires a better restart point.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Add durable stage-state persistence
- Add a dedicated persistence layer for per-recording pipeline stages using SQLite, not only transient JSON files.
- Use a new migration with the next number to add a table such as recording_pipeline_stages or equivalent, keyed by recording_id + stage_name.
- Persist at least:
  - recording_id
  - stage_name
  - status in {pending, running, completed, failed, skipped, cancelled}
  - attempt
  - started_at
  - finished_at
  - duration_ms or duration_seconds
  - error_code
  - error_text
  - metadata_json
  - updated_at
- Add DB helpers in lan_app.db for:
  - upsert stage state
  - mark stage started/completed/failed/skipped/cancelled
  - list stages for a recording in execution order
  - clear/reset stage state when explicitly requeueing from scratch
- Keep the helpers idempotent and retry-safe.

2) Define explicit stage boundaries for the current single-job pipeline
- Model the current pipeline with explicit coarse stages. Keep the set small but meaningful. A good minimum is:
  - sanitize_audio
  - precheck
  - calendar_refresh
  - asr
  - diarization
  - speaker_mapping_or_turn_build
  - language_analysis
  - metrics
  - llm_extract
  - llm_merge_or_summary_finalize
  - routing
  - export_artifacts
- You may adjust exact stage names to fit the current code structure, but keep them stable and testable.
- Do not collapse everything back into one generic pipeline stage.
- Make sure stage names map cleanly to current progress/stage codes so UI display remains understandable.

3) Refactor worker execution to resume from first incomplete stage
- In lan_app/worker_tasks.py keep process_job(...) as the single entrypoint, but stop treating a retry as a full cold start.
- Build an explicit stage orchestrator that:
  - inspects persisted stage state for the recording
  - validates whether required prior artifacts/outputs exist
  - skips already-completed stages when safe
  - resumes from the first stage that is not completed
- If a previous attempt completed ASR and diarization and then failed in llm_extract, a new attempt must resume at llm_extract without re-running ASR or diarization.
- On stage completion, persist timing/status metadata immediately.
- On stage failure, persist stage failure metadata before bubbling the error.
- Preserve the current terminal recording status behavior when the whole pipeline eventually succeeds or fails.

4) Make artifact validation explicit for resume safety
- Add a small validation layer that determines whether a completed stage can be trusted on resume.
- Do not trust only the DB row. Validate the minimal required artifact or output for resumed stages, for example:
  - sanitized audio file exists for sanitize_audio
  - precheck result artifact or derived data exists for precheck
  - transcript/segments artifacts exist for asr
  - diarization/turn artifacts exist for diarization or turn-building stages
  - summary/export artifacts exist for downstream stages where applicable
- If the DB says completed but the required artifact is missing/corrupt, the stage must be re-run and the reason must be logged.
- Keep validation deterministic and lightweight.

5) Keep progress and current UI behavior coherent
- Continue updating recording pipeline_stage and pipeline_progress in recordings so the current progress UI still works.
- When resuming, set pipeline_stage to the actual resumed stage rather than replaying from precheck.
- Keep the existing progress partial functional.
- Small UI additions are allowed to expose a list of stage states on the recording detail page, but keep them minimal in this PR.

6) Preserve retry behavior but change restart point
- Retryable failures should still be retryable according to current policy, but the next attempt must start from the first incomplete/invalid stage.
- Non-retryable failures must keep their current terminal behavior.
- Do not yet add chunk-level retry/resume here. If llm_extract as a whole fails, resume should re-enter the llm_extract stage, not redo ASR.

7) Make requeue semantics explicit
- Normal retry path should resume.
- Explicit user requeue from the UI should reset stage state and rerun from the beginning.
- Add or update a helper to clear stage state on deliberate requeue.
- Do not leave stale completed-stage rows behind after a manual requeue.

8) Logging and artifacts
- Append concise stage lifecycle lines to the step log, for example:
  - stage started: asr
  - stage completed: asr elapsed=...
  - stage resumed: llm_extract
  - stage invalidated: diarization missing artifact, rerunning
- Optionally write a lightweight JSON manifest artifact under derived/, but the SQLite stage table is the source of truth.

9) Tests with full coverage
Add deterministic offline tests for at least these scenarios:
- a first attempt completes early stages and fails in a later stage; the second attempt resumes from the correct later stage
- completed stages are skipped on resume when artifacts are present
- a stage marked completed is re-run when its required artifact is missing
- explicit user requeue clears stage state and reruns from the beginning
- stage timing/error fields are persisted correctly on success and failure
- progress/stage fields on recordings reflect the resumed stage rather than restarting at precheck
- non-retryable failures do not incorrectly resume or loop
- stale/ignored executions do not corrupt stage state
Mock heavy processing functions as needed; keep tests offline and fast.

10) Documentation and operator notes
- Update README or runbook with a short explanation that retries now resume from the first incomplete stage instead of restarting the whole recording.
- Mention that explicit requeue still forces a clean rerun.

Verification steps (must be included in PR description)
- Force a failure after ASR and diarization and confirm the next retry resumes from the later stage instead of re-running GPU-heavy stages.
- Confirm manual requeue clears checkpoints and restarts from the beginning.
- Confirm the step log and UI progress show the resumed stage accurately.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Durable per-recording pipeline stage persistence
- Stage-aware resume from the first incomplete/invalid stage
- Explicit artifact validation for resume safety
- Updated tests and docs

Success criteria
- A late-stage failure no longer causes ASR and diarization to rerun unnecessarily on the next attempt.
- Resume behavior is deterministic, visible in logs, and safe against missing artifacts.
- Manual requeue still provides a true clean rerun.
- CI remains green.
```
