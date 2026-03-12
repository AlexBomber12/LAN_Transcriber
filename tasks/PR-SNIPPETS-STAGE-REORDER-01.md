PR-SNIPPETS-STAGE-REORDER-01
================================

Branch: pr-snippets-stage-reorder-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make speaker snippets available during long-running LLM processing by moving snippet generation out of the late export_artifacts stage and into its own earlier pipeline stage. Today lan_app/worker_tasks.py generates snippets only inside _stage_export_artifacts(...), after llm_extract has already finished. This causes a bad UX for long recordings: the Speakers tab shows speaker rows but no snippet quality data while the system is still processing LLM chunks. The new behavior must generate derived/snippets_manifest.json and derived/snippets/ immediately after speaker_turns are available, before llm_extract begins, while remaining compatible with pipeline checkpoints/resume and without making snippet generation a hard blocker for summary generation.

Relevant code to inspect first
- lan_app/worker_tasks.py
- lan_app/pipeline_stages.py
- lan_app/ui_routes.py
- lan_app/templates/recording_detail.html
- lan_transcriber/pipeline_steps/snippets.py
- tests/test_pipeline_checkpoints_resume.py
- tests/test_pipeline.py
- tests/test_pipeline_steps_snippets.py
- tests/test_ui_routes.py
- tests/test_cov_lan_app_ui_routes.py
- README.md
- docs/runbook.md

Constraints
- Do not redesign snippet purity/scoring. That logic from PR-SNIPPETS-PURITY-01 must stay intact.
- Do not change speaker bank semantics or Add sample behavior beyond what is required for earlier availability.
- Do not regress checkpoint/resume behavior added in PR-PIPELINE-CHECKPOINTS-RESUME-01.
- Do not make snippet generation failure fatal for the overall processing job. Summary/metrics/routing must still continue.
- Keep 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Add a dedicated snippet_export pipeline stage
- Introduce a new stage named snippet_export.
- Register it in lan_app/pipeline_stages.py between speaker_turns and llm_extract.
- Give it a clear label like Snippet Export and a progress value between speaker_turns and llm_extract.
- Update PIPELINE_STAGE_DEFINITIONS, PIPELINE_STAGE_BY_NAME, stage_order/stage_progress behavior, and stage_artifact_paths.
- The new stage must own derived/snippets_manifest.json. export_artifacts must no longer be the stage that creates snippets.

2) Implement _stage_snippet_export(...) in lan_app/worker_tasks.py
- Add a new _stage_snippet_export(ctx: _PipelineExecutionContext) -> _StageResult.
- It must run after speaker_turns and before llm_extract.
- It should load only the prerequisites needed for snippet export:
  - sanitized audio path
  - precheck result
  - diarization segments
  - speaker_turns
  - diarization metadata or degraded-mode flag
- Use the existing export_speaker_snippets(...) and SnippetExportRequest from lan_transcriber.pipeline_steps.snippets.
- Preserve existing snippet settings: pad seconds, max duration, min duration, max snippets per speaker.

3) Make snippet_export idempotent and checkpoint-friendly
- If snippet_export already completed and derived/snippets_manifest.json is valid, resume logic must not rerun the stage.
- If llm_extract fails and the job resumes from llm_extract, snippet_export must remain completed and not regenerate snippets unnecessarily.
- Update stage validation so snippet_export is considered valid when the manifest exists and is parseable JSON.
- Remove snippets_manifest.json from export_artifacts validation ownership and move it to snippet_export ownership.

4) Keep snippet generation non-fatal
- Snippet export problems must not fail the whole recording processing pipeline.
- Handle these categories explicitly:
  - no usable speech / no speaker turns
  - degraded diarization preventing clean snippets
  - extraction failures for individual candidates
  - unexpected export exceptions
- The stage must always leave behind an inspectable manifest, even if there are zero accepted snippets.
- If an unexpected exception happens inside snippet export, catch it, log it, write an empty-or-failure manifest with enough metadata to explain what happened, and return a completed stage with warning-style metadata rather than raising and aborting llm_extract.
- Only raise if the pipeline prerequisites themselves are structurally broken, for example missing required upstream artifacts that indicate the pipeline state is invalid.

5) Remove late snippet creation from export_artifacts
- Refactor _stage_export_artifacts(...) so it no longer calls export_speaker_snippets(...).
- export_artifacts should now consume already-generated snippet artifacts if present, but it must not be responsible for generating them.
- Preserve existing transcript/summary/export behavior.
- Preserve existing no_speech/quarantine output behavior, but make sure snippet_export is the stage that owns snippet manifest creation or empty-manifest creation in those branches.

6) Ensure manifest behavior is stable for all branches
- For normal recordings, snippet_export should generate derived/snippets_manifest.json and accepted clean clips before llm_extract begins.
- For quarantined/no-speech/degraded cases, snippet_export must still produce stable inspectable output or explicit metadata so downstream code and operators do not see ambiguous missing-state behavior.
- Keep manifest format backward-compatible where practical. Extend metadata only if useful and fully tested.

7) Progress and logging
- The new stage must appear in pipeline progress and in stage rows/logs.
- Add step-log lines that make it obvious when snippet export starts, completes, and whether it produced accepted snippets.
- Include counts in metadata where useful, for example accepted snippets, speaker count, warning count, or manifest status.

8) Resume and invalidation correctness
- Update any stage ordering, later-stage clearing, and validation interactions so the new stage behaves correctly with resume/invalidation.
- If speaker_turns is invalidated, snippet_export and later stages must be invalidated as well.
- If snippet_export alone is invalidated, llm_extract and later stages must be invalidated and rerun, but earlier stages must remain intact.

9) Tests with full coverage
Add or update deterministic tests for at least these cases:
- pipeline stage definitions include snippet_export between speaker_turns and llm_extract
- stage_artifact_paths and validate_stage_artifacts treat snippet_export as the owner of snippets_manifest.json
- a normal pipeline run creates snippets_manifest.json before llm_extract begins
- export_artifacts no longer calls export_speaker_snippets(...)
- resume from llm_extract does not rerun snippet_export when its artifacts are valid
- invalidating snippet_export clears later stages but not earlier ones
- snippet export exceptions are converted into non-fatal manifest-backed results and the pipeline continues
- no_speech/quarantine paths still behave deterministically
Keep all tests offline.

10) Documentation
- Update README.md and docs/runbook.md briefly to reflect the new stage order.
- Document that snippets become available before llm_extract completes.
- Mention the main inspection artifact: derived/snippets_manifest.json.

Verification steps (must be included in PR description)
- Show that the stage order now includes snippet_export between speaker_turns and llm_extract.
- Demonstrate on a long-recording test path that snippets_manifest.json exists before llm_extract finishes.
- Confirm resume from llm_extract does not regenerate snippets when snippet_export is already valid.
- Confirm a snippet export failure does not prevent summary generation from continuing.
- Run scripts/ci.sh and keep CI green.

Deliverables
- New snippet_export stage in the worker pipeline
- snippets_manifest ownership moved out of export_artifacts
- checkpoint/resume-safe snippet availability before llm_extract
- updated tests and brief docs updates

Success criteria
- During long recordings, clean speaker snippets become available while llm_extract is still running.
- snippet_export is resumable, idempotent, and visible in progress/logs.
- Snippet problems do not abort summary processing.
- CI remains green with full coverage preserved.
```
