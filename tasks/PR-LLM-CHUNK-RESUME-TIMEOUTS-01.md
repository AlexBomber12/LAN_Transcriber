PR-LLM-CHUNK-RESUME-TIMEOUTS-01
=================================

Branch: pr-llm-chunk-resume-timeouts-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make long-transcript LLM chunk processing resilient so a timeout or transient failure in one chunk does not restart the entire recording. Today lan_transcriber/pipeline_steps/orchestrator.py::_run_chunked_llm_summary(...) writes llm_chunks_plan.json and per-chunk raw/error artifacts, but any chunk timeout raises RuntimeError and lan_app/worker_tasks.py retries the whole job. Implement chunk-level state, resume, and bounded timeout handling on top of the stage-level resume foundation from the previous PR.

Constraints
- Build on the stage-level checkpoint/resume work. Do not undo the single-job model.
- Keep the current strict JSON chunk-extract and merge contracts.
- Preserve deterministic behavior and offline tests.
- Do not redesign the entire LLM client. This PR is about chunk orchestration and timeout policy.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Persist per-chunk state durably
- Add durable persistence for chunk-level state. Prefer SQLite using a new migration with the next number, for example a table keyed by recording_id + chunk_kind + chunk_index, or another clear equivalent.
- Persist at least:
  - recording_id
  - chunk_group or phase (extract vs merge if needed)
  - chunk_index
  - chunk_total
  - status in {planned, running, completed, failed, cancelled, split}
  - attempt
  - started_at
  - finished_at
  - elapsed_seconds or duration_ms
  - error_code
  - error_text
  - parent_chunk_index if adaptive split creates children
  - metadata_json
- Add DB helpers for listing/upserting chunk state and resetting it on explicit clean requeue.

2) Resume only the failed/incomplete chunks
- Refactor _run_chunked_llm_summary(...) so it can inspect existing chunk state and artifacts.
- If chunks 1-6 are already completed and valid, a retry must resume from chunk 7, not rerun 1-6.
- Validate the required raw/extract artifacts for completed chunks before trusting them.
- Completed chunk extracts should be reused to build merge_input without re-calling the LLM for those chunks.

3) Differentiate timeout classes clearly
- There are currently at least 2 timeout concepts:
  - HTTP/request timeout inside LLM generation
  - wall-clock chunk timeout around the whole chunk call
- Make both explicit and logged distinctly.
- Introduce reason codes such as:
  - llm_chunk_timeout
  - llm_chunk_request_timeout
  - llm_chunk_parse_error
  - llm_chunk_connection_error
- Store these on chunk failure state and use them in error artifacts/logs.

4) Add adaptive split-on-timeout
- If a chunk times out, do not immediately fail the whole llm_extract stage.
- Split the failed chunk into 2 smaller child chunks when the content is still large enough to split sensibly.
- Replan only that chunk’s content into smaller child chunks while preserving ordering metadata.
- Mark the parent chunk as split and persist child chunk states.
- Do not recurse forever. Add a bounded minimum size / maximum split depth so the behavior is deterministic.
- If a chunk is already too small to split further and still times out, then fail the llm_extract stage with the specific chunk error.

5) Bound default chunk budgets more conservatively
- Update defaults and/or config handling so the long-transcript extraction path is less aggressive by default.
- Use the compact transcript/effective size semantics from the previous PR.
- Ensure the configured values continue to validate through AppSettings and pipeline Settings.
- Keep config knobs documented and test-covered.

6) Reuse completed chunk outputs to build merge input
- If a resumed attempt finds valid llm_chunk_XXX_extract.json artifacts and completed chunk-state rows, reuse them directly.
- Build llm_merge_input.json from the set of completed chunk extracts in order.
- Do not re-run successful chunk calls just because a later chunk failed previously.

7) Keep stage-level behavior coherent
- Integrate with the stage-level checkpoint system so the llm_extract stage can resume internally and then hand off to llm_merge.
- A failed chunk should not force ASR/diarization to rerun.
- A failure in llm_merge should be surfaced separately from chunk extract failures.

8) Logging and artifacts
- Extend llm_chunks_plan.json and/or related artifacts with chunk-state-friendly metadata.
- Keep per-chunk artifacts under derived/, including raw/extract/error files.
- Add concise step-log lines like:
  - llm chunk started index=3 total=10 attempt=2 chars=...
  - llm chunk resumed index=3
  - llm chunk split index=3 into 3a/3b or child indexes
  - llm chunk completed elapsed=...
  - llm chunk failed reason=llm_chunk_timeout
- Preserve operator readability.

9) Worker retry interaction
- If a chunk-level failure eventually bubbles up, the next job retry must resume from the failed/incomplete chunk set rather than restarting the full llm stage from scratch.
- Do not classify a chunk timeout as success just because earlier chunks were completed.
- Keep retry policy understandable and deterministic.

10) Tests with full coverage
Add deterministic offline tests for at least these cases:
- when chunk 2 fails, chunk 1 is reused and only chunk 2 is retried on the next attempt
- completed chunk artifacts are validated before reuse
- chunk timeout writes the correct error artifact and state row with specific reason code
- adaptive split creates child chunks and resumes processing from them
- minimum-size or max-depth guard prevents infinite splitting
- llm_merge_input is rebuilt from completed extracts without rerunning successful chunks
- a llm_merge failure is surfaced distinctly from llm_chunk failures
- explicit clean requeue clears chunk state
- config/default validation covers changed chunk-size/timeout semantics
Mock the LLM client and timing behavior; keep tests offline and fast.

11) Documentation and operator notes
- Update README/runbook briefly to explain that long-transcript processing now resumes at the failed chunk and may split oversized timed-out chunks automatically.
- Document the main timeout-related env vars and the new chunk debug artifacts/state.

Verification steps (must be included in PR description)
- Reproduce a forced timeout on one chunk and confirm the next attempt resumes from that chunk instead of restarting the entire recording.
- Confirm adaptive split reduces the failed chunk into smaller children and processing continues.
- Confirm successful earlier chunk outputs are reused.
- Confirm root-cause artifacts/logs distinguish timeout vs parse vs connection errors.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Durable per-chunk state persistence
- Resume from failed/incomplete chunk set
- Adaptive split-on-timeout
- Specific chunk error codes and clearer artifacts/logs
- Updated tests and docs

Success criteria
- A long recording no longer restarts from chunk 1 or from precheck because one chunk timed out.
- Successful chunk outputs are reused safely.
- Timeout handling degrades gracefully by splitting chunks when possible.
- CI remains green.
```
