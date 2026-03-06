PR-LLM-CHUNKING-01
==================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make long-recording LLM processing reliable and bounded by replacing the current single huge LLM call with a chunked map-reduce pipeline.
Today long recordings can remain in stage=llm for a very long time because one oversized transcript is sent to the model. We need a deterministic 2-pass flow:
1) chunk-level extraction/summarization
2) final merge into the current structured output schema

Constraints
- Keep the existing output contract as stable as possible: topic, summary, decisions, action items, emotional summary, question typing, and any current JSON fields already used by UI/export.
- Preserve current Ollama/OpenAI-compatible client usage and the existing PR-LLM-ROBUST-01 behavior (max_tokens, empty-content handling, retry on finish_reason=length).
- Do not require internet in tests.
- Keep 100% statement and branch coverage for changed/new modules.
- Keep prompt text in English inside code if you need new prompt templates.

Implementation requirements

1) Add chunking settings
Introduce configurable settings in AppSettings with sane defaults:
- LLM_CHUNK_MAX_CHARS or LLM_CHUNK_MAX_TOKENS_ESTIMATE
- LLM_CHUNK_OVERLAP_CHARS (or token estimate overlap)
- LLM_CHUNK_TIMEOUT_SECONDS
- LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS (or token estimate) to decide when to switch from single-pass to chunked mode
- LLM_MERGE_MAX_TOKENS (optional if separate from base LLM_MAX_TOKENS)
Use simple deterministic chunking by paragraph/newline boundaries when possible. If exact tokenization is not available cheaply, use a chars-based strategy with overlap.

2) Implement chunk planning and chunk extraction
Add a small helper/module, e.g. lan_transcriber/llm_chunking.py, with functions such as:
- split_transcript_for_llm(text, max_chars, overlap_chars) -> list[str]
- build_chunk_prompt(...) -> system/user prompts for per-chunk extraction
- merge_chunk_results(...) -> merged intermediate representation
Rules:
- Do not send one giant transcript when the threshold is exceeded.
- Preserve chunk order and include chunk index/total in prompts.
- Keep chunking deterministic so the same input yields the same chunk layout.

3) Implement 2-pass map-reduce LLM flow
Update the LLM stage in the pipeline/orchestrator:
- If transcript length is below threshold, keep the existing single-pass path.
- If above threshold:
  a) split into chunks
  b) for each chunk call llm.generate with a bounded timeout and chunk-specific max_tokens
  c) parse/validate each chunk result
  d) combine chunk results into a compact merge input
  e) call final merge LLM once to produce the final schema used by the rest of the app
Important:
- Chunk outputs should be compact and structured, not full prose summaries. Example intermediate JSON can include:
  - local topic candidates
  - bullet summaries
  - decisions
  - action candidates with owner/deadline if present
  - emotional cues
  - unresolved questions
- Final merge prompt should explicitly deduplicate and reconcile chunk outputs.

4) Add bounded per-chunk timeout and graceful degradation
- Each chunk call must use a bounded timeout separate from the HTTP transport timeout if needed.
- If one chunk fails after retries:
  - do not hang forever
  - either mark the whole LLM stage failed with a clear reason, or produce a partial-result artifact and fail explicitly
Pick one behavior and keep it deterministic. Do not leave the recording in Processing indefinitely.
- Add one clear error for "chunk N/M failed" including chunk index.

5) Add explicit LLM progress reporting
Update pipeline progress/stage reporting so UI can show meaningful LLM progress for long recordings:
- examples:
  - stage=llm_chunk_1_of_5
  - stage=llm_chunk_2_of_5
  - stage=llm_merge
- Keep the existing UI compatible; progress text can be new while the overall stage family remains LLM.
- Ensure the progress timestamp updates between chunks so the UI no longer looks frozen.

6) Persist chunk artifacts for debugging
Under derived/ write safe artifacts such as:
- llm_chunks_plan.json
- llm_chunk_001_raw.json
- llm_chunk_001_extract.json
- llm_merge_input.json
- llm_merge_raw.json
Do not store unnecessary prompt text if there is any privacy concern. Store enough to debug truncation/empty content issues.

7) Keep final schema compatible
- The final output consumed by UI/export must keep the existing keys/shape.
- If some chunk data is missing, final merge must degrade gracefully rather than crash.
- Preserve current summary/extract artifact names unless the existing code already supports versioned outputs.

8) Tests (100% coverage)
Add tests for:
- chunk splitting:
  - below threshold -> one chunk
  - above threshold -> multiple chunks with deterministic overlap
- long-transcript path:
  - pipeline uses chunked mode and then merge mode
  - progress/stage updates across chunks and merge
- failure handling:
  - one chunk returns empty content / raises -> pipeline fails clearly and does not hang
- merge path:
  - mocked chunk outputs merge into a final structure accepted by downstream code
Use mocked llm.generate responses. No real Ollama. Cover all branches in new helper code and orchestration glue.

9) Documentation
Update README/runbook briefly:
- long transcripts are processed in chunks
- UI may show llm_chunk_X_of_Y and llm_merge
- LLM chunk settings are configurable

Verification steps (must be included in PR description)
- Process a short recording and confirm the pipeline still uses the single-pass path.
- Process a long transcript fixture and confirm:
  - chunk artifacts are created
  - progress updates per chunk are visible
  - final structured output is produced
- Confirm no indefinite stage=llm hangs.

Deliverables
- Chunking helper/module
- Updated orchestration for single-pass vs chunked map-reduce
- Progress updates and debugging artifacts
- Tests with 100% statement and branch coverage for changed/new code
- Small docs update

Success criteria
- Long recordings no longer stall on one giant LLM request.
- UI shows visible LLM progress during chunking and merge.
- Final output remains compatible with existing UI/export consumers.
- CI remains green.
```
