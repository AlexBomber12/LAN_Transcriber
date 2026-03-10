PR-LLM-CHUNK-COMPACTION-01
===========================

Branch: pr-llm-chunk-compaction-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Reduce the cost and latency of long-transcript LLM processing by compacting the transcript representation before chunk planning and LLM submission. Today lan_transcriber/pipeline_steps/orchestrator.py builds llm_prompt_text through _speaker_turn_prompt_text(...), producing raw lines like [12.34-15.67] SPEAKER_05: text. lan_transcriber/llm_chunking.py then chunks that heavy representation directly. The result wastes prompt budget on per-line timestamps, verbose speaker labels, and unnecessary fragmentation, which contributes to long chunk times and timeouts. Create an llm-ready compact transcript format that preserves meaning, speakers, and broad chronology while being materially smaller and easier for the model to process.

Constraints
- This PR is about compacting the LLM input representation. Do not implement chunk retry/resume logic yet.
- Preserve summary quality and downstream schema contracts.
- Keep calendar title/attendee hints and multilingual behavior intact.
- Do not redesign the final non-chunked summary path unless a small shared helper makes that safer.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Introduce an explicit llm-ready transcript compaction layer
- Add a dedicated helper/module, preferably in lan_transcriber/llm_chunking.py or a nearby focused helper, that transforms speaker turns into a compact transcript optimized for LLM chunking.
- Inputs should come from structured speaker_turns, not from already-rendered raw text when possible.
- The compact representation must:
  - remove per-line timestamps from the chunk body
  - shorten speaker labels, for example SPEAKER_00 -> S1, SPEAKER_01 -> S2, or speaker aliases when they are concise
  - merge adjacent turns from the same speaker when the gap is small and no meaning is lost
  - normalize whitespace and drop empty/noise-only rows
  - preserve coarse order and speaker boundaries
- Add explicit metadata about the compacted transcript, such as total chars before/after compaction and the speaker label mapping used.

2) Keep chronology without wasting tokens
- Do not keep [start-end] on every line.
- Instead, preserve chronology through one of these bounded approaches:
  - store only coarse chunk-level time ranges outside the main transcript body
  - or keep occasional block-level anchors where useful
- The important contract is that the LLM still understands who said what and roughly in what order, but prompt size is materially reduced.

3) Use compact transcript for chunk planning
- In lan_transcriber/pipeline_steps/orchestrator.py change the long-transcript path so _run_chunked_llm_summary(...) receives compact llm-ready text or an llm-ready chunk plan input, not the old raw speaker-turn prompt text.
- Keep the short-transcript path functional. You may optionally reuse the compact representation there if it is safe and improves consistency, but do not expand scope into a large prompt rewrite.
- Write the compact transcript to a derived artifact such as llm_compact_transcript.txt and/or llm_compact_transcript.json so it is inspectable during debugging.

4) Bound effective chunk size, not only base text size
- The current plan_transcript_chunks(...) logic enforces max_chars on base chunks but overlap and prompt wrapping still inflate the actual payload.
- Add explicit accounting for effective chunk size after compaction and overlap. At minimum:
  - expose compacted base chars
  - expose effective chars after overlap
  - keep these values in llm_chunks_plan.json
- Do not yet add adaptive split on timeout here; that is next PR.

5) Improve prompt payload economy without changing schema
- Keep the strict JSON response format and current structured schema for chunk extracts and merge.
- Reduce chunk payload bloat where easy, for example by using compact speaker labels and compact transcript field names if appropriate.
- Preserve backward-compatible parse/merge behavior unless a small internal rename is clearly justified and fully updated.

6) Preserve glossary/calendar context and multilingual safety
- Calendar title and attendees must still be included in chunk and merge prompts.
- Do not drop language-awareness or confuse mixed-language transcripts. The compaction layer should be language-agnostic text handling.
- Do not destroy names/terms that glossary/corrections already stabilized.

7) Add quality-safe guardrails
- If compaction would produce an empty transcript from non-empty speaker turns, fail fast with a clear internal error instead of silently sending a broken prompt.
- Keep a mapping from compact speaker label back to original speaker/alias so debugging remains possible.
- Add a small maximum-merge-gap policy for consecutive same-speaker turns; make it deterministic and documented in tests.

8) Artifacts and observability
- Extend llm_chunks_plan.json to include enough information to understand compaction, such as:
  - source_chars
  - compact_chars
  - chunk_max_chars
  - chunk_overlap_chars
  - per-chunk base_chars/effective_chars
  - compact speaker mapping metadata
- Write a compact transcript artifact under derived/ that operators can inspect when diagnosing chunk behavior.

9) Tests with full coverage
Add deterministic offline tests for at least these cases:
- consecutive turns from the same speaker are compacted/merged correctly
- compact speaker labels are stable and reversible through the mapping metadata
- per-line timestamps are removed from compact chunk body
- empty/noise-only rows are removed safely
- compaction materially reduces size on a representative multi-speaker sample
- chunk planning reflects compacted text and records effective size metadata correctly
- the long-transcript path writes compact transcript artifacts and still produces valid prompts
- calendar title/attendee hints remain present in chunk and merge prompts
- failure paths are covered when compaction yields no usable content
Keep tests offline; do not call external LLMs.

10) Documentation and operator notes
- Update README/runbook briefly to explain that long transcripts are compacted before chunking to reduce latency and timeout risk.
- Mention the new debug artifact name(s).

Verification steps (must be included in PR description)
- Compare a representative long-recording prompt before and after compaction and show the reduction in chars.
- Confirm llm_chunks_plan.json now reports both compacted and effective chunk sizes.
- Confirm long-transcript processing still produces the same schema of chunk extracts and final summary payload.
- Run scripts/ci.sh and keep CI green.

Deliverables
- LLM-ready compact transcript representation
- Compact transcript artifacts and richer chunk plan metadata
- Long-transcript path switched to compact input
- Updated tests and docs

Success criteria
- Long transcript chunk input is materially smaller and less noisy while preserving speaker meaning.
- Chunk plan metadata reflects real effective payload size better than before.
- Summary behavior remains functionally compatible.
- CI remains green.
```
