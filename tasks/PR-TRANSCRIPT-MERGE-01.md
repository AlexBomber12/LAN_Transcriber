Run PLANNED PR

PR_ID: PR-TRANSCRIPT-MERGE-01
Branch: pr-transcript-merge-01
Title: Merge short speaker turns and add configurable merge gap to reduce transcript fragmentation

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_transcriber/pipeline_steps/speaker_turns.py (build_speaker_turns, the 1.0 sec gap threshold on line ~203)
- lan_transcriber/pipeline_steps/orchestrator.py (where build_speaker_turns is called, what params are passed)
- lan_app/config.py (AppSettings, to understand how config values are defined)
- tests/test_pipeline_steps_speaker_turns.py (existing test coverage)

Phase 2 - Implement
Implement exactly these changes. Do not add anything beyond these fixes.

CHANGE 1: Make merge gap configurable and raise default
In lan_transcriber/pipeline_steps/speaker_turns.py:
- Add a module-level constant DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC = 4.0 (was hardcoded 1.0 on line ~203).
- Add a parameter merge_gap_sec: float = DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC to build_speaker_turns.
- Replace the hardcoded 1.0 comparison with merge_gap_sec in the word-merging loop.

CHANGE 2: Post-merge pass for short turns
Add a new function merge_short_turns(turns, *, min_words=6, merge_gap_sec=DEFAULT_SPEAKER_TURN_MERGE_GAP_SEC) that runs after build_speaker_turns and:
- Iterates through turns in order.
- If a turn has fewer than min_words words (count by splitting text on whitespace) AND the previous turn has the same speaker AND the gap between them is less than merge_gap_sec, merge it into the previous turn (concatenate text, extend end timestamp).
- If a turn has fewer than min_words AND the next turn has the same speaker AND the gap is less than merge_gap_sec, merge it into the next turn (concatenate text, set start to the short turn's start).
- If a turn has fewer than min_words but neighbors are different speakers, keep it as-is (do not discard content).
- Return the merged list.

CHANGE 3: Wire into orchestrator
In the orchestrator where build_speaker_turns result is used:
- Call merge_short_turns on the result of build_speaker_turns before saving to speaker_turns.json.
- Pass through any config overrides if present.

CHANGE 4: Add config knobs
In lan_app/config.py AppSettings (or the pipeline config dict if that is how it works):
- Add SPEAKER_TURN_MERGE_GAP_SEC with default 4.0.
- Add SPEAKER_TURN_MIN_WORDS with default 6.
- Wire these values to the orchestrator call.

Phase 3 - Test and verify
- Add tests in tests/test_pipeline_steps_speaker_turns.py:
  - test_merge_gap_default: 2 words from same speaker with 2.5 sec gap should merge into 1 turn (was split with old 1.0 threshold).
  - test_merge_gap_exceeded: 2 words from same speaker with 6 sec gap should remain 2 turns.
  - test_short_turn_merged_into_previous: a 3-word turn after a 20-word turn from the same speaker with 1 sec gap merges into the 20-word turn.
  - test_short_turn_different_speakers_kept: a 2-word turn between turns of different speakers is kept.
  - test_merge_short_turns_empty: empty list returns empty list.
- Run full CI. All existing tests must pass.

Success criteria:
- The 1.0 sec hardcoded gap is replaced with a configurable default of 4.0 sec.
- Short turns (< 6 words) from the same speaker are merged with their neighbor.
- No existing tests are broken.
- New tests cover the merge gap and short-turn merging logic.
- The transcript of a single-speaker recording has significantly fewer segments compared to the old behavior.
