Run PLANNED PR

PR_ID: PR-DIARIZATION-FLICKER-01
Branch: pr-diarization-flicker-01
Title: Filter out flickering diarization speakers that appear briefly and are surrounded by the same dominant speaker

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Inspect and map
Read and confirm the current state of these files before coding:
- lan_transcriber/pipeline_steps/speaker_turns.py (_pick_speaker, _diarization_segments, build_speaker_turns)
- lan_transcriber/pipeline_steps/diarization_quality.py (existing quality checks, DiarizationProfileMetrics)
- lan_transcriber/pipeline_steps/orchestrator.py (where diarization segments are consumed and speaker_turns are built)
- lan_app/config.py (AppSettings)
- tests/test_diarization_quality.py (existing test coverage)

Phase 2 - Implement
Implement exactly these changes. Do not add anything beyond these fixes.

CHANGE 1: Add a flicker filter function
In lan_transcriber/pipeline_steps/diarization_quality.py, add a new function:

def filter_flickering_speakers(
    diar_segments: list[dict[str, Any]],
    *,
    min_total_seconds: float = 3.0,
    max_consecutive_segments: int = 2,
) -> list[dict[str, Any]]:

Logic:
- Compute total speech seconds per speaker across all diar_segments.
- Identify "flicker" speakers: speakers whose total speech is less than min_total_seconds AND who never appear in more than max_consecutive_segments consecutive diarization segments in a row.
- For each diar_segment belonging to a flicker speaker, reassign it to the nearest non-flicker speaker by time proximity (find the closest non-flicker diar_segment by start/end distance and use its speaker label).
- If all speakers are flicker speakers (edge case: very short recording), return the original segments unchanged.
- Return the cleaned list, sorted by start time.

CHANGE 2: Wire into orchestrator
In the orchestrator, call filter_flickering_speakers on the raw diarization segments BEFORE passing them to build_speaker_turns. This ensures that the flicker speaker labels never reach the word-level speaker assignment.

CHANGE 3: Add config knobs
In lan_app/config.py AppSettings (or the pipeline config dict):
- Add DIARIZATION_FLICKER_MIN_SECONDS with default 3.0.
- Add DIARIZATION_FLICKER_MAX_CONSECUTIVE with default 2.
- Wire these values to the orchestrator call.

CHANGE 4: Log flicker events
When a flicker speaker is detected and reassigned, log a warning with the speaker label, its total seconds, and how many segments were reassigned. Use the existing pipeline logging pattern.

Phase 3 - Test and verify
- Add tests in tests/test_diarization_quality.py:
  - test_flicker_speaker_reassigned: 10 segments from SPEAKER_01, 1 short segment (0.5 sec) from SPEAKER_00 in the middle. After filtering, SPEAKER_00 segment should be reassigned to SPEAKER_01.
  - test_legitimate_speaker_kept: 10 segments from SPEAKER_01, 5 segments (total 8 sec) from SPEAKER_00. SPEAKER_00 is NOT a flicker speaker and should be kept.
  - test_all_speakers_flicker_unchanged: 2 segments total, each from a different speaker, both under 3 sec. Return unchanged.
  - test_empty_segments: empty list returns empty list.
  - test_flicker_reassigned_to_nearest: SPEAKER_01 at 0-10s, SPEAKER_00 flicker at 12-12.5s, SPEAKER_02 at 15-25s. SPEAKER_00 should be reassigned to SPEAKER_01 (closer by time).
- Run full CI. All existing tests must pass.

Success criteria:
- Flicker speakers (< 3 sec total, max 2 consecutive segments) are reassigned to the nearest real speaker.
- Legitimate multi-segment speakers are never affected.
- The false SPEAKER_00 appearance seen in the Plaud vs LAN Transcriber comparison is eliminated.
- New tests cover all edge cases.
- No existing tests are broken.
