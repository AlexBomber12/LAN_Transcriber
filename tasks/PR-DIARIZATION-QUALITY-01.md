PR-DIARIZATION-QUALITY-01
=========================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Improve speaker quality after diarization is already loading successfully by adding:
1) explicit speaker-count hints
2) dialog-friendly retry strategy
3) turn smoothing/post-processing
Today speaker assignment is "a bit wrong" even when the pipeline no longer falls back to a single dummy speaker. We want noticeably better results on 2-person dialogs while staying compatible with larger meetings.

Constraints
- Do not redesign diarization from scratch.
- Keep pyannote diarization as the source of speaker segmentation.
- Preserve current artifacts and downstream compatibility.
- Keep 100% statement and branch coverage for changed/new modules.
- Any heuristics must be deterministic and configurable.

Implementation requirements

1) Add configurable speaker hints
Add settings such as:
- LAN_DIARIZATION_MIN_SPEAKERS
- LAN_DIARIZATION_MAX_SPEAKERS
- LAN_DIARIZATION_PROFILE with allowed values:
  - auto
  - dialog
  - meeting
Behavior:
- auto: no hard hints unless explicitly configured
- dialog: default min_speakers=2 and max_speakers=2
- meeting: default min_speakers=2 and max_speakers=6 (or 4 if the codebase already prefers smaller meetings)
If explicit env vars are set, they override the profile defaults.

2) Pass hints into pyannote pipeline calls
Where the actual diarization pipeline is invoked, pass min_speakers/max_speakers when available:
- pipeline({"audio": ...}, min_speakers=..., max_speakers=...)
Do not pass None values unnecessarily.
Ensure this path is covered in tests with a fake pipeline.

3) Add dialog-mode retry when output looks implausible
For dialog profile (or when max speakers is 2):
- If the first diarization result returns only 1 unique speaker but the recording has enough speech turns / duration to plausibly contain a dialog, retry once with:
  - min_speakers=2
  - max_speakers=2
Keep this retry narrow and deterministic. Do not create infinite loops.

4) Add turn smoothing/post-processing
Create a small deterministic post-processing step for diarization turns, e.g. in a dedicated helper module:
- Merge adjacent turns by the same speaker when the gap is below a threshold (configurable, e.g. 0.3 to 0.7 seconds).
- Suppress micro-turns that are likely noise:
  - if a very short turn (e.g. < 0.5s) is between two longer turns of the same neighboring speaker, absorb it into neighbors instead of flipping the speaker label.
- Optionally merge tiny islands of a different speaker if they are below a low threshold and surrounded by the same speaker.
Keep this conservative; do not over-smooth real interruptions.
Expose thresholds as settings if appropriate:
- LAN_DIARIZATION_MERGE_GAP_SECONDS
- LAN_DIARIZATION_MIN_TURN_SECONDS

5) Preserve and enrich artifacts
- Continue writing speaker_turns.json in the existing shape.
- Add a small metadata section or separate file describing:
  - diarization_profile used
  - hints applied
  - whether dialog retry was used
  - number of speakers before/after smoothing
This is important for debugging and UI transparency.

6) Do not hide degraded cases
- If diarization is degraded/fallback, preserve that signal.
- Only run smoothing on real diarization outputs, not on the fallback dummy output.

7) Tests (100% coverage)
Add tests covering:
- hint selection:
  - profile=dialog -> 2..2
  - profile=meeting -> 2..6 (or chosen default)
  - explicit env overrides profile defaults
- pipeline invocation:
  - fake pipeline receives min_speakers/max_speakers
- dialog retry:
  - first result has 1 speaker, second result has 2 speakers -> retry path is used once
  - first result already has 2 speakers -> no retry
- turn smoothing:
  - adjacent same-speaker segments merge when gap small
  - micro-turn between same-speaker neighbors is absorbed
  - larger genuine turn changes are preserved
Keep all tests offline with synthetic turn lists and fake diarization objects.

8) Documentation
Update README/runbook briefly:
- describe dialog vs meeting diarization profile
- mention speaker hints and smoothing
- mention that 2-person dialogs can be forced via LAN_DIARIZATION_PROFILE=dialog

Verification steps (include in PR description)
- Run a known 2-person dialog and confirm speaker_turns.json contains 2 speakers more reliably.
- Run a meeting sample and confirm the code still supports >2 speakers.
- Verify metadata/artifact clearly shows which hints and smoothing were applied.

Deliverables
- New settings for diarization hints/profile
- Pipeline invocation updated to pass hints
- Dialog retry logic
- Turn smoothing helper/module
- Tests with 100% statement and branch coverage for changed/new code
- Small docs update

Success criteria
- 2-person dialogs are no longer frequently collapsed or jittery in speaker labeling.
- Meetings still work without forcing everything into 2 speakers.
- Speaker-turn output is smoother and more human-plausible.
- CI remains green.
```
