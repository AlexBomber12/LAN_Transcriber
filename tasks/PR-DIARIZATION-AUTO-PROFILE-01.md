    PR-DIARIZATION-AUTO-PROFILE-01
    ==============================

    Prompt (copy as-is into Codex)
    ------------------------------

    ```text
    You are Codex Agent working on the LAN-Transcriber repository.

Goal
Automatically choose between dialog-like and meeting-like diarization behavior instead of relying on a fixed manual profile, and persist the decision so it is transparent. The current diarization quality PR exists but still behaves poorly because the system does not reliably decide whether a recording is a 2-person dialog or a larger meeting.

Constraints
- Keep pyannote diarization as the source of speaker segmentation.
- Do not require the user to manually choose dialog vs meeting.
- Preserve current artifacts and downstream compatibility.
- Maintain 100% statement and branch coverage for all changed/new modules.

Implementation requirements

1) Introduce explicit auto-profile logic
- Support diarization profiles:
  - auto
  - dialog
  - meeting
- Keep auto as the production default.
- If explicit env hints exist (min/max speakers or profile), they may override, but auto should remain the standard path.

2) Run a first-pass diarization and classify the result
- Perform the initial diarization in a general mode suitable for meetings, e.g. 1..6 or 2..6 depending on current defaults.
- Analyze the result with deterministic heuristics such as:
  - number of unique speakers
  - total speaking time share of top 2 speakers
  - count and total duration of low-mass speakers
  - turn alternation between the 2 dominant speakers
  - overlap ratio if available
- Classify the recording as dialog-like if, for example:
  - 2 dominant speakers cover most of the speech
  - remaining speakers are tiny/noisy
  - turn-taking is consistent with a dialog
Use clear thresholds in code and make them easy to tune.

3) Retry dialog-like recordings with 2-speaker constraints
- If the first pass is classified as dialog-like, retry diarization once with:
  - min_speakers=2
  - max_speakers=2
- Do not retry infinitely.
- If the dialog retry is worse by an objective metric you define (e.g. pathological segmentation, one speaker only, implausible turn pattern), keep the original result.
- Persist which result won and why.

4) Preserve meeting behavior
- If the recording does not look dialog-like, keep the meeting-oriented result.
- Do not force all recordings into 2 speakers.

5) Persist auto-profile metadata
- Write a small artifact/metadata section describing:
  - initial speaker count
  - top-speaker coverage
  - selected profile
  - whether dialog retry was attempted
  - which result was chosen
- Surface this in derived artifacts so later PRs/UI can show it.

6) Keep compatibility with existing speaker hints
- Continue supporting explicit min/max speaker env vars if they exist.
- If explicit hints are set, document and preserve precedence rules relative to auto-profile logic.

7) Tests (100% coverage)
Add deterministic tests for:
- classification as dialog-like vs meeting-like from synthetic turn inputs
- retry path triggers exactly once for dialog-like recordings
- no retry for clear meetings
- explicit overrides bypass or modify auto behavior correctly
- metadata artifact is written with the expected fields
Keep tests offline with fake diarization outputs.

8) Documentation
- Update README/runbook briefly to explain:
  - auto profile selection
  - when dialog retry is used
  - how explicit speaker hints override auto behavior

Verification steps (must be included in PR description)
- Run a known 2-person dialog and confirm the chosen profile is dialog-like and final speaker count is closer to 2.
- Run a meeting sample and confirm the profile remains meeting-like and supports >2 speakers.
- Confirm metadata/artifacts clearly show how the decision was made.

Deliverables
- Auto-profile selection logic
- One-time dialog retry logic
- Metadata/artifact describing profile selection
- Tests with 100% statement and branch coverage for changed/new code

Success criteria
- Users no longer need to manually classify recordings as dialog or meeting.
- 2-person dialogs are detected and retried with 2-speaker constraints when appropriate.
- Meetings continue to support more than 2 speakers.
- The chosen profile is explicit and debuggable.
- CI remains green.
    ```
