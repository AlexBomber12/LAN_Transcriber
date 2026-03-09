PR-SNIPPETS-PURITY-01
======================

Branch: pr-snippets-purity-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make speaker snippets trustworthy voice samples instead of broad playback windows. Right now lan_transcriber/pipeline_steps/snippets.py exports 10-20 second windows around diarization segments, contaminated clips can include multiple speakers, extraction failure can produce synthetic silence, and lan_app/ui_routes.py adds a speaker sample by blindly taking snippet_files[0]. This PR must make snippets single-speaker-first, observable, and safe for speaker-bank learning.

Constraints
- Scope is snippet generation plus the minimal route/UI changes required so Add sample uses an explicitly chosen valid snippet.
- Do not redesign the canonical speaker model or the overall recording-detail shell.
- Preserve existing transcript/export compatibility and derived/snippets artifact location under /data/recordings/<recording_id>/derived.
- Maintain 100% statement and branch coverage for every changed/new project module.
- Keep the implementation deterministic and offline-testable.

Implementation requirements

1) Replace the wide-window snippet logic with clean clip extraction
- Stop centering arbitrary 10-20 second windows around diarization segments.
- Build snippet candidates from actual single-speaker material, using diarization segments and/or finalized speaker turns as the source boundary, but always compute purity against all diarization segments for other speakers.
- Add a small configurable pad around the selected clean region, for example 0.2-0.35 seconds per side.
- Add a reasonable max clip duration cap, for example 6-8 seconds, so a long monologue does not become an oversized sample.
- Reject or heavily down-rank candidates that are too short, overlap another speaker, come from degraded/fallback diarization, or cannot be extracted from the sanitized WAV.

2) Add deterministic purity scoring and a manifest artifact
- For every candidate snippet, compute and persist metadata such as:
  - source speaker label
  - source segment/turn start and end
  - exported clip start and end
  - duration_seconds
  - overlap_seconds and overlap_ratio versus other speakers
  - purity_score
  - ranking position
  - status such as accepted, rejected_overlap, rejected_short, rejected_failed_extract, rejected_degraded
  - extraction backend used such as wave or ffmpeg
- Write a new derived/snippets_manifest.json artifact for each recording.
- Export only accepted snippets. Do not fabricate placeholder files just to reach a fixed count.
- Prefer up to 3 clean snippets per speaker, but allow 0 or 1 when the meeting quality is poor.

3) Remove silence fallback completely
- Do not write synthetic silence WAV files when extraction fails.
- If extraction fails, record the failure in snippets_manifest.json and continue safely.
- Ensure downstream code tolerates the absence of snippet files for a speaker.

4) Make Add sample explicit and safe
- In lan_app/ui_routes.py, stop using snippet_files[0] implicitly.
- Add sample must receive an explicit snippet identifier or path selected from the manifest/rendered UI.
- Validate that the selected snippet belongs to the current recording and current diarized speaker, is inside the snippets root, is a real WAV file, and is marked accepted/clean in the manifest.
- If no clean snippet exists, block Add sample with a clear message instead of creating a bad sample.

5) Surface snippet quality in the existing speaker UI with minimal targeted changes
- On the recording detail speakers tab, render available snippets with:
  - audio playback
  - clip timing
  - a clear clean/recommended badge for the best snippet
  - warnings for overlap, degraded diarization, or extraction failure
- Keep the current UI shell and tabs. Do not redesign the page.
- Make it obvious which snippet is recommended for Add sample, but do not hide alternative clean snippets.

6) Preserve speaker-bank compatibility
- Existing create_voice_sample flow must still store a real snippet path under /data.
- Existing voice-profile assignment and canonical-speaker flows must keep working.
- If exactly 1 clean snippet exists, the user must still be able to save it.
- If 0 clean snippets exist, the UI must explain why instead of silently failing.

7) Tests (100% coverage)
Add deterministic offline tests for at least these cases:
- clean single-speaker candidates outrank longer contaminated candidates
- overlap against another speaker rejects or penalizes a candidate as expected
- extraction failure does not create a silence file and is recorded in the manifest
- snippets_manifest.json is written with stable metadata and ordering
- add-sample route requires an explicit snippet selection and rejects unsafe paths/unlisted snippets
- recording detail speaker UI renders clean and warning states correctly
- no-clean-snippet case is handled with a clear message
Mock ffmpeg/process execution where needed so tests stay offline and deterministic.

8) Documentation
- Update README and/or docs/runbook.md briefly to explain:
  - snippets are now purity-ranked voice samples
  - Add sample uses an explicit chosen snippet
  - silence fallback was intentionally removed
  - snippets_manifest.json is the inspection artifact for snippet quality

Verification steps (must be included in PR description)
- Process a recording that previously produced mixed snippets and confirm only clean single-speaker clips are accepted.
- On the recording detail page, add a sample from a specific recommended snippet and confirm the saved sample path matches the chosen clip.
- Confirm a speaker with no clean clip is blocked with a visible message rather than getting a silent/garbage sample.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Purity-based snippet selection and extraction
- snippets_manifest.json artifact
- Removal of silence fallback
- Explicit snippet selection for Add sample
- Minimal recording-detail UI quality indicators for snippets
- Tests with 100% statement and branch coverage for changed/new modules

Success criteria
- Speaker snippets used for the bank are no longer broad mixed-speaker windows by default.
- Add sample never silently grabs the first available snippet without quality checks.
- Extraction failure no longer creates fake silence samples.
- Users can see which snippet is clean and recommended.
- CI remains green.
```
