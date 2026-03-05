PR-AUDIO-SANITIZE-FFMPEG-01
===========================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Fix hangs and unstable processing caused by damaged or unusual input audio files (especially MP3) by adding a mandatory audio sanitization step with ffmpeg before VAD, ASR, and diarization.
Current production symptoms include:
- torchaudio/libmpg123 warnings and errors such as:
  "MPEG_LAYER_III subtype is unknown"
  "part2_3_length ... too large for available bit count"
- recordings stuck in Processing / stage=diarize for a long time
- partial artifacts only (e.g. diarization_status.json and metrics.json exist, but transcript/speaker_turns are missing)
- inability to evaluate diarization quality because the pipeline never reaches stable decoding

The fix is to normalize incoming audio into a known-good WAV format once, then run the rest of the pipeline only on that sanitized file.

Constraints
- Micro PR focused on pre-pipeline audio sanitization only.
- Do not redesign the pipeline.
- Do not require users to convert files manually.
- Keep original uploaded/raw file untouched for audit/debug.
- Preserve 100% statement and branch coverage for changed/new code.

Implementation requirements

1) Add a dedicated sanitize-audio step using ffmpeg
- Introduce a small helper function/module in the app codebase, e.g.:
  - lan_transcriber/audio_sanitize.py
  - sanitize_audio_for_pipeline(input_path: Path, output_path: Path) -> Path
- Behavior:
  - Use ffmpeg to transcode the input file into a normalized WAV:
    - pcm_s16le
    - 16000 Hz
    - mono
  - Example command shape:
    ffmpeg -y -i <input> -ar 16000 -ac 1 -c:a pcm_s16le <output.wav>
  - Capture stderr/stdout for diagnostics.
  - If ffmpeg exits non-zero, raise a clear exception that includes the command and truncated stderr.
  - If input already appears to be a clean PCM WAV at the target format, it is acceptable either to:
    - still run ffmpeg for consistency, or
    - short-circuit and reuse the original file.
  Pick one approach and keep it deterministic.

2) Integrate sanitization before VAD/ASR/diarization
- Locate the precheck pipeline entry where raw audio is currently passed into WhisperX / Silero / diarization.
- Before any VAD/ASR/diarization work starts:
  - create a sanitized working file under the recording directory, e.g.:
    /data/recordings/<trs_id>/derived/audio_sanitized.wav
    or another stable path under derived/ or tmp/
  - call sanitize_audio_for_pipeline(raw_audio, sanitized_audio)
  - pass sanitized_audio path to all downstream steps:
    - Silero VAD
    - WhisperX ASR
    - pyannote diarization
- Ensure downstream artifacts still belong to the same recording id.
- Keep raw/audio.mp3 unchanged.

3) Observability and artifacts
- Add a small artifact or metadata record describing sanitization, e.g.:
  derived/audio_sanitize.json
  with fields like:
    {
      "input_path": "...",
      "output_path": "...",
      "ffmpeg_used": true,
      "sample_rate": 16000,
      "channels": 1,
      "codec": "pcm_s16le"
    }
- Log one INFO line at pipeline start:
  "audio sanitized to wav" with the output path (no excessive verbosity).
- On failure, log a clear warning/error and mark the recording terminally failed or quarantined according to existing project conventions.
- Do not leave the recording stuck in Processing if sanitization fails.

4) Make hangs impossible in this step
- Run ffmpeg with a bounded execution timeout (e.g. 120s or configurable by env if the project already supports timeouts).
- If timeout is hit:
  - kill the subprocess
  - raise a clear exception
  - ensure the job does not remain stuck

5) Tests (100% coverage)
- Add unit tests for the new sanitize helper:
  a) happy path: subprocess succeeds and returns output path
  b) ffmpeg non-zero exit: raises clear exception with stderr snippet
  c) timeout path: raises clear exception
- Add integration-style unit tests for pipeline wiring (mock subprocess, no real ffmpeg needed):
  a) precheck pipeline passes sanitized wav path to downstream components instead of raw mp3 path
  b) when sanitization fails, the job ends in a non-processing state and does not create partial downstream artifacts
- If you create audio_sanitize.json, assert it is written on success.
- Keep 100% statement and branch coverage for all new/modified modules.

6) Documentation
- Update README/runbook briefly:
  - uploaded audio is automatically normalized to wav before processing
  - this protects against broken/odd MP3 files
- No user action should be required.

Verification steps (include in PR description)
1) Upload a known-good MP3 and confirm:
   - derived/audio_sanitized.wav exists
   - pipeline completes normally
2) Re-run a previously problematic MP3 and confirm:
   - no libmpg123 decode spam in worker logs
   - recording no longer hangs indefinitely in Processing
3) Confirm raw/audio.mp3 is preserved.

Deliverables
- New sanitize helper/module using ffmpeg
- Pipeline wired to use sanitized wav for all downstream steps
- Optional audio_sanitize.json artifact
- Tests covering success, failure, timeout, and downstream wiring
- Small docs update

Success criteria
- Users no longer need to convert audio manually.
- Broken/odd MP3 files do not cause the pipeline to hang at diarize/transcribe.
- Downstream steps operate on a normalized wav and become deterministic.
- CI remains green with 100% statement and branch coverage.
```
