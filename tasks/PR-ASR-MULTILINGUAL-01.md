    PR-ASR-MULTILINGUAL-01
    ======================

    Prompt (copy as-is into Codex)
    ------------------------------

    ```text
    You are Codex Agent working on the LAN-Transcriber repository.

Goal
Improve ASR on mixed-language recordings by moving from whole-recording language detection to segment/chunk-level language identification with language-aware transcription.
The current behavior works on long single-language recordings but fails on bilingual recordings: the dominant language may be correct, while the secondary language is transcribed as the wrong language or even phonetically in the dominant script.

Constraints
- Keep the current ASR stack unless a very small helper is needed.
- Do not require the user to manually mark a recording as bilingual.
- Preserve existing transcript/export structures as much as possible.
- Maintain 100% statement and branch coverage for changed/new modules.

Implementation requirements

1) Add a mixed-language processing mode
- Introduce a configurable mode, e.g.:
  - auto
  - force_single_language
  - force_multilingual
- Default should be auto.
- Auto should choose mixed-language behavior when the recording shows credible language switches.

2) Perform language ID at chunk/segment level
- After sanitize/VAD segmentation, run language detection per chunk or grouped segment rather than only once for the whole recording.
- Persist language spans/artifacts, e.g.:
  - dominant language
  - per-segment/chunk language label
  - confidence score if available
- Keep chunk boundaries deterministic.

3) Run ASR with language hints per chunk
- For each chunk, call ASR with the detected language hint when confidence is sufficient.
- Group adjacent chunks with the same language to reduce overhead.
- If language confidence is weak, preserve an "uncertain" marker and avoid silently forcing a wrong language.

4) Merge chunk transcripts back into one coherent transcript
- Preserve timestamps and speaker attribution alignment as much as possible.
- The final transcript/export should remain compatible with existing UI/export consumers.
- Store language labels per segment if feasible.

5) Add review signaling for multilingual uncertainty
- If multiple chunks are uncertain or conflict strongly, mark the recording or those segments as review-worthy with an explicit reason, not a silent bad transcript.

6) Tests (100% coverage)
Add tests covering:
- single-language recording still uses the simpler path
- bilingual synthetic input yields at least 2 language spans
- ASR is invoked with per-chunk language hints
- uncertain language path sets review/degraded metadata instead of silently forcing a wrong language
Keep all tests offline by mocking language detection and ASR calls.

7) Documentation
- Update README/runbook briefly:
  - mixed-language handling now happens automatically
  - language spans may appear in artifacts
  - uncertain multilingual segments can trigger review reasons

Verification steps (must be included in PR description)
- Run a long single-language recording and confirm behavior is unchanged.
- Run a bilingual sample and confirm secondary-language chunks are no longer transcribed entirely as the dominant language.
- Confirm language span artifacts exist and are inspectable.

Deliverables
- Segment-level language identification flow
- Language-aware chunk transcription and merge
- Review signaling for uncertain multilingual segments
- Tests with 100% statement and branch coverage

Success criteria
- Bilingual recordings are no longer forced through one dominant language only.
- Secondary-language segments are transcribed with the correct language more often.
- Single-language recordings continue to work cleanly.
- CI remains green.
    ```
