    PR-UI-PIPELINE-UX-01
    ====================

    Prompt (copy as-is into Codex)
    ------------------------------

    ```text
    You are Codex Agent working on the LAN-Transcriber repository.

Goal
Polish the pipeline UX and operational transparency so the app clearly explains what happened, shows correct timings, and behaves like a finished product. This PR must fix the current issues:
- recordings can end in Needs Review without showing why
- duration may show as unknown even though sanitized audio exists
- progress jumps feel arbitrary, e.g. diarization immediately showing 50%
- displayed timestamps are off by one hour for the Europe/Rome user
- the recording detail page requires a manual refresh before Export buttons and final artifacts appear
- deleting a recording from Actions -> Delete does not fully remove raw/derived artifacts from disk

Constraints
- This PR is UI/ops polish only. Do not redesign ASR, diarization, or LLM behavior.
- Preserve current data model unless a small schema extension is needed for review reasons or progress metadata.
- Maintain 100% statement and branch coverage for all changed/new project modules.
- Keep behavior deterministic and observable.

Implementation requirements

1) Make Needs Review explicit
- Introduce explicit review reason fields in persistent state, e.g.:
  - review_reason_code
  - review_reason_text
- Populate them whenever a terminal status is Needs Review, examples:
  - llm_truncated
  - llm_empty_content
  - diarization_degraded
  - multilingual_uncertain
  - speaker_assignment_low_confidence
- If there is no reason, do not leave the user with an unexplained Needs Review.
- On the recording detail page and list page, show the human-readable reason text next to the status badge.

2) Fix duration source of truth
- Make duration derive from the normalized/sanitized audio if it exists, otherwise fall back to the raw file.
- Prefer ffprobe on derived/audio_sanitized.wav after sanitize.
- Persist duration once computed so the UI does not repeatedly re-guess it.
- Ensure the detail page never shows Duration: unknown when a sanitized wav exists and ffprobe succeeds.

3) Replace the coarse progress model with a realistic staged model
- Revisit the progress mapping so the bar reflects actual work more closely.
- Suggested phase weights:
  - sanitize: 5
  - ASR/VAD: 30
  - diarization: 25
  - metrics/postprocess: 10
  - llm chunking: 20
  - llm merge: 10
- The exact values can differ, but eliminate large misleading jumps.
- Preserve current stage names but support finer-grained values.
- Ensure the progress timestamp updates between meaningful substeps.

4) Fix timezone rendering
- Keep persisted timestamps in UTC.
- Render them in the UI using the browser local timezone, or explicitly convert to Europe/Rome if the current UI is server-rendered only.
- Ensure the displayed time matches the user's local time and is no longer shifted by one hour.

5) Auto-refresh terminal state UI
- When a recording transitions to DONE, FAILED, or NEEDS_REVIEW, automatically refresh the relevant UI fragment so:
  - Export buttons appear without manual browser refresh
  - final summary/status/review reason becomes visible
- Use the existing HTMX/polling model if present. Do not add a heavy frontend framework.

6) Delete must remove disk artifacts
- In the delete action, remove:
  - DB rows associated with the recording
  - /data/recordings/<trs_id>/raw
  - /data/recordings/<trs_id>/derived
  - /data/recordings/<trs_id>/logs
  - any temp files owned by that recording
- Keep delete safe:
  - only delete inside the recording root for that trs_id
  - do not allow path traversal
- If disk cleanup partially fails, surface a clear error rather than pretending delete succeeded.

7) Tests (100% coverage)
Add/extend tests covering:
- Needs Review shows explicit reason text
- duration is computed from sanitized wav metadata
- progress mapping produces expected values/stages
- timezone rendering is correct for Europe/Rome/local browser conversion path
- terminal-state auto-refresh exposes Export without manual reload
- delete removes artifacts from a temp recording directory and DB state
Keep tests offline and deterministic.

8) Documentation
- Update README/runbook briefly to explain:
  - review reasons
  - duration source
  - delete removes disk artifacts
  - timestamps are shown in local timezone

Verification steps (must be included in PR description)
- Process a recording end-to-end and confirm:
  - duration is present
  - progress feels monotonic and sensible
  - Export appears without manual refresh
  - any Needs Review status includes a visible reason
- Delete the recording and confirm its data directory is gone from disk.

Deliverables
- Updated pipeline/UI status handling with explicit review reasons
- Correct duration handling from sanitized audio
- Improved progress model
- Timezone fix
- Auto-refresh of terminal states
- Safe full delete of recording artifacts
- Tests with 100% statement and branch coverage for changed/new code

Success criteria
- No unexplained Needs Review states remain.
- Duration is no longer unknown when sanitized audio exists.
- Progress bar reflects actual work better.
- Displayed times match Europe/Rome/local time.
- Export appears automatically after completion.
- Delete removes both DB state and disk artifacts.
- CI remains green.
    ```
