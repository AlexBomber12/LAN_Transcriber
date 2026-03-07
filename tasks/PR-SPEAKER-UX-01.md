PR-SPEAKER-UX-01
================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Add the user-facing speaker management and remapping UX on top of the canonical speaker backend. The current pain points are:
- one real voice may appear as multiple records
- speaker mapping in a recording may be wrong
- there is no practical way to remap S1/S2/... to the correct canonical speaker
- degraded diarization/speaker assignment is not obvious enough in the UI

Constraints
- Build on the canonical speaker backend introduced earlier.
- Do not redesign the whole UI shell.
- Preserve current exports/transcripts while allowing remap corrections.
- Maintain 100% statement and branch coverage for changed/new modules.

Implementation requirements

1) Add a canonical speakers management page or section
- Show canonical speakers, sample counts, and basic metadata.
- Surface potential duplicate candidates if the backend can provide them.
- Add actions for:
  - merge speaker A into speaker B
  - inspect speaker samples/snippets if already available

2) Add per-recording speaker remap UI
- On the recording detail page, show diarized speakers (S1, S2, etc.) with:
  - current canonical mapping (if any)
  - confidence / low-confidence indicator
  - dropdown or selector to remap to a different canonical speaker
  - option to leave unmatched
- Saving remap should update downstream display/export artifacts or the rendering layer so the corrected speaker names appear consistently.

3) Make degraded/fallback state visible
- If diarization was degraded/fallback or assignment confidence is low, show a clear badge/message in the recording UI.
- Do not make the user guess why speaker results look suspicious.

4) Support duplicate merge flow in UI
- Wire the backend merge operation into a simple UI action.
- Keep it safe: confirmation required, and the target speaker must be explicit.
- After merge, the page should refresh to the canonical surviving speaker.

5) Preserve auditability / reversibility if already supported
- If the project already has an operation log or metadata notes, use it.
- If not, at least log merge/remap operations clearly in server logs and keep DB updates explicit.

6) Tests (100% coverage)
Add tests for:
- speakers page rendering
- remap form submission and persistence
- merge action wiring to backend service
- degraded/fallback badge visibility
- export/detail view reflects remapped speaker names
Keep tests offline and deterministic using existing UI test patterns.

7) Documentation
- Update README/runbook briefly:
  - canonical speakers page
  - how to remap speakers for a recording
  - how duplicate merge works

Verification steps (must be included in PR description)
- Open a recording with imperfect mapping, remap S1/S2 to the correct canonical speakers, and confirm the detail/export view reflects the correction.
- Merge duplicate speakers and confirm one canonical record remains with combined samples.
- Confirm degraded diarization is visible in the UI.

Deliverables
- Canonical speaker management UI
- Per-recording remap UI
- Duplicate merge UI
- Degraded/fallback visibility improvements
- Tests with 100% statement and branch coverage

Success criteria
- Users can correct bad speaker mappings without DB surgery.
- Duplicate speaker records can be merged through the UI.
- The app visibly distinguishes healthy vs degraded speaker results.
- CI remains green.
```
