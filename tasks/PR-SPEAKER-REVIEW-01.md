PR-SPEAKER-REVIEW-01
=====================

Branch: pr-speaker-review-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Redesign the speaker-review flow so operators can make explicit, safe decisions when diarization snippets or suggested matches are uncertain. Right now the UI mostly offers 2 paths: map to a canonical speaker or leave unmatched. That is not expressive enough. The operator needs 4 clear outcomes:
- confirm match to a canonical speaker
- keep unknown on purpose
- local label only for this recording
- add trusted sample only when the snippet is genuinely good
This PR must separate identity decisions from sample-training decisions and avoid contaminating the canonical speaker bank.

Relevant code to inspect first
- lan_app/ui_routes.py
- lan_app/templates/recording_detail.html
- lan_app/templates/voices.html
- lan_app/db.py
- lan_app/migrations/
- lan_app/exporter.py
- lan_app/speaker_bank.py
- tests/test_ui_routes.py
- tests/test_speaker_bank.py
- tests/test_cov_lan_app_db.py
- tests/test_cov_lan_app_ui_routes.py

Constraints
- Build on top of the current canonical speaker backend and speaker UX.
- Keep mapping and sample addition as separate actions. Do not auto-create training samples during a remap.
- Do not remove the ability to keep a diarized speaker unmatched.
- Prefer safety over aggressive auto-labeling. It is better to preserve an unknown speaker than to store a wrong canonical identity.
- Keep exports and display labels consistent after manual review.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Introduce explicit speaker review state
Add a durable way to distinguish between these cases:
- unreviewed / system-suggested
- confirmed canonical match
- intentionally kept unknown
- local-label-only
The current schema may need a focused migration, for example on speaker_assignments, to support explicit review semantics and an optional local display label. Keep the schema small and explainable.

2) Add "Keep unknown" as a first-class operator action
The operator must be able to mark a diarized speaker as intentionally unknown even after inspecting candidate matches or bad snippets.
This action must:
- clear the pending-review ambiguity
- avoid adding a canonical speaker
- avoid adding a sample
- remain visible in the UI as a deliberate decision, not as "nothing happened"

3) Add "Local label only" for per-recording naming
Support a local display label that applies only to the current recording and its exports/detail views. This is for cases where the operator recognizes the person in this meeting but does not want to create or pollute a canonical global speaker record.
Local label must:
- display in the current recording detail view
- display in exports for this recording
- not create a canonical voice profile
- not affect future recordings automatically

4) Keep canonical match explicit
Retain the current remap flow to a canonical speaker, but make the UI wording clearer so the operator understands that this is a global identity mapping, not just a local display tweak.
Surface enough context to avoid accidental global assignment.

5) Keep sample training separate and manual
The "Add sample from this recording" flow must remain separate and must be clearly framed as speaker-bank training input.
Improve the messaging so operators understand:
- mapping a speaker is not the same as training the speaker bank
- bad snippets should not be added as samples
- the action is disabled only when there are no acceptable clean snippets

6) Update rendering logic for local labels and explicit unknowns
Any place that currently derives a speaker display name only from voice_profile_name must be updated to respect the new reviewed states and local labels. This includes at minimum:
- recording detail speakers tab
- any per-recording speaker display helpers in ui_routes.py
- export rendering in lan_app/exporter.py
Keep the output deterministic.

7) Improve operator wording in the Speakers tab
Replace ambiguous wording such as "Leave unmatched" where it hides intent. The UI should clearly distinguish:
- unknown by choice
- still needs review
- local label only
- mapped globally
Do not redesign the whole page, but make the choices self-explanatory.

8) Tests with full coverage
Add or update deterministic tests for at least these cases:
- confirm canonical match persists and displays correctly
- keep unknown persists as an intentional review state
- local label only displays in detail and export for that recording
- add sample remains separate from remap/local-label decisions
- existing speaker-bank flows still work
- old rows without the new fields degrade safely
Keep tests offline.

9) Documentation
Update README or docs/runbook.md briefly with operator guidance:
- when to keep unknown
- when to use local label only
- when to create a canonical speaker
- when to add a trusted sample

Verification steps (must be included in PR description)
- Mark a diarized speaker as Keep unknown and confirm the row no longer reads as unresolved ambiguity.
- Assign a Local label only and confirm exports for that recording use the local label without creating a canonical speaker.
- Confirm a canonical remap still works and Add sample remains a separate explicit action.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Explicit speaker review state model
- Keep unknown action
- Local label only flow
- Updated display/export helpers
- Tests with 100% statement and branch coverage

Success criteria
- Speaker review becomes explicit and safe instead of ambiguous.
- Operators can avoid polluting the canonical speaker bank.
- Local-only naming is possible when global identity is not appropriate.
- CI remains green.
```