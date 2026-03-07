PR-SPEAKER-BANK-CANONICAL-01
============================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Introduce a canonical speaker bank backend so one real person maps to one canonical speaker record, with many embeddings/samples attached to that person. This fixes the current problem where the same real voice can appear as multiple duplicate records, and multiple diarized speakers can incorrectly map to the same bank entry.

Constraints
- This PR is backend/model/assignment only. A richer UI comes later.
- Preserve current recording artifacts and transcript compatibility.
- Maintain 100% statement and branch coverage for changed/new modules.
- Keep migrations/data changes safe and explicit.

Implementation requirements

1) Introduce a canonical speaker data model
- Add or migrate schema so the system distinguishes:
  - canonical speaker (one person)
  - speaker samples / embeddings (many per person)
- A canonical speaker must have:
  - stable id
  - display name
  - metadata fields needed by current app
- A speaker sample must reference the canonical speaker and store:
  - embedding / snippet metadata
  - provenance (recording id, timestamp if applicable)

2) Migrate existing voice data safely
- If the current schema stores one row per voice/profile, provide a migration strategy that:
  - creates canonical speaker rows
  - reattaches existing embeddings/samples
  - preserves existing names where possible
- Keep migration deterministic and testable.

3) Implement one-to-one assignment per recording
- Replace any greedy independent matching with a one-to-one assignment algorithm for diarized speakers inside a single recording.
- Use a bipartite assignment approach (Hungarian / linear sum assignment or an equivalent deterministic algorithm) over similarity scores so:
  - one diarized speaker maps to at most one canonical speaker
  - one canonical speaker is not assigned to multiple diarized speakers in the same recording unless explicitly allowed by policy
- If confidence is low, leave a diarized speaker unmatched instead of forcing a bad match.

4) Implement duplicate-merge backend operations
- Add a backend operation/service to merge canonical speaker A into canonical speaker B:
  - move all samples/embeddings from A to B
  - migrate references
  - preserve auditability if the project already tracks it
- This backend operation should not require the final UI yet, but it must exist so later UI can call it.

5) Improve speaker assignment observability
- Persist assignment confidence and the list of candidate matches for a diarized speaker if feasible.
- If confidence is below threshold, mark the mapping as low-confidence and preserve that for UI/review.
- Ensure downstream artifacts can still render using unmatched speakers (e.g. keep S1/S2 if no canonical match is chosen).

6) Do not create new duplicates by default
- Adjust speaker-bank update logic so a new sample is only attached to an existing canonical speaker when confidence is sufficiently high.
- Otherwise create an unmatched or review-required state instead of silently duplicating.

7) Tests (100% coverage)
Add tests for:
- migration from existing voice/profile rows to canonical speaker + samples
- one-to-one assignment:
  - two diarized speakers competing for one canonical speaker should not both map to it
  - low-confidence cases remain unmatched
- merge operation:
  - samples are moved correctly
  - references point to the target canonical speaker
- update logic:
  - high-confidence attaches to existing canonical speaker
  - low-confidence does not silently create bad duplicates
Keep tests offline and deterministic.

8) Documentation
- Update README/runbook briefly:
  - one person = one canonical speaker
  - many samples per speaker
  - backend now supports duplicate merge operations

Verification steps (must be included in PR description)
- Seed synthetic duplicate speakers and verify they can be merged backend-side.
- Run assignment on a recording with 2 diarized speakers and confirm one-to-one matching behavior.
- Confirm low-confidence cases remain reviewable instead of being auto-merged.

Deliverables
- Canonical speaker schema/backend
- Migration for existing data
- One-to-one assignment logic
- Duplicate merge backend operation
- Tests with 100% statement and branch coverage

Success criteria
- One real person is no longer represented by multiple unrelated speaker records by default.
- One-to-one assignment prevents obviously wrong duplicate mapping within a recording.
- Duplicate speakers can be merged safely.
- CI remains green.
```
