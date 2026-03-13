PR-CORRECTIONS-UX-01
=====================

Branch: pr-corrections-ux-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Redesign the current Glossary UI into a simpler Corrections / ASR Memory experience that ordinary operators can understand quickly. The backend glossary and correction-memory system already exists, but the current page feels too technical and too disconnected from the daily review flow. This PR must simplify the UX, rename the operator-facing concept, and add a practical quick-entry path from a recording.

Relevant code to inspect first
- lan_app/ui_routes.py
- lan_app/templates/base.html
- lan_app/templates/glossary.html
- lan_app/templates/recording_detail.html
- lan_app/asr_glossary.py
- lan_app/db.py
- tests/test_ui_routes.py
- tests/test_asr_glossary.py
- tests/test_cov_lan_app_ui_routes.py

Constraints
- Keep the existing glossary/correction-memory backend model unless a very small route/view helper change is needed.
- Do not turn this into a full transcript editor.
- Do not remove the existing /glossary route. Backward compatibility matters.
- Operator-facing naming can change to Corrections or ASR Memory, but routes can stay stable underneath.
- Keep the UI utilitarian and consistent with the current app style.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Rename the operator-facing concept
Change the main UI wording from "Glossary" to a clearer operator-facing name such as "Corrections" or "ASR Memory".
At minimum update:
- navigation label
- page heading
- helper copy
- recording detail quick links
Do not break the /glossary route.

2) Simplify the create/edit form
Rework the current form into 2 layers:
- basic mode with the fields an operator truly needs most often
- advanced mode for the less common metadata
A good basic mode should emphasize:
- correct term
- wrong variants / common mistaken spellings
- optional note
Keep kind/source/recording-id/enabled available, but de-emphasize them behind an advanced section or otherwise lower their cognitive load.

3) Make the page self-explanatory
The page should explain in plain operator language that this feature:
- stores spelling/correction memory for future ASR prompts
- does not retrain the model
- helps future recordings reuse known terms
Avoid overly technical backend language.

4) Add a quick-entry path from a recording
From the recording detail experience, add a practical path to create a correction with context from the current recording. A valid implementation can:
- link to /glossary with prefilled query params
- or open an embedded quick form
The quick-entry path should prefill recording context when possible and reduce manual retyping.

5) Improve list readability
Make stored entries easier to scan. At minimum improve:
- operator-facing labels
- enabled/disabled wording
- display of variants
- visibility of where an entry came from
Keep the underlying data model intact.

6) Keep ASR glossary visibility in recording detail aligned
Update the recording detail overview so the quick link and explanatory text match the new Corrections / ASR Memory wording. Do not remove the inspectable per-recording glossary context.

7) Backward compatibility
Preserve existing CRUD behavior and /glossary route semantics. If you add a /corrections alias, keep /glossary working too.

8) Tests with full coverage
Add or update deterministic tests for at least these cases:
- the renamed page renders with the simpler operator-facing wording
- basic create/edit flow still works
- advanced fields still persist correctly when used
- recording-detail quick entry prefills the correction form correctly
- existing CRUD behavior remains intact
Keep tests offline.

9) Documentation
Update README or docs/runbook.md briefly to explain the operator-facing meaning of Corrections / ASR Memory and how to add a new correction from a recording.

Verification steps (must be included in PR description)
- Open the renamed page and confirm the purpose is understandable without reading code-level terminology.
- Add a correction from the recording-detail quick path and confirm the form is prefilled sensibly.
- Confirm old /glossary links still work.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Simpler Corrections / ASR Memory page
- Reduced-cognitive-load create/edit form
- Quick-entry path from recording detail
- Updated operator-facing wording in nav and overview
- Tests with 100% statement and branch coverage

Success criteria
- Operators understand the purpose of the feature within a few seconds.
- Adding a correction becomes faster and less technical.
- Existing glossary backend behavior remains intact.
- CI remains green.
```