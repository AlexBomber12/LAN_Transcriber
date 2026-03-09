PR-GLOSSARY-CORRECTIONS-01
==========================

Branch: pr-glossary-corrections-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Improve name and domain-term recognition without model fine-tuning by adding a real ASR glossary and correction-memory flow. Right now the WhisperX integration in lan_transcriber/pipeline_steps/orchestrator.py does not feed initial_prompt/hotwords/glossary context into ASR, so names like Sander can drift to Sandia. This PR must add a deterministic multi-source glossary, persist manual corrections, and feed that context into the existing ASR stack safely.

Constraints
- Do not implement ML training or model fine-tuning. This PR is glossary/correction memory only.
- Keep the current WhisperX + multilingual ASR stack. Any new helper must integrate with the existing signature-compat layer and must not regress WhisperX API-drift hardening.
- Keep prompt/context size bounded and deterministic.
- Maintain 100% statement and branch coverage for every changed/new project module.
- Keep tests offline by mocking ASR/transcribe calls.

Implementation requirements

1) Add a persistent glossary/correction data model
- Introduce a SQLite-backed glossary model that supports at least:
  - canonical term text
  - aliases/variants or observed wrong spellings
  - type/kind such as person, company, product, project, term
  - source such as manual, correction, speaker_bank, calendar, system
  - enabled/disabled state
  - timestamps/metadata needed for auditability
- The design may use one table with aliases_json or separate tables if that is cleaner, but it must support both manual glossary entries and correction memory.
- Add safe migrations and DB helpers in lan_app/db.py.

2) Build a deterministic multi-source ASR glossary helper
- Add a dedicated helper module, for example under lan_transcriber/pipeline_steps or lan_app, that composes a per-recording glossary from these sources:
  - manual glossary entries from the DB
  - correction-memory entries from the DB
  - canonical speaker names already known in the speaker bank
  - selected calendar event title/participants when available
  - optionally selected project name or project keywords if already present and cheap to include
- Normalize and deduplicate case-insensitively while preserving canonical display text.
- Sort deterministically, bound total term count and total character budget, and avoid prompt bloat.
- Emit an inspectable per-recording artifact such as derived/asr_glossary.json or derived/asr_prompt_context.json that shows which terms were used and where they came from.

3) Feed glossary context into ASR safely
- Update the WhisperX transcribe integration so it can pass glossary context through supported kwargs such as initial_prompt and/or hotwords when the underlying callable supports them.
- Reuse the existing compatibility helpers that already filter unsupported kwargs. Do not introduce a brittle direct call signature assumption.
- Apply this in both legacy and modern WhisperX paths, and ensure the multilingual chunked path also forwards the glossary context for every chunk.
- If the transcribe callable does not support any glossary-related kwargs, log the degraded behavior and continue rather than failing the recording.
- Keep language hints and glossary hints compatible. Do not accidentally force English-only behavior when the recording is multilingual.

4) Add a minimal glossary management UI
- Add a simple Glossary page or equivalent management section in the existing server-rendered UI.
- Support create, edit, enable/disable, and delete for glossary entries.
- Support entering a canonical term and one or more aliases/observed variants.
- Keep the UI simple and utilitarian. No large redesign.
- Expose enough metadata to understand where an entry came from and whether it is active.

5) Make correction memory practical
- Provide a clear path to store a manual correction as future ASR memory without building a full transcript editor in this PR.
- A valid implementation is a glossary/correction UI that lets the user store entries such as canonical=Sander, aliases=[Sandia].
- Optionally add a quick action from a recording page to prefill a glossary/correction form, but do not turn this PR into a full transcript-editing system.

6) Surface per-recording glossary context
- On the recording detail page, show a lightweight list or summary of glossary terms actually used for that recording.
- Make it possible to verify that calendar names, speaker names, and manual terms were included.
- Keep this section read-mostly; the management UI can live on a dedicated page.

7) Tests (100% coverage)
Add deterministic offline tests for at least these cases:
- glossary builder merges manual, speaker, and calendar sources deterministically
- aliases/variants deduplicate correctly and prompt size caps are enforced
- supported transcribe kwargs receive initial_prompt/hotwords/context when available
- unsupported transcribe signatures still work without breaking the recording
- multilingual chunk path forwards glossary context per chunk
- glossary CRUD routes and UI render correctly
- per-recording glossary artifact is written and detail page shows it

8) Documentation
- Update README and/or docs/runbook.md briefly to explain:
  - the app now uses a glossary/correction-memory flow instead of model training
  - glossary sources include manual terms, speakers, and calendar context
  - manual corrections are stored as canonical term plus aliases/variants
  - recordings write an inspectable ASR glossary artifact

Verification steps (must be included in PR description)
- Add a manual glossary entry such as canonical=Sander with alias=Sandia, process a recording, and confirm the per-recording glossary artifact includes it.
- Confirm ASR transcribe calls receive glossary context when supported by the underlying callable and degrade safely when not supported.
- Open the recording detail page and confirm the glossary terms used for that recording are visible.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Persistent glossary/correction-memory schema and DB helpers
- Deterministic multi-source ASR glossary builder
- Safe glossary prompt injection into WhisperX/legacy/multilingual transcribe paths
- Minimal glossary management UI
- Per-recording glossary artifact and detail-page visibility
- Tests with 100% statement and branch coverage for changed/new modules

Success criteria
- Rare names and internal terms no longer rely only on raw acoustics.
- Manual corrections influence future recordings through glossary memory without model training.
- ASR glossary context is inspectable, bounded, and deterministic.
- WhisperX signature compatibility remains intact.
- CI remains green.
```
