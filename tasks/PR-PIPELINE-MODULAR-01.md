PR-PIPELINE-MODULAR-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/pipeline-modular-01
PR title: PR-PIPELINE-MODULAR-01 Modularize pipeline and harden LLM parsing
Base branch: main

Goal:
1) Break lan_transcriber/pipeline.py into testable modules.
2) Consolidate duplicated utilities like safe_float and normalise_language_code into a single shared module.
3) Make LLM output parsing robust with schema validation and debug artifacts.
4) Improve coverage by removing pipeline.py from coverage omit, while keeping coverage threshold unchanged.

A) Module split
- Create a new package: lan_transcriber/pipeline_steps/
- Move logic from lan_transcriber/pipeline.py into modules:
  - lan_transcriber/pipeline_steps/precheck.py
  - lan_transcriber/pipeline_steps/language.py
  - lan_transcriber/pipeline_steps/speaker_turns.py
  - lan_transcriber/pipeline_steps/snippets.py
  - lan_transcriber/pipeline_steps/summary_builder.py
  - lan_transcriber/pipeline_steps/artifacts.py
- Each module must expose 1-3 public functions with clear inputs and outputs.
- Prefer dataclasses or Pydantic models for step input and output so interfaces are explicit and testable.
- Keep lan_transcriber/pipeline.py but reduce it to a thin orchestrator:
  - Under 300 lines
  - No large sets of private helpers

B) Consolidate utilities
- Add lan_transcriber/utils.py.
- Move duplicated helpers there:
  - safe_float
  - normalise_language_code
  - any other duplicates found during refactor
- Update all call sites to import from lan_transcriber/utils.py.
- Add focused unit tests for these utilities.

C) Harden LLM parsing with schema validation and debug artifacts
- In summary_builder.py introduce Pydantic models:
  - SummaryResponse
  - ActionItem
  - Question
- Parse flow:
  1) Extract candidate JSON dict from model output.
  2) Validate using SummaryResponse.model_validate.
  3) On success: return validated structure.
  4) On failure:
     - write derived artifacts:
       - derived/llm_raw.txt
       - derived/llm_extract.json
       - derived/llm_validation_error.json
     - set a flag in summary.json like parse_error=true and include a short reason string.
     - do not silently drop actions/questions.

D) Coverage and tests
- Update .coveragerc:
  - Stop omitting lan_transcriber/pipeline.py.
  - Do not omit new pipeline_steps modules.
- Add tests for pipeline_steps modules:
  - precheck: VAD edge cases, duration thresholds.
  - language: normalisation and dominant language selection.
  - speaker_turns: turns and interruptions for a small synthetic segment list.
  - summary_builder: valid and invalid JSON cases, artifacts on failure.
  - artifacts: atomic write behavior for JSON outputs.
- Keep existing integration tests passing.
- Keep coverage threshold unchanged and green.

E) No behavior regressions
- Derived artifacts and filenames remain the same on successful runs.
- Document any intentional behavior changes.

Local verification:
- scripts/ci.sh

Success criteria:
- pipeline.py is thin and readable.
- Duplicated utilities are removed.
- LLM parsing is schema-validated with debug artifacts on failure.
- Coverage includes pipeline orchestrator and new modules; scripts/ci.sh is green.
```
