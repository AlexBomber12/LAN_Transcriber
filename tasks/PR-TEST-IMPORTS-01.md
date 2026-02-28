PR-TEST-IMPORTS-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/test-imports-01
PR title: PR-TEST-IMPORTS-01 Expand import smoke tests to cover critical modules (catch breaking dependency changes early)
Base branch: main

Goal
Catch common breakages at import-time (missing attributes, renamed modules, optional deps wired incorrectly) using fast unit tests.

Scope
- Only tests.
- Must remain light: no torch/whisperx/pyannote downloads.

Implementation

A) Update tests/test_imports.py
- Keep the existing sys.path insert of repo root.
- Expand the module list to include critical modules that should be importable without heavy deps:
  - lan_app.api
  - lan_app.worker
  - lan_app.worker_tasks
  - lan_app.ui_routes
  - lan_app.db
  - lan_app.exporter
  - lan_transcriber.pipeline
  - lan_transcriber.pipeline_steps.orchestrator
  - lan_transcriber.pipeline_steps.precheck
  - lan_transcriber.pipeline_steps.summary_builder

- Ensure imports do not implicitly import whisperx at module import time.
- If any module currently imports heavy deps at import time, refactor that module so heavy deps are imported inside functions.

B) Add 1 targeted import for the whisperx helper module
- Importing the orchestrator module should be safe because it must import whisperx only inside the ASR function.

Local verification
- scripts/ci.sh

Success criteria
- The expanded import smoke test runs quickly and passes.
- If a future change introduces an import-time dependency failure, this test fails immediately.
```
