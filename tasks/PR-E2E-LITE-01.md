    PR-E2E-LITE-01
    ==============

    Prompt (copy as-is into Codex)
    ------------------------------

    ```text
    You are Codex Agent working on the LAN-Transcriber repository.

Goal
Add a CI-friendly "E2E-lite" test that runs the processing flow on a real audio file path end-to-end, without requiring:
- internet access
- Hugging Face gated models
- GPU
- downloading heavy ASR/diarization models
The test must run quickly and deterministically in CI.

Principles
- Generate a small WAV file during the test (no binary fixture committed).
- Monkeypatch/mock heavy components (ASR load/inference, diarization load, LLM calls) so the test exercises:
  - file handling
  - pipeline orchestration glue
  - artifact writing and status transitions
  - error handling paths
- Prefer in-process orchestration (no Redis/RQ). If unavoidable, mark as @pytest.mark.e2e and keep default CI path in-process.

Implementation requirements

1) Add a new pytest test module (runs in CI by default)
- Create tests/test_e2e_lite_processing.py
- Generate a 1-2 second WAV file via the wave module into tmp_path.
- Invoke the production entrypoint used for single-file processing (pipeline/orchestrator function), not an internal helper.
- Ensure it completes successfully and writes expected artifacts into a temp output directory.

2) Mock heavy dependencies
- Monkeypatch the model entrypoints that would download/run ML:
  - whisperx.load_model (or your internal wrapper) -> return a fake model producing deterministic segments
  - diarizer builder/loader -> force _FallbackDiariser or fake diariser with deterministic annotation
  - any LLM calls -> deterministic stub
- Avoid importing heavyweight modules at import time; if necessary, move imports behind functions in production code (minimal change, fully covered).

3) Assertions
At minimum assert:
- pipeline returns a success outcome (status/return tuple) and does not quarantine
- transcript and summary artifacts exist (or whatever your export-only mode guarantees)
- diarization field exists (fallback or fake) so downstream does not crash
- artifacts are non-empty where appropriate

4) Stability and speed
- Hard timeout per test (15-30 seconds).
- No background workers left running.
- No network calls (enforce by mocking or by refusing when env lacks HF_TOKEN).

5) Coverage
- Any new production glue must be fully covered (100% statement and branch).
- The E2E-lite test itself must be deterministic and not flaky.

Deliverables
- New E2E-lite test running by default in CI.
- Minimal supporting glue for dependency injection/mocking, fully covered.

Success criteria
- `pytest -q` passes in CI without HF_TOKEN and without internet.
- The E2E-lite test exercises real file path processing and verifies artifacts are produced.
    ```
