PR-FIX-LLM-MODEL-DEFAULT-01
===========================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Eliminate the recurring production failure where the worker unexpectedly uses model "llama3:8b" and Ollama returns:
{"message":"model 'llama3:8b' not found", ...}
Root cause: a hardcoded default llm_model = "llama3:8b" exists in lan_transcriber/pipeline_steps/orchestrator.py, and tests also reference it.

We will remove this hardcoded default and enforce a strict rule:
- There is no default LLM model.
- LLM_MODEL environment variable is REQUIRED.
- If missing, the app/worker fails fast with a clear, actionable error message.

Constraints
- Micro PR: focus only on LLM model configuration and tests/docs updates.
- Do not change prompt content or LLM call semantics.
- Do not require a running Ollama in CI; tests stay fully mocked/offline.
- Preserve 100% statement and branch coverage gate; add tests for new branches.

Implementation requirements

1) Remove hardcoded model default in orchestrator
- Locate the config/struct in lan_transcriber/pipeline_steps/orchestrator.py where llm_model is defined with a default:
  llm_model: str = "llama3:8b"
- Remove the default. Options (choose the cleanest for the codebase):
  A) Make it required:
     llm_model: str
  B) Allow Optional but enforce later:
     llm_model: str | None = None
     and validate before use.
- Ensure the pipeline cannot proceed to any LLM call without a non-empty model string.

2) Make AppSettings the single source of truth and require env
- Locate AppSettings (likely lan_app/settings.py or similar).
- Add/ensure a required field:
  - name: llm_model
  - env var: LLM_MODEL
  - no default (required)
- If the project supports reading from .env, keep it. But the key requirement: if LLM_MODEL is missing, settings initialization or validation must fail.
- Provide a clear error message:
  - "LLM_MODEL is required. Set it in .env (e.g., LLM_MODEL=gpt-oss:120b)."
- Ensure any previous fallbacks (e.g., llama3:8b) are removed.

3) Propagate settings.llm_model into the pipeline config
- Identify where pipeline config is built (PipelineConfig/PipelineSettings/OrchestratorConfig).
- Ensure llm_model is taken from AppSettings.llm_model and passed into orchestrator/run_pipeline config.
- Add a validation guard at the boundary (recommended even if settings enforces it):
  - if not llm_model.strip(): raise RuntimeError with the same clear message
This avoids silent misconfiguration if config objects are constructed in tests.

4) Update tests that referenced llama3:8b
- We already found these references:
  - tests/test_ui_routes.py around lines ~1166 and ~1262
  - orchestrator hardcode at ~line 61
- Replace "llama3:8b" with a test constant, e.g. "test-llm-model".
- Ensure tests set LLM_MODEL in their environment fixture before AppSettings is created.
  - Use monkeypatch.setenv("LLM_MODEL", "test-llm-model")
- Update any expected JSON in tests accordingly.

5) Add explicit tests for fail-fast behavior (100% coverage)
- Add a new unit test module, e.g. tests/test_settings_llm_model_required.py
- Test cases:
  a) With LLM_MODEL set, AppSettings initializes and exposes llm_model.
  b) Without LLM_MODEL, AppSettings initialization fails with a clear error message.
- If AppSettings is Pydantic, assert the exception includes "LLM_MODEL" and "required".
- If the code validates later (runtime guard), add a test for that guard too.

6) Update docs / env template
- Update .env.example (and README/runbook if present) to include:
  - LLM_MODEL=...
- Add a short "Required" note so this never surprises again.

7) Verification steps (include in PR description)
- Build and start stack with LLM_MODEL set; run one recording; verify the worker uses the configured model.
- Unset LLM_MODEL and confirm worker fails fast at startup with the clear message (no hidden fallback to llama3:8b).

Deliverables
- No hardcoded "llama3:8b" anywhere in runtime code.
- LLM_MODEL required in settings; fail-fast on missing/empty.
- Tests updated and new tests added; 100% coverage maintained.
- Docs updated to mark LLM_MODEL as required.

Success criteria
- Worker never sends Ollama requests with model "llama3:8b" unless explicitly configured.
- If LLM_MODEL is missing, the app/worker fails fast with an actionable error.
- CI stays green with 100% statement and branch coverage.
```
