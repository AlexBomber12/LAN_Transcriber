PR-FIX-TORCH-SAFEGLOBALS-03
===========================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Problem
Worker still crashes while WhisperX initializes pyannote VAD under PyTorch >= 2.6 defaults (torch.load weights_only=True).
Logs show:
- _pickle.UnpicklingError: Weights only load failed
- WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.base.ContainerMetadata

PR-FIX-TORCH-SAFEGLOBALS-01 and PR-FIX-TORCH-SAFEGLOBALS-02 are marked DONE, yet production still fails. The most likely reasons are:
- torch serialization API signature mismatch: safe_globals/add_safe_globals may accept dict mapping rather than list (or vice versa), so our call silently does nothing.
- the allowlisting is not applied in the right scope/thread (the checkpoint load happens inside whisperx.load_model and may run inside a worker thread).
We must fix this in a robust, verifiable way.

Goal
Make the OmegaConf allowlisting:
- correct for all relevant OmegaConf types (ListConfig, DictConfig, ContainerMetadata)
- robust across torch API variants (list vs dict argument shapes)
- scoped to the exact load (prefer torch.serialization.safe_globals context manager)
- with a narrow one-time retry only for "Unsupported global" errors under the "omegaconf." namespace

Security constraints
- Do NOT set weights_only=False.
- Do NOT allowlist arbitrary classes; only allowlist types under the omegaconf namespace.
- Do NOT vendor-patch whisperx/pyannote/lightning.
- The fix must be deterministic and low-noise.

Implementation tasks

1) Harden safe-globals helper (list+dict compatibility)
- Locate lan_transcriber/torch_safe_globals.py (or equivalent) created by previous PRs.
- Ensure it supports both scoped and global forms:
  - Prefer torch.serialization.safe_globals
  - Fallback to torch.serialization.add_safe_globals
- When calling either API, attempt both argument shapes:
  A) list form: [ListConfig, DictConfig, ContainerMetadata]
  B) dict form:
     {
       "omegaconf.listconfig.ListConfig": ListConfig,
       "omegaconf.dictconfig.DictConfig": DictConfig,
       "omegaconf.base.ContainerMetadata": ContainerMetadata,
     }
- Try list first. If it raises TypeError, try dict. If both fail, log exactly one warning (include exception class + message) and continue (do not crash).
- The helper must be safe to call multiple times and must not print.

2) Ensure allowlisting includes ContainerMetadata
- Import and include the following when available:
  - omegaconf.listconfig.ListConfig
  - omegaconf.dictconfig.DictConfig
  - omegaconf.base.ContainerMetadata
- If any import fails, skip that symbol but continue.

3) Apply allowlisting in the correct scope
- In lan_transcriber/pipeline_steps/orchestrator.py wrap whisperx.load_model with the safe-globals context manager so that it is active during the checkpoint load.
- Ensure this wrapper executes in the same thread that runs whisperx.load_model (if you use asyncio.to_thread, the context must be inside the function passed to to_thread, not outside).

4) Narrow one-time retry for OmegaConf-only unsupported globals
- Add a small parser to extract the FQN from error messages like:
  "Unsupported global: GLOBAL omegaconf.base.ContainerMetadata"
- Only accept if the FQN starts with "omegaconf.".
- On first failure with such an error, attempt one retry that:
  - imports the type by FQN
  - allowlists it (again using the hardened list/dict API handling)
  - retries whisperx.load_model once
- Never retry for non-omegaconf FQNs. Never retry more than once.

5) Tests (100% coverage)
- Create/extend tests/test_torch_safe_globals.py.
- Tests must not import real torch/omegaconf. Use sys.modules injection for stubs.
- Cover:
  a) safe_globals(list) works
  b) safe_globals(list) TypeError, safe_globals(dict) works
  c) add_safe_globals(list) works
  d) add_safe_globals(list) TypeError, add_safe_globals(dict) works
  e) torch missing -> no crash
  f) omegaconf missing -> no crash
  g) parser extracts OmegaConf FQN and rejects non-omegaconf FQN
  h) auto-retry path succeeds on second attempt after allowlisting specific FQN
- Ensure 100% statement and branch coverage for all modified/new code.

6) Verification (include in PR description)
- docker compose build --no-cache worker
- docker compose up -d --force-recreate --no-deps worker
- Process a short recording and confirm the worker no longer fails with Unsupported global errors.

Deliverables
- Hardened safe-globals helper supporting list+dict argument variants
- Correct scoping of the context manager around whisperx.load_model
- Narrow one-time retry for OmegaConf-only unsupported globals
- Tests with 100% statement+branch coverage for changed/new modules

Success criteria
- WhisperX VAD checkpoint load succeeds under torch weights_only=True without Unsupported global errors.
- Fix is secure (OmegaConf-only allowlisting) and deterministic.
- CI is green with 100% coverage enforcement.
```
