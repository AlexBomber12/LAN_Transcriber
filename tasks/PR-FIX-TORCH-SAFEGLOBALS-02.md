PR-FIX-TORCH-SAFEGLOBALS-02
===========================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Fix the recurring worker crash when WhisperX initializes pyannote VAD on recent PyTorch versions where torch.load defaults to weights_only=True.

Current failure (from worker logs)
- _pickle.UnpicklingError: Weights only load failed
- WeightsUnpickler error: Unsupported global: omegaconf.base.ContainerMetadata
Previously observed unsupported globals also include:
- omegaconf.listconfig.ListConfig
- omegaconf.dictconfig.DictConfig

This happens while loading the pyannote VAD checkpoint inside:
whisperx.vads.pyannote -> pyannote.audio.core.model.Model.from_pretrained -> lightning_fabric.utilities.cloud_io -> torch.load

Background
A previous PR added allowlisting for OmegaConf ListConfig/DictConfig, but runtime still fails on ContainerMetadata. We need a robust, future-proof fix:
- Allowlist the full set of OmegaConf types used in these checkpoints (including ContainerMetadata)
- Use scoped allowlisting via torch.serialization.safe_globals when available
- Add a single auto-retry when an Unsupported global error is detected, so future OmegaConf-only types do not require another PR
- Keep security: do not allowlist arbitrary types; only allowlist types under the "omegaconf" Python package.

Constraints
- Micro PR, minimal change set.
- Do not set weights_only=False.
- Do not vendor-patch whisperx/pyannote/lightning.
- Must remain compatible with torch versions where torch.serialization.safe_globals / add_safe_globals may not exist.
- Tests must not import real torch or omegaconf; use sys.modules stubs.
- Maintain 100% statement and branch coverage for new/changed modules.

Implementation tasks

1) Extend / harden the torch safe-globals helper
- Locate the existing helper module created by PR-FIX-TORCH-SAFEGLOBALS-01 (likely lan_transcriber/torch_safe_globals.py).
- Change it to a scoped context manager that supports:
  - torch.serialization.safe_globals (preferred)
  - torch.serialization.add_safe_globals (fallback)
- Ensure it allowlists all of the following OmegaConf types if available:
  - omegaconf.listconfig.ListConfig
  - omegaconf.dictconfig.DictConfig
  - omegaconf.base.ContainerMetadata
- The helper must:
  - be safe to call multiple times
  - never raise (fail open: do nothing on import errors)
  - not print; optional debug logging only if a project logger already exists

Suggested API:
- @contextlib.contextmanager
  def omegaconf_safe_globals_for_torch_load(extra_fqns: list[str] | None = None): ...

Where extra_fqns are fully-qualified names (strings) that you may import+allowlist if they belong to the omegaconf namespace.

2) Add robust "Unsupported global" parsing utility (OmegaConf-only)
- Implement a tiny helper to parse the Unsupported global message and return the fully-qualified name (FQN), e.g.:
  - "omegaconf.base.ContainerMetadata"
- Only accept FQNs that start with "omegaconf.".
- Attempt to import the type by FQN and add it to the allowlist for the retry.
- This must not allowlist non-omegaconf symbols.

3) Wrap WhisperX load_model with the safe-globals context + auto-retry
- In lan_transcriber/pipeline_steps/orchestrator.py, locate _whisperx_asr where whisperx.load_model is called.
- Wrap the call with:
  - with omegaconf_safe_globals_for_torch_load(): ...
- Add one retry:
  - Call whisperx.load_model once.
  - If it raises an UnpicklingError / RuntimeError containing "Unsupported global:" and the FQN is in the omegaconf namespace:
    - re-enter the context with extra_fqns=[that_fqn]
    - call whisperx.load_model again
  - If it still fails, re-raise original exception.
- Keep the retry narrow and deterministic (only for the specific Unsupported global error).

4) Tests (100% coverage for touched modules)
Create/extend tests so the entire new logic is covered:
- Use sys.modules stubs to create:
  - fake torch module with:
    - torch.serialization.safe_globals context manager (optional)
    - torch.serialization.add_safe_globals function (optional)
  - fake omegaconf modules/classes:
    - omegaconf.listconfig.ListConfig
    - omegaconf.dictconfig.DictConfig
    - omegaconf.base.ContainerMetadata
- Test cases:
  a) safe_globals exists: context wraps and is called with expected classes
  b) safe_globals missing, add_safe_globals exists: add_safe_globals called with expected classes
  c) torch missing: no crash
  d) omegaconf missing: no crash
  e) Unsupported global parsing returns the FQN and rejects non-omegaconf FQNs
  f) Auto-retry path:
     - first call raises a fake UnpicklingError containing "Unsupported global: GLOBAL omegaconf.base.ContainerMetadata"
     - second call succeeds
     - assert that the extra_fqn allowlist was attempted
- Ensure 100% statement and branch coverage for any newly added helper functions.

5) Verification steps (must be included in PR description)
- Rebuild and restart worker:
  - docker compose build --no-cache worker
  - docker compose up -d --force-recreate --no-deps worker
- Process a short recording.
- Confirm worker no longer fails with any of:
  - Unsupported global: omegaconf.base.ContainerMetadata
  - Unsupported global: omegaconf.listconfig.ListConfig
  - Unsupported global: omegaconf.dictconfig.DictConfig

Deliverables
- Updated safe-globals helper including ContainerMetadata and supporting scoped safe_globals
- Narrow auto-retry for OmegaConf-only Unsupported global errors
- New/updated unit tests with 100% statement and branch coverage

Success criteria
- WhisperX VAD initialization no longer fails with weights_only / Unsupported global OmegaConf errors in production logs.
- The fix is secure (only allowlists omegaconf types) and does not disable weights_only.
- CI passes with 100% statement and branch coverage.
```
