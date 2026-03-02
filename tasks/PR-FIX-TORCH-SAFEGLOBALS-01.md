PR-FIX-TORCH-SAFEGLOBALS-01
===========================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Fix the worker crash when WhisperX initializes pyannote VAD on recent PyTorch versions where torch.load defaults to weights_only=True.
Current failure:
- _pickle.UnpicklingError: Weights only load failed
- Unsupported global: omegaconf.listconfig.ListConfig (and/or omegaconf.dictconfig.DictConfig)
This happens while loading the pyannote VAD checkpoint inside whisperx.vads.pyannote -> pyannote.audio.core.model.Model.from_pretrained -> lightning_fabric.utilities.cloud_io -> torch.load.

We want a minimal, safe fix: allowlist the OmegaConf container types for the weights-only unpickler via torch.serialization.add_safe_globals, before calling whisperx.load_model.

Constraints
- Keep change minimal (micro PR).
- Do not disable weights_only globally or set weights_only=False.
- Do not modify vendor packages (whisperx, pyannote, lightning).
- Must remain compatible with older torch versions where torch.serialization.add_safe_globals may not exist.
- Must not require torch to be importable in unit tests (tests should use sys.modules stubs).
- Maintain 100% statement and branch coverage for new/changed modules.

Implementation tasks

1) Add a small helper module to apply safe globals
- Create a new module, for example: lan_transcriber/torch_safe_globals.py
- Implement a single function, for example:
  - def allowlist_omegaconf_for_weights_only() -> None
- Behavior:
  - Try to import torch.
  - Get add_safe_globals = getattr(torch.serialization, "add_safe_globals", None). If not callable, return without error.
  - Try to import:
    - from omegaconf.listconfig import ListConfig
    - from omegaconf.dictconfig import DictConfig
    If either import fails, return without error.
  - Call add_safe_globals([ListConfig, DictConfig]) (or a dict mapping names to types if required by the torch version).
  - Do not print; use logger at debug level only if a logger is available in the project. Do not spam logs.
  - The function must be idempotent and safe to call multiple times.

2) Call the helper before WhisperX model init
- In lan_transcriber/pipeline_steps/orchestrator.py locate the place where WhisperX ASR model is created:
  - whisperx.load_model(cfg.asr_model, device, compute_type=...)
- Immediately before that call, call:
  - allowlist_omegaconf_for_weights_only()
- Also ensure this code path is executed for both CPU and GPU device selections (even if CPU not used in production, tests may exercise it).

3) Add regression tests (no real torch required)
- Create a new test module, for example: tests/test_torch_safe_globals.py
- Use monkeypatch to inject fake modules into sys.modules:
  - a fake torch module with torch.serialization.add_safe_globals function that records arguments
  - fake omegaconf.listconfig.ListConfig and omegaconf.dictconfig.DictConfig classes
- Test cases (each must assert no exception and correct behavior):
  a) add_safe_globals present and both OmegaConf types present -> add_safe_globals called once with both types
  b) torch module missing -> function returns without error and does not call anything
  c) add_safe_globals missing -> function returns without error
  d) OmegaConf modules missing -> function returns without error
- Ensure 100% statement and branch coverage for lan_transcriber/torch_safe_globals.py.

4) Optional: guard against future regressions in Docker smoke
- If the repository has docker smoke tests that already import whisperx.asr, do not change them.
- If not, add a lightweight in-container check that imports torch.serialization and ensures no crash on import, but do not attempt to load checkpoints (no network, no HF token).

Verification steps
- Rebuild worker image and restart worker:
  - docker compose build --no-cache worker
  - docker compose up -d --force-recreate --no-deps worker
- Start processing a short recording. The worker must not crash with:
  - Unsupported global: omegaconf.listconfig.ListConfig
- Run unit tests:
  - scripts/ci.sh
  - ensure coverage stays at 100% statement and branch.

Deliverables
- New helper module applying torch safe globals allowlist for OmegaConf types
- Orchestrator calls helper before whisperx.load_model
- New unit tests covering all branches, 100% coverage for the new helper

Success criteria
- Worker no longer fails during WhisperX VAD initialization with WeightsUnpickler / Unsupported global OmegaConf errors.
- No vendor patching required.
- Tests pass with 100% statement and branch coverage.
```
