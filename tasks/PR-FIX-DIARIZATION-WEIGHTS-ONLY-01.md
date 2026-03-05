PR-FIX-DIARIZATION-WEIGHTS-ONLY-01
==================================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make pyannote diarization work reliably under PyTorch versions where torch.load defaults to weights_only=True (PyTorch 2.6+ behavior).
Today diarization falls back to a single-speaker dummy output because pyannote Pipeline.from_pretrained fails with errors like:
- Weights only load failed
- Unsupported global: GLOBAL torch.torch_version.TorchVersion
- Unsupported global: GLOBAL omegaconf.*
- Unsupported global: GLOBAL pyannote.audio.core.task.Specifications
This causes all recordings to end up with speaker_turns containing only speaker "S1".

We must:
1) Load the pyannote diarization pipeline successfully (pyannote/speaker-diarization-3.1) under weights_only=True by using scoped torch.serialization.safe_globals / add_safe_globals allowlisting.
2) Make the allowlisting robust: automatically allowlist new Unsupported globals ONLY if they are in trusted namespaces (pyannote.* or omegaconf.*) plus TorchVersion.
3) Keep security posture: do NOT set weights_only=False, do NOT allowlist arbitrary classes outside the trusted namespaces.
4) Improve observability: clearly log whether diarization used the real pyannote pipeline or fell back, and include the failure cause.
5) Add regression tests to prevent returning to "always S1" due to diarizer init failures.

Constraints
- Micro PR: diarization init + safe-globals + tests + logs only.
- Do not vendor-patch pyannote/lightning.
- Do not change WhisperX VAD (it already uses Silero).
- Keep 100% statement and branch coverage enforcement intact.

Implementation plan

A) Implement a dedicated safe-globals context manager for checkpoint loading
1) Create a new helper module (or extend an existing one if it already exists and is appropriate):
- Suggested: lan_transcriber/torch_safe_globals.py (if it exists) or lan_transcriber/torch_safe_globals_diarization.py
- Provide a context manager that takes a list of types and applies them to:
  - torch.serialization.safe_globals (preferred, scoped), else
  - torch.serialization.add_safe_globals (fallback)
- Robust across torch API variants:
  - Try list-of-types form first
  - If TypeError, try dict mapping of fully-qualified-name -> type
- Must never crash if torch is missing or APIs are absent; just yield without changes.
- Must not print; use debug logs only.

2) Build the base allowlist set (trusted only)
Always attempt to import and include these types (if importable):
- from torch.torch_version import TorchVersion  (FQN torch.torch_version.TorchVersion)
- from omegaconf.listconfig import ListConfig
- from omegaconf.dictconfig import DictConfig
- from omegaconf.base import ContainerMetadata
- from pyannote.audio.core.task import Specifications  (FQN pyannote.audio.core.task.Specifications)
If some imports fail, skip them safely.

B) Implement a robust diarization pipeline loader with OmegaConf/pyannote-only auto-retry
1) Locate diarization builder/loader code (the place that does Pipeline.from_pretrained for LAN_DIARIZATION_MODEL_ID).
2) Wrap Pipeline.from_pretrained in the safe-globals context manager using the base allowlist from section A.
3) Add a narrow retry loop (max 3 attempts total):
- Attempt to load pipeline.
- If it fails with an UnpicklingError / RuntimeError containing "Unsupported global:":
  - Extract the fully-qualified name (FQN) after "Unsupported global: GLOBAL ".
  - Accept ONLY if FQN starts with "pyannote." or "omegaconf." or equals "torch.torch_version.TorchVersion".
  - Import the type by FQN (use importlib). If import succeeds, add to allowlist and retry.
  - If FQN is outside trusted namespaces, do not allowlist it; re-raise.
- If it fails with other exceptions, re-raise (so we see real problems like missing token/gated access).
This eliminates the current whack-a-mole cycle for new pyannote/omegaconf classes.

4) Keep existing fallback diarizer, but make it explicit
- If diarization init fails after retries, fall back to _FallbackDiariser as today, BUT:
  - Log a clear warning: "diariser init failed, falling back", including the exception class and message.
  - Store a simple flag in artifacts (or in speaker_turns metadata) that diarization was degraded/fallback, so UI can show this.

C) Add speaker-count hints for dialogs (optional but high value)
- Add optional env vars:
  - LAN_DIARIZATION_MIN_SPEAKERS
  - LAN_DIARIZATION_MAX_SPEAKERS
- If set, pass them to pipeline call:
  pipeline({"audio": ...}, min_speakers=..., max_speakers=...)
- Keep default behavior unchanged if env vars are not set.
This helps 2-speaker dialogs once diarization loads.

D) Tests (100% coverage)
Add tests that do not require real pyannote downloads:
1) Unit test for the loader retry logic:
- Monkeypatch Pipeline.from_pretrained to raise a fake UnpicklingError containing:
  "Unsupported global: GLOBAL pyannote.audio.core.task.Specifications"
  on first call, then succeed on second.
- Ensure the loader:
  - parses FQN
  - imports the stubbed type
  - safe_globals/add_safe_globals called with the imported type
  - retries and returns success

2) Unit test that rejects unsafe namespaces:
- Error contains "Unsupported global: GLOBAL builtins.eval" (or similar).
- Ensure loader does NOT allowlist and raises.

3) Unit test that fallback is used only after retries and logs a warning:
- Make Pipeline.from_pretrained always fail with the same supported-global error but import fails.
- Ensure fallback diariser is returned AND a warning log is emitted.

4) Unit test for allowlist API variants:
- Fake torch.serialization.safe_globals that accepts dict but raises TypeError on list; assert dict fallback used.
- Fake torch.serialization.add_safe_globals similarly.

5) If you implement LAN_DIARIZATION_MIN/MAX_SPEAKERS, add a unit test verifying kwargs passed to pipeline call.

E) Verification steps (include in PR description)
1) After deploying, diarization should no longer log init failures.
2) Reprocess a 2-speaker dialog and confirm speaker_turns.json contains at least 2 unique speakers.
3) Confirm fallback is explicit only when truly necessary.

Deliverables
- Robust safe-globals + retry diarization loader
- Explicit fallback logging and optional degraded flag
- Optional min/max speakers configuration
- Unit tests covering all branches with 100% coverage for new/modified modules

Success criteria
- pyannote diarization initializes successfully under torch weights_only=True without manual allowlisting updates for each new pyannote/omegaconf class.
- speaker_turns.json no longer collapses to only "S1" for normal 2-speaker recordings.
- If diarization truly cannot load (e.g., gated/token issues), fallback is explicit and diagnosable.
- CI remains green with 100% statement and branch coverage.
```
