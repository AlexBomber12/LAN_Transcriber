PR-VAD-SILERO-01
================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Switch WhisperX voice activity detection (VAD) to Silero by default, so the pipeline no longer depends on Pyannote VAD checkpoints loaded via torch.load.
This eliminates the recurring class of runtime failures caused by PyTorch weights_only unpickler restrictions (Unsupported global: omegaconf.*) triggered during WhisperX Pyannote VAD initialization.

Scope
This PR only changes the VAD used for segmentation inside WhisperX ASR initialization. Speaker diarization may still use pyannote (separately) as currently implemented.
Keep the change minimal and deterministic.

Non-goals
- Do not remove pyannote diarization pipeline support.
- Do not vendor-patch whisperx/pyannote/lightning.
- Do not disable torch weights_only or set weights_only=False.
- Do not introduce internet-only behavior; must work offline once models are cached.

Implementation requirements

1) Add a first-class config for VAD method (default: silero)
- Add a setting, e.g. in AppSettings:
  - env var: LAN_VAD_METHOD
  - allowed values: silero, pyannote
  - default: silero
- Document it in README/runbook and .env.example.

2) Use Silero VAD explicitly when loading WhisperX model
- Locate lan_transcriber/pipeline_steps/orchestrator.py where WhisperX ASR model is created (whisperx.load_model).
- Inspect the installed whisperx version and determine the correct argument name(s) to select VAD.
  - If whisperx.load_model supports vad_method="silero", use that.
  - If it uses vad_model or vad_options, set it accordingly so Pyannote VAD is not constructed.
- Ensure we do not pass any pyannote-only token parameters when using Silero.
- Ensure this change applies to all code paths that initialize WhisperX.

3) Add a small runtime log hint (low noise)
- When building ASR model, log one INFO line:
  - "ASR VAD method: silero" (or pyannote)
This helps confirm behavior in production logs without being chatty.

4) Tests (must keep CI green under 100% coverage enforcement)
- Add unit tests that verify:
  a) Default VAD method is silero when LAN_VAD_METHOD is unset.
  b) orchestrator calls whisperx.load_model with the correct argument(s) selecting silero.
     - Use monkeypatch to replace whisperx.load_model with a stub and capture kwargs.
  c) When LAN_VAD_METHOD=pyannote, the code passes the argument selecting pyannote (backward compatible), but do not actually download models.
- Ensure tests do not import heavy ML stacks unnecessarily:
  - If orchestrator imports whisperx at module import time, patch at the right import location.
- Achieve 100% statement and branch coverage for any new setting parsing and the new VAD selection branch.

5) Optional hardening: fail-fast if whisperx API changes
- Add a small contract-style test that introspects whisperx.load_model signature in the test environment and asserts we are using a supported parameter name.
- If signature differs, the test should fail with a clear message, prompting a code update.

6) Verification steps (include in PR description)
- Build worker image without cache and restart:
  - docker compose build --no-cache worker
  - docker compose up -d --force-recreate --no-deps worker
- Process a short recording.
- Confirm worker logs show "ASR VAD method: silero".
- Confirm the previous Pyannote VAD torch.load weights_only errors no longer appear.

Deliverables
- Configurable VAD method with default silero
- Orchestrator uses silero explicitly in whisperx.load_model
- Updated docs and env example
- Unit tests and any signature contract test required for stability

Success criteria
- Processing no longer fails due to "Weights only load failed" and "Unsupported global: omegaconf.*" originating from WhisperX Pyannote VAD.
- CI passes with 100% statement and branch coverage.
- Users can still opt back into Pyannote VAD via LAN_VAD_METHOD=pyannote if desired.
```
