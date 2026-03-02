PR-VAD-SILERO-02
================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Problem
After switching VAD to Silero, worker crashes during transcription with:
TypeError: 'str' object is not callable
Stack points to whisperx/asr.py transcribe where it executes:
vad_segments = self.vad_model({...})
So model.vad_model is a string (likely "silero") instead of a callable VAD object.

We inspected whisperx.asr.load_model signature in the running container:
asr.load_model(..., vad_model: Optional[whisperx.vads.Vad] = None, vad_method: Optional[str] = 'pyannote', vad_options: Optional[dict] = None, ...)

Root cause
Our integration likely passed a string into vad_model (e.g., vad_model="silero") or otherwise set model.vad_model to a string.
Correct usage is:
- pass vad_method="silero" (string), and leave vad_model as None
- optionally pass vad_options dict
WhisperX will then create a callable VAD internally.

Goal
Fix the Silero VAD integration so:
- WhisperX ASR model is loaded with vad_method="silero" (configurable)
- model.vad_model is always callable before any transcription starts
- we fail fast with a clear error if VAD is misconfigured (instead of crashing deep inside whisperx)
- add regression tests so this never returns

Constraints
- Micro PR, minimal change.
- Do not vendor-patch whisperx.
- Keep existing 100% statement and branch coverage enforcement.
- Tests must not download models or require GPU.

Implementation tasks

1) Update config and defaults (if not already)
- Ensure AppSettings has a VAD method setting:
  - env var: LAN_VAD_METHOD
  - allowed: silero, pyannote
  - default: silero
- Ensure docs and .env.example mention LAN_VAD_METHOD.

2) Use whisperx.asr.load_model and pass the correct parameters
- In lan_transcriber/pipeline_steps/orchestrator.py locate the code that creates the WhisperX model.
- Prefer importing whisperx.asr as wx_asr and calling wx_asr.load_model directly, so we rely on the real signature (not the wrapper whisperx.load_model).
- When building kwargs:
  - Always pass vad_method=<configured method> (string)
  - Never pass vad_model as a string
  - Only pass vad_model if you are passing an actual callable VAD object (not needed for silero)
  - Pass vad_options if you already have them (dict) or keep None

Example shape (adapt to existing variables):
model = wx_asr.load_model(
    whisper_arch=cfg.asr_model,
    device=device,
    device_index=0,
    compute_type=compute_type,
    language=cfg.language,
    vad_method=vad_method,
    vad_options=vad_options,
)

3) Add a fail-fast guard right after model creation
- Immediately after load_model returns, validate:
  - vad = getattr(model, "vad_model", None)
  - if not callable(vad): raise RuntimeError with a clear message containing:
    - vad_method value
    - type(vad)
This prevents deep whisperx stack crashes and makes debugging obvious.

4) Tests (must keep CI green under 100% coverage)
- Add/extend unit tests that do not import heavy ML:
  - Monkeypatch the import location used in orchestrator (wx_asr.load_model) with a stub to capture kwargs and return a fake model object.
  - Fake model should expose vad_model as a callable for the happy path.
- Test cases:
  a) Default config (no LAN_VAD_METHOD) passes vad_method="silero" and does not pass vad_model as string.
  b) LAN_VAD_METHOD=pyannote passes vad_method="pyannote".
  c) If load_model returns a model with vad_model="silero" (string), orchestrator raises RuntimeError with the clear message (fail-fast guard).
- Ensure 100% statement and branch coverage for any new/modified code.

5) Verification steps (include in PR description)
- Rebuild worker image and restart:
  - docker compose build --no-cache worker
  - docker compose up -d --force-recreate --no-deps worker
- Process a short recording.
- Confirm the previous error "TypeError: 'str' object is not callable" no longer appears.
- Confirm logs show the selected VAD method (optional single INFO line).

Deliverables
- Correct Silero integration using vad_method, not vad_model string
- Fail-fast guard for callable vad_model
- Unit tests preventing regression with 100% coverage for modified/new code

Success criteria
- Worker successfully transcribes recordings without TypeError 'str' object is not callable in whisperx/asr.py.
- model.vad_model is callable at runtime when transcription starts.
- CI passes with 100% statement and branch coverage.
```
