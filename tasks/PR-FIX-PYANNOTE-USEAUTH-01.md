PR-FIX-PYANNOTE-USEAUTH-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-pyannote-useauth-01
PR title: PR-FIX-PYANNOTE-USEAUTH-01 Fix whisperx VAD crash with pyannote: ignore unsupported use_auth_token in Inference.__init__
Base branch: main

Problem
Pipeline fails during ASR model load (whisperx.load_model) with:

  TypeError: Inference.__init__() got an unexpected keyword argument 'use_auth_token'

Stacktrace shows the call originates from whisperx VAD integration:
  whisperx/vads/pyannote.py -> pyannote.audio.pipelines.voice_activity_detection.VoiceActivityDetection
  which forwards use_auth_token into pyannote.audio.Inference(...)

In our environment, pyannote.audio.Inference.__init__ does not accept use_auth_token, so the process crashes.

Goal
1) Prevent the pipeline from crashing on this incompatibility.
2) Keep behavior unchanged otherwise (we do not use use_auth_token today; whisperx passes None).
3) Implement a small compatibility shim that is safe across versions:
   - If Inference already supports use_auth_token, do nothing.
   - If not, wrap Inference.__init__ to accept use_auth_token and ignore it.
4) Add a fast unit test to lock the behavior.

Non-goals
- Do not download models or require Hugging Face auth.
- Do not change diarization pipeline in this PR.
- Do not pin whisperx/pyannote versions here (keep it a runtime compatibility fix).

Implementation

A) Add a compatibility helper module
- Create lan_transcriber/compat/pyannote_compat.py (or lan_app/compat if orchestrator lives there; keep close to orchestrator).
- Implement:

  def patch_pyannote_inference_ignore_use_auth_token() -> bool:
      - Try to import: from pyannote.audio import Inference
      - Inspect signature of Inference.__init__ using inspect.signature
      - If "use_auth_token" is already present, return False (no patch needed)
      - If already patched (sentinel attribute on the function), return False
      - Else:
          orig_init = Inference.__init__
          def patched_init(self, *args, use_auth_token=None, **kwargs):
              return orig_init(self, *args, **kwargs)
          set sentinel: patched_init._lan_ignore_use_auth_token = True
          assign: Inference.__init__ = patched_init
          return True

Notes:
- Keep the patch minimal; ignore only use_auth_token.
- The patch is safe because whisperx passes use_auth_token=None today, and we are only ignoring it.

B) Call the patch before whisperx loads VAD
- In lan_transcriber/pipeline_steps/orchestrator.py:
  - At the start of _whisperx_asr (or whichever function calls whisperx.load_model):
      from lan_transcriber.compat.pyannote_compat import patch_pyannote_inference_ignore_use_auth_token
      patch_pyannote_inference_ignore_use_auth_token()
  - Ensure this call happens before whisperx.load_model(...)

C) Optional: add a small debug log
- If you have step logs, append one line when the patch is applied.
- Do not spam logs if patch is not needed.

D) Tests
- Add tests/test_pyannote_compat.py:
  - Create a dummy class with __init__(self, *args, **kwargs) that does NOT accept use_auth_token explicitly.
  - Monkeypatch pyannote.audio.Inference to this dummy class.
  - Call patch_pyannote_inference_ignore_use_auth_token()
  - Assert it returns True.
  - Instantiate DummyInference(use_auth_token=None) and assert it does not raise.
  - Call patch again and assert it returns False (idempotent).

If importing pyannote.audio in tests is heavy:
- Use sys.modules monkeypatch to provide a lightweight fake pyannote.audio module with Inference.

Verification
- scripts/ci.sh
- Manual:
  - Process a recording that previously failed and confirm whisperx.load_model no longer crashes.

Success criteria
- No more TypeError about use_auth_token in Inference.__init__.
- Patch is applied only when needed and is idempotent.
- Unit tests are green.
- scripts/ci.sh is green.
```
