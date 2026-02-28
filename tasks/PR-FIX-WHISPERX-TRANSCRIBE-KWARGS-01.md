PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-whisperx-transcribe-kwargs-01
PR title: PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-01 Make WhisperX transcribe calls signature-compatible (avoid unexpected kwargs like vad_filter)
Base branch: main

Problem
Processing fails with:
  FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'vad_filter'

This is API drift: our code passes kwargs that may or may not be accepted by the installed whisperx or faster-whisper wrapper.

Goals
1) Fix the immediate crash by never passing unsupported kwargs into transcribe().
2) Make the code robust across whisperx or faster-whisper versions by filtering kwargs by callable signature.
3) Keep behavior close:
   - If vad_filter is supported, pass it.
   - If not supported, drop it silently (no crash).
4) Add small, fast unit tests for the kwarg-filter helper.

Non-goals
- Do not download models.
- Do not change diarization logic.
- Do not add new env vars in this PR (unless already present).

Implementation

A) Add a generic helper for signature-based kwarg filtering
- Create: lan_transcriber/compat/call_compat.py
- Implement:
  filter_kwargs_for_callable(fn, kwargs) -> dict
    - inspect.signature(fn)
    - if **kwargs present, return kwargs unchanged
    - else keep only keys present in signature.parameters
  call_with_supported_kwargs(fn, *args, **kwargs)
    - filtered = filter_kwargs_for_callable(fn, kwargs)
    - return fn(*args, **filtered)

B) Use helper for WhisperX transcribe calls
- Locate ASR path (typically lan_transcriber/pipeline_steps/orchestrator.py, function _whisperx_asr or similar).
- Replace direct calls to:
  - model.transcribe(...)
  - whisperx.transcribe(...)
  with call_with_supported_kwargs(...).

- Ensure the helper wraps the exact callable you call:
  call_with_supported_kwargs(model.transcribe, audio, ..., vad_filter=True, ...)

C) Optional: debug line when dropping kwargs
- If you have a per-recording step log, append once:
  "whisperx transcribe: dropped unsupported kwargs: vad_filter"
- Do not log if nothing was dropped.

Tests

D) Add unit tests for the helper (fast, no external libs)
- Add tests/test_call_compat.py with:
  1) function without **kwargs drops vad_filter
  2) function with **kwargs keeps vad_filter
  3) call_with_supported_kwargs calls with filtered kwargs

Verification
- scripts/ci.sh
- Manual: run processing and confirm no 'vad_filter' TypeError.

Success criteria
- The 'vad_filter' unexpected kwarg crash is eliminated.
- Helper is covered by unit tests.
- scripts/ci.sh is green.
```
