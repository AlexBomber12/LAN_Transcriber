PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-02

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-whisperx-transcribe-kwargs-02
PR title: PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-02 Robustly drop unsupported transcribe kwargs even when inspect.signature is unavailable (fix vad_filter crash)
Base branch: main

Problem
Runtime fails with:
  FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'vad_filter'

This persists when inspect.signature(fn) is not inspectable, so signature-based filtering cannot remove kwargs.

Goals
1) Never crash due to unsupported kwargs (vad_filter now, future kwargs later).
2) Preserve behavior: pass kwargs when supported, drop only offending kwargs when TypeError indicates an unexpected kwarg.
3) Add unit tests that reproduce this failure mode.

Implementation
- In lan_transcriber/compat/call_compat.py, enhance call_with_supported_kwargs:
  - attempt signature filtering first
  - call inside retry loop catching TypeError
  - if message matches "unexpected keyword argument 'X'", drop X and retry
  - cap retries
  - re-raise non-matching TypeError

- Add _extract_unexpected_kwarg_name(msg) helper using regex:
  r"unexpected keyword argument '([^']+)'"

Tests
- Add tests where inspect.signature is monkeypatched to raise and the callable rejects vad_filter.
- Assert retry succeeds after dropping vad_filter.
- Assert non-matching TypeError re-raises.

Success criteria
- No more vad_filter kwarg crash.
- Unit tests cover retry-and-drop behavior.
- scripts/ci.sh is green.
```
