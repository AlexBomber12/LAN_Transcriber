PR-TEST-WHISPERX-CONTRACT-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/test-whisperx-contract-01
PR title: PR-TEST-WHISPERX-CONTRACT-01 Add contract tests for WhisperX transcribe API drift (with or without vad_filter)
Base branch: main

Context
We had repeated runtime failures from external API drift. We need contract tests that fail fast in CI without models, GPU, or network.

Goals
1) Add offline, fast tests simulating different transcribe() signatures.
2) Cover:
   - transcribe() without vad_filter
   - transcribe() with vad_filter
   - transcribe() with **kwargs
3) Ensure our code path uses call_with_supported_kwargs (PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-01).

Implementation

A) Make ASR wrapper testable (minimal refactor if needed)
- If orchestrator imports whisperx inline and is hard to call in tests:
  - Extract the "call transcribe" logic into lan_transcriber/asr/whisperx_adapter.py
  - Provide run_asr(...) -> (segments, info)
- Keep refactor minimal. Focus only on the transcribe call boundary.

B) Fake whisperx module via sys.modules
- In tests, inject a fake whisperx module implementing:
  - load_audio(path) -> "AUDIO"
  - load_model(...) -> FakeModel
- FakeModel.transcribe signatures:
  1) no vad_filter parameter
  2) has vad_filter parameter
  3) accepts **kwargs
- Each returns a dict with segments and language (no real inference).

C) Tests
- Add tests/test_whisperx_contract.py:
  1) no vad_filter: does not crash, segments returned
  2) has vad_filter: assert vad_filter True
  3) **kwargs: assert vad_filter present

Verification
- scripts/ci.sh

Success criteria
- CI fails if we reintroduce passing unsupported kwargs to transcribe().
- Tests are fast and offline.
- scripts/ci.sh is green.
```
