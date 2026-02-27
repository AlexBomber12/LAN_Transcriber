PR-FIX-WHISPERX-API-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-whisperx-api-01
PR title: PR-FIX-WHISPERX-API-01 Fix WhisperX API usage (no whisperx.transcribe) and add modern-path unit test
Base branch: main

Problem
Transcription fails with:
  module 'whisperx' has no attribute 'transcribe'
The repo pins whisperx==3.4.2, where the supported flow is:
  whisperx.load_model(...).transcribe(...)
  whisperx.load_audio(...)
Optionally align to get word timestamps.

Goals
1) Stop runtime failures by removing the hard dependency on whisperx.transcribe.
2) Keep backward compatibility for older WhisperX or tests that provide whisperx.transcribe, by using it when present.
3) Prefer modern WhisperX flow when whisperx.transcribe is missing.
4) Do not fail the recording if alignment cannot be downloaded or run. Continue with segment-level timestamps.
5) Add at least 1 unit test that exercises the modern path (no whisperx.transcribe).

Non goals
- Do not change the overall pipeline structure, DB schema, or UI.
- Do not unpin whisperx version in this PR.

Implementation

A) Extend pipeline Settings with ASR options
File: lan_transcriber/pipeline_steps/orchestrator.py
- In class Settings(BaseSettings), add fields with defaults:
  - asr_model: str = "large-v3"
  - asr_device: str = "auto"  
    Allowed values: auto, cuda, cpu
  - asr_compute_type: str | None = None
  - asr_batch_size: int = 16
  - asr_enable_align: bool = True
- These must be configurable via env due to env_prefix="LAN_".

B) Add a reusable WhisperX ASR helper
File: lan_transcriber/pipeline_steps/orchestrator.py
- Add top-level helpers near other utilities:
  1) _select_asr_device(cfg: Settings) -> str
     - If cfg.asr_device is "cpu" or "cuda", return it.
     - If cfg.asr_device is "auto":
       - Try import torch and use "cuda" when torch.cuda.is_available() else "cpu".
       - If torch import fails, use "cpu".

  2) _select_compute_type(cfg: Settings, device: str) -> str
     - If cfg.asr_compute_type is set and non-empty, use it.
     - Else use "float16" for cuda and "int8" for cpu.

  3) _whisperx_asr(audio_path: Path, *, override_lang: str | None, cfg: Settings) -> tuple[list[dict[str, Any]], dict[str, Any]]
     - Import whisperx inside the function.
     - language handling:
       - If override_lang is None, pass language=None for modern path.
       - If override_lang is set, pass that language code string.
       - Never pass language="auto" to the modern path.
     - If hasattr(whisperx, "transcribe"):
       - Keep current behavior to preserve compatibility:
         - Call whisperx.transcribe(str(audio_path), word_timestamps=True, vad_filter=True, language=override_lang or "auto")
         - Fallback without word_timestamps on TypeError
       - Return list(segments), dict(info or {})
     - Else use modern WhisperX:
       - device = _select_asr_device(cfg)
       - compute_type = _select_compute_type(cfg, device)
       - audio = whisperx.load_audio(str(audio_path))
       - model = whisperx.load_model(cfg.asr_model, device, compute_type=compute_type)
         - If TypeError for compute_type, retry without compute_type
       - result = model.transcribe(
           audio,
           batch_size=cfg.asr_batch_size,
           vad_filter=True,
           language=(override_lang if override_lang else None),
         )
       - segments = list(result.get("segments", []))
       - info = {"language": result.get("language") or (override_lang or "unknown")}
       - If cfg.asr_enable_align is True:
         - Try to run alignment to populate per-word timestamps
         - align_lang = normalise_language_code(info.get("language")) or "en"
         - model_a, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
         - aligned = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
           - If TypeError due to signature mismatch, retry with fewer args
         - segments = list(aligned.get("segments", segments))
         - If any alignment step fails, swallow the exception and keep the non-aligned segments
       - Return segments, info

C) Use the helper in run_pipeline
File: lan_transcriber/pipeline_steps/orchestrator.py
- Inside run_pipeline, remove the nested _asr() that calls whisperx.transcribe.
- Replace it with calling the new _whisperx_asr helper:
  - raw_segments, info = _whisperx_asr(audio_path, override_lang=override_lang, cfg=cfg)
- Keep the existing progress emission stages and error handling.

D) Unit tests
Add a new test file: tests/test_whisperx_api.py
- Create a minimal fake whisperx module without attribute transcribe.
- The fake must provide:
  - load_audio(path) -> "audio"
  - load_model(model_name, device, compute_type=...) -> object with transcribe(...) returning {"segments": [...], "language": "en"}
  - load_align_model(language_code, device) -> ("align_model", {"lang": language_code})
  - align(segments, model_a, metadata, audio, device, return_char_alignments=False) -> {"segments": segments_with_words}
- Use monkeypatch.setitem(sys.modules, "whisperx", fake_module)
- Call lan_transcriber.pipeline_steps.orchestrator._whisperx_asr with:
  - cfg = pipeline.Settings(asr_device="cpu", asr_enable_align=True, tmp_root=tmp_path, recordings_root=tmp_path / "recordings")
  - audio_path = tmp_path / "a.wav" (create an empty file)
- Assert:
  - returned segments is a list and not empty
  - each segment contains start, end, text
  - at least 1 segment has words list after align
  - info["language"] is "en"

E) Local verification
- scripts/ci.sh

Success criteria
- Runtime no longer crashes with "whisperx has no attribute transcribe".
- When whisperx.transcribe exists, the code still uses it and existing tests remain valid.
- When whisperx.transcribe does not exist, the code uses load_model().transcribe() and proceeds.
- Alignment failures do not fail the recording.
- New unit test for modern path passes.
- scripts/ci.sh is green.
```
