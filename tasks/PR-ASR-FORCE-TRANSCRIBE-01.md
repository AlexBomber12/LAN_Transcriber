Run PLANNED PR

PR_ID: PR-ASR-FORCE-TRANSCRIBE-01
Branch: pr-asr-force-transcribe-01
Title: Force task=transcribe in WhisperX to prevent translation on multilingual recordings

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
On multilingual recordings (e.g. Dutch lesson in Russian), WhisperX without explicit task="transcribe" sometimes switches to translation mode, producing Hebrew, Spanish, English, Japanese text instead of the actual spoken Dutch/Russian. The root cause: Whisper's default behavior on short chunks with uncertain language detection is to translate rather than transcribe. Adding task="transcribe" forces Whisper to always output text in the original spoken language.

Phase 1 - Inspect
Read:
- lan_transcriber/pipeline_steps/orchestrator.py: _modern_transcribe (line ~1656), transcribe_kwargs dict (line ~1661)
- lan_transcriber/pipeline_steps/orchestrator.py: _legacy_transcribe if it exists, check if it also needs the fix
- lan_transcriber/pipeline_steps/multilingual_asr.py: any transcribe calls that pass kwargs

Phase 2 - Implement

FIX 1: Add task="transcribe" to _modern_transcribe
In orchestrator.py _modern_transcribe, add "task": "transcribe" to transcribe_kwargs:

transcribe_kwargs: dict[str, Any] = {
    "batch_size": cfg.asr_batch_size,
    "vad_filter": True,
    "task": "transcribe",
    "language": (override_lang if override_lang else None),
    **glossary_kwargs,
}

FIX 2: Check _legacy_transcribe and multilingual_asr.py
Search for all other places where model.transcribe or transcribe_fn is called. If any of them pass kwargs without task="transcribe", add it. This includes:
- _legacy_transcribe in orchestrator.py (if present)
- transcribe_fn calls in multilingual_asr.py
- Any _transcribe_chunk functions

Phase 3 - Test and verify
- Run full CI.
- Reprocess the Dutch/Russian lesson recording and verify output is clean ru/nl text without Hebrew/Spanish/English/Japanese artifacts.

Success criteria:
- All transcribe calls include task="transcribe".
- Multilingual recordings produce text only in the actually spoken languages.
- No translation artifacts (Hebrew, Spanish, Japanese, etc.) in output.
- No existing tests break.
