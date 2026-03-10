from __future__ import annotations

import os

import pytest

_DEFAULT_TEST_LLM_MODEL = os.environ.get("LLM_MODEL", "").strip() or "test-llm-model"
os.environ["LLM_MODEL"] = _DEFAULT_TEST_LLM_MODEL
os.environ.setdefault("LAN_LLM_MODEL", _DEFAULT_TEST_LLM_MODEL)


@pytest.fixture(autouse=True)
def _clear_asr_model_cache_between_tests():
    if os.environ.get("SMOKE_IMAGE", "").strip():
        yield
        return

    from lan_transcriber.pipeline_steps import orchestrator as pipeline

    pipeline.clear_asr_model_cache()
    if hasattr(pipeline._whisperx_transcriber_state, "transcribe_audio"):  # noqa: SLF001
        delattr(pipeline._whisperx_transcriber_state, "transcribe_audio")  # noqa: SLF001
    if hasattr(pipeline._whisperx_transcriber_state, "use_session_transcriber"):  # noqa: SLF001
        delattr(pipeline._whisperx_transcriber_state, "use_session_transcriber")  # noqa: SLF001
    yield
    pipeline.clear_asr_model_cache()
    if hasattr(pipeline._whisperx_transcriber_state, "transcribe_audio"):  # noqa: SLF001
        delattr(pipeline._whisperx_transcriber_state, "transcribe_audio")  # noqa: SLF001
    if hasattr(pipeline._whisperx_transcriber_state, "use_session_transcriber"):  # noqa: SLF001
        delattr(pipeline._whisperx_transcriber_state, "use_session_transcriber")  # noqa: SLF001
