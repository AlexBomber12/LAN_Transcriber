from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from lan_transcriber.pipeline_steps import orchestrator as pipeline


def _test_cfg(tmp_path: Path) -> pipeline.Settings:
    return pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )


def test_whisperx_none_transcribe_falls_back_to_modern_path(tmp_path: Path, monkeypatch) -> None:
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.transcribe = None

    class _FakeModel:
        def transcribe(
            self,
            audio: str,
            *,
            batch_size: int,
            vad_filter: bool,
            language: str | None,
        ) -> dict[str, object]:
            assert audio == "audio"
            assert batch_size == 16
            assert vad_filter is True
            assert language is None
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

    def _load_audio(path: str) -> str:
        assert path.endswith("a.wav")
        return "audio"

    def _load_model(model_name: str, device: str, compute_type: str = "int8") -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        return _FakeModel()

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")

    segments, info = pipeline._whisperx_asr(audio_path, override_lang=None, cfg=_test_cfg(tmp_path))

    assert segments and segments[0]["text"] == "hello"
    assert info["language"] == "en"


def test_whisperx_non_callable_transcribe_raises_clear_typeerror(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.transcribe = object()
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")

    with pytest.raises(TypeError, match="whisperx\\.transcribe must be callable or None"):
        pipeline._whisperx_asr(audio_path, override_lang=None, cfg=_test_cfg(tmp_path))
