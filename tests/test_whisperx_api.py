from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from lan_transcriber.pipeline_steps import orchestrator as pipeline


def test_whisperx_asr_modern_path(tmp_path: Path, monkeypatch):
    fake_whisperx = ModuleType("whisperx")

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

    def _load_align_model(language_code: str, device: str) -> tuple[str, dict[str, str]]:
        assert language_code == "en"
        assert device == "cpu"
        return "align_model", {"lang": language_code}

    def _align(
        segments: list[dict[str, object]],
        model_a: str,
        metadata: dict[str, str],
        audio: str,
        device: str,
        return_char_alignments: bool = False,
    ) -> dict[str, object]:
        assert model_a == "align_model"
        assert metadata == {"lang": "en"}
        assert audio == "audio"
        assert device == "cpu"
        assert return_char_alignments is False
        enriched = []
        for segment in segments:
            row = dict(segment)
            row["words"] = [{"word": "hello", "start": 0.0, "end": 1.0}]
            enriched.append(row)
        return {"segments": enriched}

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    fake_whisperx.load_align_model = _load_align_model
    fake_whisperx.align = _align
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=True,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    segments, info = pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)

    assert isinstance(segments, list)
    assert segments
    assert all("start" in segment and "end" in segment and "text" in segment for segment in segments)
    assert any(segment.get("words") for segment in segments)
    assert info["language"] == "en"
