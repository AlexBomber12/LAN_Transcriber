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


def test_whisperx_asr_modern_path_drops_unsupported_kwargs(tmp_path: Path, monkeypatch):
    fake_whisperx = ModuleType("whisperx")

    class _FakeModel:
        def transcribe(
            self,
            audio: str,
            *,
            batch_size: int,
            language: str | None,
        ) -> dict[str, object]:
            assert audio == "audio"
            assert batch_size == 16
            assert language == "es"
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}], "language": "es"}

    def _load_audio(path: str) -> str:
        assert path.endswith("a.wav")
        return "audio"

    # Intentionally omit compute_type to exercise the fallback load_model branch.
    def _load_model(model_name: str, device: str) -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        return _FakeModel()

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    step_log: list[str] = []

    segments, info = pipeline._whisperx_asr(
        audio_path,
        override_lang="es",
        cfg=cfg,
        step_log_callback=step_log.append,
    )

    assert segments and segments[0]["text"] == "hola"
    assert info["language"] == "es"
    assert any("dropped unsupported kwargs: vad_filter" in line for line in step_log)


def test_whisperx_asr_retry_without_word_timestamps_on_typeerror(tmp_path: Path, monkeypatch):
    fake_whisperx = ModuleType("whisperx")
    calls: list[dict[str, object]] = []

    def _transcribe(audio_path: str, *args: object, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        del args
        calls.append(dict(kwargs))
        if "word_timestamps" in kwargs:
            raise TypeError("unexpected keyword argument 'word_timestamps'")
        assert audio_path.endswith("a.wav")
        assert kwargs == {"vad_filter": True, "language": "auto"}
        return ([{"start": 0.0, "end": 1.0, "text": "hello"}], {"language": "en"})

    fake_whisperx.transcribe = _transcribe
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    step_log: list[str] = []

    segments, info = pipeline._whisperx_asr(
        audio_path,
        override_lang=None,
        cfg=cfg,
        step_log_callback=step_log.append,
    )

    assert segments and segments[0]["text"] == "hello"
    assert info["language"] == "en"
    assert len(calls) == 2
    assert "word_timestamps" in calls[0]
    assert "word_timestamps" not in calls[1]


def test_log_dropped_kwargs_returns_early_when_nothing_dropped() -> None:
    messages: list[str] = []

    pipeline._log_dropped_kwargs(
        callback=messages.append,
        scope="whisperx transcribe",
        attempted={"vad_filter": True},
        filtered={"vad_filter": True},
    )

    assert messages == []


def test_log_dropped_kwargs_ignores_callback_errors() -> None:
    def _raise(_message: str) -> None:
        raise RuntimeError("boom")

    pipeline._log_dropped_kwargs(
        callback=_raise,
        scope="whisperx transcribe",
        attempted={"vad_filter": True},
        filtered={},
    )
