from __future__ import annotations

import contextlib
import pickle
import sys
from pathlib import Path
from types import ModuleType

import pytest

from lan_transcriber.pipeline_steps import orchestrator as pipeline


def test_whisperx_asr_modern_path(tmp_path: Path, monkeypatch, caplog):
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

    def _load_model(
        model_name: str,
        device: str,
        compute_type: str = "int8",
        vad_method: str = "silero",
    ) -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        assert vad_method == "silero"
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

    with caplog.at_level("INFO"):
        segments, info = pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)

    assert isinstance(segments, list)
    assert segments
    assert all("start" in segment and "end" in segment and "text" in segment for segment in segments)
    assert any(segment.get("words") for segment in segments)
    assert info["language"] == "en"
    assert "ASR VAD method: silero" in caplog.text


def test_whisperx_asr_modern_path_uses_vad_model_fallback(tmp_path: Path, monkeypatch):
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

    def _load_model(
        model_name: str,
        device: str,
        compute_type: str = "int8",
        vad_model: str = "silero",
    ) -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        assert vad_model == "pyannote"
        return _FakeModel()

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(pipeline, "patch_pyannote_inference_ignore_use_auth_token", lambda: True)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        vad_method="pyannote",
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
    assert any("pyannote compat" in line for line in step_log)


def test_whisperx_asr_modern_path_uses_vad_options_fallback(tmp_path: Path, monkeypatch):
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

    def _load_model(
        model_name: str,
        device: str,
        compute_type: str = "int8",
        vad_options: dict[str, object] | None = None,
    ) -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        assert vad_options == {"vad_method": "silero"}
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

    segments, info = pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)

    assert segments and segments[0]["text"] == "hello"
    assert info["language"] == "en"


def test_whisperx_asr_modern_path_uses_vad_fallback_keys_with_var_kwargs(tmp_path: Path, monkeypatch):
    fake_whisperx = ModuleType("whisperx")
    seen_kwargs: dict[str, object] = {}

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

    def _load_model(
        model_name: str,
        device: str,
        compute_type: str = "int8",
        **kwargs: object,
    ) -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        seen_kwargs.update(kwargs)
        # Simulate implementations that actually consume vad_model.
        assert kwargs.get("vad_model") == "pyannote"
        return _FakeModel()

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(pipeline, "patch_pyannote_inference_ignore_use_auth_token", lambda: False)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        vad_method="pyannote",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    segments, info = pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)

    assert segments and segments[0]["text"] == "hello"
    assert info["language"] == "en"
    assert seen_kwargs["vad_method"] == "pyannote"
    assert seen_kwargs["vad_model"] == "pyannote"
    assert seen_kwargs["vad_options"] == {"vad_method": "pyannote"}


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


def test_whisperx_asr_retries_once_for_omegaconf_unsupported_global(tmp_path: Path, monkeypatch) -> None:
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.transcribe = None
    load_attempts: list[str] = []
    context_calls: list[list[str]] = []

    class _FakeModel:
        def transcribe(
            self,
            audio: str,
            *,
            batch_size: int,
            vad_filter: bool,
            language: str | None,
        ) -> dict[str, object]:
            del batch_size, vad_filter, language
            assert audio == "audio"
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}

    def _load_audio(path: str) -> str:
        assert path.endswith("a.wav")
        return "audio"

    def _load_model(_model_name: str, _device: str, compute_type: str = "int8") -> _FakeModel:
        assert compute_type == "int8"
        load_attempts.append("call")
        if len(load_attempts) == 1:
            raise pickle.UnpicklingError(
                "Weights only load failed. Unsupported global: GLOBAL omegaconf.base.ContainerMetadata"
            )
        return _FakeModel()

    @contextlib.contextmanager
    def _fake_safe_globals(extra_fqns: list[str] | None = None):
        context_calls.append(list(extra_fqns or []))
        yield

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(pipeline, "omegaconf_safe_globals_for_torch_load", _fake_safe_globals)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    segments, info = pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)

    assert segments and segments[0]["text"] == "hello"
    assert info["language"] == "en"
    assert len(load_attempts) == 2
    assert context_calls == [[], ["omegaconf.base.ContainerMetadata"]]


def test_whisperx_asr_does_not_retry_for_non_unsupported_global_errors(tmp_path: Path, monkeypatch) -> None:
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.transcribe = None
    context_calls: list[list[str]] = []

    class _FakeModel:
        def transcribe(self, *args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            return {"segments": [], "language": "en"}

    def _load_audio(path: str) -> str:
        assert path.endswith("a.wav")
        return "audio"

    def _load_model(_model_name: str, _device: str, compute_type: str = "int8") -> _FakeModel:
        assert compute_type == "int8"
        raise RuntimeError("some other load failure")

    @contextlib.contextmanager
    def _fake_safe_globals(extra_fqns: list[str] | None = None):
        context_calls.append(list(extra_fqns or []))
        yield

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(pipeline, "omegaconf_safe_globals_for_torch_load", _fake_safe_globals)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    with pytest.raises(RuntimeError, match="some other load failure"):
        pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)

    assert context_calls == [[]]


def test_whisperx_asr_reraises_original_error_when_retry_fails(tmp_path: Path, monkeypatch) -> None:
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.transcribe = None
    context_calls: list[list[str]] = []
    first_error = pickle.UnpicklingError(
        "Weights only load failed. Unsupported global: GLOBAL omegaconf.base.ContainerMetadata"
    )

    class _FakeModel:
        def transcribe(self, *args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            return {"segments": [], "language": "en"}

    def _load_audio(path: str) -> str:
        assert path.endswith("a.wav")
        return "audio"

    def _load_model(_model_name: str, _device: str, compute_type: str = "int8") -> _FakeModel:
        assert compute_type == "int8"
        if len(context_calls) == 1:
            raise first_error
        raise RuntimeError("retry failed with another reason")

    @contextlib.contextmanager
    def _fake_safe_globals(extra_fqns: list[str] | None = None):
        context_calls.append(list(extra_fqns or []))
        yield

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(pipeline, "omegaconf_safe_globals_for_torch_load", _fake_safe_globals)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    with pytest.raises(pickle.UnpicklingError, match="ContainerMetadata"):
        pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)

    assert context_calls == [[], ["omegaconf.base.ContainerMetadata"]]


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


def test_whisperx_load_model_vad_kwargs_when_signature_unavailable(monkeypatch) -> None:
    def _load_model(*_args: object, **_kwargs: object) -> object:
        return object()

    def _raise_signature_error(_fn: object) -> object:
        raise TypeError("signature unavailable")

    monkeypatch.setattr(pipeline.inspect, "signature", _raise_signature_error)

    kwargs = pipeline._whisperx_load_model_vad_kwargs(_load_model, "pyannote")

    assert kwargs == {
        "vad_method": "pyannote",
        "vad_model": "pyannote",
        "vad_options": {"vad_method": "pyannote"},
    }
