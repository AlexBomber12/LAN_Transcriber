from __future__ import annotations

import contextlib
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lan_transcriber.pipeline_steps import orchestrator as pipeline


def test_whisperx_asr_modern_path(tmp_path: Path, monkeypatch, caplog):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")
    seen_kwargs: dict[str, object] = {}

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

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
        **kwargs: object,
    ) -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        assert vad_method == "silero"
        assert kwargs == {}
        seen_kwargs.update({"compute_type": compute_type, "vad_method": vad_method, **kwargs})
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

    def _wrapper_load_model(*_args: object, **_kwargs: object) -> _FakeModel:
        raise AssertionError("expected whisperx.asr.load_model to be used")

    fake_whisperx.load_audio = _load_audio
    fake_whisperx.load_model = _wrapper_load_model
    fake_whisperx.load_align_model = _load_align_model
    fake_whisperx.align = _align
    fake_asr.load_model = _load_model
    fake_whisperx.asr = fake_asr
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)

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
    assert seen_kwargs["vad_method"] == "silero"
    assert "vad_model" not in seen_kwargs
    assert "ASR VAD method: silero" in caplog.text


def test_whisperx_asr_modern_path_uses_pyannote_vad_method(tmp_path: Path, monkeypatch):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")
    seen_kwargs: dict[str, object] = {}

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

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
        **kwargs: object,
    ) -> _FakeModel:
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        assert vad_method == "pyannote"
        assert kwargs == {}
        seen_kwargs.update({"compute_type": compute_type, "vad_method": vad_method, **kwargs})
        return _FakeModel()

    fake_whisperx.load_audio = _load_audio
    fake_asr.load_model = _load_model
    fake_whisperx.asr = fake_asr
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)
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
    assert seen_kwargs["vad_method"] == "pyannote"
    assert "vad_model" not in seen_kwargs
    assert any("pyannote compat" in line for line in step_log)


def test_whisperx_asr_modern_path_fails_fast_on_non_callable_vad_model(tmp_path: Path, monkeypatch):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")

    class _FakeModel:
        vad_model = "silero"

        def transcribe(
            self,
            _audio: str,
            **_kwargs: object,
        ) -> dict[str, object]:
            raise AssertionError("transcribe should not run before vad_model validation")

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

    fake_whisperx.load_audio = _load_audio
    fake_asr.load_model = _load_model
    fake_whisperx.asr = fake_asr
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    with pytest.raises(
        RuntimeError,
        match="WhisperX VAD misconfigured:.*vad_method='silero'.*type\\(vad_model\\)=<class 'str'>",
    ):
        pipeline._whisperx_asr(audio_path, override_lang=None, cfg=cfg)


def test_whisperx_asr_modern_path_drops_unsupported_kwargs(tmp_path: Path, monkeypatch):
    fake_whisperx = ModuleType("whisperx")

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

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
        vad_model = staticmethod(lambda _payload: [])

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
        vad_model = staticmethod(lambda _payload: [])

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
        vad_model = staticmethod(lambda _payload: [])

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


def test_glossary_transcribe_helper_edge_paths() -> None:
    assert pipeline._glossary_transcribe_kwargs(None) == {}  # noqa: SLF001
    assert pipeline._glossary_transcribe_kwargs({"initial_prompt": "Prompt"}) == {  # noqa: SLF001
        "initial_prompt": "Prompt"
    }
    assert pipeline._glossary_transcribe_kwargs({"hotwords": "Term"}) == {  # noqa: SLF001
        "hotwords": "Term"
    }

    messages: list[str] = []
    pipeline._log_glossary_context_unsupported(  # noqa: SLF001
        callback=messages.append,
        glossary_kwargs={"hotwords": "Term"},
        filtered_kwargs={"hotwords": "Term"},
        state={},
    )
    assert messages == []

    pipeline._log_glossary_context_unsupported(  # noqa: SLF001
        callback=messages.append,
        glossary_kwargs={"initial_prompt": "Prompt", "hotwords": "Term"},
        filtered_kwargs={"initial_prompt": "Prompt"},
        state={},
    )
    assert messages == [
        "whisperx transcribe: glossary context unsupported for some kwargs; "
        "continuing with supported ASR hints only"
    ]

    runtime_state = pipeline._new_glossary_runtime_state(  # noqa: SLF001
        {"initial_prompt": "Prompt", "hotwords": "Term"}
    )
    pipeline._update_glossary_runtime_state(  # noqa: SLF001
        state=runtime_state,
        glossary_kwargs={"initial_prompt": "Prompt", "hotwords": "Term"},
        final_kwargs={"initial_prompt": "Prompt"},
        dropped_kwargs=("hotwords",),
    )
    pipeline._update_glossary_runtime_state(  # noqa: SLF001
        state={"checked": False, "applied_keys": [], "dropped_keys": ()},
        glossary_kwargs={"initial_prompt": "Prompt"},
        final_kwargs={},
        dropped_kwargs=("initial_prompt",),
    )

    def transcribe_audio(*_args, **_kwargs) -> None:
        return None

    transcribe_audio.glossary_runtime_state = runtime_state  # type: ignore[attr-defined]
    assert pipeline._glossary_runtime_metadata(transcribe_audio) == {  # noqa: SLF001
        "checked": True,
        "requested_keys": ["hotwords", "initial_prompt"],
        "applied_keys": ["initial_prompt"],
        "dropped_keys": ["hotwords"],
    }
    assert pipeline._glossary_runtime_metadata(object()) == {}  # noqa: SLF001
    transcribe_audio.glossary_runtime_state = {  # type: ignore[attr-defined]
        "requested_keys": (),
        "applied_keys": set(),
        "dropped_keys": set(),
        "checked": False,
    }
    assert pipeline._glossary_runtime_metadata(transcribe_audio) == {}  # noqa: SLF001

    glossary_payload = {
        "version": 1,
        "recording_id": "rec-1",
        "terms": ["Sander", "Sandia"],
        "entry_count": 1,
        "term_count": 2,
        "initial_prompt": "Glossary: Sander; Sandia",
        "hotwords": "Sander, Sandia",
    }
    assert pipeline._effective_asr_glossary_artifact(  # noqa: SLF001
        asr_glossary=glossary_payload,
        runtime_metadata=None,
    ) == glossary_payload
    assert pipeline._effective_asr_glossary_artifact(  # noqa: SLF001
        asr_glossary=glossary_payload,
        runtime_metadata={"requested_keys": ("initial_prompt",), "applied_keys": []},
    ) == glossary_payload
    assert pipeline._effective_asr_glossary_artifact(  # noqa: SLF001
        asr_glossary=glossary_payload,
        runtime_metadata={
            "requested_keys": ["initial_prompt", "hotwords"],
            "applied_keys": ["initial_prompt"],
            "dropped_keys": ["hotwords"],
        },
    ) == {
        "version": 1,
        "recording_id": "rec-1",
        "terms": ["Sander", "Sandia"],
        "entry_count": 1,
        "term_count": 2,
        "initial_prompt": "Glossary: Sander; Sandia",
        "applied_kwargs": ["initial_prompt"],
        "dropped_kwargs": ["hotwords"],
    }
    assert pipeline._effective_asr_glossary_artifact(  # noqa: SLF001
        asr_glossary=glossary_payload,
        runtime_metadata={
            "requested_keys": ["initial_prompt", "hotwords"],
            "applied_keys": [],
            "dropped_keys": ["initial_prompt", "hotwords"],
        },
    ) is None
    assert pipeline._effective_asr_glossary_artifact(  # noqa: SLF001
        asr_glossary=glossary_payload,
        runtime_metadata={
            "requested_keys": ["other"],
            "applied_keys": [],
            "dropped_keys": [],
        },
    ) == {
        "version": 1,
        "recording_id": "rec-1",
        "terms": ["Sander", "Sandia"],
        "entry_count": 1,
        "term_count": 2,
    }


def test_build_whisperx_transcriber_modern_path_forwards_glossary_kwargs(
    tmp_path: Path,
    monkeypatch,
):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")
    seen_kwargs: dict[str, object] = {}

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

        def transcribe(
            self,
            audio: str,
            *,
            batch_size: int,
            vad_filter: bool,
            language: str | None,
            initial_prompt: str,
            hotwords: str,
        ) -> dict[str, object]:
            assert audio == "audio"
            assert batch_size == 16
            assert vad_filter is True
            assert language == "es"
            seen_kwargs["initial_prompt"] = initial_prompt
            seen_kwargs["hotwords"] = hotwords
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}], "language": "es"}

    def _load_audio(_path: str) -> str:
        return "audio"

    def _load_model(
        _model_name: str,
        _device: str,
        compute_type: str = "int8",
        vad_method: str = "silero",
    ) -> _FakeModel:
        assert compute_type == "int8"
        assert vad_method == "silero"
        return _FakeModel()

    fake_whisperx.load_audio = _load_audio
    fake_asr.load_model = _load_model
    fake_whisperx.asr = fake_asr
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)

    transcribe_audio = pipeline._build_whisperx_transcriber(
        cfg=pipeline.Settings(
            asr_device="cpu",
            asr_enable_align=False,
            tmp_root=tmp_path,
            recordings_root=tmp_path / "recordings",
        ),
        asr_glossary={
            "initial_prompt": "Glossary: Sander; Sandia",
            "hotwords": "Sander, Sandia",
        },
    )

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    segments, info = transcribe_audio(audio_path, "es")

    assert segments and segments[0]["text"] == "hola"
    assert info["language"] == "es"
    assert seen_kwargs == {
        "initial_prompt": "Glossary: Sander; Sandia",
        "hotwords": "Sander, Sandia",
    }


def test_build_whisperx_transcriber_modern_path_reuses_cached_model_across_builds(
    tmp_path: Path,
    monkeypatch,
):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")
    load_calls: list[tuple[str, str, str]] = []

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

        def transcribe(self, _audio: str, *, batch_size: int, vad_filter: bool, language: str | None):
            assert batch_size == 16
            assert vad_filter is True
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "ok"}], "language": language or "en"}

    fake_whisperx.load_audio = lambda _path: "audio"
    fake_asr.load_model = lambda model_name, device, *, compute_type="int8", vad_method="silero": (
        load_calls.append((model_name, device, compute_type)) or _FakeModel()
    )
    fake_whisperx.asr = fake_asr
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)
    pipeline.clear_asr_model_cache()

    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    first = pipeline._build_whisperx_transcriber(cfg=cfg)
    second = pipeline._build_whisperx_transcriber(cfg=cfg)

    audio_path = tmp_path / "cached.wav"
    audio_path.write_bytes(b"")
    assert first(audio_path, None)[0][0]["text"] == "ok"
    assert second(audio_path, "en")[0][0]["text"] == "ok"
    assert load_calls == [("large-v3", "cpu", "int8")]
    pipeline.clear_asr_model_cache()


def test_build_whisperx_transcriber_retries_gpu_oom_once_with_smaller_compute_type(
    tmp_path: Path,
    monkeypatch,
):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")
    compute_types: list[str] = []

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

        def transcribe(self, _audio: str, *, batch_size: int, vad_filter: bool, language: str | None):
            assert batch_size == 16
            assert vad_filter is True
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "gpu"}], "language": language or "en"}

    def _load_model(_model_name: str, _device: str, *, compute_type="float16", vad_method="silero"):
        compute_types.append(compute_type)
        assert vad_method == "silero"
        if len(compute_types) == 1:
            raise RuntimeError("CUDA out of memory")
        return _FakeModel()

    fake_whisperx.load_audio = lambda _path: "audio"
    fake_asr.load_model = _load_model
    fake_whisperx.asr = fake_asr
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)
    monkeypatch.setattr(
        pipeline,
        "collect_cuda_runtime_facts",
        lambda: SimpleNamespace(
            is_available=True,
            device_count=1,
            visible_devices="0",
            torch_cuda_version="12.4",
        ),
    )
    step_log: list[str] = []
    pipeline.clear_asr_model_cache()

    transcribe_audio = pipeline._build_whisperx_transcriber(
        cfg=pipeline.Settings(
            asr_device="auto",
            asr_enable_align=False,
            tmp_root=tmp_path,
            recordings_root=tmp_path / "recordings",
        ),
        step_log_callback=step_log.append,
    )

    audio_path = tmp_path / "gpu-oom.wav"
    audio_path.write_bytes(b"")
    segments, info = transcribe_audio(audio_path, None)

    assert segments[0]["text"] == "gpu"
    assert info["language"] == "en"
    assert compute_types == ["float16", "int8_float16"]
    assert any("retrying once with compute_type=int8_float16" in message for message in step_log)
    pipeline.clear_asr_model_cache()


def test_load_cached_whisperx_model_raises_plain_load_errors(
    tmp_path: Path,
    monkeypatch,
):
    fake_asr = SimpleNamespace(
        load_model=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("plain load failure"))
    )
    monkeypatch.setattr(pipeline, "_log_cuda_memory_snapshot", lambda **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "omegaconf_safe_globals_for_torch_load",
        contextlib.nullcontext,
    )
    pipeline.clear_asr_model_cache()

    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    with pytest.raises(RuntimeError, match="plain load failure"):
        pipeline._load_cached_whisperx_model(  # noqa: SLF001
            cfg=cfg,
            wx_asr=fake_asr,
            device="cpu",
            compute_type="int8",
            step_log_callback=None,
        )


def test_load_cached_whisperx_model_prefers_cache_entry_inserted_after_load(
    tmp_path: Path,
    monkeypatch,
):
    load_calls: list[tuple[object, ...]] = []
    inserted_model = object()
    loaded_model = object()

    def _load_model(*args: object, **kwargs: object) -> object:
        load_calls.append(args + (kwargs["compute_type"], kwargs["vad_method"]))
        return loaded_model

    fake_asr = SimpleNamespace(load_model=_load_model)
    cfg = pipeline.Settings(
        asr_device="cpu",
        asr_enable_align=False,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    cache_key = pipeline._asr_model_cache_key(  # noqa: SLF001
        cfg=cfg,
        device="cpu",
        compute_type="int8",
        load_model_callable=_load_model,
    )

    class _RaceLock:
        def __init__(self) -> None:
            self.enter_count = 0

        def __enter__(self):
            self.enter_count += 1
            if self.enter_count == 2:
                pipeline._ASR_MODEL_CACHE[cache_key] = inserted_model  # noqa: SLF001
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(pipeline, "_log_cuda_memory_snapshot", lambda **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "omegaconf_safe_globals_for_torch_load",
        contextlib.nullcontext,
    )
    pipeline.clear_asr_model_cache()
    monkeypatch.setattr(pipeline, "_ASR_MODEL_CACHE_LOCK", _RaceLock())

    cached_model, selected_compute_type = pipeline._load_cached_whisperx_model(  # noqa: SLF001
        cfg=cfg,
        wx_asr=fake_asr,
        device="cpu",
        compute_type="int8",
        step_log_callback=None,
    )

    assert cached_model is inserted_model
    assert selected_compute_type == "int8"
    assert load_calls == [("large-v3", "cpu", "int8", "silero")]
    pipeline.clear_asr_model_cache()


def test_build_whisperx_transcriber_modern_path_uses_safe_fallback_when_call_details_are_unavailable(
    tmp_path: Path,
    monkeypatch,
):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

        def transcribe(
            self,
            audio: str,
            *,
            batch_size: int,
            vad_filter: bool,
            language: str | None,
            initial_prompt: str,
        ) -> dict[str, object]:
            assert audio == "audio"
            assert batch_size == 16
            assert vad_filter is True
            assert language == "es"
            assert initial_prompt == "Glossary: Sander"
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}], "language": "es"}

    fake_whisperx.load_audio = lambda _path: "audio"
    fake_asr.load_model = lambda *_args, **_kwargs: _FakeModel()
    fake_whisperx.asr = fake_asr
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)
    monkeypatch.setattr(
        pipeline,
        "call_with_supported_kwargs",
        lambda fn, *args, **kwargs: fn(*args, **pipeline.filter_kwargs_for_callable(fn, kwargs)),
    )

    transcribe_audio = pipeline._build_whisperx_transcriber(
        cfg=pipeline.Settings(
            asr_device="cpu",
            asr_enable_align=False,
            tmp_root=tmp_path,
            recordings_root=tmp_path / "recordings",
        ),
        asr_glossary={"initial_prompt": "Glossary: Sander"},
    )

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    segments, info = transcribe_audio(audio_path, "es")

    assert segments and segments[0]["text"] == "hola"
    assert info["language"] == "es"


def test_build_whisperx_transcriber_legacy_path_logs_glossary_degrade_when_unsupported(
    tmp_path: Path,
    monkeypatch,
):
    fake_whisperx = ModuleType("whisperx")

    def _transcribe(audio_path: str, *, language: str) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert audio_path.endswith("a.wav")
        assert language == "auto"
        return ([{"start": 0.0, "end": 1.0, "text": "hello"}], {"language": "en"})

    fake_whisperx.transcribe = _transcribe
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    step_log: list[str] = []
    transcribe_audio = pipeline._build_whisperx_transcriber(
        cfg=pipeline.Settings(
            asr_device="cpu",
            asr_enable_align=False,
            tmp_root=tmp_path,
            recordings_root=tmp_path / "recordings",
        ),
        step_log_callback=step_log.append,
        asr_glossary={
            "initial_prompt": "Glossary: Sander; Sandia",
            "hotwords": "Sander, Sandia",
        },
    )

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    segments, info = transcribe_audio(audio_path, None)

    assert segments and segments[0]["text"] == "hello"
    assert info["language"] == "en"
    assert any("glossary context unsupported" in line for line in step_log)


def test_build_whisperx_transcriber_legacy_path_logs_partial_glossary_degrade_after_runtime_fallback(
    tmp_path: Path,
    monkeypatch,
):
    fake_whisperx = ModuleType("whisperx")
    calls: list[dict[str, object]] = []

    def _transcribe(audio_path: str, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        calls.append(dict(kwargs))
        assert audio_path.endswith("a.wav")
        if "hotwords" in kwargs:
            raise TypeError("FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'hotwords'")
        assert kwargs == {
            "vad_filter": True,
            "language": "auto",
            "word_timestamps": True,
            "initial_prompt": "Glossary: Sander; Sandia",
        }
        return ([{"start": 0.0, "end": 1.0, "text": "hello"}], {"language": "en"})

    fake_whisperx.transcribe = _transcribe
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    step_log: list[str] = []
    transcribe_audio = pipeline._build_whisperx_transcriber(
        cfg=pipeline.Settings(
            asr_device="cpu",
            asr_enable_align=False,
            tmp_root=tmp_path,
            recordings_root=tmp_path / "recordings",
        ),
        step_log_callback=step_log.append,
        asr_glossary={
            "initial_prompt": "Glossary: Sander; Sandia",
            "hotwords": "Sander, Sandia",
        },
    )

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    segments, info = transcribe_audio(audio_path, None)

    assert segments and segments[0]["text"] == "hello"
    assert info["language"] == "en"
    assert calls == [
        {
            "vad_filter": True,
            "language": "auto",
            "word_timestamps": True,
            "initial_prompt": "Glossary: Sander; Sandia",
            "hotwords": "Sander, Sandia",
        },
        {
            "vad_filter": True,
            "language": "auto",
            "word_timestamps": True,
            "initial_prompt": "Glossary: Sander; Sandia",
        },
    ]
    assert any(
        "glossary context unsupported for some kwargs" in line for line in step_log
    )
