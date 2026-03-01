from __future__ import annotations

from typing import Any

import pytest

from lan_transcriber.compat import call_compat
from lan_transcriber.compat.call_compat import call_with_supported_kwargs, filter_kwargs_for_callable


def test_filter_kwargs_drops_unsupported_keys_without_var_kwargs() -> None:
    def _fn(audio: str, *, batch_size: int, language: str | None = None) -> None:
        del audio, batch_size, language

    filtered = filter_kwargs_for_callable(
        _fn,
        {"batch_size": 16, "language": "en", "vad_filter": True},
    )

    assert filtered == {"batch_size": 16, "language": "en"}


def test_filter_kwargs_keeps_keys_with_var_kwargs() -> None:
    def _fn(audio: str, **kwargs: object) -> None:
        del audio, kwargs

    filtered = filter_kwargs_for_callable(
        _fn,
        {"batch_size": 16, "language": "en", "vad_filter": True},
    )

    assert filtered == {"batch_size": 16, "language": "en", "vad_filter": True}


def test_filter_kwargs_returns_empty_dict_for_empty_input() -> None:
    def _fn(audio: str, *, batch_size: int) -> None:
        del audio, batch_size

    assert filter_kwargs_for_callable(_fn, {}) == {}


def test_filter_kwargs_keeps_input_when_signature_introspection_fails(monkeypatch) -> None:
    def _raise_signature_error(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("signature unavailable")

    monkeypatch.setattr(call_compat.inspect, "signature", _raise_signature_error)

    kwargs = {"vad_filter": True}
    assert filter_kwargs_for_callable(object(), kwargs) == kwargs


def test_call_with_supported_kwargs_filters_before_call() -> None:
    captured: dict[str, object] = {}

    def _fn(audio: str, *, batch_size: int, language: str | None = None) -> str:
        captured["audio"] = audio
        captured["batch_size"] = batch_size
        captured["language"] = language
        return "ok"

    result = call_with_supported_kwargs(
        _fn,
        "audio-bytes",
        batch_size=8,
        language="es",
        vad_filter=True,
    )

    assert result == "ok"
    assert captured == {
        "audio": "audio-bytes",
        "batch_size": 8,
        "language": "es",
    }


def test_call_with_supported_kwargs_retries_when_signature_is_unavailable(monkeypatch) -> None:
    def _raise_signature_error(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("signature unavailable")

    class _RetryTarget:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, audio: str, **kwargs: Any) -> int:
            del audio
            self.calls.append(dict(kwargs))
            if "vad_filter" in kwargs:
                raise TypeError("FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'vad_filter'")
            return int(kwargs["batch_size"])

    target = _RetryTarget()
    monkeypatch.setattr(call_compat.inspect, "signature", _raise_signature_error)

    result = call_with_supported_kwargs(
        target,
        "audio-bytes",
        batch_size=8,
        vad_filter=True,
    )

    assert result == 8
    assert target.calls == [
        {"batch_size": 8, "vad_filter": True},
        {"batch_size": 8},
    ]


def test_call_with_supported_kwargs_reraises_non_matching_type_error(monkeypatch) -> None:
    def _raise_signature_error(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("signature unavailable")

    def _fn(audio: str, **kwargs: Any) -> None:
        del audio, kwargs
        raise TypeError("call failed for unrelated reason")

    monkeypatch.setattr(call_compat.inspect, "signature", _raise_signature_error)

    with pytest.raises(TypeError, match="unrelated reason"):
        call_with_supported_kwargs(
            _fn,
            "audio-bytes",
            batch_size=8,
            vad_filter=True,
        )
