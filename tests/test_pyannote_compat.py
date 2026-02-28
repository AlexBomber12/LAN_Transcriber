from __future__ import annotations

import sys
from types import ModuleType

from lan_transcriber.compat.pyannote_compat import patch_pyannote_inference_ignore_use_auth_token


def test_patch_pyannote_inference_ignore_use_auth_token(monkeypatch):
    fake_pyannote = ModuleType("pyannote")
    fake_audio = ModuleType("pyannote.audio")

    class DummyInference:
        def __init__(self, *args, **kwargs):
            assert "use_auth_token" not in kwargs
            self.args = args
            self.kwargs = kwargs

    fake_audio.Inference = DummyInference
    fake_pyannote.audio = fake_audio
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    assert patch_pyannote_inference_ignore_use_auth_token() is True

    instance = DummyInference("model-id", use_auth_token=None, sample_rate=16000)
    assert instance.args == ("model-id",)
    assert instance.kwargs == {"sample_rate": 16000}

    assert patch_pyannote_inference_ignore_use_auth_token() is False
