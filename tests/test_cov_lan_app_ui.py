from __future__ import annotations

import importlib
import runpy
import sys
import types
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from lan_app import ui


class _DummyResult:
    def __init__(self) -> None:
        self.summary = "Summary"
        self.friendly = 7
        self.body = "Transcript body"
        self.summary_path = Path("summary.md")
        self.body_path = Path("full.md")
        self.unknown_chunks = [Path("chunk-1.wav")]


class _DummySettings:
    def __init__(self, root: Path) -> None:
        self.voices_dir = root / "voices"
        self.recordings_root = root / "recordings"
        self.tmp_root = root / "tmp"


def test_transcribe_empty_audio_returns_placeholder() -> None:
    result = ui.transcribe("")
    assert result[0].startswith("### Summary")
    assert result[1].startswith("### Friendly-score: **0**")
    assert result[5] == "—"


def test_transcribe_happy_path_and_fallback_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    recorded: list[_DummyResult] = []

    class _Diar:
        def to(self, _device: str) -> "_Diar":
            return self

    def _from_pretrained(*args: object, **kwargs: object) -> _Diar:
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise TypeError("old-signature")
        return _Diar()

    async def _process_recording(**_kwargs: object) -> _DummyResult:
        return _DummyResult()

    monkeypatch.setattr(ui, "split_repo_id_and_revision", lambda *_a, **_k: ("repo/test", None))
    monkeypatch.setattr(ui.Pipeline, "from_pretrained", _from_pretrained)
    monkeypatch.setattr(ui.pipeline, "Settings", lambda: _DummySettings(tmp_path))
    monkeypatch.setattr(ui, "process_recording", _process_recording)
    monkeypatch.setattr(ui.llm_client, "LLMClient", lambda: object())
    monkeypatch.setattr(ui, "set_current_result", lambda value: recorded.append(value))

    result = ui.transcribe(str(tmp_path / "meeting.wav"))
    assert "Summary" in result[0]
    assert "Friendly-score: **7**" in result[1]
    assert result[2] == Path("summary.md")
    assert result[5] == "chunk-1.wav"
    assert calls[0] == (("repo/test",), {})
    assert calls[1] == (("repo/test",), {})
    assert len(recorded) == 1


def test_enroll_speaker_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui.pipeline, "Settings", lambda: _DummySettings(tmp_path))
    missing = ui.enroll_speaker("", "")
    assert "Upload voice sample" in missing

    source = tmp_path / "voice.wav"
    source.write_bytes(b"wav")
    ok = ui.enroll_speaker(str(source), "Alex")
    assert "Alex" in ok
    assert (tmp_path / "voices" / "Alex.wav").exists()


def test_ui_fallback_root_route_is_callable() -> None:
    response = TestClient(ui.app).get("/")
    assert response.status_code == 200


def _install_ui_import_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mount_mode: str,
    launch_calls: list[dict[str, object]] | None = None,
) -> None:
    launch_calls = launch_calls if launch_calls is not None else []

    class _Component:
        def click(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _Ctx:
        def __enter__(self) -> "_Ctx":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    class _Blocks(_Ctx):
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load(self, *_args: object, **_kwargs: object) -> None:
            return None

        def launch(self, **kwargs: object) -> None:
            launch_calls.append(kwargs)

    def _component(*_args: object, **_kwargs: object) -> _Component:
        return _Component()

    def _ctx(*_args: object, **_kwargs: object) -> _Ctx:
        return _Ctx()

    gradio = types.ModuleType("gradio")
    gradio.Blocks = _Blocks  # type: ignore[attr-defined]
    gradio.Markdown = _component  # type: ignore[attr-defined]
    gradio.Row = _ctx  # type: ignore[attr-defined]
    gradio.Column = _ctx  # type: ignore[attr-defined]
    gradio.Audio = _component  # type: ignore[attr-defined]
    gradio.Button = _component  # type: ignore[attr-defined]
    gradio.File = _component  # type: ignore[attr-defined]
    gradio.Accordion = _ctx  # type: ignore[attr-defined]
    gradio.Textbox = _component  # type: ignore[attr-defined]

    if mount_mode == "raise":
        def _mount_raise(*_args: object, **_kwargs: object) -> FastAPI:
            raise RuntimeError("mount failed")

        gradio.mount_gradio_app = _mount_raise  # type: ignore[attr-defined]
    elif mount_mode == "with_root":
        def _mount_with_root(app: FastAPI, *_args: object, **_kwargs: object) -> FastAPI:
            @app.get("/")
            async def _root() -> str:
                return "mounted"

            return app

        gradio.mount_gradio_app = _mount_with_root  # type: ignore[attr-defined]

    torch_mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_mod.cuda = _Cuda()  # type: ignore[attr-defined]

    pyannote_mod = types.ModuleType("pyannote")
    pyannote_audio_mod = types.ModuleType("pyannote.audio")

    class _Pipeline:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> "_Pipeline":
            return _Pipeline()

        def to(self, _device: str) -> "_Pipeline":
            return self

    pyannote_audio_mod.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gradio", gradio)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_mod)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio_mod)
    monkeypatch.setenv("CI", "false")


def test_ui_reload_mount_exception_and_fallback_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ui_import_stubs(monkeypatch, mount_mode="raise")
    reloaded = importlib.reload(ui)
    response = TestClient(reloaded.app).get("/")
    assert response.status_code == 200
    assert "LAN Transcriber" in response.text


def test_ui_reload_when_mount_has_root_skips_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ui_import_stubs(monkeypatch, mount_mode="with_root")
    reloaded = importlib.reload(ui)
    response = TestClient(reloaded.app).get("/")
    assert response.status_code == 200
    assert response.text.strip('"') == "mounted"


def test_ui_module_main_guard_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    launch_calls: list[dict[str, object]] = []
    _install_ui_import_stubs(monkeypatch, mount_mode="with_root", launch_calls=launch_calls)
    runpy.run_module("lan_app.ui", run_name="__main__")
    assert launch_calls
    assert launch_calls[0]["server_name"] == "0.0.0.0"
    assert launch_calls[0]["share"] is False
