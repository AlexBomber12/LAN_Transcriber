from __future__ import annotations

import builtins
import struct
import sys
from pathlib import Path
from types import ModuleType

import pytest

from lan_transcriber import aliases, artifacts, native_fixups, normalizer, runtime_paths, utils
from lan_transcriber.compat import call_compat, pyannote_compat
from lan_transcriber.compat.call_compat import call_with_supported_kwargs
from lan_transcriber.compat.pyannote_compat import patch_pyannote_inference_ignore_use_auth_token
from lan_transcriber.models import TranscriptResult

_PT_GNU_STACK = 0x6474E551


def _elf_ident(*, elf_class: int, elf_data: int) -> bytes:
    return b"\x7fELF" + bytes([elf_class, elf_data, 1, 0, 0]) + bytes(7)


def _elf64_header(*, e_phoff: int, e_phentsize: int, e_phnum: int) -> bytes:
    return struct.pack(
        "<HHIQQQIHHHHHH",
        2,
        62,
        1,
        0,
        e_phoff,
        0,
        0,
        64,
        e_phentsize,
        e_phnum,
        0,
        0,
        0,
    )


def test_alias_load_returns_empty_for_non_mapping_payload(tmp_path: Path) -> None:
    path = tmp_path / "db.yaml"
    path.write_text("- not-a-dict\n", encoding="utf-8")
    assert aliases.load_aliases(path) == {}


def test_build_recording_artifacts_defaults_audio_extension(tmp_path: Path) -> None:
    built = artifacts.build_recording_artifacts(tmp_path, "id", audio_ext=None)
    assert built.raw_audio_path.suffix == ".bin"


def test_stage_raw_audio_continues_when_path_resolution_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    src.write_bytes(b"abc")
    original_resolve = Path.resolve

    def _patched_resolve(path_obj: Path, *args: object, **kwargs: object) -> Path:
        if path_obj == src:
            raise FileNotFoundError("synthetic race")
        return original_resolve(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _patched_resolve)
    out = artifacts.stage_raw_audio(src, dst)
    assert out == dst
    assert dst.read_bytes() == b"abc"


def test_atomic_write_bytes_unlinks_temp_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "out.txt"
    captured_tmp_name: dict[str, str] = {}
    original_replace = artifacts.os.replace

    def _replace_with_failure(src: str, dst: Path) -> None:
        captured_tmp_name["name"] = src
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts.os, "replace", _replace_with_failure)
    with pytest.raises(OSError, match="replace failed"):
        artifacts._atomic_write_bytes(target, b"payload")
    assert "name" in captured_tmp_name
    assert not Path(captured_tmp_name["name"]).exists()
    monkeypatch.setattr(artifacts.os, "replace", original_replace)


def test_call_with_supported_kwargs_reaches_final_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fn(*_args: object, **kwargs: object) -> str:
        calls.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(call_compat, "range", lambda *_args: [], raising=False)
    result = call_with_supported_kwargs(_fn, "audio", vad_filter=True)
    assert result == "ok"
    assert calls == [{"vad_filter": True}]


def test_pyannote_patch_returns_false_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _guarded_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "pyannote.audio":
            raise ImportError("synthetic import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    assert patch_pyannote_inference_ignore_use_auth_token() is False


def test_pyannote_patch_returns_false_when_inference_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyannote = ModuleType("pyannote")
    fake_audio = ModuleType("pyannote.audio")

    class DummyInference:
        __init__ = None

    fake_audio.Inference = DummyInference
    fake_pyannote.audio = fake_audio
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    assert patch_pyannote_inference_ignore_use_auth_token() is False


def test_pyannote_patch_returns_false_when_signature_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyannote = ModuleType("pyannote")
    fake_audio = ModuleType("pyannote.audio")

    class DummyInference:
        def __init__(self, *_args: object, **_kwargs: object):
            return None

    fake_audio.Inference = DummyInference
    fake_pyannote.audio = fake_audio
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    def _boom(*_args: object, **_kwargs: object):
        raise TypeError("signature unavailable")

    monkeypatch.setattr(pyannote_compat.inspect, "signature", _boom)
    assert patch_pyannote_inference_ignore_use_auth_token() is False


def test_pyannote_patch_returns_false_when_use_auth_token_is_already_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pyannote = ModuleType("pyannote")
    fake_audio = ModuleType("pyannote.audio")

    class DummyInference:
        def __init__(self, *_args: object, use_auth_token: object | None = None, **_kwargs: object):
            del use_auth_token

    fake_audio.Inference = DummyInference
    fake_pyannote.audio = fake_audio
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    assert patch_pyannote_inference_ignore_use_auth_token() is False


def test_transcript_result_empty_factory() -> None:
    result = TranscriptResult.empty("summary")
    assert result.summary == "summary"
    assert result.body == ""
    assert result.friendly == 0
    assert result.segments == []


def test_clear_execstack_flag_rejects_non_little_endian_elf(tmp_path: Path) -> None:
    target = tmp_path / "bad-data.so"
    target.write_bytes(_elf_ident(elf_class=2, elf_data=2))
    assert native_fixups.clear_execstack_flag(target) is False


def test_clear_execstack_flag_rejects_unknown_elf_class(tmp_path: Path) -> None:
    target = tmp_path / "bad-class.so"
    target.write_bytes(_elf_ident(elf_class=3, elf_data=1))
    assert native_fixups.clear_execstack_flag(target) is False


def test_clear_execstack_flag_rejects_short_header_payload(tmp_path: Path) -> None:
    target = tmp_path / "short-header.so"
    target.write_bytes(_elf_ident(elf_class=2, elf_data=1) + b"\x00" * 8)
    assert native_fixups.clear_execstack_flag(target) is False


def test_clear_execstack_flag_rejects_invalid_program_header_geometry(tmp_path: Path) -> None:
    target = tmp_path / "bad-geometry.so"
    target.write_bytes(
        _elf_ident(elf_class=2, elf_data=1)
        + _elf64_header(e_phoff=0, e_phentsize=56, e_phnum=1)
    )
    assert native_fixups.clear_execstack_flag(target) is False


def test_clear_execstack_flag_handles_short_program_header_type_field(tmp_path: Path) -> None:
    target = tmp_path / "short-type.so"
    target.write_bytes(
        _elf_ident(elf_class=2, elf_data=1)
        + _elf64_header(e_phoff=64, e_phentsize=56, e_phnum=1)
    )
    assert native_fixups.clear_execstack_flag(target) is False


def test_clear_execstack_flag_handles_short_program_header_flags_field(tmp_path: Path) -> None:
    target = tmp_path / "short-flags.so"
    target.write_bytes(
        _elf_ident(elf_class=2, elf_data=1)
        + _elf64_header(e_phoff=64, e_phentsize=56, e_phnum=1)
        + struct.pack("<I", _PT_GNU_STACK)
    )
    assert native_fixups.clear_execstack_flag(target) is False


def test_ensure_ctranslate2_no_execstack_ignores_unmodified_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a.so"
    b = tmp_path / "b.so"
    a.write_bytes(b"x")
    b.write_bytes(b"x")

    monkeypatch.setattr(native_fixups, "_PATCH_RAN", False)
    monkeypatch.setattr(native_fixups, "_PATCHED_PATHS", ())
    monkeypatch.setattr(native_fixups, "find_libctranslate2_candidates", lambda: [a, b])
    monkeypatch.setattr(native_fixups, "clear_execstack_flag", lambda p: p == b)

    assert native_fixups.ensure_ctranslate2_no_execstack() == [str(b)]


def test_normalizer_does_not_fuzzy_compare_long_sentences() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(30)) + "."
    text = f"{long_sentence} {long_sentence}"
    assert normalizer.dedup(text) == text


def test_default_data_root_prefers_data_mount_when_writable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAN_DATA_ROOT", raising=False)
    monkeypatch.setattr(runtime_paths.Path, "exists", lambda self: str(self) == "/data")
    monkeypatch.setattr(runtime_paths.os, "access", lambda *_args, **_kwargs: True)
    assert runtime_paths.default_data_root() == Path("/data")


def test_normalise_language_code_handles_empty_and_long_tokens() -> None:
    assert utils.normalise_language_code("   ") is None
    assert utils.normalise_language_code("english") is None


def test_normalise_text_items_covers_strip_disabled_and_empty_after_strip() -> None:
    assert utils.normalise_text_items(["- keep"], max_items=3, strip_bullets=False) == ["- keep"]
    assert utils.normalise_text_items(["-   ", "ok"], max_items=3) == ["ok"]
