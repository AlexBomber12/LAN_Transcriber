from __future__ import annotations

import struct
from pathlib import Path

from lan_transcriber import native_fixups
from lan_transcriber.native_fixups import clear_execstack_flag

_PT_GNU_STACK = 0x6474E551
_PF_RWX = 0x7
_PF_RW = 0x6


def _write_synthetic_elf64(path: Path, *, p_flags: int, p_type: int = _PT_GNU_STACK) -> None:
    e_ident = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + bytes(7)
    elf_header = struct.pack(
        "<HHIQQQIHHHHHH",
        2,  # e_type
        62,  # e_machine (x86_64)
        1,  # e_version
        0,  # e_entry
        64,  # e_phoff
        0,  # e_shoff
        0,  # e_flags
        64,  # e_ehsize
        56,  # e_phentsize
        1,  # e_phnum
        0,  # e_shentsize
        0,  # e_shnum
        0,  # e_shstrndx
    )
    program_header = struct.pack(
        "<IIQQQQQQ",
        p_type,
        p_flags,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    path.write_bytes(e_ident + elf_header + program_header)


def _read_elf64_program_flags(path: Path) -> int:
    payload = path.read_bytes()
    # p_flags is the second 32-bit field in the first ELF64 program header.
    return struct.unpack("<I", payload[68:72])[0]


def _write_synthetic_elf32(path: Path, *, p_flags: int) -> None:
    e_ident = b"\x7fELF" + bytes([1, 1, 1, 0, 0]) + bytes(7)
    elf_header = struct.pack(
        "<HHIIIIIHHHHHH",
        2,  # e_type
        3,  # e_machine (x86)
        1,  # e_version
        0,  # e_entry
        52,  # e_phoff
        0,  # e_shoff
        0,  # e_flags
        52,  # e_ehsize
        32,  # e_phentsize
        1,  # e_phnum
        0,  # e_shentsize
        0,  # e_shnum
        0,  # e_shstrndx
    )
    program_header = struct.pack(
        "<IIIIIIII",
        _PT_GNU_STACK,
        0,
        0,
        0,
        0,
        0,
        p_flags,
        0,
    )
    path.write_bytes(e_ident + elf_header + program_header)


def test_clear_execstack_flag_patches_synthetic_elf64(tmp_path: Path) -> None:
    shared_obj = tmp_path / "libctranslate2.so.test"
    _write_synthetic_elf64(shared_obj, p_flags=_PF_RWX)

    changed = clear_execstack_flag(shared_obj)

    assert changed is True
    assert _read_elf64_program_flags(shared_obj) == _PF_RW


def test_clear_execstack_flag_noop_when_exec_bit_already_cleared(tmp_path: Path) -> None:
    shared_obj = tmp_path / "libctranslate2.so.test"
    _write_synthetic_elf64(shared_obj, p_flags=_PF_RW)

    changed = clear_execstack_flag(shared_obj)

    assert changed is False
    assert _read_elf64_program_flags(shared_obj) == _PF_RW


def test_clear_execstack_flag_noop_on_non_elf(tmp_path: Path) -> None:
    payload = tmp_path / "not-elf.bin"
    payload.write_bytes(b"not an elf")

    changed = clear_execstack_flag(payload)

    assert changed is False


def test_clear_execstack_flag_noop_when_program_header_is_not_gnu_stack(tmp_path: Path) -> None:
    shared_obj = tmp_path / "libctranslate2.so.non-gnu-stack"
    _write_synthetic_elf64(shared_obj, p_flags=_PF_RWX, p_type=1)

    changed = clear_execstack_flag(shared_obj)

    assert changed is False
    assert _read_elf64_program_flags(shared_obj) == _PF_RWX


def test_clear_execstack_flag_patches_synthetic_elf32(tmp_path: Path) -> None:
    shared_obj = tmp_path / "libctranslate2.so.elf32"
    _write_synthetic_elf32(shared_obj, p_flags=_PF_RWX)

    changed = clear_execstack_flag(shared_obj)

    assert changed is True
    payload = shared_obj.read_bytes()
    # ELF32 p_flags is the seventh 32-bit field in the first program header.
    assert struct.unpack("<I", payload[76:80])[0] == _PF_RW


def test_clear_execstack_flag_returns_false_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.so"
    assert clear_execstack_flag(missing) is False


def test_find_libctranslate2_candidates_filters_and_sorts(tmp_path: Path, monkeypatch) -> None:
    site_a = tmp_path / "site-a"
    site_b = tmp_path / "site-b"
    file_b = site_b / "pkg" / "libctranslate2.so.4"
    file_a = site_a / "nested" / "libctranslate2-cuda.so.4.1"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_b.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_bytes(b"x")
    file_b.write_bytes(b"x")
    (site_a / "nested" / "libctranslate2-dir.so").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        native_fixups.site,
        "getsitepackages",
        lambda: [str(site_b), str(site_a), str(tmp_path / "missing-site")],
    )

    candidates = native_fixups.find_libctranslate2_candidates()

    assert candidates == sorted([file_a.resolve(), file_b.resolve()], key=lambda item: str(item))


def test_find_libctranslate2_candidates_returns_empty_on_site_failure(monkeypatch) -> None:
    def _boom() -> list[str]:
        raise RuntimeError("site broken")

    monkeypatch.setattr(native_fixups.site, "getsitepackages", _boom)
    assert native_fixups.find_libctranslate2_candidates() == []


def test_ensure_ctranslate2_no_execstack_patches_once_and_ignores_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target_ok = tmp_path / "libctranslate2-ok.so"
    target_fail = tmp_path / "libctranslate2-fail.so"
    target_ok.write_bytes(b"x")
    target_fail.write_bytes(b"x")

    monkeypatch.setattr(native_fixups, "_PATCH_RAN", False)
    monkeypatch.setattr(native_fixups, "_PATCHED_PATHS", ())
    monkeypatch.setattr(
        native_fixups,
        "find_libctranslate2_candidates",
        lambda: [target_ok, target_fail],
    )

    def _fake_clear(path: Path) -> bool:
        if path == target_fail:
            raise RuntimeError("patch failed")
        return True

    monkeypatch.setattr(native_fixups, "clear_execstack_flag", _fake_clear)

    patched_first = native_fixups.ensure_ctranslate2_no_execstack()
    patched_second = native_fixups.ensure_ctranslate2_no_execstack()

    assert patched_first == [str(target_ok)]
    assert patched_second == [str(target_ok)]


def test_ensure_ctranslate2_no_execstack_handles_candidate_scan_failure(monkeypatch) -> None:
    monkeypatch.setattr(native_fixups, "_PATCH_RAN", False)
    monkeypatch.setattr(native_fixups, "_PATCHED_PATHS", ())

    def _boom() -> list[Path]:
        raise RuntimeError("scan failed")

    monkeypatch.setattr(native_fixups, "find_libctranslate2_candidates", _boom)

    assert native_fixups.ensure_ctranslate2_no_execstack() == []
