from __future__ import annotations

import logging
import site
import struct
import threading
from pathlib import Path

_LOG = logging.getLogger(__name__)

_ELF_MAGIC = b"\x7fELF"
_ELFCLASS32 = 1
_ELFCLASS64 = 2
_ELFDATA2LSB = 1

_PT_GNU_STACK = 0x6474E551
_PF_X = 0x1

_PATCH_LOCK = threading.Lock()
_PATCH_RAN = False
_PATCHED_PATHS: tuple[str, ...] = ()


def clear_execstack_flag(path: Path) -> bool:
    """Clear PF_X from PT_GNU_STACK in an ELF file.

    Returns True when a patch was applied; otherwise False.
    """

    try:
        with path.open("r+b") as fh:
            ident = fh.read(16)
            if len(ident) < 16 or ident[:4] != _ELF_MAGIC:
                return False
            ei_class = ident[4]
            ei_data = ident[5]
            if ei_data != _ELFDATA2LSB:
                return False

            if ei_class == _ELFCLASS64:
                header_fmt = "<HHIQQQIHHHHHH"
                p_type_off = 0
                p_flags_off = 4
            elif ei_class == _ELFCLASS32:
                header_fmt = "<HHIIIIIHHHHHH"
                p_type_off = 0
                p_flags_off = 24
            else:
                return False

            header_size = struct.calcsize(header_fmt)
            header_data = fh.read(header_size)
            if len(header_data) != header_size:
                return False

            header = struct.unpack(header_fmt, header_data)
            e_phoff = int(header[4])
            e_phentsize = int(header[8])
            e_phnum = int(header[9])

            min_entry_size = max(p_type_off + 4, p_flags_off + 4)
            if e_phoff <= 0 or e_phentsize < min_entry_size or e_phnum <= 0:
                return False

            patched = False
            for idx in range(e_phnum):
                base = e_phoff + idx * e_phentsize
                fh.seek(base + p_type_off)
                p_type_data = fh.read(4)
                if len(p_type_data) != 4:
                    return patched
                p_type = struct.unpack("<I", p_type_data)[0]
                if p_type != _PT_GNU_STACK:
                    continue

                fh.seek(base + p_flags_off)
                p_flags_data = fh.read(4)
                if len(p_flags_data) != 4:
                    return patched
                p_flags = struct.unpack("<I", p_flags_data)[0]
                if p_flags & _PF_X:
                    fh.seek(base + p_flags_off)
                    fh.write(struct.pack("<I", p_flags & ~_PF_X))
                    patched = True
            return patched
    except (OSError, struct.error):
        return False


def find_libctranslate2_candidates() -> list[Path]:
    """Find libctranslate2 shared objects installed in site-packages."""

    found: set[Path] = set()
    try:
        package_roots = [Path(root) for root in site.getsitepackages()]
    except Exception:
        return []
    for root in package_roots:
        if not root.exists():
            continue
        for candidate in root.glob("**/libctranslate2*.so*"):
            if not candidate.is_file():
                continue
            found.add(candidate.resolve(strict=False))
    return sorted(found, key=lambda item: str(item))


def ensure_ctranslate2_no_execstack() -> list[str]:
    """Patch ctranslate2 shared objects once per process."""

    global _PATCH_RAN, _PATCHED_PATHS
    with _PATCH_LOCK:
        if _PATCH_RAN:
            return list(_PATCHED_PATHS)

        patched: list[str] = []
        try:
            candidates = find_libctranslate2_candidates()
        except Exception:
            _LOG.exception("failed to enumerate ctranslate2 candidates")
            candidates = []

        for candidate in candidates:
            try:
                if clear_execstack_flag(candidate):
                    patched.append(str(candidate))
            except Exception:
                _LOG.warning(
                    "failed to clear executable-stack flag for %s",
                    candidate,
                    exc_info=True,
                )

        _PATCHED_PATHS = tuple(sorted(set(patched)))
        _PATCH_RAN = True
        return list(_PATCHED_PATHS)


__all__ = [
    "clear_execstack_flag",
    "find_libctranslate2_candidates",
    "ensure_ctranslate2_no_execstack",
]
