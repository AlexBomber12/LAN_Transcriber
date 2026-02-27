from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import BinaryIO

ALLOWED_UPLOAD_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".aac",
    ".ogg",
    ".flac",
}

_DEFAULT_SUFFIX = ".mp3"
_FILENAME_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PLAUD_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2})[_\-](\d{2})[_\-](\d{2})"
)
_COPY_CHUNK_SIZE = 1024 * 1024


def suffix_from_name(filename: str) -> str:
    name = str(filename or "")
    suffix = Path(name).suffix.strip().lower()
    if suffix:
        return suffix
    return _DEFAULT_SUFFIX


def safe_filename(filename: str) -> str:
    raw = str(filename or "").replace("\\", "/").split("/")[-1].strip()
    if not raw:
        return "upload"
    sanitized = _FILENAME_SAFE_CHARS_RE.sub("_", raw).strip("._- ")
    return sanitized or "upload"


def parse_plaud_captured_at(filename: str) -> str | None:
    match = _PLAUD_RE.search(str(filename or ""))
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(group) for group in match.groups())
    try:
        parsed = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def infer_captured_at(filename: str) -> str:
    parsed = parse_plaud_captured_at(filename)
    if parsed is not None:
        return parsed
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_upload_to_path(upload, dest: Path, *, max_bytes: int | None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    fh: BinaryIO
    try:
        upload.file.seek(0)
    except Exception:
        pass

    try:
        with dest.open("wb") as fh:
            while True:
                chunk = upload.file.read(_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if max_bytes is not None and bytes_written > max_bytes:
                    raise ValueError("max upload size exceeded")
                fh.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    return bytes_written


__all__ = [
    "ALLOWED_UPLOAD_EXTENSIONS",
    "infer_captured_at",
    "parse_plaud_captured_at",
    "safe_filename",
    "suffix_from_name",
    "write_upload_to_path",
]
