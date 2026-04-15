from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import stat
from typing import BinaryIO
from zoneinfo import ZoneInfo

from .config import AppSettings

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
    r"(\d{4})-(\d{2})-(\d{2})[\s_-]+(\d{2})[_\-](\d{2})[_\-](\d{2})"
)
_COPY_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CaptureTimeInference:
    captured_at: str
    captured_at_source: str | None
    captured_at_timezone: str | None
    captured_at_inferred_from_filename: bool


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


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def parse_plaud_captured_local_datetime(filename: str) -> datetime | None:
    match = _PLAUD_RE.search(str(filename or ""))
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(group) for group in match.groups())
    try:
        parsed = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    return parsed


def normalize_plaud_captured_at(
    local_datetime: datetime,
    *,
    upload_capture_timezone: ZoneInfo,
) -> str:
    aware_local = local_datetime.replace(tzinfo=upload_capture_timezone)
    return _utc_iso(aware_local)


def parse_plaud_captured_at(
    filename: str,
    *,
    upload_capture_timezone: ZoneInfo,
) -> str | None:
    parsed = parse_plaud_captured_local_datetime(filename)
    if parsed is not None:
        return normalize_plaud_captured_at(
            parsed,
            upload_capture_timezone=upload_capture_timezone,
        )
    return None


def infer_upload_capture_time(
    filename: str,
    *,
    upload_capture_timezone: ZoneInfo,
) -> CaptureTimeInference:
    parsed = parse_plaud_captured_local_datetime(filename)
    if parsed is None:
        return CaptureTimeInference(
            captured_at=_utc_iso(datetime.now(tz=timezone.utc)),
            captured_at_source=None,
            captured_at_timezone=upload_capture_timezone.key,
            captured_at_inferred_from_filename=False,
        )
    return CaptureTimeInference(
        captured_at=normalize_plaud_captured_at(
            parsed,
            upload_capture_timezone=upload_capture_timezone,
        ),
        captured_at_source=parsed.isoformat(timespec="seconds"),
        captured_at_timezone=upload_capture_timezone.key,
        captured_at_inferred_from_filename=True,
    )


def infer_captured_at(
    filename: str,
    *,
    upload_capture_timezone: ZoneInfo,
) -> str:
    return infer_upload_capture_time(
        filename,
        upload_capture_timezone=upload_capture_timezone,
    ).captured_at


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_COPY_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_matching_upload_recording(
    upload_path: Path,
    *,
    settings: AppSettings,
) -> str | None:
    try:
        upload_stat = upload_path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(upload_stat.st_mode):
        return None
    upload_size = upload_stat.st_size
    upload_digest = _sha256_file(upload_path)

    for raw_audio in sorted(settings.recordings_root.glob("*/raw/audio.*")):
        if raw_audio == upload_path:
            continue
        try:
            if raw_audio.stat().st_size != upload_size:
                continue
            if _sha256_file(raw_audio) == upload_digest:
                return raw_audio.parent.parent.name
        except OSError:
            continue
    return None


__all__ = [
    "ALLOWED_UPLOAD_EXTENSIONS",
    "CaptureTimeInference",
    "find_matching_upload_recording",
    "infer_captured_at",
    "infer_upload_capture_time",
    "normalize_plaud_captured_at",
    "parse_plaud_captured_at",
    "parse_plaud_captured_local_datetime",
    "safe_filename",
    "suffix_from_name",
    "write_upload_to_path",
]
