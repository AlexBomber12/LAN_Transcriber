"""Google Drive ingest: Service Account + shared Inbox folder.

Polls a shared Drive folder for new audio files, downloads them,
creates recording rows and enqueues the processing pipeline.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .config import AppSettings
from .constants import (
    JOB_TYPE_PRECHECK,
    JOB_TYPE_STT,
    JOB_TYPE_DIARIZE,
    JOB_TYPE_ALIGN,
    JOB_TYPE_LANGUAGE,
    JOB_TYPE_LLM,
    JOB_TYPE_METRICS,
    RECORDING_STATUS_QUEUED,
)
import shutil

from .db import connect, create_recording, init_db

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_PIPELINE_STEPS = (
    JOB_TYPE_PRECHECK,
    JOB_TYPE_STT,
    JOB_TYPE_DIARIZE,
    JOB_TYPE_ALIGN,
    JOB_TYPE_LANGUAGE,
    JOB_TYPE_LLM,
    JOB_TYPE_METRICS,
)

# Plaud filename patterns:
#   "2026-02-18 16_01_43.mp3"
#   "2026-02-18 16-01-43.mp3"
_PLAUD_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2})[_\-](\d{2})[_\-](\d{2})"
)


def parse_plaud_captured_at(filename: str) -> str | None:
    """Parse captured_at ISO-8601 timestamp from a Plaud-style filename.

    Returns ISO-8601 string (UTC) on success, None if no match.
    """
    m = _PLAUD_RE.search(filename)
    if m is None:
        return None
    year, month, day, hour, minute, second = (int(g) for g in m.groups())
    try:
        dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def build_drive_service(sa_json_path: Path) -> Any:
    """Create an authenticated Google Drive API v3 service."""
    creds = Credentials.from_service_account_file(str(sa_json_path), scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_inbox_files(service: Any, folder_id: str) -> list[dict[str, Any]]:
    """List files in the Inbox folder.

    Returns a list of Drive file metadata dicts with id, name,
    md5Checksum, mimeType, createdTime.
    """
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "files(id,name,md5Checksum,mimeType,createdTime)"
    results: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        resp = (
            service.files()
            .list(
                q=query,
                fields=f"nextPageToken,{fields}",
                pageToken=page_token,
                pageSize=100,
            )
            .execute()
        )
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def _known_drive_file_ids(settings: AppSettings) -> set[str]:
    """Return set of drive_file_id values already in the DB."""
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute(
            "SELECT drive_file_id FROM recordings WHERE drive_file_id IS NOT NULL"
        ).fetchall()
    return {row[0] for row in rows}


def download_file(service: Any, file_id: str, dest: Path) -> Path:
    """Download a Drive file to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with dest.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest


def _suffix_from_name(name: str) -> str:
    """Extract file extension from filename, default to .mp3."""
    idx = name.rfind(".")
    if idx > 0:
        return name[idx:]
    return ".mp3"


def _enqueue_pipeline_jobs(
    recording_id: str,
    settings: AppSettings,
) -> list[str]:
    """Enqueue the first pipeline step into Redis/RQ and create DB
    placeholder rows for the remaining steps.

    The first step (precheck) is pushed onto the worker queue so
    processing starts automatically.  Later PRs will chain subsequent
    steps.
    """
    from .db import create_job
    from .jobs import enqueue_recording_job

    job_ids: list[str] = []

    # First step: enqueue into Redis/RQ so the worker picks it up
    first_step = _PIPELINE_STEPS[0]
    rj = enqueue_recording_job(
        recording_id, job_type=first_step, settings=settings
    )
    job_ids.append(rj.job_id)

    # Remaining steps: DB placeholders only (chaining not yet wired)
    for step in _PIPELINE_STEPS[1:]:
        job_id = uuid4().hex
        create_job(
            job_id=job_id,
            recording_id=recording_id,
            job_type=step,
            settings=settings,
        )
        job_ids.append(job_id)
    return job_ids


def ingest_once(
    settings: AppSettings | None = None,
    *,
    service: Any | None = None,
) -> list[dict[str, Any]]:
    """Run a single ingest cycle.

    Returns a list of dicts describing each newly ingested recording.
    """
    cfg = settings or AppSettings()

    if not cfg.gdrive_sa_json_path or not str(cfg.gdrive_sa_json_path).strip():
        raise ValueError("GDRIVE_SA_JSON_PATH is not configured")
    if not cfg.gdrive_inbox_folder_id or not cfg.gdrive_inbox_folder_id.strip():
        raise ValueError("GDRIVE_INBOX_FOLDER_ID is not configured")

    init_db(cfg)

    if service is None:
        service = build_drive_service(cfg.gdrive_sa_json_path)

    inbox_files = list_inbox_files(service, cfg.gdrive_inbox_folder_id)
    known_ids = _known_drive_file_ids(cfg)
    new_files = [f for f in inbox_files if f["id"] not in known_ids]

    ingested: list[dict[str, Any]] = []

    for drive_file in new_files:
        file_id = drive_file["id"]
        file_name = drive_file["name"]
        drive_md5 = drive_file.get("md5Checksum")

        recording_id = f"trs_{uuid4().hex[:8]}"
        ext = _suffix_from_name(file_name)
        raw_dir = cfg.recordings_root / recording_id / "raw"
        dest = raw_dir / f"audio{ext}"

        try:
            download_file(service, file_id, dest)
        except Exception:
            logger.exception("Failed to download %s (%s)", file_name, file_id)
            # Clean up the partially written recording directory
            rec_dir = cfg.recordings_root / recording_id
            shutil.rmtree(rec_dir, ignore_errors=True)
            continue

        # Parse captured_at from Plaud filename; fall back to Drive createdTime
        captured_at = parse_plaud_captured_at(file_name)
        capture_warning = None
        if captured_at is None:
            created_time = drive_file.get("createdTime")
            if created_time:
                captured_at = created_time
            else:
                captured_at = (
                    datetime.now(tz=timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            capture_warning = "captured_at parsed from Drive createdTime (filename parse failed)"

        create_recording(
            recording_id,
            source="gdrive",
            source_filename=file_name,
            captured_at=captured_at,
            status=RECORDING_STATUS_QUEUED,
            drive_file_id=file_id,
            drive_md5=drive_md5,
            settings=cfg,
        )

        job_ids = _enqueue_pipeline_jobs(recording_id, cfg)

        result: dict[str, Any] = {
            "recording_id": recording_id,
            "drive_file_id": file_id,
            "source_filename": file_name,
            "captured_at": captured_at,
            "jobs_created": len(job_ids),
        }
        if capture_warning:
            result["warning"] = capture_warning

        ingested.append(result)
        logger.info(
            "Ingested %s -> %s (%d jobs)",
            file_name,
            recording_id,
            len(job_ids),
        )

    return ingested


__all__ = [
    "build_drive_service",
    "download_file",
    "ingest_once",
    "list_inbox_files",
    "parse_plaud_captured_at",
]
