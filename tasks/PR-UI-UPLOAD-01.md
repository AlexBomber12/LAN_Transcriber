PR-UI-UPLOAD-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/ui-upload-01
PR title: PR-UI-UPLOAD-01 Upload ingest API (multipart) and recording creation
Base branch: main

Goal:
1) Add a first-class upload ingestion endpoint that creates a recording from an uploaded audio file.
2) Store the uploaded file under /data/recordings/<recording_id>/raw/audio.<ext>.
3) Create a DB recording row with source=upload and enqueue the existing precheck pipeline job.
4) Add focused tests that do not require Redis.

Scope:
- API only. No UI page in this PR.
- Keep existing Google Drive ingest and Connections page unchanged for now.

A) Upload utilities (new module)
- Add lan_app/uploads.py with:
  1) ALLOWED_UPLOAD_EXTENSIONS set:
     - .mp3, .wav, .m4a, .mp4, .aac, .ogg, .flac
  2) suffix_from_name(filename: str) -> str
     - Return extension including leading dot.
     - Default to .mp3 when missing.
  3) safe_filename(filename: str) -> str
     - Strip directory parts and unsafe characters.
     - Preserve original basename where possible.
     - Fall back to "upload" when empty.
  4) parse_plaud_captured_at(filename: str) -> str | None
     - Parse Plaud patterns:
       - "YYYY-MM-DD HH_MM_SS"
       - "YYYY-MM-DD HH-MM-SS"
       - Allow the pattern to appear anywhere inside the filename.
     - Return "YYYY-MM-DDTHH:MM:SSZ" on success.
     - Return None on no match or invalid date/time.
  5) infer_captured_at(filename: str) -> str
     - Use parse_plaud_captured_at first.
     - Else return current UTC timestamp as ISO with Z.
  6) write_upload_to_path(upload, dest: Path, *, max_bytes: int | None) -> int
     - Stream copy in chunks (do not read entire file into memory).
     - Create parent directory.
     - Enforce max_bytes if provided:
       - If exceeded, delete partial file and raise ValueError("max upload size exceeded").
     - Return bytes_written.

B) Runtime config (optional max size)
- Extend lan_app/config.py AppSettings with:
  - upload_max_bytes: int | None
  - Env name: UPLOAD_MAX_BYTES
  - Validation: when set, must be >= 1
- Keep default None.

C) API endpoint
- In lan_app/api.py:
  1) Add imports:
     - from fastapi import File, UploadFile
     - from uuid import uuid4
     - from lan_app.uploads import (
         ALLOWED_UPLOAD_EXTENSIONS,
         safe_filename,
         suffix_from_name,
         infer_captured_at,
         write_upload_to_path,
       )
  2) Add POST /api/uploads
     - Accept multipart form field "file" as UploadFile.
     - Generate recording_id as f"trs_{uuid4().hex[:8]}".
     - ext = suffix_from_name(upload.filename)
     - Validate ext in ALLOWED_UPLOAD_EXTENSIONS, else 422.
     - recording_dir = settings.recordings_root / recording_id
     - raw_dir = recording_dir / "raw"
     - dest = raw_dir / f"audio{ext}"
     - bytes_written = write_upload_to_path(upload, dest, max_bytes=settings.upload_max_bytes)
     - captured_at = infer_captured_at(upload.filename or "")
     - create_recording(...):
       - source="upload"
       - source_filename=safe_filename(upload.filename or "")
       - captured_at=captured_at
       - status=RECORDING_STATUS_QUEUED
     - enqueue_recording_job(recording_id, job_type=JOB_TYPE_PRECHECK, settings=settings)
     - Return JSON:
       - recording_id
       - job_id
       - captured_at
       - bytes_written
  3) Error handling
     - On any failure after creating recording_dir, remove the entire recording_dir with shutil.rmtree(..., ignore_errors=True).
     - When max size exceeded, return 413 with clear message.
     - When queue enqueue fails, return 503 with clear message.

D) Tests (no Redis)
- Add tests/test_upload.py
  1) Reuse the pattern from tests/test_ui_routes.py:
     - Build a temp AppSettings pointing data_root and db_path to tmp_path.
     - monkeypatch api._settings to cfg.
     - init_db(cfg).
  2) Stub queueing:
     - monkeypatch api.enqueue_recording_job to a DB-only stub that:
       - creates a queued job row in DB (create_job)
       - sets recording status to Queued
       - returns a RecordingJob with job_id
  3) Test success:
     - POST /api/uploads with files={"file": ("2026-02-18 16_01_43.mp3", b"abc", "audio/mpeg")}
     - Assert 200 and response contains recording_id starting with "trs_".
     - Assert file exists at cfg.recordings_root/<id>/raw/audio.mp3.
     - Assert DB recording exists:
       - source == "upload"
       - source_filename contains ".mp3"
       - captured_at == "2026-02-18T16:01:43Z"
     - Assert 1 queued job exists for recording.
  4) Test unsupported extension:
     - Upload "bad.exe" and expect 422.

Local verification:
- scripts/ci.sh

Success criteria:
- POST /api/uploads creates a new recording directory, writes the raw audio file, inserts a DB row, and enqueues a precheck job.
- Upload size limit works when UPLOAD_MAX_BYTES is set.
- tests/test_upload.py is green.
- scripts/ci.sh is green.
```
