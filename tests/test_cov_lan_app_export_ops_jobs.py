from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from lan_app.config import AppSettings
from lan_app.constants import (
    DEFAULT_REQUEUE_JOB_TYPE,
    JOB_STATUS_QUEUED,
    JOB_TYPE_STT,
    RECORDING_STATUS_READY,
)
from lan_app.db import create_recording, init_db
import lan_app.exporter as exporter
import lan_app.jobs as jobs
import lan_app.ops as ops
import lan_app.reaper as reaper


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_exporter_json_loaders_handle_invalid_and_wrong_types(tmp_path: Path):
    bad_dict = tmp_path / "bad-dict.json"
    bad_dict.write_text("{", encoding="utf-8")
    assert exporter._load_json_dict(bad_dict) == {}

    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2]", encoding="utf-8")
    assert exporter._load_json_dict(list_json) == {}

    bad_list = tmp_path / "bad-list.json"
    bad_list.write_text("{", encoding="utf-8")
    assert exporter._load_json_list(bad_list) == []

    dict_json = tmp_path / "dict.json"
    dict_json.write_text('{"k": 1}', encoding="utf-8")
    assert exporter._load_json_list(dict_json) == []


def test_exporter_normalise_text_items_covers_limits_and_empty_after_bullet(monkeypatch):
    lines = exporter._normalise_text_items(
        " first line \n\n- second line\nthird line",
        max_items=2,
    )
    assert lines == ["first line", "second line"]

    values = iter(["- ", "kept"])
    monkeypatch.setattr(exporter, "_normalize_text", lambda _row: next(values))
    assert exporter._normalise_text_items(["x", "y"], max_items=10) == ["kept"]


def test_exporter_section_helpers_cover_optional_paths():
    metadata = exporter._metadata_lines(
        {
            "id": "rec-1",
            "source_filename": "meeting.wav",
            "duration_sec": "unknown-ish",
        },
        language="de",
    )
    assert "- Duration: unknown-ish" in metadata
    assert "- Language: de" in metadata

    action_items = exporter._action_items_section(
        {
            "action_items": [
                {"task": "   ", "owner": "skip-me"},
                {"task": "Ship update", "deadline": "2026-01-01"},
                {"task": "Review notes", "owner": "Alex"},
                "  Free-form follow up  ",
                "   ",
            ]
        }
    )
    assert "- [ ] Ship update (deadline: 2026-01-01)" in action_items
    assert "- [ ] Review notes (owner: Alex)" in action_items
    assert "- [ ] Free-form follow up" in action_items

    questions = exporter._questions_section(
        {"questions": {"extracted": ["  Why now?  "]}}
    )
    assert questions == ["## Questions", "- Why now?"]

    metrics = exporter._metrics_section(
        {
            "meeting": {
                "total_interruptions": 2,
                "total_questions": None,
                "decisions_count": 1,
                "action_items_count": 0,
            }
        }
    )
    assert metrics == [
        "## Metrics",
        "- Interruptions: 2",
        "- Decisions: 1",
        "- Action items: 0",
    ]
    assert (
        exporter._metrics_section(
            {
                "meeting": {
                    "total_interruptions": None,
                    "total_questions": None,
                    "decisions_count": None,
                    "action_items_count": None,
                }
            }
        )
        == []
    )

    transcript = exporter._transcript_section(
        {"text": "fallback transcript"},
        [{"speaker": "S1", "text": "   "}],
    )
    assert transcript == ["## Transcript", "fallback transcript"]


def test_build_onenote_markdown_language_detection_paths(tmp_path: Path):
    cfg = _cfg(tmp_path)

    rec_with_detected = "rec-lang-detected"
    derived_detected = cfg.recordings_root / rec_with_detected / "derived"
    derived_detected.mkdir(parents=True, exist_ok=True)
    (derived_detected / "transcript.json").write_text(
        json.dumps({"language": {"detected": " fr "}, "text": "bonjour"}),
        encoding="utf-8",
    )
    markdown_detected = exporter.build_onenote_markdown(
        {"id": rec_with_detected, "source_filename": "detected.wav"},
        settings=cfg,
    )
    assert "- Language: fr" in markdown_detected

    rec_with_dominant = "rec-lang-dominant"
    derived_dominant = cfg.recordings_root / rec_with_dominant / "derived"
    derived_dominant.mkdir(parents=True, exist_ok=True)
    (derived_dominant / "transcript.json").write_text(
        json.dumps({"language": "not-a-dict", "dominant_language": "es", "text": "hola"}),
        encoding="utf-8",
    )
    markdown_dominant = exporter.build_onenote_markdown(
        {"id": rec_with_dominant, "source_filename": "dominant.wav"},
        settings=cfg,
    )
    assert "- Language: es" in markdown_dominant

    rec_language_auto = "rec-lang-auto"
    derived_auto = cfg.recordings_root / rec_language_auto / "derived"
    derived_auto.mkdir(parents=True, exist_ok=True)
    (derived_auto / "transcript.json").write_text(
        json.dumps({"language": {"detected": "ignored"}, "dominant_language": "ignored"}),
        encoding="utf-8",
    )
    markdown_auto = exporter.build_onenote_markdown(
        {
            "id": rec_language_auto,
            "source_filename": "auto.wav",
            "language_auto": "en",
        },
        settings=cfg,
    )
    assert "- Language: en" in markdown_auto


def test_try_read_bytes_returns_none_on_oserror(tmp_path: Path, monkeypatch):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"payload")
    path_type = type(target)
    real_read_bytes = path_type.read_bytes

    def _patched_read_bytes(self):
        if self == target:
            raise OSError("read failed")
        return real_read_bytes(self)

    monkeypatch.setattr(path_type, "read_bytes", _patched_read_bytes)
    assert exporter._try_read_bytes(target) is None


def test_build_export_zip_bytes_missing_recording_and_snippet_paths(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    init_db(cfg)

    with pytest.raises(KeyError):
        exporter.build_export_zip_bytes("missing-recording", settings=cfg)

    recording_id = "rec-export-snippets"
    create_recording(
        recording_id,
        source="upload",
        source_filename="snippets.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    derived = cfg.recordings_root / recording_id / "derived"
    snippets = derived / "snippets"
    snippets.mkdir(parents=True, exist_ok=True)
    inside = snippets / "inside.wav"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")

    path_type = type(snippets)
    real_rglob = path_type.rglob

    def _patched_rglob(self, pattern):
        if self == snippets and pattern == "*.wav":
            return [outside, inside]
        return real_rglob(self, pattern)

    monkeypatch.setattr(path_type, "rglob", _patched_rglob)

    payload = exporter.build_export_zip_bytes(
        recording_id,
        settings=cfg,
        include_snippets=True,
    )
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = set(archive.namelist())
    assert f"derived/snippets/{inside.name}" in names
    assert "outside.wav" not in names


def test_build_export_zip_bytes_include_snippets_non_dir_and_unreadable_file(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    recording_id = "rec-export-unreadable"
    create_recording(
        recording_id,
        source="upload",
        source_filename="unreadable.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    snippets_as_file = derived / "snippets"
    snippets_as_file.write_text("not-a-directory", encoding="utf-8")
    payload_no_dir = exporter.build_export_zip_bytes(
        recording_id,
        settings=cfg,
        include_snippets=True,
    )
    archive_no_dir = zipfile.ZipFile(io.BytesIO(payload_no_dir))
    assert all(not name.startswith("derived/snippets/") for name in archive_no_dir.namelist())

    snippets_as_file.unlink()
    snippets_dir = derived / "snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    unreadable = snippets_dir / "unreadable.wav"
    unreadable.write_bytes(b"wav")

    real_try_read_bytes = exporter._try_read_bytes

    def _patched_try_read_bytes(path: Path):
        if path == unreadable:
            return None
        return real_try_read_bytes(path)

    monkeypatch.setattr(exporter, "_try_read_bytes", _patched_try_read_bytes)

    payload_unreadable = exporter.build_export_zip_bytes(
        recording_id,
        settings=cfg,
        include_snippets=True,
    )
    archive_unreadable = zipfile.ZipFile(io.BytesIO(payload_unreadable))
    assert all("unreadable.wav" not in name for name in archive_unreadable.namelist())


def test_jobs_get_queue_status_and_pending_job_paths(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)

    class _FakeQueueCtor:
        def __init__(self, *, name: str, connection: object):
            self.name = name
            self.connection = connection

    monkeypatch.setattr(jobs, "AppSettings", lambda: cfg)
    monkeypatch.setattr(jobs.Redis, "from_url", lambda url: {"url": url})
    monkeypatch.setattr(jobs, "Queue", _FakeQueueCtor)

    queue_from_settings = jobs.get_queue()
    assert queue_from_settings.name == cfg.rq_queue_name
    assert queue_from_settings.connection == {"url": cfg.redis_url}

    assert jobs._status_value(None) is None

    class _FakeRQJob:
        def __init__(self, status: str):
            self.status = status
            self.deleted = False

        def get_status(self, refresh: bool = False):
            assert refresh is True
            return SimpleNamespace(value=self.status)

        def delete(self, *, remove_from_queue: bool = True, delete_dependents: bool = True):
            self.deleted = remove_from_queue and delete_dependents

    class _FakeQueue:
        def __init__(self):
            self.jobs = {
                "done": _FakeRQJob("finished"),
                "deferred": _FakeRQJob("deferred"),
                "scheduled": _FakeRQJob("scheduled"),
            }
            self.removed: list[str] = []

        def fetch_job(self, job_id: str):
            return self.jobs.get(job_id)

        def remove(self, job_id: str):
            self.removed.append(job_id)

    deferred_removed: list[tuple[str, bool]] = []
    scheduled_removed: list[tuple[str, bool]] = []

    class _DeferredRegistry:
        def __init__(self, *, queue):
            self.queue = queue

        def remove(self, job_id: str, delete_job: bool = False):
            deferred_removed.append((job_id, delete_job))

    class _ScheduledRegistry:
        def __init__(self, *, queue):
            self.queue = queue

        def remove(self, job_id: str, delete_job: bool = False):
            scheduled_removed.append((job_id, delete_job))

    monkeypatch.setattr(jobs, "DeferredJobRegistry", _DeferredRegistry)
    monkeypatch.setattr(jobs, "ScheduledJobRegistry", _ScheduledRegistry)

    queue = _FakeQueue()
    assert jobs._purge_pending_queue_job(queue, "missing") is False
    assert jobs._purge_pending_queue_job(queue, "done") is False
    assert jobs._purge_pending_queue_job(queue, "deferred") is True
    assert jobs._purge_pending_queue_job(queue, "scheduled") is True
    assert deferred_removed == [("deferred", False)]
    assert scheduled_removed == [("scheduled", False)]


def test_jobs_purge_pending_queue_job_covers_scheduled_false_branch(monkeypatch):
    class _WeirdStatus:
        def __init__(self):
            self.calls = 0

        def __hash__(self):
            return hash("scheduled")

        def __eq__(self, other):
            self.calls += 1
            if self.calls == 1 and other == "scheduled":
                return True
            return False

    status = _WeirdStatus()

    class _FakeJob:
        def __init__(self):
            self.deleted = False

        def get_status(self, refresh: bool = False):
            assert refresh is True
            return SimpleNamespace(value="scheduled")

        def delete(self, *, remove_from_queue: bool = True, delete_dependents: bool = True):
            self.deleted = remove_from_queue and delete_dependents

    class _FakeQueue:
        def __init__(self):
            self.job = _FakeJob()

        def fetch_job(self, _job_id: str):
            return self.job

        def remove(self, _job_id: str):
            raise AssertionError("queue.remove should not be called")

    monkeypatch.setattr(jobs, "_status_value", lambda _status: status)
    queue = _FakeQueue()
    assert jobs._purge_pending_queue_job(queue, "job-weird") is True
    assert queue.job.deleted is True


def test_purge_pending_recording_jobs_handles_empty_and_paginated_results(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(jobs, "init_db", lambda _cfg: None)
    monkeypatch.setattr(jobs, "get_queue", lambda _cfg: object())

    monkeypatch.setattr(
        jobs,
        "list_jobs",
        lambda **_kwargs: ([], 0),
    )
    assert jobs.purge_pending_recording_jobs("rec-empty", settings=cfg) == 0

    offsets: list[int] = []

    def _list_jobs_paginated(**kwargs):
        offset = int(kwargs["offset"])
        offsets.append(offset)
        if offset == 0:
            return ([{"id": "job-a"}, {"id": "job-b"}], 3)
        if offset == 2:
            return ([{"id": "job-c"}], 3)
        return ([], 3)

    monkeypatch.setattr(jobs, "list_jobs", _list_jobs_paginated)
    monkeypatch.setattr(
        jobs,
        "_purge_pending_queue_job",
        lambda _queue, job_id: job_id != "job-b",
    )

    removed = jobs.purge_pending_recording_jobs("rec-paged", settings=cfg)
    assert removed == 2
    assert offsets == [0, 2]


def test_cancel_pending_queue_job_returns_false_on_errors(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(jobs, "get_queue", lambda _cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    assert jobs.cancel_pending_queue_job("job-1", settings=cfg) is False

    monkeypatch.setattr(jobs, "get_queue", lambda _cfg: object())
    monkeypatch.setattr(
        jobs,
        "_purge_pending_queue_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert jobs.cancel_pending_queue_job("job-2", settings=cfg) is False


def test_enqueue_recording_job_rejects_invalid_type_and_missing_recording(tmp_path: Path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="Unsupported job type"):
        jobs.enqueue_recording_job("rec-x", job_type="unsupported", settings=cfg)

    with pytest.raises(jobs.RecordingNotFoundError, match="Recording not found"):
        jobs.enqueue_recording_job("rec-missing", settings=cfg)


def test_enqueue_recording_job_allows_empty_existing_job_id(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(jobs, "init_db", lambda _cfg: None)
    monkeypatch.setattr(jobs, "get_recording", lambda *_a, **_k: {"id": "rec-1"})
    monkeypatch.setattr(
        jobs,
        "create_job_if_no_active_for_recording",
        lambda **_kwargs: (False, {"id": "   "}),
    )
    monkeypatch.setattr(jobs, "set_recording_status", lambda *_a, **_k: True)

    class _QueueOK:
        def enqueue(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(jobs, "get_queue", lambda _cfg: _QueueOK())

    job = jobs.enqueue_recording_job("rec-1", settings=cfg)
    assert job.recording_id == "rec-1"
    assert job.job_type == DEFAULT_REQUEUE_JOB_TYPE


def test_enqueue_recording_job_uses_create_job_for_non_default_type(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(jobs, "init_db", lambda _cfg: None)
    monkeypatch.setattr(jobs, "get_recording", lambda *_a, **_k: {"id": "rec-2"})
    monkeypatch.setattr(
        jobs,
        "create_job_if_no_active_for_recording",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    created: list[dict[str, object]] = []
    monkeypatch.setattr(
        jobs,
        "create_job",
        lambda **kwargs: created.append(kwargs),
    )
    monkeypatch.setattr(jobs, "set_recording_status", lambda *_a, **_k: True)

    class _QueueOK:
        def enqueue(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(jobs, "get_queue", lambda _cfg: _QueueOK())

    job = jobs.enqueue_recording_job("rec-2", job_type=JOB_TYPE_STT, settings=cfg)
    assert job.job_type == JOB_TYPE_STT
    assert len(created) == 1
    assert created[0]["job_type"] == JOB_TYPE_STT
    assert created[0]["status"] == JOB_STATUS_QUEUED


def test_enqueue_recording_job_reraises_when_fail_job_also_fails(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(jobs, "init_db", lambda _cfg: None)
    monkeypatch.setattr(jobs, "get_recording", lambda *_a, **_k: {"id": "rec-3"})
    monkeypatch.setattr(
        jobs,
        "create_job_if_no_active_for_recording",
        lambda **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        jobs,
        "fail_job",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fail-job-down")),
    )

    class _BrokenQueue:
        def enqueue(self, *_args, **_kwargs):
            raise RuntimeError("redis-down")

    monkeypatch.setattr(jobs, "get_queue", lambda _cfg: _BrokenQueue())

    with pytest.raises(RuntimeError, match="redis-down"):
        jobs.enqueue_recording_job("rec-3", settings=cfg)


def test_ops_parse_and_iteration_branches(tmp_path: Path, monkeypatch):
    assert ops._parse_utc(None) is None
    assert ops._parse_utc("   ") is None
    assert ops._parse_utc("not-a-date") is None

    parsed_naive = ops._parse_utc("2026-01-02T03:04:05")
    assert parsed_naive is not None
    assert parsed_naive.tzinfo == timezone.utc

    cfg = _cfg(tmp_path)
    offsets: list[int] = []

    def _paged_list_recordings(**kwargs):
        offset = int(kwargs["offset"])
        offsets.append(offset)
        if offset == 0:
            return ([{"id": "rec-1"}], 2)
        if offset == 1:
            return ([{"id": "rec-2"}], 2)
        return ([], 2)

    monkeypatch.setattr(ops, "list_recordings", _paged_list_recordings)
    rows = ops._iter_recordings(settings=cfg)
    assert rows == [{"id": "rec-1"}, {"id": "rec-2"}]
    assert offsets == [0, 1]

    monkeypatch.setattr(ops, "list_recordings", lambda **_kwargs: ([], 0))
    assert ops._iter_recordings(settings=cfg) == []


def test_ops_delete_path_and_cleanup_skip_paths(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    assert ops._delete_path(tmp_path / "does-not-exist") is False

    monkeypatch.setattr(
        ops,
        "_iter_recordings",
        lambda **_kwargs: [
            {"id": "   ", "updated_at": "2000-01-01T00:00:00Z"},
            {"id": "rec-old", "updated_at": "2000-01-01T00:00:00Z"},
        ],
    )
    monkeypatch.setattr(ops, "_delete_path", lambda _path: False)
    monkeypatch.setattr(ops, "delete_recording", lambda *_a, **_k: False)

    summary = ops.run_retention_cleanup(settings=cfg)
    assert summary == {
        "quarantine_recordings_deleted": 0,
        "quarantine_directories_deleted": 0,
        "tmp_entries_deleted": 0,
    }


def test_ops_cleanup_handles_tmp_stat_errors_and_failed_deletes(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ops, "_iter_recordings", lambda **_kwargs: [])

    tmp_root = cfg.data_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    bad_entry = tmp_root / "bad.tmp"
    old_entry = tmp_root / "old.tmp"
    bad_entry.write_text("x", encoding="utf-8")
    old_entry.write_text("y", encoding="utf-8")
    stale_timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
    old_entry.touch()
    bad_entry.touch()
    os.utime(old_entry, (stale_timestamp, stale_timestamp))
    os.utime(bad_entry, (stale_timestamp, stale_timestamp))

    path_type = type(bad_entry)
    real_stat = path_type.stat

    def _patched_stat(self, *args, **kwargs):
        if self == bad_entry:
            raise OSError("stat failed")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "stat", _patched_stat)
    monkeypatch.setattr(ops, "_delete_path", lambda _path: False)

    summary = ops.run_retention_cleanup(settings=cfg)
    assert summary["tmp_entries_deleted"] == 0


def test_reaper_now_default_and_missing_identifier_paths(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.stuck_job_seconds = 10

    monkeypatch.setattr(
        reaper,
        "list_stale_started_jobs",
        lambda **_kwargs: [{"id": "", "recording_id": "rec-stale", "type": "precheck"}],
    )
    monkeypatch.setattr(
        reaper,
        "list_processing_recordings_without_started_job",
        lambda **_kwargs: [
            {"id": "", "active_job_id": "job-1", "active_job_type": "precheck"},
            {"id": "rec-no-job", "active_job_id": "", "active_job_type": " "},
        ],
    )
    monkeypatch.setattr(
        reaper,
        "fail_job_if_started",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    monkeypatch.setattr(
        reaper,
        "fail_job_if_queued",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    monkeypatch.setattr(
        reaper,
        "set_recording_status_if_current_in_and_no_started_job",
        lambda recording_id, *_a, **_k: recording_id == "rec-no-job",
    )
    appended: list[str] = []
    monkeypatch.setattr(
        reaper,
        "_append_step_log_best_effort",
        lambda _path, message, *, now: appended.append(message),
    )

    summary = reaper.run_stuck_job_reaper_once(settings=cfg)
    assert summary["stale_started_jobs"] == 1
    assert summary["processing_without_started"] == 2
    assert summary["recovered_jobs"] == 0
    assert summary["recovered_recordings"] == 1
    assert summary["job_ids"] == []
    assert summary["recording_ids"] == ["rec-no-job"]
    assert len(appended) == 1
    assert "job=none" in appended[0]
