from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lan_app import worker_tasks
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FAILED,
    JOB_STATUS_STARTED,
    JOB_TYPE_CLEANUP,
    JOB_TYPE_PRECHECK,
    JOB_TYPE_PUBLISH,
    JOB_TYPE_STT,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_READY,
)
from lan_app.db import create_recording, init_db
from lan_transcriber.pipeline import PrecheckResult


def _db_settings(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _lightweight_settings(tmp_path: Path, *, max_job_attempts: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        max_job_attempts=max_job_attempts,
    )


def _patch_lightweight_process_job_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    max_job_attempts: int = 3,
) -> SimpleNamespace:
    settings = _lightweight_settings(tmp_path, max_job_attempts=max_job_attempts)
    monkeypatch.setattr(worker_tasks, "AppSettings", lambda: settings)
    monkeypatch.setattr(worker_tasks, "init_db", lambda _settings: None)
    return settings


def _patch_precheck_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_status: str = RECORDING_STATUS_READY,
) -> None:
    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(worker_tasks, "_job_attempt", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(worker_tasks, "set_recording_status", lambda *_a, **_k: True)
    monkeypatch.setattr(worker_tasks, "_append_step_log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        worker_tasks,
        "_run_precheck_pipeline",
        lambda **_kwargs: (final_status, None),
    )
    monkeypatch.setattr(worker_tasks, "clear_recording_progress", lambda *_a, **_k: True)


def test_restore_status_from_precheck_log_handles_read_errors_and_non_terminal_statuses(
    tmp_path: Path,
):
    cfg = _db_settings(tmp_path)

    unreadable_path = worker_tasks._step_log_path("rec-restore-unreadable", JOB_TYPE_PRECHECK, cfg)
    unreadable_path.mkdir(parents=True, exist_ok=True)
    assert worker_tasks._restore_status_from_precheck_log(
        "rec-restore-unreadable",
        cfg,
    ) == (None, None)

    precheck_log = worker_tasks._step_log_path("rec-restore-nonterminal", JOB_TYPE_PRECHECK, cfg)
    precheck_log.parent.mkdir(parents=True, exist_ok=True)
    precheck_log.write_text(
        "\n".join(
            [
                "noise line",
                "quarantined reason=manual_hold",
                "finished recording_status=Queued",
            ]
        ),
        encoding="utf-8",
    )

    assert worker_tasks._restore_status_from_precheck_log(
        "rec-restore-nonterminal",
        cfg,
    ) == (None, "manual_hold")


def test_has_queued_precheck_job_skips_current_job_and_non_precheck_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _db_settings(tmp_path)

    monkeypatch.setattr(
        worker_tasks,
        "list_jobs",
        lambda **_kwargs: (
            [
                {"id": "job-current", "type": JOB_TYPE_PRECHECK},
                {"id": "job-legacy", "type": JOB_TYPE_STT},
            ],
            2,
        ),
    )

    assert (
        worker_tasks._has_queued_precheck_job(
            "rec-queued-check",
            settings=cfg,
            exclude_job_id="job-current",
        )
        is False
    )


def test_success_status_mapping_covers_all_single_job_outputs():
    assert worker_tasks._success_status(JOB_TYPE_PUBLISH) == "Published"
    assert worker_tasks._success_status(JOB_TYPE_CLEANUP) == "Quarantine"
    assert worker_tasks._success_status(JOB_TYPE_PRECHECK) == "Ready"


def test_job_attempt_returns_zero_for_invalid_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _db_settings(tmp_path)
    monkeypatch.setattr(worker_tasks, "get_job", lambda *_a, **_k: {"attempt": "oops"})
    assert worker_tasks._job_attempt("job-attempt-invalid", cfg) == 0


def test_log_stale_inflight_execution_ignores_log_write_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    log_path = tmp_path / "logs" / "step.log"
    monkeypatch.setattr(
        worker_tasks,
        "_append_step_log",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    worker_tasks._log_stale_inflight_execution(
        job_id="job-stale-1",
        job_type=JOB_TYPE_PRECHECK,
        log_path=log_path,
        detail="status=failed",
    )


def test_start_job_or_ignore_stale_execution_handles_oserror_and_missing_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    log_path = tmp_path / "logs" / "step.log"
    settings = _lightweight_settings(tmp_path)

    monkeypatch.setattr(worker_tasks, "start_job", lambda *_a, **_k: False)
    monkeypatch.setattr(worker_tasks, "get_job", lambda *_a, **_k: {"status": JOB_STATUS_FAILED})
    monkeypatch.setattr(
        worker_tasks,
        "_append_step_log",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("readonly")),
    )
    assert (
        worker_tasks._start_job_or_ignore_stale_execution(
            job_id="job-stale-2",
            recording_id="rec-2",
            job_type=JOB_TYPE_PRECHECK,
            settings=settings,
            log_path=log_path,
        )
        is False
    )

    monkeypatch.setattr(worker_tasks, "get_job", lambda *_a, **_k: {})
    monkeypatch.setattr(worker_tasks, "_append_step_log", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="Job not found: job-missing"):
        worker_tasks._start_job_or_ignore_stale_execution(
            job_id="job-missing",
            recording_id="rec-missing",
            job_type=JOB_TYPE_PRECHECK,
            settings=settings,
            log_path=log_path,
        )


def test_retry_delay_seconds_handles_zero_and_out_of_range_attempts():
    policy = worker_tasks.RetryPolicy(max_attempts=2, backoff_seconds=(4,))
    assert worker_tasks._retry_delay_seconds(policy, 0) == 0
    assert worker_tasks._retry_delay_seconds(policy, 5) == 0


def test_record_retry_handles_requeue_and_log_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    settings = _lightweight_settings(tmp_path)
    log_path = tmp_path / "logs" / "step.log"

    monkeypatch.setattr(
        worker_tasks,
        "requeue_job_if_started",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        worker_tasks._record_retry(
            job_id="job-retry-1",
            job_type=JOB_TYPE_PRECHECK,
            recording_id="rec-retry-1",
            attempt=1,
            max_attempts=3,
            delay_seconds=1,
            settings=settings,
            log_path=log_path,
            exc=RuntimeError("r"),
        )
        is False
    )

    monkeypatch.setattr(worker_tasks, "requeue_job_if_started", lambda *_a, **_k: True)
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_status",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("set failed")),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_append_step_log",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("log failed")),
    )
    assert (
        worker_tasks._record_retry(
            job_id="job-retry-2",
            job_type=JOB_TYPE_PRECHECK,
            recording_id="rec-retry-2",
            attempt=1,
            max_attempts=3,
            delay_seconds=1,
            settings=settings,
            log_path=log_path,
            exc=RuntimeError("r"),
        )
        is True
    )


def test_record_failure_and_max_attempts_helpers_swallow_secondary_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    settings = _lightweight_settings(tmp_path)
    log_path = tmp_path / "logs" / "step.log"

    monkeypatch.setattr(
        worker_tasks,
        "fail_job",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_status",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("set")),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_append_step_log",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("log")),
    )

    worker_tasks._record_failure(
        job_id="job-fail-1",
        job_type=JOB_TYPE_PRECHECK,
        recording_id="rec-fail-1",
        settings=settings,
        log_path=log_path,
        exc=RuntimeError("primary"),
    )
    worker_tasks._record_max_attempts_exceeded(
        job_id="job-fail-2",
        job_type=JOB_TYPE_PRECHECK,
        recording_id="rec-fail-2",
        settings=settings,
        log_path=log_path,
    )


def test_clean_language_value_rejects_blank_strings():
    assert worker_tasks._clean_language_value("  en  ") == "en"
    assert worker_tasks._clean_language_value("   ") is None


def test_load_transcript_language_payload_parses_valid_and_invalid_json(tmp_path: Path):
    cfg = _db_settings(tmp_path)
    transcript_path = cfg.recordings_root / "rec-lang-1" / "derived" / "transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps({"dominant_language": " en ", "target_summary_language": "  "}),
        encoding="utf-8",
    )
    assert worker_tasks._load_transcript_language_payload("rec-lang-1", cfg) == ("en", None)

    transcript_path.write_text("{not-json", encoding="utf-8")
    assert worker_tasks._load_transcript_language_payload("rec-lang-1", cfg) == (None, None)


def test_load_calendar_summary_context_handles_invalid_candidate_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _db_settings(tmp_path)

    monkeypatch.setattr(
        worker_tasks,
        "get_calendar_match",
        lambda *_a, **_k: {"selected_event_id": "evt-1", "candidates_json": "{bad"},
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-1", cfg) == (None, [])

    monkeypatch.setattr(
        worker_tasks,
        "get_calendar_match",
        lambda *_a, **_k: {"selected_event_id": "evt-1", "candidates_json": "{}"},
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-2", cfg) == (None, [])

    monkeypatch.setattr(
        worker_tasks,
        "get_calendar_match",
        lambda *_a, **_k: {"selected_event_id": "evt-1", "candidates_json": 123},
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-3", cfg) == (None, [])

    monkeypatch.setattr(
        worker_tasks,
        "get_calendar_match",
        lambda *_a, **_k: {
            "selected_event_id": "evt-1",
            "candidates_json": '[{"event_id":"evt-1","subject":"Team Sync","attendees":[]}]',
        },
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-3b", cfg) == (
        "Team Sync",
        [],
    )

    monkeypatch.setattr(
        worker_tasks,
        "get_calendar_match",
        lambda *_a, **_k: {
            "selected_event_id": "evt-1",
            "candidates_json": ["not-dict", {"event_id": "evt-2"}],
        },
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-4", cfg) == (None, [])


def test_load_calendar_summary_context_returns_title_without_attendees_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _db_settings(tmp_path)
    monkeypatch.setattr(
        worker_tasks,
        "get_calendar_match",
        lambda *_a, **_k: {
            "selected_event_id": "evt-1",
            "candidates_json": [
                {"event_id": "evt-1", "subject": "Weekly Sync", "attendees": "not-a-list"}
            ],
        },
    )

    assert worker_tasks._load_calendar_summary_context("rec-cal-5", cfg) == (
        "Weekly Sync",
        [],
    )


def test_fallback_diariser_annotation_itertracks_covers_both_modes():
    diariser = worker_tasks._FallbackDiariser(1.5)
    annotation = asyncio.run(diariser(Path("/tmp/audio.wav")))
    yielded_with_label = list(annotation.itertracks(yield_label=True))
    yielded_without_label = list(annotation.itertracks(yield_label=False))
    assert yielded_with_label[0][0].start == 0.0
    assert yielded_with_label[0][1] == "S1"
    assert yielded_without_label[0][0].end == 1.5


def test_pyannote_diariser_retries_with_string_path_for_signature_mismatch():
    class _Model:
        def __init__(self):
            self.calls: list[object] = []

        def __call__(self, payload: object):
            self.calls.append(payload)
            if isinstance(payload, dict):
                raise TypeError("missing required positional argument")
            return {"ok": True}

    model = _Model()
    diariser = worker_tasks._PyannoteDiariser(model)
    result = asyncio.run(diariser(Path("/tmp/input.wav")))
    assert result == {"ok": True}
    assert model.calls == [{"audio": "/tmp/input.wav"}, "/tmp/input.wav"]


def test_pyannote_diariser_does_not_swallow_non_signature_type_errors():
    class _Model:
        def __call__(self, _payload: object):
            raise TypeError("internal type mismatch")

    diariser = worker_tasks._PyannoteDiariser(_Model())
    with pytest.raises(TypeError, match="internal type mismatch"):
        asyncio.run(diariser(Path("/tmp/input.wav")))


def test_pyannote_diariser_rejects_non_callable_model():
    with pytest.raises(TypeError, match="pipeline_model must be a callable"):
        worker_tasks._PyannoteDiariser(None)


def test_build_diariser_uses_fallback_when_pyannote_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyannote.audio":
            err = ModuleNotFoundError("No module named 'pyannote'")
            err.name = "pyannote"
            raise err
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("pyannote", None)
    sys.modules.pop("pyannote.audio", None)

    diariser = worker_tasks._build_diariser(duration_sec=12.0)
    assert isinstance(diariser, worker_tasks._FallbackDiariser)


def test_run_precheck_pipeline_records_step_logs_and_explicit_summary_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-step-log-1",
        source="test",
        source_filename="audio.wav",
        target_summary_language="fr",
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-step-log-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"\x00")

    monkeypatch.setattr(
        worker_tasks,
        "run_precheck",
        lambda *_a, **_k: PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
    )
    monkeypatch.setattr(worker_tasks, "_build_diariser", lambda *_a, **_k: object())
    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_metrics",
        lambda *_a, **_k: {"participants": [], "meeting": {"total_interruptions": 0}},
    )
    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_routing",
        lambda *_a, **_k: {
            "suggested_project_id": None,
            "confidence": 0.0,
            "threshold": 0.5,
            "auto_selected": False,
            "status_after_routing": RECORDING_STATUS_READY,
        },
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_transcript_language_payload",
        lambda *_a, **_k: ("en", "fr"),
    )

    observed_updates: dict[str, object] = {}

    def _capture_language_updates(recording_id: str, *, settings=None, **kwargs):
        observed_updates["recording_id"] = recording_id
        observed_updates["kwargs"] = kwargs
        return True

    monkeypatch.setattr(
        worker_tasks,
        "set_recording_language_settings",
        _capture_language_updates,
    )

    async def _run_pipeline_with_callbacks(*_args, **kwargs):
        progress_callback = kwargs["progress_callback"]
        step_log_callback = kwargs["step_log_callback"]
        progress_callback("stt", 0.25)
        step_log_callback("synthetic step callback message")
        return None

    monkeypatch.setattr(worker_tasks, "run_pipeline", _run_pipeline_with_callbacks)

    final_status, quarantine_reason = worker_tasks._run_precheck_pipeline(
        recording_id="rec-step-log-1",
        settings=cfg,
        log_path=cfg.recordings_root / "rec-step-log-1" / "logs" / "step-precheck.log",
    )

    assert final_status == RECORDING_STATUS_READY
    assert quarantine_reason is None
    assert observed_updates == {
        "recording_id": "rec-step-log-1",
        "kwargs": {"language_auto": "en", "target_summary_language": "fr"},
    }


def test_process_job_rejects_unknown_job_type():
    with pytest.raises(ValueError, match="Unsupported job type"):
        worker_tasks.process_job("job-unknown-1", "rec-unknown-1", "unknown")


def test_process_job_legacy_paths_cover_ignore_and_fail_job_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_lightweight_process_job_env(monkeypatch, tmp_path)
    monkeypatch.setattr(worker_tasks, "get_recording", lambda *_a, **_k: {"status": RECORDING_STATUS_READY})

    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: False,
    )
    ignored = worker_tasks.process_job("job-legacy-ignored", "rec-legacy-ignored", JOB_TYPE_STT)
    assert ignored["status"] == "ignored"

    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        worker_tasks,
        "_append_step_log",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("readonly")),
    )
    monkeypatch.setattr(worker_tasks, "fail_job", lambda *_a, **_k: True)
    stale = worker_tasks.process_job("job-legacy-stale", "rec-legacy-stale", JOB_TYPE_STT)
    assert stale["status"] == "ignored"

    monkeypatch.setattr(worker_tasks, "_append_step_log", lambda *_a, **_k: None)
    monkeypatch.setattr(worker_tasks, "fail_job", lambda *_a, **_k: False)
    with pytest.raises(ValueError, match="Job not found: job-legacy-missing"):
        worker_tasks.process_job("job-legacy-missing", "rec-legacy-missing", JOB_TYPE_STT)


def test_process_job_legacy_queued_paths_cover_precheck_pending_and_restore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_lightweight_process_job_env(monkeypatch, tmp_path)
    monkeypatch.setattr(worker_tasks, "get_recording", lambda *_a, **_k: {"status": RECORDING_STATUS_QUEUED})
    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(worker_tasks, "fail_job", lambda *_a, **_k: True)
    monkeypatch.setattr(
        worker_tasks,
        "_append_step_log",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("readonly")),
    )

    monkeypatch.setattr(worker_tasks, "_has_queued_precheck_job", lambda *_a, **_k: True)
    pending = worker_tasks.process_job("job-legacy-pending", "rec-legacy-pending", JOB_TYPE_STT)
    assert pending["status"] == "ignored"

    monkeypatch.setattr(worker_tasks, "_has_queued_precheck_job", lambda *_a, **_k: False)
    monkeypatch.setattr(
        worker_tasks,
        "_restore_status_from_precheck_log",
        lambda *_a, **_k: (RECORDING_STATUS_READY, None),
    )
    monkeypatch.setattr(worker_tasks, "set_recording_status", lambda *_a, **_k: True)
    restored = worker_tasks.process_job("job-legacy-restored", "rec-legacy-restored", JOB_TYPE_STT)
    assert restored["status"] == "ignored"


def test_process_job_precheck_raises_when_recording_status_update_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_lightweight_process_job_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(worker_tasks, "_job_attempt", lambda *_a, **_k: 1)
    monkeypatch.setattr(worker_tasks, "set_recording_status", lambda *_a, **_k: False)
    monkeypatch.setattr(worker_tasks, "clear_recording_progress", lambda *_a, **_k: True)
    monkeypatch.setattr(worker_tasks, "_job_status", lambda *_a, **_k: JOB_STATUS_STARTED)
    monkeypatch.setattr(worker_tasks, "_is_retryable_exception", lambda _exc: False)
    monkeypatch.setattr(worker_tasks, "_record_failure", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="Recording not found"):
        worker_tasks.process_job("job-precheck-missing-recording", "rec-precheck-missing-recording", JOB_TYPE_PRECHECK)


def test_process_job_precheck_stale_paths_cover_race_conditions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_lightweight_process_job_env(monkeypatch, tmp_path)
    _patch_precheck_happy_path(monkeypatch)

    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_FAILED])
    monkeypatch.setattr(
        worker_tasks,
        "_job_status",
        lambda *_a, **_k: next(status_iter, JOB_STATUS_STARTED),
    )
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_status_if_current_in_and_job_started",
        lambda *_a, **_k: False,
    )

    ignored = worker_tasks.process_job("job-precheck-stale-1", "rec-precheck-stale-1", JOB_TYPE_PRECHECK)
    assert ignored["status"] == "ignored"

    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_STARTED])
    monkeypatch.setattr(
        worker_tasks,
        "_job_status",
        lambda *_a, **_k: next(status_iter, JOB_STATUS_STARTED),
    )
    monkeypatch.setattr(worker_tasks, "get_recording", lambda *_a, **_k: {"status": RECORDING_STATUS_READY})
    monkeypatch.setattr(
        worker_tasks,
        "fail_job_if_started",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("db race")),
    )
    ignored_with_error = worker_tasks.process_job(
        "job-precheck-stale-2",
        "rec-precheck-stale-2",
        JOB_TYPE_PRECHECK,
    )
    assert ignored_with_error["status"] == "ignored"

    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_STARTED])
    monkeypatch.setattr(
        worker_tasks,
        "_job_status",
        lambda *_a, **_k: next(status_iter, JOB_STATUS_STARTED),
    )
    monkeypatch.setattr(worker_tasks, "get_recording", lambda *_a, **_k: {})
    with pytest.raises(ValueError, match="Recording not found"):
        worker_tasks.process_job("job-precheck-stale-3", "rec-precheck-stale-3", JOB_TYPE_PRECHECK)


def test_process_job_precheck_finish_job_false_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_lightweight_process_job_env(monkeypatch, tmp_path)
    _patch_precheck_happy_path(monkeypatch)
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_status_if_current_in_and_job_started",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(worker_tasks, "finish_job_if_started", lambda *_a, **_k: False)

    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_FAILED])
    monkeypatch.setattr(
        worker_tasks,
        "_job_status",
        lambda *_a, **_k: next(status_iter, JOB_STATUS_STARTED),
    )
    ignored = worker_tasks.process_job("job-precheck-finish-1", "rec-precheck-finish-1", JOB_TYPE_PRECHECK)
    assert ignored["status"] == "ignored"

    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_STARTED])
    monkeypatch.setattr(
        worker_tasks,
        "_job_status",
        lambda *_a, **_k: next(status_iter, JOB_STATUS_STARTED),
    )
    with pytest.raises(ValueError, match="Job not found"):
        worker_tasks.process_job("job-precheck-finish-2", "rec-precheck-finish-2", JOB_TYPE_PRECHECK)


def test_process_job_precheck_exception_paths_cover_stale_and_retry_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _patch_lightweight_process_job_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(worker_tasks, "_job_attempt", lambda *_a, **_k: 1)
    monkeypatch.setattr(worker_tasks, "set_recording_status", lambda *_a, **_k: True)
    monkeypatch.setattr(worker_tasks, "_append_step_log", lambda *_a, **_k: None)
    monkeypatch.setattr(worker_tasks, "clear_recording_progress", lambda *_a, **_k: True)
    monkeypatch.setattr(
        worker_tasks,
        "_run_precheck_pipeline",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(worker_tasks, "_log_stale_inflight_execution", lambda **_kwargs: None)

    monkeypatch.setattr(worker_tasks, "_job_status", lambda *_a, **_k: JOB_STATUS_FAILED)
    stale = worker_tasks.process_job("job-precheck-exc-1", "rec-precheck-exc-1", JOB_TYPE_PRECHECK)
    assert stale["status"] == "ignored"

    start_iter = iter([True, False])
    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: next(start_iter),
    )
    monkeypatch.setattr(worker_tasks, "_job_status", lambda *_a, **_k: JOB_STATUS_STARTED)
    monkeypatch.setattr(worker_tasks, "_is_retryable_exception", lambda _exc: True)
    monkeypatch.setattr(worker_tasks, "_retry_delay_seconds", lambda *_a, **_k: 0)
    monkeypatch.setattr(worker_tasks, "_record_retry", lambda **_kwargs: True)
    monkeypatch.setattr(
        worker_tasks,
        "time",
        SimpleNamespace(sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("sleep not expected"))),
    )
    retry_no_sleep = worker_tasks.process_job("job-precheck-exc-2", "rec-precheck-exc-2", JOB_TYPE_PRECHECK)
    assert retry_no_sleep["status"] == "ignored"

    monkeypatch.setattr(
        worker_tasks,
        "_start_job_or_ignore_stale_execution",
        lambda **_kwargs: True,
    )
    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_FAILED])
    monkeypatch.setattr(worker_tasks, "_job_status", lambda *_a, **_k: next(status_iter))
    monkeypatch.setattr(worker_tasks, "_record_retry", lambda **_kwargs: False)
    retry_stale = worker_tasks.process_job("job-precheck-exc-3", "rec-precheck-exc-3", JOB_TYPE_PRECHECK)
    assert retry_stale["status"] == "ignored"

    monkeypatch.setattr(worker_tasks, "_job_status", lambda *_a, **_k: JOB_STATUS_STARTED)
    monkeypatch.setattr(worker_tasks, "_record_failure", lambda **_kwargs: None)
    with pytest.raises(RuntimeError, match="boom"):
        worker_tasks.process_job("job-precheck-exc-4", "rec-precheck-exc-4", JOB_TYPE_PRECHECK)
