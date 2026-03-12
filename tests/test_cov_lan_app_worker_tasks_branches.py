from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType, SimpleNamespace
import wave

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
    RECORDING_STATUS_STOPPED,
    RECORDING_STATUS_STOPPING,
)
from lan_app.db import create_recording, get_recording, init_db, set_recording_cancel_request
from lan_transcriber.llm_client import LLMEmptyContentError, LLMTruncatedResponseError
from lan_transcriber.pipeline import PrecheckResult


def _db_settings(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _write_pcm_wav(path: Path, *, sample_rate: int = 16000, duration_sec: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(sample_rate * duration_sec), 1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


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
    monkeypatch.setattr(worker_tasks, "set_recording_status_if_current_in", lambda *_a, **_k: True)
    monkeypatch.setattr(worker_tasks, "get_recording", lambda *_a, **_k: {})
    monkeypatch.setattr(worker_tasks, "_append_step_log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        worker_tasks,
        "_run_precheck_pipeline",
        lambda **_kwargs: worker_tasks.PipelineTerminalState(status=final_status),
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


def test_stop_request_helpers_build_metadata_and_log_output(tmp_path: Path) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-stop-helper-1",
        source="test",
        source_filename="stop.wav",
        status=RECORDING_STATUS_STOPPING,
        settings=cfg,
    )
    set_recording_cancel_request(
        "rec-stop-helper-1",
        requested_by="user",
        reason_code="user_stop",
        reason_text="Stop requested by user",
        settings=cfg,
    )

    stop_request = worker_tasks._recording_stop_request(  # noqa: SLF001
        "rec-stop-helper-1",
        settings=cfg,
    )
    assert stop_request is not None
    stop = worker_tasks.RecordingStopRequested(
        stage_name="llm_extract",
        checkpoint="after_llm_chunk",
        stop_request=stop_request,
        chunk_index="1",
        chunk_total=2,
    )

    metadata = worker_tasks._cancelled_stage_metadata(label="LLM Extract", stop=stop)  # noqa: SLF001
    assert metadata["cancelled_by_user"] is True
    assert metadata["cancel_chunk_index"] == "1"
    assert metadata["cancel_chunk_total"] == 2

    log_path = tmp_path / "logs" / "step.log"
    worker_tasks._log_stop_requested(log_path=log_path, stop=stop)  # noqa: SLF001
    assert "cancelled_by_user stage=llm_extract checkpoint=after_llm_chunk" in log_path.read_text(
        encoding="utf-8"
    )


def test_cancel_aware_helpers_stop_llm_requests_and_chunk_terminal_paths(
    tmp_path: Path,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-stop-helper-2",
        source="test",
        source_filename="stop2.wav",
        settings=cfg,
    )

    class _SyncClient:
        identity = "sync-client"

        def generate(self, *_args, **_kwargs):
            return {"status": "ok"}

    sync_client = worker_tasks._CancelAwareLLMClient(  # noqa: SLF001
        base_client=_SyncClient(),
        recording_id="rec-stop-helper-2",
        settings=cfg,
        stage_name="llm_extract",
    )
    assert sync_client.generate() == {"status": "ok"}
    assert sync_client.identity == "sync-client"

    class _BaseStore:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def list_states(self, *, chunk_group: str):
            self.calls.append(f"list:{chunk_group}")
            return []

        def upsert_state(self, **kwargs):
            self.calls.append(f"upsert:{kwargs['chunk_index']}")
            return None

        def mark_started(self, **kwargs):
            self.calls.append(f"started:{kwargs['chunk_index']}")
            return {"status": "running"}

        def mark_completed(self, **kwargs):
            self.calls.append(f"completed:{kwargs['chunk_index']}")
            return {"status": "completed"}

        def mark_failed(self, **kwargs):
            self.calls.append(f"failed:{kwargs['chunk_index']}")
            return {"status": "failed"}

        def mark_split(self, **kwargs):
            self.calls.append(f"split:{kwargs['chunk_index']}")
            return {"status": "split"}

        def mark_cancelled(self, **kwargs):
            self.calls.append(f"cancelled:{kwargs['chunk_index']}")
            return {"status": "cancelled"}

        def clear_states(self, *, chunk_group: str | None = None):
            self.calls.append(f"clear:{chunk_group}")
            return 0

    base_store = _BaseStore()
    store = worker_tasks._CancelAwareChunkStateStore(  # noqa: SLF001
        base_store=base_store,
        recording_id="rec-stop-helper-2",
        settings=cfg,
        stage_name="llm_extract",
    )

    assert store.list_states(chunk_group="extract") == []
    assert (
        store.upsert_state(
            chunk_group="extract",
            chunk_index="0",
            chunk_total=2,
            metadata={"chunk_id": "0"},
        )
        is None
    )
    assert store.mark_started(chunk_group="extract", chunk_index="1", chunk_total=2) == {
        "status": "running"
    }
    assert store.mark_completed(chunk_group="extract", chunk_index="1", chunk_total=2) == {
        "status": "completed"
    }
    assert store.mark_failed(chunk_group="extract", chunk_index="1b", chunk_total=2) == {
        "status": "failed"
    }
    assert store.mark_split(chunk_group="extract", chunk_index="1c", chunk_total=2) == {
        "status": "split"
    }
    assert store.clear_states(chunk_group="extract") == 0

    set_recording_cancel_request(
        "rec-stop-helper-2",
        requested_by="user",
        reason_code="user_stop",
        reason_text="Stop requested by user",
        settings=cfg,
    )
    with pytest.raises(worker_tasks.RecordingStopRequested):
        sync_client.generate()
    with pytest.raises(worker_tasks.RecordingStopRequested):
        store.mark_failed(chunk_group="extract", chunk_index="2", chunk_total=2)
    with pytest.raises(worker_tasks.RecordingStopRequested):
        store.mark_split(chunk_group="extract", chunk_index="3", chunk_total=3)
    assert "cancelled:2" in base_store.calls
    assert "cancelled:3" in base_store.calls


def test_stop_request_helpers_cover_optional_metadata_and_log_paths(tmp_path: Path) -> None:
    stop = worker_tasks.RecordingStopRequested(
        stage_name="llm_extract",
        checkpoint="before_llm_request",
        stop_request=worker_tasks._RecordingStopRequest(  # noqa: SLF001
            requested_at=None,
            requested_by=None,
            reason_code=None,
            reason_text=None,
        ),
    )

    metadata = worker_tasks._cancelled_stage_metadata(label="LLM Extract", stop=stop)  # noqa: SLF001
    assert metadata == {
        "label": "LLM Extract",
        "cancelled_by_user": True,
        "cancel_checkpoint": "before_llm_request",
        "cancel_requested_by": "user",
        "cancel_reason_code": "user_stop",
        "cancel_reason_text": "Cancelled by user",
    }

    log_path = tmp_path / "logs" / "step.log"
    worker_tasks._log_stop_requested(log_path=log_path, stop=stop)  # noqa: SLF001
    assert log_path.read_text(encoding="utf-8").strip().endswith(
        "cancelled_by_user stage=llm_extract checkpoint=before_llm_request requested_by=user"
    )


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
        exc=RuntimeError("max attempts exceeded"),
    )


def test_clean_language_value_rejects_blank_strings():
    assert worker_tasks._clean_language_value("  en  ") == "en"
    assert worker_tasks._clean_language_value("   ") is None


def test_build_pipeline_settings_propagates_runtime_llm_and_vad_settings(tmp_path: Path):
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
        llm_max_tokens=1536,
        llm_max_tokens_retry=3072,
        llm_chunk_max_chars=4096,
        llm_chunk_overlap_chars=256,
        llm_chunk_timeout_seconds=45.0,
        llm_chunk_split_min_chars=900,
        llm_chunk_split_max_depth=3,
        llm_long_transcript_threshold_chars=8192,
        llm_merge_max_tokens=2048,
        diarization_profile="meeting",
        diarization_min_speakers=3,
        diarization_max_speakers=5,
        diarization_dialog_retry_min_duration_seconds=12.0,
        diarization_dialog_retry_min_turns=4,
        diarization_merge_gap_seconds=0.6,
        diarization_min_turn_seconds=0.4,
        vad_method="pyannote",
    )

    pipeline_cfg = worker_tasks._build_pipeline_settings(cfg)

    assert pipeline_cfg.vad_method == "pyannote"
    assert pipeline_cfg.llm_model == cfg.llm_model
    assert pipeline_cfg.llm_max_tokens == 1536
    assert pipeline_cfg.llm_max_tokens_retry == 3072
    assert pipeline_cfg.llm_chunk_max_chars == 4096
    assert pipeline_cfg.llm_chunk_overlap_chars == 256
    assert pipeline_cfg.llm_chunk_timeout_seconds == 45.0
    assert pipeline_cfg.llm_chunk_split_min_chars == 900
    assert pipeline_cfg.llm_chunk_split_max_depth == 3
    assert pipeline_cfg.llm_long_transcript_threshold_chars == 8192
    assert pipeline_cfg.llm_merge_max_tokens == 2048
    assert pipeline_cfg.diarization_merge_gap_seconds == 0.6
    assert pipeline_cfg.diarization_min_turn_seconds == 0.4
    assert pipeline_cfg.diarization_profile == "meeting"
    assert pipeline_cfg.diarization_min_speakers == 3
    assert pipeline_cfg.diarization_max_speakers == 5
    assert pipeline_cfg.diarization_dialog_retry_min_duration_seconds == 12.0
    assert pipeline_cfg.diarization_dialog_retry_min_turns == 4


def test_db_chunk_state_store_wrapper_round_trips(tmp_path: Path) -> None:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    worker_tasks.init_db(cfg)
    from lan_app.db import create_recording

    create_recording(
        "rec-store-1",
        source="upload",
        source_filename="input.wav",
        settings=cfg,
    )
    store = worker_tasks._DbChunkStateStore("rec-store-1", cfg)  # noqa: SLF001

    assert store.list_states(chunk_group="extract") == []
    planned = store.upsert_state(
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
        status="planned",
        metadata={"order_path": [1], "text": "chunk one", "base_text": "chunk one"},
    )
    assert planned is not None
    started = store.mark_started(
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
    )
    assert started is not None
    completed = store.mark_completed(
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
    )
    assert completed is not None
    failed = store.mark_failed(
        chunk_group="extract",
        chunk_index="2",
        chunk_total=2,
        error_code="llm_chunk_timeout",
        error_text="boom",
        metadata={"order_path": [2], "text": "chunk two", "base_text": "chunk two"},
    )
    assert failed is not None
    split = store.mark_split(
        chunk_group="extract",
        chunk_index="3",
        chunk_total=3,
        error_code="llm_chunk_timeout",
        error_text="split",
        metadata={"order_path": [3], "text": "chunk three", "base_text": "chunk three"},
    )
    assert split is not None
    cancelled = store.mark_cancelled(
        chunk_group="extract",
        chunk_index="4",
        chunk_total=4,
        metadata={"order_path": [4], "text": "chunk four", "base_text": "chunk four"},
    )
    assert cancelled is not None
    assert len(store.list_states(chunk_group="extract")) == 4
    assert store.clear_states(chunk_group="extract") == 4


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
        "calendar_summary_context",
        lambda *_a, **_k: (None, []),
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-1", cfg) == (None, [])

    monkeypatch.setattr(
        worker_tasks,
        "calendar_summary_context",
        lambda *_a, **_k: (None, []),
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-2", cfg) == (None, [])

    monkeypatch.setattr(
        worker_tasks,
        "calendar_summary_context",
        lambda *_a, **_k: (None, []),
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-3", cfg) == (None, [])

    monkeypatch.setattr(
        worker_tasks,
        "calendar_summary_context",
        lambda *_a, **_k: ("Team Sync", []),
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-3b", cfg) == (
        "Team Sync",
        [],
    )

    monkeypatch.setattr(
        worker_tasks,
        "calendar_summary_context",
        lambda *_a, **_k: (None, []),
    )
    assert worker_tasks._load_calendar_summary_context("rec-cal-4", cfg) == (None, [])


def test_load_calendar_summary_context_returns_title_without_attendees_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _db_settings(tmp_path)
    monkeypatch.setattr(
        worker_tasks,
        "calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", []),
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


def test_pyannote_diariser_retries_when_dict_payload_typeerror_expects_path():
    class _Model:
        def __init__(self):
            self.calls: list[object] = []

        def __call__(self, payload: object):
            self.calls.append(payload)
            if isinstance(payload, dict):
                raise TypeError("expected str, bytes or os.PathLike object, not dict")
            return {"ok": "path"}

    model = _Model()
    diariser = worker_tasks._PyannoteDiariser(model)
    result = asyncio.run(diariser(Path("/tmp/input.wav")))
    assert result == {"ok": "path"}
    assert model.calls == [{"audio": "/tmp/input.wav"}, "/tmp/input.wav"]


def test_pyannote_diariser_passes_optional_speaker_hint_kwargs():
    class _Model:
        def __init__(self):
            self.calls: list[tuple[object, dict[str, object]]] = []

        def __call__(self, payload: object, **kwargs: object):
            self.calls.append((payload, dict(kwargs)))
            return {"ok": True}

    model = _Model()
    diariser = worker_tasks._PyannoteDiariser(
        model,
        min_speakers=2,
        max_speakers=4,
    )
    result = asyncio.run(diariser(Path("/tmp/input.wav")))
    assert result == {"ok": True}
    assert model.calls == [
        (
            {"audio": "/tmp/input.wav"},
            {"min_speakers": 2, "max_speakers": 4},
        )
    ]
    assert diariser.last_run_metadata == {
        "requested_profile": "auto",
        "diarization_profile": "auto",
        "initial_profile": "meeting",
        "selected_profile": None,
        "auto_profile_enabled": True,
        "override_reason": None,
        "initial_hints": {"min_speakers": 2, "max_speakers": 4},
        "retry_hints": None,
        "effective_hints": {"min_speakers": 2, "max_speakers": 4},
        "profile_selection": None,
        "dialog_retry_used": False,
        "speaker_count_before_retry": 0,
        "speaker_count_after_retry": 0,
        "effective_device": None,
    }


def test_pyannote_diariser_falls_back_when_lazy_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    warnings: list[str] = []
    attempts: list[str] = []

    def _raise_loader():
        attempts.append("load")
        raise RuntimeError("auth failed")

    monkeypatch.setattr(
        worker_tasks._logger,
        "warning",
        lambda template, *args: warnings.append(template % args),
    )
    diariser = worker_tasks._PyannoteDiariser(
        pipeline_loader=_raise_loader,
        fallback_duration_sec=12.0,
    )

    first_annotation = asyncio.run(diariser(Path("/tmp/auth-failed.wav")))
    second_annotation = asyncio.run(diariser(Path("/tmp/auth-failed.wav")))
    first_tracks = list(first_annotation.itertracks(yield_label=True))
    second_tracks = list(second_annotation.itertracks(yield_label=True))

    assert attempts == ["load"]
    assert diariser.mode == "fallback"
    assert diariser.last_run_metadata["effective_device"] == "cpu"
    assert first_tracks[0][0].end == 12.0
    assert second_tracks[0][1] == "S1"
    assert warnings == [
        "pyannote diarization load failed; using fallback diariser: RuntimeError: auth failed"
    ]


def test_pyannote_diariser_reraises_invalid_lazy_loader_state():
    diariser = worker_tasks._PyannoteDiariser(
        pipeline_loader=lambda: object(),
    )

    with pytest.raises(TypeError, match="pipeline_model must be a callable"):
        asyncio.run(diariser(Path("/tmp/not-callable.wav")))


def test_pyannote_diariser_reraises_forced_gpu_loader_device_errors():
    diariser = worker_tasks._PyannoteDiariser(
        pipeline_loader=lambda: (_ for _ in ()).throw(
            RuntimeError("Failed to move pyannote diarization pipeline to cuda:0: boom")
        ),
        requested_device="cuda:0",
        fallback_duration_sec=12.0,
    )

    with pytest.raises(RuntimeError, match="Failed to move pyannote diarization pipeline to cuda:0: boom"):
        asyncio.run(diariser(Path("/tmp/forced-gpu.wav")))


def test_pyannote_diariser_treats_gpu_alias_as_forced_gpu_device():
    diariser = worker_tasks._PyannoteDiariser(
        pipeline_loader=lambda: (_ for _ in ()).throw(
            RuntimeError("Requested diarization device cuda but CUDA is unavailable.")
        ),
        requested_device="gpu",
        fallback_duration_sec=12.0,
    )

    with pytest.raises(
        RuntimeError,
        match="Requested diarization device cuda but CUDA is unavailable.",
    ):
        asyncio.run(diariser(Path("/tmp/gpu-alias.wav")))


def test_pyannote_diariser_reraises_invalid_requested_device_errors_without_eager_validation():
    diariser = worker_tasks._PyannoteDiariser(
        pipeline_loader=lambda: (_ for _ in ()).throw(
            ValueError("Device must be one of auto, cpu, cuda, or cuda:<index>.")
        ),
        requested_device="metal",
        fallback_duration_sec=12.0,
    )

    with pytest.raises(
        ValueError,
        match="Device must be one of auto, cpu, cuda, or cuda:<index>.",
    ):
        asyncio.run(diariser(Path("/tmp/invalid-device.wav")))


def test_resolve_diarization_speaker_hints_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "auto")
    monkeypatch.delenv("LAN_DIARIZATION_MIN_SPEAKERS", raising=False)
    monkeypatch.delenv("LAN_DIARIZATION_MAX_SPEAKERS", raising=False)
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.profile == "auto"
    assert hints.initial_profile == "meeting"
    assert hints.auto_profile_enabled is True
    assert hints.override_reason is None
    assert hints.min_speakers == 2
    assert hints.max_speakers == 6

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "dialog")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.profile == "dialog"
    assert hints.initial_profile == "dialog"
    assert hints.auto_profile_enabled is False
    assert hints.override_reason == "profile_forced_dialog"
    assert hints.min_speakers == 2
    assert hints.max_speakers == 2

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "meeting")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_SPEAKERS", "3")
    monkeypatch.setenv("LAN_DIARIZATION_MAX_SPEAKERS", "4")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_DURATION_SECONDS", "9.5")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_TURNS", "5")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.profile == "meeting"
    assert hints.initial_profile == "meeting"
    assert hints.auto_profile_enabled is False
    assert hints.override_reason == "profile_forced_meeting"
    assert hints.min_speakers == 3
    assert hints.max_speakers == 4
    assert hints.dialog_retry_min_duration_seconds == 9.5
    assert hints.dialog_retry_min_turns == 5

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "auto")
    monkeypatch.delenv("LAN_DIARIZATION_MIN_SPEAKERS")
    monkeypatch.setenv("LAN_DIARIZATION_MAX_SPEAKERS", "2")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.initial_profile == "dialog"
    assert hints.auto_profile_enabled is False
    assert hints.override_reason == "explicit_speaker_hints"
    assert hints.min_speakers == 2
    assert hints.max_speakers == 2

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "dialog")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_SPEAKERS", "5")
    monkeypatch.setenv("LAN_DIARIZATION_MAX_SPEAKERS", "2")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.min_speakers is None
    assert hints.max_speakers is None

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "meeting")
    monkeypatch.delenv("LAN_DIARIZATION_MIN_SPEAKERS")
    monkeypatch.setenv("LAN_DIARIZATION_MAX_SPEAKERS", "1")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.min_speakers is None
    assert hints.max_speakers == 1

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "dialog")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_SPEAKERS", "5")
    monkeypatch.delenv("LAN_DIARIZATION_MAX_SPEAKERS")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.min_speakers == 5
    assert hints.max_speakers is None

    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "unknown")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_SPEAKERS", "abc")
    monkeypatch.setenv("LAN_DIARIZATION_MAX_SPEAKERS", "0")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_DURATION_SECONDS", "-1")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_TURNS", "0")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.profile == "auto"
    assert hints.initial_profile == "meeting"
    assert hints.auto_profile_enabled is True
    assert hints.min_speakers == 2
    assert hints.max_speakers == 6
    assert hints.dialog_retry_min_duration_seconds == 20.0
    assert hints.dialog_retry_min_turns == 6

    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_DURATION_SECONDS", "oops")
    hints = worker_tasks._resolve_diarization_speaker_hints()
    assert hints.dialog_retry_min_duration_seconds == 20.0


def test_pyannote_diariser_retry_dialog_forces_two_speakers_once():
    segment = SimpleNamespace(start=0.0, end=1.0)

    class _Model:
        def __init__(self):
            self.calls: list[tuple[object, dict[str, object]]] = []

        def __call__(self, payload: object, **kwargs: object):
            self.calls.append((payload, dict(kwargs)))

            class _Annotation:
                def itertracks(self, yield_label: bool = False):
                    if yield_label:
                        yield segment, "S1"
                        if kwargs.get("max_speakers") == 2:
                            yield segment, "S2"

            return _Annotation()

    diariser = worker_tasks._PyannoteDiariser(
        _Model(),
        profile="dialog",
        min_speakers=2,
        max_speakers=4,
        dialog_retry_min_duration_seconds=8.0,
        dialog_retry_min_turns=3,
    )

    asyncio.run(diariser(Path("/tmp/input.wav")))
    retried = asyncio.run(diariser.retry_dialog(Path("/tmp/input.wav")))

    assert len(list(retried.itertracks(yield_label=True))) == 2
    assert diariser.last_run_metadata == {
        "requested_profile": "dialog",
        "diarization_profile": "dialog",
        "initial_profile": "dialog",
        "selected_profile": "dialog",
        "auto_profile_enabled": False,
        "override_reason": None,
        "initial_hints": {"min_speakers": 2, "max_speakers": 4},
        "retry_hints": {"min_speakers": 2, "max_speakers": 2},
        "effective_hints": {"min_speakers": 2, "max_speakers": 2},
        "profile_selection": None,
        "dialog_retry_used": True,
        "speaker_count_before_retry": 1,
        "speaker_count_after_retry": 2,
        "effective_device": None,
    }


def test_write_diarization_status_artifact_writes_payload_and_ignores_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _db_settings(tmp_path)
    worker_tasks._write_diarization_status_artifact(
        recording_id="rec-diar-mode-1",
        mode="pyannote",
        reason=None,
        settings=cfg,
    )
    artifact_path = (
        cfg.recordings_root
        / "rec-diar-mode-1"
        / "derived"
        / "diarization_status.json"
    )
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {
        "mode": "pyannote",
        "degraded": False,
    }

    monkeypatch.setattr(
        worker_tasks,
        "atomic_write_json",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("readonly")),
    )
    worker_tasks._write_diarization_status_artifact(
        recording_id="rec-diar-mode-2",
        mode="fallback",
        reason="RuntimeError: boom",
        settings=cfg,
    )


def test_pyannote_diariser_rejects_non_callable_model():
    with pytest.raises(TypeError, match="pipeline_loader is required|pipeline_model must be a callable"):
        worker_tasks._PyannoteDiariser(None)

    diariser = worker_tasks._PyannoteDiariser(lambda *_a, **_k: {"ok": True}, initial_profile="weird")
    assert diariser.initial_profile == "meeting"


def test_pyannote_diariser_rejects_invalid_lazy_pipeline_states():
    with pytest.raises(TypeError, match="pipeline_model must be a callable"):
        worker_tasks._PyannoteDiariser(123)

    diariser = worker_tasks._PyannoteDiariser(lambda *_a, **_k: {"ok": True})
    diariser._pipeline_model = None
    diariser._pipeline_loader = None
    with pytest.raises(TypeError, match="pipeline_loader must be provided"):
        diariser._ensure_pipeline_model()

    lazy_diariser = worker_tasks._PyannoteDiariser(
        pipeline_loader=lambda: object(),
    )
    with pytest.raises(TypeError, match="pipeline_model must be a callable"):
        lazy_diariser._ensure_pipeline_model()


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


def test_build_diariser_applies_profile_defaults_from_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _Model:
        def __init__(self):
            self.calls: list[tuple[object, dict[str, object]]] = []

        def __call__(self, payload: object, **kwargs: object):
            self.calls.append((payload, dict(kwargs)))
            return {"ok": True}

    model = _Model()
    cfg = _db_settings(tmp_path)
    cfg.diarization_profile = "dialog"
    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = object()  # type: ignore[attr-defined]
    pyannote_pkg = ModuleType("pyannote")
    pyannote_pkg.audio = pyannote_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    monkeypatch.setattr(worker_tasks, "load_pyannote_pipeline", lambda **_kwargs: model)

    diariser = worker_tasks._build_diariser(duration_sec=30.0, settings=cfg)
    assert asyncio.run(diariser(Path("/tmp/dialog.wav"))) == {"ok": True}
    assert model.calls == [
        (
            {"audio": "/tmp/dialog.wav"},
            {"min_speakers": 2, "max_speakers": 2},
        )
    ]


def test_build_diariser_loads_pipeline_lazily_and_forwards_device_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    class _Model:
        def __init__(self):
            self.calls: list[tuple[object, dict[str, object]]] = []
            self._lan_effective_device = "cpu"

        def __call__(self, payload: object, **kwargs: object):
            self.calls.append((payload, dict(kwargs)))
            return {"ok": True}

    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = object()  # type: ignore[attr-defined]
    pyannote_pkg = ModuleType("pyannote")
    pyannote_pkg.audio = pyannote_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)

    model = _Model()
    load_calls: list[dict[str, object]] = []

    def _load_pyannote_pipeline(**kwargs):
        load_calls.append(dict(kwargs))
        return model

    monkeypatch.setattr(worker_tasks, "load_pyannote_pipeline", _load_pyannote_pipeline)

    diariser = worker_tasks._build_diariser(
        duration_sec=30.0,
        model_id="repo/test",
        settings=SimpleNamespace(
            diarization_profile="auto",
            diarization_min_speakers=None,
            diarization_max_speakers=None,
            diarization_dialog_retry_min_duration_seconds=1.0,
            diarization_dialog_retry_min_turns=2,
            diarization_device="cpu",
            gpu_scheduler_mode="auto",
        ),
    )

    assert load_calls == []
    result = asyncio.run(diariser(Path("/tmp/lazy.wav")))
    assert result == {"ok": True}
    assert load_calls == [
        {
            "model_id": "repo/test",
            "device": "cpu",
            "scheduler_mode": "auto",
        }
    ]
    assert diariser.last_run_metadata["effective_device"] == "cpu"


def test_log_gpu_execution_policy_writes_step_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "step-precheck.log"
    pipeline_settings = worker_tasks.PipelineSettings(
        recordings_root=tmp_path / "recordings",
        tmp_root=tmp_path / "tmp",
        speaker_db=tmp_path / "aliases.yaml",
        llm_model="model",
        asr_device="cpu",
        diarization_device="cuda:1",
        gpu_scheduler_mode="parallel",
    )
    monkeypatch.setattr(
        worker_tasks,
        "collect_cuda_runtime_facts",
        lambda: SimpleNamespace(
            is_available=True,
            device_count=2,
            visible_devices="0,1",
            torch_cuda_version="12.4",
        ),
    )

    worker_tasks._log_gpu_execution_policy(  # noqa: SLF001
        pipeline_settings=pipeline_settings,
        diariser=SimpleNamespace(mode="pyannote"),
        log_path=log_path,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "gpu policy" in log_text
    assert "asr_device=cpu" in log_text
    assert "diarization_device=cuda:1" in log_text


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
    _write_pcm_wav(raw_audio)

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

    terminal_state = worker_tasks._run_precheck_pipeline(
        recording_id="rec-step-log-1",
        settings=cfg,
        log_path=cfg.recordings_root / "rec-step-log-1" / "logs" / "step-precheck.log",
    )

    assert terminal_state.status == RECORDING_STATUS_READY
    assert terminal_state.quarantine_reason is None
    assert observed_updates == {
        "recording_id": "rec-step-log-1",
        "kwargs": {"language_auto": "en", "target_summary_language": "fr"},
    }


def test_run_precheck_pipeline_uses_sanitized_audio_for_precheck_and_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-sanitize-wire-1",
        source="test",
        source_filename="audio.mp3",
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-sanitize-wire-1" / "raw" / "audio.mp3"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    raw_audio.write_bytes(b"fake")

    observed_paths: dict[str, Path] = {}

    def _fake_sanitize(input_path: Path, output_path: Path, **_kwargs) -> Path:
        observed_paths["sanitize_input"] = input_path
        observed_paths["sanitize_output"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")
        return output_path

    monkeypatch.setattr(
        worker_tasks,
        "sanitize_audio_for_pipeline",
        _fake_sanitize,
    )

    def _fake_run_precheck(audio_path: Path, *_args, **_kwargs) -> PrecheckResult:
        observed_paths["precheck"] = audio_path
        return PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.6,
            quarantine_reason=None,
        )

    monkeypatch.setattr(worker_tasks, "run_precheck", _fake_run_precheck)
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
        lambda *_a, **_k: (None, None),
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_recording_asr_glossary",
        lambda *_a, **_k: {
            "entry_count": 1,
            "term_count": 2,
            "truncated": False,
            "initial_prompt": "Glossary: Sander; Sandia",
            "hotwords": "Sander, Sandia",
        },
    )

    async def _fake_run_pipeline(*_args, **kwargs):
        observed_paths["pipeline"] = kwargs["audio_path"]
        observed_paths["glossary"] = kwargs["asr_glossary"]
        return None

    monkeypatch.setattr(worker_tasks, "run_pipeline", _fake_run_pipeline)

    terminal_state = worker_tasks._run_precheck_pipeline(
        recording_id="rec-sanitize-wire-1",
        settings=cfg,
        log_path=cfg.recordings_root / "rec-sanitize-wire-1" / "logs" / "step-precheck.log",
    )

    assert terminal_state.status == RECORDING_STATUS_READY
    assert terminal_state.quarantine_reason is None
    assert observed_paths["sanitize_input"] == raw_audio
    assert observed_paths["sanitize_output"] == (
        cfg.recordings_root / "rec-sanitize-wire-1" / "derived" / "audio_sanitized.wav"
    )
    assert observed_paths["precheck"] == observed_paths["sanitize_output"]
    assert observed_paths["pipeline"] == observed_paths["sanitize_output"]
    assert observed_paths["glossary"]["term_count"] == 2

    sanitize_payload = json.loads(
        (
            cfg.recordings_root
            / "rec-sanitize-wire-1"
            / "derived"
            / "audio_sanitize.json"
        ).read_text(encoding="utf-8")
    )
    assert sanitize_payload == {
        "input_path": str(raw_audio),
        "output_path": str(observed_paths["sanitize_output"]),
        "ffmpeg_used": True,
        "sample_rate": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
    }
    step_log = (
        cfg.recordings_root
        / "rec-sanitize-wire-1"
        / "logs"
        / "step-precheck.log"
    ).read_text(encoding="utf-8")
    assert "asr glossary entries=1 terms=2 truncated=False" in step_log
    recording = get_recording("rec-sanitize-wire-1", settings=cfg)
    assert recording is not None
    assert recording["duration_sec"] == 30.0


def test_run_precheck_pipeline_survives_calendar_refresh_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-calendar-refresh-fail-1",
        source="test",
        source_filename="calendar-refresh.wav",
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-calendar-refresh-fail-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm_wav(raw_audio)

    monkeypatch.setattr(worker_tasks, "_resolve_raw_audio_path", lambda *_a, **_k: raw_audio)
    monkeypatch.setattr(worker_tasks, "sanitize_audio_for_pipeline", lambda src, _dst: src)
    monkeypatch.setattr(
        worker_tasks,
        "run_precheck",
        lambda *_a, **_k: PrecheckResult(
            duration_sec=20.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
    )
    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_calendar_match",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("calendar boom")),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_build_diariser",
        lambda *_a, **_k: worker_tasks._FallbackDiariser(20.0),
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_recording_asr_glossary",
        lambda *_a, **_k: {
            "entry_count": 0,
            "term_count": 0,
            "truncated": False,
            "initial_prompt": "",
            "hotwords": "",
        },
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_transcript_language_payload",
        lambda *_a, **_k: (None, None),
    )
    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_metrics",
        lambda *_a, **_k: {"participants": [], "meeting": {}},
    )
    monkeypatch.setattr(worker_tasks, "refresh_recording_routing", lambda *_a, **_k: {})
    monkeypatch.setattr(worker_tasks, "run_pipeline", lambda *_a, **_k: asyncio.sleep(0))

    state = worker_tasks._run_precheck_pipeline(
        recording_id="rec-calendar-refresh-fail-1",
        settings=cfg,
        log_path=cfg.recordings_root
        / "rec-calendar-refresh-fail-1"
        / "logs"
        / "step-precheck.log",
    )

    assert state.status == RECORDING_STATUS_READY
    recording = get_recording("rec-calendar-refresh-fail-1", settings=cfg)
    assert recording is not None
    assert recording["duration_sec"] == 20.0


def test_run_precheck_pipeline_marks_fallback_when_builder_returns_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-step-log-fallback-1",
        source="test",
        source_filename="audio.wav",
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-step-log-fallback-1" / "raw" / "audio.wav"
    raw_audio.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm_wav(raw_audio)

    monkeypatch.setattr(
        worker_tasks,
        "run_precheck",
        lambda *_a, **_k: PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_build_diariser",
        lambda *_a, **_k: worker_tasks._FallbackDiariser(30.0),
    )
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
        "run_pipeline",
        lambda *_a, **_k: asyncio.sleep(0),
    )

    terminal_state = worker_tasks._run_precheck_pipeline(
        recording_id="rec-step-log-fallback-1",
        settings=cfg,
        log_path=cfg.recordings_root
        / "rec-step-log-fallback-1"
        / "logs"
        / "step-precheck.log",
    )

    assert terminal_state.status == RECORDING_STATUS_READY
    assert terminal_state.quarantine_reason is None
    log_text = (
        cfg.recordings_root
        / "rec-step-log-fallback-1"
        / "logs"
        / "step-precheck.log"
    ).read_text(encoding="utf-8")
    assert "diariser mode=fallback reason=pyannote_unavailable" in log_text
    status_payload = json.loads(
        (
            cfg.recordings_root
            / "rec-step-log-fallback-1"
            / "derived"
            / "diarization_status.json"
        ).read_text(encoding="utf-8")
    )
    assert status_payload == {
        "mode": "fallback",
        "degraded": True,
        "reason": "pyannote_unavailable",
    }


def test_review_reason_helpers_cover_exception_and_routing_paths(tmp_path: Path) -> None:
    cfg = _db_settings(tmp_path)
    assert worker_tasks._load_json_dict(tmp_path / "missing.json") == {}  # noqa: SLF001
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert worker_tasks._load_json_dict(broken) == {}  # noqa: SLF001
    as_list = tmp_path / "list.json"
    as_list.write_text("[]", encoding="utf-8")
    assert worker_tasks._load_json_dict(as_list) == {}  # noqa: SLF001

    truncated = worker_tasks._review_reason_from_exception(  # noqa: SLF001
        LLMTruncatedResponseError(
            host="localhost",
            model="test-model",
            max_tokens=123,
            request_id="req-1",
            raw_response={},
        )
    )
    assert truncated[0] == "llm_truncated"

    empty = worker_tasks._review_reason_from_exception(  # noqa: SLF001
        LLMEmptyContentError(
            host="localhost",
            model="test-model",
            max_tokens=123,
            finish_reason="stop",
            request_id="req-2",
            raw_response={},
        )
    )
    assert empty[0] == "llm_empty_content"

    gpu_oom = worker_tasks._review_reason_from_exception(  # noqa: SLF001
        RuntimeError("CUDA out of memory while loading faster-whisper")
    )
    assert gpu_oom == (
        "gpu_oom",
        "The worker ran out of GPU memory while loading or running a heavy model; manual review required.",
    )

    generic = worker_tasks._review_reason_from_exception(RuntimeError("boom"))  # noqa: SLF001
    assert generic[0] == "job_retry_limit_reached"

    derived = cfg.recordings_root / "rec-review-reason-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "review": {
                    "required": True,
                    "reason_code": "multilingual_uncertain",
                    "reason_text": (
                        "Language detection conflicted across multilingual chunks."
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    assert worker_tasks._review_reason_from_routing(  # noqa: SLF001
        recording_id="rec-review-reason-1",
        settings=cfg,
        routing={"confidence": 0.25, "threshold": 0.5},
    ) == (
        "multilingual_uncertain",
        "Language detection conflicted across multilingual chunks.",
    )

    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "review": {
                    "required": True,
                    "reason_code": "multilingual_uncertain",
                }
            }
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text("{}", encoding="utf-8")
    (derived / "diarization_metadata.json").write_text("{}", encoding="utf-8")
    assert worker_tasks._review_reason_from_routing(  # noqa: SLF001
        recording_id="rec-review-reason-1",
        settings=cfg,
        routing={"confidence": 0.95, "threshold": 0.5},
    ) == (
        "multilingual_uncertain",
        "Multilingual transcript review is required.",
    )

    (derived / "transcript.json").write_text("{}", encoding="utf-8")
    (derived / "summary.json").write_text(
        json.dumps({"parse_error_reason": "json_object_not_found"}),
        encoding="utf-8",
    )
    assert worker_tasks._review_reason_from_routing(  # noqa: SLF001
        recording_id="rec-review-reason-1",
        settings=cfg,
        routing={"confidence": 0.25, "threshold": 0.5},
    ) == (
        "llm_empty_content",
        "LLM output was empty or invalid JSON; manual review required.",
    )

    (derived / "summary.json").write_text(
        json.dumps({"parse_error_reason": "summary_bullets: required"}),
        encoding="utf-8",
    )
    invalid_payload_reason = worker_tasks._review_reason_from_routing(  # noqa: SLF001
        recording_id="rec-review-reason-1",
        settings=cfg,
        routing={"confidence": 0.25, "threshold": 0.5},
    )
    assert invalid_payload_reason[0] == "llm_output_invalid"

    (derived / "summary.json").write_text("{}", encoding="utf-8")
    (derived / "diarization_metadata.json").write_text(
        json.dumps({"degraded": True}),
        encoding="utf-8",
    )
    assert worker_tasks._review_reason_from_routing(  # noqa: SLF001
        recording_id="rec-review-reason-1",
        settings=cfg,
        routing={"confidence": 0.25, "threshold": 0.5},
    ) == (
        "diarization_degraded",
        "Diarization ran in degraded mode; manual review required.",
    )

    (derived / "diarization_metadata.json").write_text(
        json.dumps({"degraded": False}),
        encoding="utf-8",
    )
    assert worker_tasks._review_reason_from_routing(  # noqa: SLF001
        recording_id="rec-review-reason-1",
        settings=cfg,
        routing={"confidence": 0.25, "threshold": 0.5},
    ) == (
        "routing_low_confidence",
        "Project routing confidence 0.25 is below threshold 0.50; manual review required.",
    )


def test_set_recording_duration_best_effort_swallows_write_errors(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _db_settings(tmp_path)
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_duration",
        lambda *_a, **_k: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )

    with caplog.at_level("WARNING"):
        worker_tasks._set_recording_duration_best_effort(  # noqa: SLF001
            "rec-duration-warning-1",
            duration_sec=5.0,
            settings=cfg,
        )

    assert "Failed to persist duration for recording rec-duration-warning-1" in caplog.text


def test_run_precheck_pipeline_skips_duration_persist_when_duration_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-no-duration-1",
        source="test",
        source_filename="audio.wav",
        settings=cfg,
    )

    raw_audio = cfg.recordings_root / "rec-no-duration-1" / "raw" / "audio.wav"
    _write_pcm_wav(raw_audio)

    monkeypatch.setattr(
        worker_tasks,
        "run_precheck",
        lambda *_a, **_k: PrecheckResult(
            duration_sec=None,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
    )
    monkeypatch.setattr(worker_tasks, "_build_diariser", lambda *_a, **_k: object())
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_duration",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("duration should not be persisted")
        ),
    )
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
        lambda *_a, **_k: (None, None),
    )
    monkeypatch.setattr(worker_tasks, "run_pipeline", lambda *_a, **_k: asyncio.sleep(0))

    terminal_state = worker_tasks._run_precheck_pipeline(
        recording_id="rec-no-duration-1",
        settings=cfg,
        log_path=cfg.recordings_root / "rec-no-duration-1" / "logs" / "step-precheck.log",
    )
    assert terminal_state.status == RECORDING_STATUS_READY


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
    monkeypatch.setattr(worker_tasks, "set_recording_status_if_current_in", lambda *_a, **_k: False)
    monkeypatch.setattr(worker_tasks, "get_recording", lambda *_a, **_k: {})
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
    monkeypatch.setattr(worker_tasks, "set_recording_status_if_current_in", lambda *_a, **_k: False)
    monkeypatch.setattr(worker_tasks, "get_recording", lambda *_a, **_k: {"status": RECORDING_STATUS_STOPPED})
    finish_calls: list[str | None] = []
    stale_details: list[str] = []
    monkeypatch.setattr(
        worker_tasks,
        "finish_job_if_started",
        lambda _job_id, *, settings=None, error=None: finish_calls.append(error) or True,
    )
    monkeypatch.setattr(
        worker_tasks,
        "_log_stale_inflight_execution",
        lambda **kwargs: stale_details.append(str(kwargs["detail"])),
    )
    ignored_stopped = worker_tasks.process_job(
        "job-precheck-stale-stop",
        "rec-precheck-stale-stop",
        JOB_TYPE_PRECHECK,
    )
    assert ignored_stopped["status"] == "ignored"
    assert finish_calls == ["cancelled_by_user"]
    assert stale_details == ["recording_status=Stopped"]

    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_STARTED])
    monkeypatch.setattr(
        worker_tasks,
        "_job_status",
        lambda *_a, **_k: next(status_iter, JOB_STATUS_STARTED),
    )
    monkeypatch.setattr(
        worker_tasks,
        "finish_job_if_started",
        lambda _job_id, settings=None, error=None: (_ for _ in ()).throw(RuntimeError("db race")),
    )
    stale_details.clear()
    ignored_stopped_with_error = worker_tasks.process_job(
        "job-precheck-stale-stop-error",
        "rec-precheck-stale-stop-error",
        JOB_TYPE_PRECHECK,
    )
    assert ignored_stopped_with_error["status"] == "ignored"
    assert stale_details == ["recording_status=Stopped"]

    status_iter = iter([JOB_STATUS_STARTED, JOB_STATUS_STARTED])
    monkeypatch.setattr(
        worker_tasks,
        "_job_status",
        lambda *_a, **_k: next(status_iter, JOB_STATUS_STARTED),
    )
    monkeypatch.setattr(worker_tasks, "set_recording_status_if_current_in", lambda *_a, **_k: True)
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_status_if_current_in_and_job_started",
        lambda *_a, **_k: False,
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
    monkeypatch.setattr(worker_tasks, "set_recording_status_if_current_in", lambda *_a, **_k: True)
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


def test_run_precheck_pipeline_stop_before_finalization_when_no_stages_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording("rec-stop-finalize-empty", source="test", source_filename="empty.wav", settings=cfg)
    monkeypatch.setattr(worker_tasks, "PIPELINE_STAGE_DEFINITIONS", ())

    stop = worker_tasks.RecordingStopRequested(
        stage_name="finalize",
        checkpoint="before_finalization",
        stop_request=worker_tasks._RecordingStopRequest(  # noqa: SLF001
            requested_at=None,
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
        ),
    )

    def _raise_if_stop_requested(**kwargs):
        if kwargs["stage_name"] == "finalize":
            raise stop

    monkeypatch.setattr(worker_tasks, "_raise_if_stop_requested", _raise_if_stop_requested)

    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-stop-finalize-empty",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-stop-finalize-empty", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == "Stopped"


def test_run_precheck_pipeline_marks_started_stage_cancelled_on_stop_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording("rec-stop-cancelled-stage", source="test", source_filename="cancel.wav", settings=cfg)
    stage = worker_tasks.PIPELINE_STAGE_DEFINITIONS[0]
    monkeypatch.setattr(worker_tasks, "PIPELINE_STAGE_DEFINITIONS", (stage,))
    monkeypatch.setattr(worker_tasks, "_raise_if_stop_requested", lambda **_kwargs: None)

    stop = worker_tasks.RecordingStopRequested(
        stage_name=stage.name,
        checkpoint="after_llm_chunk",
        stop_request=worker_tasks._RecordingStopRequest(  # noqa: SLF001
            requested_at="2026-01-10T10:06:00Z",
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
        ),
        chunk_index="1",
        chunk_total=2,
    )
    monkeypatch.setitem(
        worker_tasks._PIPELINE_STAGE_RUNNERS,
        stage.name,
        lambda _ctx: (_ for _ in ()).throw(stop),
    )

    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-stop-cancelled-stage",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-stop-cancelled-stage", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == "Stopped"
    rows = worker_tasks.list_recording_pipeline_stages("rec-stop-cancelled-stage", settings=cfg)
    assert rows[0]["status"] == "cancelled"


def test_run_precheck_pipeline_stop_before_stage_skips_cancelled_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording("rec-stop-before-stage", source="test", source_filename="before.wav", settings=cfg)
    stage = worker_tasks.PIPELINE_STAGE_DEFINITIONS[0]
    monkeypatch.setattr(worker_tasks, "PIPELINE_STAGE_DEFINITIONS", (stage,))

    stop = worker_tasks.RecordingStopRequested(
        stage_name=stage.name,
        checkpoint="before_stage",
        stop_request=worker_tasks._RecordingStopRequest(  # noqa: SLF001
            requested_at=None,
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
        ),
    )

    def _raise_if_stop_requested(**kwargs):
        if kwargs["stage_name"] == stage.name and kwargs["checkpoint"] == "before_stage":
            raise stop

    monkeypatch.setattr(worker_tasks, "_raise_if_stop_requested", _raise_if_stop_requested)

    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-stop-before-stage",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-stop-before-stage", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == "Stopped"
    assert worker_tasks.list_recording_pipeline_stages("rec-stop-before-stage", settings=cfg) == []


def test_run_precheck_pipeline_stops_after_skipped_stage_when_stop_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording("rec-stop-skipped-stage", source="test", source_filename="skip.wav", settings=cfg)
    stage = worker_tasks.PIPELINE_STAGE_DEFINITIONS[0]
    monkeypatch.setattr(worker_tasks, "PIPELINE_STAGE_DEFINITIONS", (stage,))

    def _runner(_ctx):
        set_recording_cancel_request(
            "rec-stop-skipped-stage",
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
            settings=cfg,
        )
        return worker_tasks._StageResult(status="skipped", metadata={"skip_reason": "manual"})  # noqa: SLF001

    monkeypatch.setitem(worker_tasks._PIPELINE_STAGE_RUNNERS, stage.name, _runner)

    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-stop-skipped-stage",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-stop-skipped-stage", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == "Stopped"


def test_run_precheck_pipeline_stop_during_final_finalize_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording("rec-stop-finalize-tail", source="test", source_filename="final.wav", settings=cfg)
    stage = worker_tasks.PIPELINE_STAGE_DEFINITIONS[0]
    monkeypatch.setattr(worker_tasks, "PIPELINE_STAGE_DEFINITIONS", (stage,))
    monkeypatch.setitem(
        worker_tasks._PIPELINE_STAGE_RUNNERS,
        stage.name,
        lambda _ctx: worker_tasks._StageResult(status="completed"),
    )

    stop = worker_tasks.RecordingStopRequested(
        stage_name="finalize",
        checkpoint="before_finalization",
        stop_request=worker_tasks._RecordingStopRequest(  # noqa: SLF001
            requested_at=None,
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
        ),
    )

    def _raise_if_stop_requested(**kwargs):
        if kwargs["stage_name"] == "finalize":
            raise stop

    monkeypatch.setattr(worker_tasks, "_raise_if_stop_requested", _raise_if_stop_requested)

    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-stop-finalize-tail",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-stop-finalize-tail", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == "Stopped"


def test_process_job_converts_completed_terminal_state_to_stopped_when_stop_request_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_lightweight_process_job_env(monkeypatch, tmp_path)
    _patch_precheck_happy_path(monkeypatch, final_status=RECORDING_STATUS_READY)

    stop_request = worker_tasks._RecordingStopRequest(  # noqa: SLF001
        requested_at=None,
        requested_by="user",
        reason_code="user_stop",
        reason_text="Stop requested by user",
    )
    monkeypatch.setattr(worker_tasks, "_recording_stop_request", lambda *_a, **_k: stop_request)

    captured: dict[str, str | None] = {}

    def _set_terminal_status(_recording_id, status, **_kwargs):
        captured["status"] = status
        return True

    def _finish_job(_job_id, *, settings=None, error=None):
        captured["error"] = error
        return True

    monkeypatch.setattr(
        worker_tasks,
        "set_recording_status_if_current_in_and_job_started",
        _set_terminal_status,
    )
    monkeypatch.setattr(worker_tasks, "finish_job_if_started", _finish_job)
    monkeypatch.setattr(
        worker_tasks,
        "_stop_terminal_state",
        lambda **_kwargs: worker_tasks.PipelineTerminalState(status="Stopped"),
    )
    monkeypatch.setattr(worker_tasks, "_job_status", lambda *_a, **_k: JOB_STATUS_STARTED)

    result = worker_tasks.process_job(
        "job-precheck-stop-after-pipeline",
        "rec-precheck-stop-after-pipeline",
        JOB_TYPE_PRECHECK,
    )
    assert result["status"] == "ok"
    assert captured == {
        "status": "Stopped",
        "error": "cancelled_by_user",
    }
