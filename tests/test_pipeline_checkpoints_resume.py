from __future__ import annotations

import json
from pathlib import Path
import wave

from fastapi.testclient import TestClient
import pytest

from lan_app import api
from lan_app import jobs as jobs_module
from lan_app import ui_routes
from lan_app import worker_tasks
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_QUEUED,
    JOB_TYPE_PRECHECK,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_READY,
)
from lan_app.db import (
    clear_recording_pipeline_stages,
    create_job,
    create_recording,
    get_recording,
    init_db,
    list_recording_pipeline_stages,
    mark_recording_pipeline_stage_cancelled,
    mark_recording_pipeline_stage_completed,
    mark_recording_pipeline_stage_failed,
    mark_recording_pipeline_stage_started,
    mark_recording_pipeline_stage_skipped,
    upsert_recording_pipeline_stage,
)
from lan_app.pipeline_stages import (
    PIPELINE_STAGE_STATUS_SKIPPED,
    stage_artifact_paths,
    validate_stage_artifacts,
)
from lan_transcriber.pipeline import PrecheckResult
from lan_transcriber.pipeline_steps.diarization_quality import SpeakerTurnSmoothingResult


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    cfg.max_job_attempts = 3
    return cfg


def _write_pcm_wav(path: Path, *, sample_rate: int = 16000, duration_sec: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(sample_rate * duration_sec), 1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _valid_llm_json() -> str:
    return json.dumps(
        {
            "topic": "Weekly Sync",
            "summary_bullets": ["Discussed the work plan."],
            "decisions": ["Proceed with stage checkpoints."],
            "action_items": [{"task": "Write tests", "owner": "Ada", "deadline": None}],
            "emotional_summary": "Focused and calm.",
            "questions": {
                "total_count": 0,
                "types": {
                    "open": 0,
                    "yes_no": 0,
                    "clarification": 0,
                    "status": 0,
                    "decision_seeking": 0,
                },
                "extracted": [],
            },
        }
    )


def _patch_pipeline_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg: AppSettings,
    llm_outcomes: list[object] | None = None,
    quarantine_reason: str | None = None,
    no_speech: bool = False,
    routing_status: str = RECORDING_STATUS_READY,
    routing_confidence: float = 0.91,
) -> dict[str, int]:
    counters = {"asr": 0, "diarization": 0, "llm": 0}
    llm_values = iter(llm_outcomes or [_valid_llm_json()])

    monkeypatch.setattr(worker_tasks, "AppSettings", lambda: cfg)
    monkeypatch.setattr(worker_tasks, "run_precheck", lambda *_a, **_k: PrecheckResult(42.0, 0.8, quarantine_reason))
    monkeypatch.setattr(worker_tasks, "refresh_recording_calendar_match", lambda *_a, **_k: None)
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", ["Ada", "Bob"]),
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_recording_asr_glossary",
        lambda *_a, **_k: {
            "entry_count": 1,
            "term_count": 2,
            "truncated": False,
            "initial_prompt": "checkpoint",
            "hotwords": "resume",
        },
    )
    monkeypatch.setattr(worker_tasks, "_log_gpu_execution_policy", lambda **_kwargs: None)
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_build_whisperx_transcriber",
        lambda **_kwargs: (lambda *_args, **_kw: ([], {})),
    )

    def _fake_run_language_aware_asr(*_args, **_kwargs):
        counters["asr"] += 1
        if no_speech:
            segments = []
        else:
            segments = [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Discussed the work plan.",
                    "language": "en",
                }
            ]
        return (
            segments,
            {"language": "en"},
            {
                "configured_mode": "auto",
                "selected_mode": "single_language",
                "selection_reason": "test",
                "used_multilingual_path": False,
                "chunks": [],
            },
        )

    monkeypatch.setattr(worker_tasks, "run_language_aware_asr", _fake_run_language_aware_asr)

    class _DummyDiariser:
        def __init__(self) -> None:
            self.mode = "pyannote"
            self.last_run_metadata = {
                "diarization_profile": "auto",
                "requested_profile": "auto",
                "initial_profile": "meeting",
                "selected_profile": "meeting",
                "auto_profile_enabled": False,
                "override_reason": None,
                "initial_hints": {"min_speakers": 2, "max_speakers": 6},
                "effective_hints": {"min_speakers": 2, "max_speakers": 6},
                "profile_selection": {
                    "selected_profile": "meeting",
                    "selected_result": "initial_pass",
                    "initial_metrics": {},
                },
                "dialog_retry_used": False,
                "speaker_count_before_retry": 1,
                "speaker_count_after_retry": 1,
                "effective_device": "cpu",
                "scheduler_mode": "sequential",
                "scheduler_reason": "test",
            }

        async def __call__(self, _audio_path: Path) -> object:
            counters["diarization"] += 1
            return object()

    monkeypatch.setattr(worker_tasks, "_build_diariser", lambda *_a, **_k: _DummyDiariser())

    async def _fake_retry_dialog_diarization(**kwargs):
        return kwargs["diarization"]

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_maybe_retry_dialog_diarization",
        _fake_retry_dialog_diarization,
    )
    monkeypatch.setattr(
        worker_tasks,
        "_diarization_segments",
        lambda _annotation: [{"speaker": "S1", "start": 0.0, "end": 1.0}],
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_speaker_turns",
        lambda *_a, **_k: (
            []
            if no_speech
            else [{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "Discussed the work plan."}]
        ),
    )
    monkeypatch.setattr(
        worker_tasks,
        "smooth_speaker_turns",
        lambda turns, **_kwargs: SpeakerTurnSmoothingResult(
            turns=list(turns),
            adjacent_merges=0,
            micro_turn_absorptions=0,
            turn_count_before=len(turns),
            turn_count_after=len(turns),
            speaker_count_before=1 if turns else 0,
            speaker_count_after=1 if turns else 0,
        ),
    )
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_sentiment_score", lambda _text: 7)
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_use_chunked_llm", lambda *_a, **_k: False)
    monkeypatch.setattr(worker_tasks, "build_structured_summary_prompts", lambda *_a, **_k: ("sys", "user"))

    async def _fake_generate_llm_message(*_args, **_kwargs):
        counters["llm"] += 1
        outcome = next(llm_values)
        if isinstance(outcome, Exception):
            raise outcome
        return {"content": str(outcome)}

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_generate_llm_message",
        _fake_generate_llm_message,
    )

    def _fake_export(request):
        request.snippets_dir.mkdir(parents=True, exist_ok=True)
        clip_path = request.snippets_dir / "s1-1.wav"
        clip_path.write_bytes(b"clip")
        manifest_path = request.snippets_dir.parent / "snippets_manifest.json"
        manifest = {
            "version": 1,
            "source_kind": "segment",
            "degraded_diarization": False,
            "pad_seconds": request.pad_seconds,
            "max_clip_duration_seconds": request.max_clip_duration_sec,
            "min_clip_duration_seconds": request.min_clip_duration_sec,
            "max_snippets_per_speaker": request.max_snippets_per_speaker,
            "speakers": {"S1": [{"path": clip_path.name}]},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return [clip_path]

    monkeypatch.setattr(worker_tasks, "export_speaker_snippets", _fake_export)
    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_metrics",
        lambda *_a, **_k: {"participants": ["S1"], "meeting": {"total_interruptions": 0}},
    )
    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_routing",
        lambda *_a, **_k: {
            "suggested_project_id": 1,
            "suggested_project_name": "Alpha",
            "confidence": routing_confidence,
            "threshold": 0.5,
            "rationale": ["test"],
            "status_after_routing": routing_status,
            "auto_selected": routing_status == RECORDING_STATUS_READY,
        },
    )
    return counters


def test_process_job_resumes_from_failed_llm_stage_and_persists_stage_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-resume-1", source="test", source_filename="resume.wav", settings=cfg)
    _write_pcm_wav(cfg.recordings_root / "rec-resume-1" / "raw" / "audio.wav")
    create_job(
        job_id="job-resume-1",
        recording_id="rec-resume-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_QUEUED,
        settings=cfg,
    )
    counters = _patch_pipeline_dependencies(
        monkeypatch,
        cfg=cfg,
        llm_outcomes=[RuntimeError("llm exploded"), _valid_llm_json()],
    )

    result = worker_tasks.process_job("job-resume-1", "rec-resume-1", JOB_TYPE_PRECHECK)

    assert result["status"] == "ok"
    assert counters["asr"] == 1
    assert counters["diarization"] == 1
    assert counters["llm"] == 2

    stages = {row["stage_name"]: row for row in list_recording_pipeline_stages("rec-resume-1", settings=cfg)}
    assert stages["asr"]["attempt"] == 1
    assert stages["diarization"]["attempt"] == 1
    assert stages["llm_extract"]["attempt"] == 2
    assert stages["routing"]["status"] == "completed"
    assert get_recording("rec-resume-1", settings=cfg)["status"] == RECORDING_STATUS_READY

    log_text = (cfg.recordings_root / "rec-resume-1" / "logs" / "step-precheck.log").read_text(encoding="utf-8")
    assert "stage resumed: llm_extract" in log_text


def test_run_precheck_pipeline_quarantine_skips_downstream_stages_and_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-quarantine-1", source="test", source_filename="quarantine.wav", settings=cfg)
    _write_pcm_wav(cfg.recordings_root / "rec-quarantine-1" / "raw" / "audio.wav")
    _patch_pipeline_dependencies(
        monkeypatch,
        cfg=cfg,
        quarantine_reason="manual_hold",
    )

    outcome = worker_tasks._run_precheck_pipeline(
        recording_id="rec-quarantine-1",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-quarantine-1", JOB_TYPE_PRECHECK, cfg),
    )

    assert outcome.status == RECORDING_STATUS_QUARANTINE
    summary_payload = json.loads(
        (cfg.recordings_root / "rec-quarantine-1" / "derived" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["status"] == "quarantined"
    stages = {row["stage_name"]: row for row in list_recording_pipeline_stages("rec-quarantine-1", settings=cfg)}
    assert stages["asr"]["status"] == "skipped"
    assert stages["routing"]["status"] == "skipped"


def test_run_precheck_pipeline_no_speech_skips_llm_and_writes_no_speech_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-no-speech-1", source="test", source_filename="silent.wav", settings=cfg)
    _write_pcm_wav(cfg.recordings_root / "rec-no-speech-1" / "raw" / "audio.wav")
    _patch_pipeline_dependencies(
        monkeypatch,
        cfg=cfg,
        no_speech=True,
    )

    outcome = worker_tasks._run_precheck_pipeline(
        recording_id="rec-no-speech-1",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-no-speech-1", JOB_TYPE_PRECHECK, cfg),
    )

    assert outcome.status == RECORDING_STATUS_READY
    summary_payload = json.loads(
        (cfg.recordings_root / "rec-no-speech-1" / "derived" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["status"] == "no_speech"
    stages = {row["stage_name"]: row for row in list_recording_pipeline_stages("rec-no-speech-1", settings=cfg)}
    assert stages["llm_extract"]["status"] == "skipped"


def test_pipeline_stage_db_helpers_and_ui_display_paths(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-stage-db-1", source="test", source_filename="stage.wav", settings=cfg)
    artifacts = stage_artifact_paths("rec-stage-db-1", settings=cfg)
    ui_routes._settings = cfg  # noqa: SLF001

    upserted = upsert_recording_pipeline_stage(
        "rec-stage-db-1",
        stage_name="sanitize_audio",
        status="pending",
        metadata={"label": "Sanitize Audio"},
        settings=cfg,
    )
    assert upserted is not None
    started = mark_recording_pipeline_stage_started(
        "rec-stage-db-1",
        stage_name="sanitize_audio",
        metadata={"label": "Sanitize Audio"},
        settings=cfg,
    )
    assert started["attempt"] == 1
    completed = mark_recording_pipeline_stage_completed(
        "rec-stage-db-1",
        stage_name="sanitize_audio",
        metadata={"label": "Sanitize Audio"},
        settings=cfg,
    )
    assert completed["status"] == "completed"
    failed = mark_recording_pipeline_stage_failed(
        "rec-stage-db-1",
        stage_name="precheck",
        error_code="RuntimeError",
        error_text="boom",
        settings=cfg,
    )
    assert failed["status"] == "failed"
    skipped = mark_recording_pipeline_stage_skipped(
        "rec-stage-db-1",
        stage_name="routing",
        metadata={"skip_reason": "manual_test"},
        settings=cfg,
    )
    assert skipped["status"] == PIPELINE_STAGE_STATUS_SKIPPED
    cancelled = mark_recording_pipeline_stage_cancelled(
        "rec-stage-db-1",
        stage_name="metrics",
        metadata={"skip_reason": "cancelled"},
        settings=cfg,
    )
    assert cancelled["status"] == "cancelled"

    artifacts["sanitize_audio"][0].parent.mkdir(parents=True, exist_ok=True)
    artifacts["sanitize_audio"][0].write_bytes(b"wav")
    artifacts["sanitize_audio"][1].write_text(
        json.dumps({"output_path": str(artifacts["sanitize_audio"][0])}),
        encoding="utf-8",
    )
    ok, reason = validate_stage_artifacts(
        "rec-stage-db-1",
        stage_name="sanitize_audio",
        status="completed",
        metadata={},
        settings=cfg,
    )
    assert ok is True
    assert reason is None
    ok, reason = validate_stage_artifacts(
        "rec-stage-db-1",
        stage_name="routing",
        status="skipped",
        metadata={"skip_reason": "manual_test"},
        settings=cfg,
    )
    assert ok is True
    assert reason is None
    ok, reason = validate_stage_artifacts(
        "rec-stage-db-1",
        stage_name="llm_extract",
        status="completed",
        metadata={},
        settings=cfg,
    )
    assert ok is False
    assert reason == "llm_extract missing artifact summary.json"

    display_rows = ui_routes._pipeline_stage_rows_for_display("rec-stage-db-1")  # noqa: SLF001
    assert [row["stage_name"] for row in display_rows] == [
        "sanitize_audio",
        "precheck",
        "metrics",
        "routing",
    ]
    assert display_rows[0]["stage_label"] == "Sanitize Audio"

    cleared = clear_recording_pipeline_stages(
        "rec-stage-db-1",
        settings=cfg,
        from_stage="metrics",
    )
    assert cleared == 2


def test_enqueue_recording_job_clears_existing_pipeline_stage_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-requeue-1", source="test", source_filename="requeue.wav", settings=cfg)
    mark_recording_pipeline_stage_started(
        "rec-requeue-1",
        stage_name="sanitize_audio",
        settings=cfg,
    )
    mark_recording_pipeline_stage_completed(
        "rec-requeue-1",
        stage_name="sanitize_audio",
        settings=cfg,
    )

    class _Queue:
        def enqueue(self, *_args, **_kwargs) -> None:
            return None

    monkeypatch.setattr(jobs_module, "get_queue", lambda _cfg=None: _Queue())

    job = jobs_module.enqueue_recording_job(
        "rec-requeue-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )

    assert job.job_type == JOB_TYPE_PRECHECK
    assert list_recording_pipeline_stages("rec-requeue-1", settings=cfg) == []


def test_ui_recording_detail_renders_pipeline_stages_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-ui-stage-1", source="test", source_filename="ui.wav", settings=cfg)
    mark_recording_pipeline_stage_started(
        "rec-ui-stage-1",
        stage_name="sanitize_audio",
        settings=cfg,
    )
    mark_recording_pipeline_stage_completed(
        "rec-ui-stage-1",
        stage_name="sanitize_audio",
        settings=cfg,
    )

    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    client = TestClient(api.app)

    response = client.get("/recordings/rec-ui-stage-1")

    assert response.status_code == 200
    assert "Pipeline Stages" in response.text
    assert "Sanitize Audio" in response.text
