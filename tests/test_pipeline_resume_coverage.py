from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from lan_app import db as db_module
from lan_app import pipeline_stages
from lan_app import worker_tasks
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_TYPE_PRECHECK,
    RECORDING_STATUS_NEEDS_REVIEW,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_READY,
    RECORDING_STATUS_STOPPED,
    RECORDING_STATUS_STOPPING,
)
from lan_app.db import (
    create_recording,
    init_db,
    list_recording_llm_chunk_states,
    list_recording_pipeline_stages,
    mark_recording_pipeline_stage_completed,
    mark_recording_pipeline_stage_skipped,
    set_recording_cancel_request,
)
from lan_transcriber.pipeline import PrecheckResult
from lan_transcriber.pipeline_steps import orchestrator as pipeline_orchestrator
from lan_transcriber.pipeline_steps.diarization_quality import SpeakerTurnSmoothingResult


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.llm_model = "test-model"
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    cfg.max_job_attempts = 3
    return cfg


def _write_pcm_wav(path: Path, *, sample_rate: int = 16000, duration_sec: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(sample_rate * duration_sec), 1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _new_ctx(
    tmp_path: Path,
    recording_id: str,
    *,
    create_raw_audio: bool = False,
) -> tuple[AppSettings, worker_tasks._PipelineExecutionContext]:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        recording_id,
        source="test",
        source_filename=f"{recording_id}.wav",
        settings=cfg,
    )
    if create_raw_audio:
        _write_pcm_wav(cfg.recordings_root / recording_id / "raw" / "audio.wav")
    ctx = worker_tasks._new_pipeline_context(
        recording_id=recording_id,
        settings=cfg,
        log_path=worker_tasks._step_log_path(recording_id, JOB_TYPE_PRECHECK, cfg),
    )
    ctx.pipeline_settings.speaker_db = tmp_path / "aliases.json"
    return cfg, ctx


def test_db_pipeline_stage_internal_guard_paths(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording("rec-db-guard", source="test", source_filename="db.wav", settings=cfg)

    assert db_module._normalise_pipeline_stage_metadata({"bad": {1}}) == {}  # noqa: SLF001
    assert db_module._duration_ms_between("bad-start", "bad-finish") is None  # noqa: SLF001

    with pytest.raises(ValueError, match="Terminal stage status must not be running"):
        db_module._mark_recording_pipeline_stage_terminal(  # noqa: SLF001
            "rec-db-guard",
            stage_name="precheck",
            status="running",
            settings=cfg,
        )


def test_pipeline_stage_validation_and_artifact_edge_cases(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    assert pipeline_stages.stage_progress("snippet_export") == 0.84
    assert pipeline_stages.stage_order("snippet_export") == 75
    assert pipeline_stages.stage_order("llm_extract") == 80
    assert pipeline_stages.stage_order("export_artifacts") == 90
    assert pipeline_stages.stage_progress("metrics") == 0.98
    assert pipeline_stages.stage_label("snippet_export") == "Snippet Export"
    assert pipeline_stages.stage_label("routing") == "Routing"

    with pytest.raises(ValueError, match="Unsupported pipeline stage"):
        pipeline_stages.validate_pipeline_stage_name("bad-stage")
    with pytest.raises(ValueError, match="Unsupported pipeline stage status"):
        pipeline_stages.validate_pipeline_stage_status("bad-status")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert pipeline_stages._load_json(bad_json) is None  # noqa: SLF001

    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"wav")
    assert pipeline_stages._validate_path_payload(wav_path) is True  # noqa: SLF001

    ok, reason = pipeline_stages.validate_stage_artifacts(
        "rec-stage-edge",
        stage_name="routing",
        status="skipped",
        metadata={},
        settings=cfg,
    )
    assert ok is False
    assert reason == "routing missing skip_reason metadata"

    ok, reason = pipeline_stages.validate_stage_artifacts(
        "rec-stage-edge",
        stage_name="sanitize_audio",
        status="completed",
        metadata={"raw_audio_missing": True},
        settings=cfg,
    )
    assert ok is True
    assert reason is None

    artifacts = pipeline_stages.stage_artifact_paths("rec-stage-edge", settings=cfg)
    assert artifacts["snippet_export"][0].name == "snippets_manifest.json"
    assert artifacts["snippet_export"][0] not in artifacts["export_artifacts"]
    artifacts["sanitize_audio"][1].parent.mkdir(parents=True, exist_ok=True)
    artifacts["sanitize_audio"][1].write_text(
        json.dumps({"output_path": str(tmp_path / "missing.wav")}),
        encoding="utf-8",
    )
    ok, reason = pipeline_stages.validate_stage_artifacts(
        "rec-stage-edge",
        stage_name="sanitize_audio",
        status="completed",
        metadata={},
        settings=cfg,
    )
    assert ok is False
    assert reason == "sanitize_audio missing artifact audio_sanitized.wav"


def test_worker_helper_functions_cover_loading_logging_and_audio_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-helper-1", create_raw_audio=True)
    raw_audio_path = cfg.recordings_root / "rec-helper-1" / "raw" / "audio.wav"

    bad_json = tmp_path / "list-bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    as_dict = tmp_path / "list-dict.json"
    as_dict.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert worker_tasks._load_json_list(bad_json) == []  # noqa: SLF001
    assert worker_tasks._load_json_list(as_dict) == []  # noqa: SLF001
    assert worker_tasks._stage_metadata({"metadata_json": []}) == {}  # noqa: SLF001
    assert worker_tasks._load_precheck_artifact(ctx) is None  # noqa: SLF001

    logged: list[str] = []
    monkeypatch.setattr(worker_tasks, "_append_step_log", lambda _path, message: logged.append(message))
    worker_tasks._log_stage_completed(tmp_path / "stage.log", "asr", duration_ms=None)  # noqa: SLF001
    worker_tasks._log_stage_invalidated(tmp_path / "stage.log", "precheck", reason="artifact missing")  # noqa: SLF001
    assert logged == [
        "stage completed: asr",
        "stage invalidated: precheck artifact missing, rerunning",
    ]

    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(tmp_path / "missing.wav")}),
        encoding="utf-8",
    )
    ctx.artifacts.sanitized_audio_path.write_bytes(b"wav")
    assert worker_tasks._working_audio_path(ctx) == ctx.artifacts.sanitized_audio_path  # noqa: SLF001

    ctx.artifacts.sanitized_audio_path.unlink()
    ctx.raw_audio_path = raw_audio_path
    assert worker_tasks._working_audio_path(ctx) == raw_audio_path  # noqa: SLF001

    ctx.raw_audio_path = None
    assert worker_tasks._working_audio_path(ctx) == raw_audio_path  # noqa: SLF001


def test_build_diarization_metadata_payload_normalizes_optional_fields(tmp_path: Path) -> None:
    cfg = worker_tasks._build_pipeline_settings(_cfg(tmp_path))  # noqa: SLF001
    smoothing_result = SpeakerTurnSmoothingResult(
        turns=[{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}],
        adjacent_merges=0,
        micro_turn_absorptions=0,
        turn_count_before=1,
        turn_count_after=1,
        speaker_count_before=1,
        speaker_count_after=1,
    )

    payload = worker_tasks._build_diarization_metadata_payload(  # noqa: SLF001
        runtime={
            "mode": "fallback",
            "effective_hints": "bad",
            "initial_hints": "bad",
            "profile_selection": "bad",
        },
        cfg=cfg,
        smoothing_result=smoothing_result,
    )
    assert payload["hints_applied"] == {}
    assert "initial_hints" not in payload
    assert "profile_selection" not in payload

    payload = worker_tasks._build_diarization_metadata_payload(  # noqa: SLF001
        runtime={
            "mode": "pyannote",
            "effective_hints": {"min_speakers": 2},
            "initial_hints": {"min_speakers": 1},
            "profile_selection": {},
        },
        cfg=cfg,
        smoothing_result=smoothing_result,
    )
    assert payload["initial_hints"] == {"min_speakers": 1}
    assert "profile_selection" not in payload


def test_worker_diarization_metadata_wrapper_delegates_to_orchestrator(
    tmp_path: Path,
) -> None:
    """Regression guard for PR-ARTIFACT-SINGLE-WRITER-01.

    Ensures ``worker_tasks._build_diarization_metadata_payload`` is only a
    thin delegate to ``pipeline_orchestrator._build_diarization_metadata_payload``
    so that any new key added to the orchestrator's shared builder shows up
    in the stage-based worker path without a parallel code change. This is
    the exact bug that PR-SPEAKER-MERGE-DIAGNOSTICS-HOTFIX-01 had to fix by
    hand before consolidation.
    """

    cfg = worker_tasks._build_pipeline_settings(_cfg(tmp_path))  # noqa: SLF001
    smoothing_result = SpeakerTurnSmoothingResult(
        turns=[{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}],
        adjacent_merges=0,
        micro_turn_absorptions=0,
        turn_count_before=1,
        turn_count_after=1,
        speaker_count_before=1,
        speaker_count_after=1,
    )
    runtime = {
        "mode": "pyannote",
        "effective_hints": {"min_speakers": 2},
        "initial_hints": {"min_speakers": 2},
        "profile_selection": {},
        "used_dummy_fallback": False,
    }
    speaker_merges = {"S2": "S1"}
    speaker_merge_diagnostics = {
        "embedding_model_available": True,
        "speakers_found": ["S1", "S2"],
        "centroids_computed": ["S1", "S2"],
        "pairwise_scores": [{"a": "S1", "b": "S2", "score": 0.93}],
        "merges_applied": {"S2": "S1"},
        "skipped_reason": None,
    }
    worker_payload = worker_tasks._build_diarization_metadata_payload(  # noqa: SLF001
        runtime=runtime,
        cfg=cfg,
        smoothing_result=smoothing_result,
        speaker_merges=speaker_merges,
        speaker_merge_diagnostics=speaker_merge_diagnostics,
    )
    orchestrator_payload = pipeline_orchestrator._build_diarization_metadata_payload(  # noqa: SLF001
        runtime=runtime,
        cfg=cfg,
        smoothing_result=smoothing_result,
        used_dummy_fallback=False,
        speaker_merges=speaker_merges,
        speaker_merge_diagnostics=speaker_merge_diagnostics,
    )
    assert worker_payload == orchestrator_payload
    assert worker_payload["speaker_merges"] == speaker_merges
    assert worker_payload["speaker_merge_diagnostics"] == speaker_merge_diagnostics
    # Field added by orchestrator is preserved end-to-end.
    assert "speaker_merge_diagnostics" in worker_payload


def test_orchestrator_diarization_metadata_payload_new_field_survives_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate the PR-140 scenario: a new field added to the orchestrator's
    shared builder must survive the full worker_tasks write path.

    We monkeypatch the shared orchestrator builder to add an extra key and
    then call the worker delegate; the key must be present on the returned
    payload, proving the worker never rebuilds the payload independently.
    """

    cfg = worker_tasks._build_pipeline_settings(_cfg(tmp_path))  # noqa: SLF001
    smoothing_result = SpeakerTurnSmoothingResult(
        turns=[],
        adjacent_merges=0,
        micro_turn_absorptions=0,
        turn_count_before=0,
        turn_count_after=0,
        speaker_count_before=0,
        speaker_count_after=0,
    )
    original_builder = pipeline_orchestrator._build_diarization_metadata_payload  # noqa: SLF001

    def _augmented_builder(**kwargs):
        payload = original_builder(**kwargs)
        payload["pr140_simulated_new_field"] = "present"
        return payload

    monkeypatch.setattr(
        pipeline_orchestrator,
        "_build_diarization_metadata_payload",
        _augmented_builder,
    )

    payload = worker_tasks._build_diarization_metadata_payload(  # noqa: SLF001
        runtime={"mode": "pyannote"},
        cfg=cfg,
        smoothing_result=smoothing_result,
    )
    assert payload["pr140_simulated_new_field"] == "present"


def test_finalize_transcript_payload_merges_speaker_lines_and_review() -> None:
    base = {
        "recording_id": "rec-1",
        "segments": [],
        "speakers": ["S1"],
        "text": "hello",
    }
    finalized = pipeline_orchestrator._finalize_transcript_payload(  # noqa: SLF001
        base,
        speaker_lines=["[0.00-1.00] **S1:** hello"],
        asr_execution={"used_multilingual_path": False, "selected_mode": "mono"},
        review={"required": True, "reason_code": "low_confidence"},
    )
    assert finalized is base  # finalization mutates in place and returns the same dict
    assert finalized["speaker_lines"] == ["[0.00-1.00] **S1:** hello"]
    assert finalized["multilingual_asr"] == {
        "used_multilingual_path": False,
        "selected_mode": "mono",
    }
    assert finalized["review"] == {"required": True, "reason_code": "low_confidence"}
    # Empty/None asr execution and review collapse to empty dicts.
    finalized = pipeline_orchestrator._finalize_transcript_payload(  # noqa: SLF001
        {"recording_id": "rec-2"},
        speaker_lines=[],
        asr_execution=None,  # type: ignore[arg-type]
        review=None,  # type: ignore[arg-type]
    )
    assert finalized["speaker_lines"] == []
    assert finalized["multilingual_asr"] == {}
    assert finalized["review"] == {}


def test_stage_precheck_with_missing_duration_and_calendar_refresh_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-precheck-edge", create_raw_audio=True)
    ctx.raw_audio_path = cfg.recordings_root / "rec-precheck-edge" / "raw" / "audio.wav"

    duration_updates: list[float] = []
    monkeypatch.setattr(
        worker_tasks,
        "run_precheck",
        lambda *_a, **_k: PrecheckResult(None, 0.33, None),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_set_recording_duration_best_effort",
        lambda _recording_id, duration_sec, settings: duration_updates.append(duration_sec),
    )
    result = worker_tasks._stage_precheck(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert duration_updates == []

    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_calendar_match",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("calendar boom")),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", ["Ada"]),
    )
    result = worker_tasks._stage_calendar_refresh(ctx)  # noqa: SLF001
    assert result.metadata["warning"] == "calendar boom"


@pytest.mark.parametrize(
    ("stage_fn", "recording_id"),
    [
        (worker_tasks._stage_asr, "rec-missing-asr"),
        (worker_tasks._stage_diarization, "rec-missing-diar"),
        (worker_tasks._stage_language_analysis, "rec-missing-lang"),
        (worker_tasks._stage_speaker_turns, "rec-missing-turns"),
        (worker_tasks._stage_snippet_export, "rec-missing-snippets"),
        (worker_tasks._stage_llm_extract, "rec-missing-llm"),
        (worker_tasks._stage_export_artifacts, "rec-missing-export"),
        (worker_tasks._stage_routing, "rec-missing-routing"),
    ],
)
def test_stage_functions_require_precheck_artifact(
    tmp_path: Path,
    stage_fn,
    recording_id: str,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, recording_id)
    with pytest.raises(RuntimeError, match="Missing precheck artifact"):
        stage_fn(ctx)


def test_stage_asr_raises_when_working_audio_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-asr-no-audio")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", []),
    )

    with pytest.raises(RuntimeError, match="Missing sanitized audio"):
        worker_tasks._stage_asr(ctx)  # noqa: SLF001


def test_stage_asr_uses_custom_whisperx_path_when_monkeypatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-asr-custom", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.raw_audio_path = cfg.recordings_root / "rec-asr-custom" / "raw" / "audio.wav"

    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", ["Ada"]),
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_recording_asr_glossary",
        lambda *_a, **_k: {"entry_count": 0, "term_count": 0, "truncated": False},
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_write_asr_glossary_artifact",
        lambda **_kwargs: None,
    )

    def _fake_custom_whisperx(
        _audio_path: Path,
        *,
        override_lang: str | None,
        cfg,
        step_log_callback=None,
    ):
        if step_log_callback is not None:
            step_log_callback(f"custom whisperx lang={override_lang}")
        return (
            [{"start": 0.0, "end": 1.0, "text": "hello"}],
            {"language": "en"},
        )

    def _fake_run_language_aware_asr(
        audio_path: Path,
        *,
        transcribe_fn,
        step_log_callback=None,
        **_kwargs,
    ):
        if step_log_callback is not None:
            step_log_callback("language-aware")
        segments, info = transcribe_fn(audio_path, "en")
        return segments, info, {"used_multilingual_path": False}

    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_whisperx_asr", _fake_custom_whisperx)
    monkeypatch.setattr(worker_tasks, "run_language_aware_asr", _fake_run_language_aware_asr)

    result = worker_tasks._stage_asr(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert result.metadata["segment_count"] == 1


def test_stage_asr_restores_existing_whisperx_state_and_runtime_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-asr-default", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.raw_audio_path = cfg.recordings_root / "rec-asr-default" / "raw" / "audio.wav"

    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", ["Ada"]),
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_recording_asr_glossary",
        lambda *_a, **_k: {"entry_count": 0, "term_count": 0, "truncated": False},
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_write_asr_glossary_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_build_whisperx_transcriber",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_glossary_runtime_metadata",
        lambda _transcribe_audio: {"used_prompt": True},
    )

    def _fake_default_whisperx(
        _audio_path: Path,
        *,
        override_lang: str | None,
        cfg,
        step_log_callback=None,
    ):
        if step_log_callback is not None:
            step_log_callback(f"default whisperx lang={override_lang}")
        return (
            [{"start": 0.0, "end": 1.0, "text": "hello"}],
            {"language": "en"},
        )

    def _fake_run_language_aware_asr(
        audio_path: Path,
        *,
        transcribe_fn,
        step_log_callback=None,
        **_kwargs,
    ):
        if step_log_callback is not None:
            step_log_callback("language-aware")
        segments, info = transcribe_fn(audio_path, "en")
        return segments, info, {"used_multilingual_path": False}

    state = worker_tasks.pipeline_orchestrator._whisperx_transcriber_state
    previous_transcriber = object()
    monkeypatch.setattr(state, "transcribe_audio", previous_transcriber, raising=False)
    monkeypatch.setattr(state, "use_session_transcriber", True, raising=False)
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_whisperx_asr", _fake_default_whisperx)
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_DEFAULT_WHISPERX_ASR",
        _fake_default_whisperx,
    )
    monkeypatch.setattr(worker_tasks, "run_language_aware_asr", _fake_run_language_aware_asr)

    result = worker_tasks._stage_asr(ctx)  # noqa: SLF001

    assert result.status == "completed"
    assert ctx.asr_execution["glossary_runtime"] == {"used_prompt": True}
    assert state.transcribe_audio is previous_transcriber
    assert state.use_session_transcriber is True


def test_stage_asr_default_path_handles_missing_session_transcriber_attr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-asr-default-cleanup", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.raw_audio_path = cfg.recordings_root / "rec-asr-default-cleanup" / "raw" / "audio.wav"

    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", []),
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_recording_asr_glossary",
        lambda *_a, **_k: {"entry_count": 0, "term_count": 0, "truncated": False},
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_write_asr_glossary_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_build_whisperx_transcriber",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_glossary_runtime_metadata",
        lambda _transcribe_audio: {},
    )

    def _fake_default_whisperx(
        _audio_path: Path,
        *,
        override_lang: str | None,
        cfg,
        step_log_callback=None,
    ):
        if step_log_callback is not None:
            step_log_callback(f"default whisperx lang={override_lang}")
        return (
            [{"start": 0.0, "end": 1.0, "text": "hello"}],
            {"language": "en"},
        )

    def _fake_run_language_aware_asr(
        audio_path: Path,
        *,
        transcribe_fn,
        step_log_callback=None,
        **_kwargs,
    ):
        state = worker_tasks.pipeline_orchestrator._whisperx_transcriber_state
        if hasattr(state, "transcribe_audio"):
            delattr(state, "transcribe_audio")
        if step_log_callback is not None:
            step_log_callback("language-aware")
        segments, info = transcribe_fn(audio_path, "en")
        return segments, info, {"used_multilingual_path": False}

    state = worker_tasks.pipeline_orchestrator._whisperx_transcriber_state
    if hasattr(state, "transcribe_audio"):
        delattr(state, "transcribe_audio")
    if hasattr(state, "use_session_transcriber"):
        delattr(state, "use_session_transcriber")

    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_whisperx_asr", _fake_default_whisperx)
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_DEFAULT_WHISPERX_ASR",
        _fake_default_whisperx,
    )
    monkeypatch.setattr(worker_tasks, "run_language_aware_asr", _fake_run_language_aware_asr)

    result = worker_tasks._stage_asr(ctx)  # noqa: SLF001

    assert result.status == "completed"
    assert hasattr(state, "transcribe_audio") is False
    assert hasattr(state, "use_session_transcriber") is False


def test_stage_diarization_fallback_and_dummy_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-diar-fallback", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.raw_audio_path = cfg.recordings_root / "rec-diar-fallback" / "raw" / "audio.wav"
    ctx.artifacts.asr_segments_json_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(worker_tasks, "_build_diariser", lambda *_a, **_k: worker_tasks._FallbackDiariser(1.0))
    monkeypatch.setattr(worker_tasks, "_log_gpu_execution_policy", lambda **_kwargs: None)
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_diariser_runtime_metadata",
        lambda _diariser: {},
    )

    async def _fake_retry_dialog_diarization(**kwargs):
        kwargs["step_log_callback"]("retry diarization")
        return kwargs["diarization"]

    calls = {"count": 0}

    def _fake_diarization_segments(_annotation):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return [{"speaker": "S1", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_maybe_retry_dialog_diarization",
        _fake_retry_dialog_diarization,
    )
    monkeypatch.setattr(worker_tasks, "_diarization_segments", _fake_diarization_segments)

    result = worker_tasks._stage_diarization(ctx)  # noqa: SLF001
    assert result.metadata["mode"] == "fallback"
    assert result.metadata["used_dummy_fallback"] is True


def test_stage_diarization_init_failure_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-diar-init-fail", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.raw_audio_path = cfg.recordings_root / "rec-diar-init-fail" / "raw" / "audio.wav"
    ctx.artifacts.asr_segments_json_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        worker_tasks,
        "_build_diariser",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("diariser boom")),
    )
    monkeypatch.setattr(worker_tasks, "_log_gpu_execution_policy", lambda **_kwargs: None)
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_diariser_runtime_metadata",
        lambda _diariser: {},
    )
    monkeypatch.setattr(
        worker_tasks,
        "_diarization_segments",
        lambda _annotation: [{"speaker": "S1", "start": 0.0, "end": 1.0}],
    )

    async def _fake_retry_dialog_diarization(**kwargs):
        return kwargs["diarization"]

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_maybe_retry_dialog_diarization",
        _fake_retry_dialog_diarization,
    )

    result = worker_tasks._stage_diarization(ctx)  # noqa: SLF001
    assert result.metadata["mode"] == "fallback"


def test_stage_diarization_raises_when_working_audio_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-diar-no-audio")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    monkeypatch.setattr(worker_tasks, "_build_diariser", lambda *_a, **_k: worker_tasks._FallbackDiariser(1.0))
    monkeypatch.setattr(worker_tasks, "_log_gpu_execution_policy", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="Missing sanitized audio"):
        worker_tasks._stage_diarization(ctx)  # noqa: SLF001


def test_stage_language_analysis_updates_detected_language_from_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-language-analysis")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.artifacts.asr_segments_json_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "hola"}]),
        encoding="utf-8",
    )
    ctx.artifacts.asr_info_json_path.write_text(json.dumps({"language": "unknown"}), encoding="utf-8")
    ctx.artifacts.asr_execution_json_path.write_text(
        json.dumps({"used_multilingual_path": False}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        worker_tasks,
        "analyse_languages",
        lambda *_a, **_k: SimpleNamespace(
            dominant_language="es",
            distribution={"es": 88.0},
            spans=[{"start": 0.0, "end": 1.0, "language": "es"}],
            segments=[{"start": 0.0, "end": 1.0, "text": "hola", "language": "es"}],
            review_required=False,
            review_reason_code=None,
            review_reason_text=None,
            uncertain_segment_count=0,
            conflict_segment_count=0,
        ),
    )
    monkeypatch.setattr(
        worker_tasks,
        "resolve_target_summary_language",
        lambda *_a, **_k: "es",
    )

    result = worker_tasks._stage_language_analysis(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert ctx.language_payload["language"]["detected"] == "es"
    assert ctx.language_payload["language"]["confidence"] == 0.88

    _cfg_value, ctx_missing_percent = _new_ctx(tmp_path, "rec-language-analysis-missing-percent")
    ctx_missing_percent.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx_missing_percent.artifacts.asr_segments_json_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "hola"}]),
        encoding="utf-8",
    )
    ctx_missing_percent.artifacts.asr_info_json_path.write_text(
        json.dumps({"language": "unknown"}),
        encoding="utf-8",
    )
    ctx_missing_percent.artifacts.asr_execution_json_path.write_text(
        json.dumps({"used_multilingual_path": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker_tasks,
        "analyse_languages",
        lambda *_a, **_k: SimpleNamespace(
            dominant_language="es",
            distribution={"fr": 100.0},
            spans=[],
            segments=[{"start": 0.0, "end": 1.0, "text": "hola", "language": "es"}],
            review_required=False,
            review_reason_code=None,
            review_reason_text=None,
            uncertain_segment_count=0,
            conflict_segment_count=0,
        ),
    )

    result = worker_tasks._stage_language_analysis(ctx_missing_percent)  # noqa: SLF001
    assert result.status == "completed"
    assert ctx_missing_percent.language_payload["language"]["detected"] == "es"
    assert ctx_missing_percent.language_payload["language"]["confidence"] is None


def test_stage_speaker_turns_requires_language_artifact_and_supports_unsmoothed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-speaker-turns")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)

    with pytest.raises(RuntimeError, match="Missing language analysis artifact"):
        worker_tasks._stage_speaker_turns(ctx)  # noqa: SLF001

    ctx.artifacts.language_analysis_json_path.write_text(
        json.dumps(
            {
                "dominant_language": "en",
                "language": {"detected": "en", "confidence": 0.95},
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "language": "en"}],
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S2", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_runtime_json_path.write_text(
        json.dumps({"mode": "fallback", "used_dummy_fallback": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_speaker_turns",
        lambda *_a, **_k: [{"speaker": "S2", "start": 0.0, "end": 1.0, "text": "hello"}],
    )
    monkeypatch.setattr(
        worker_tasks,
        "smooth_speaker_turns",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not smooth")),
    )

    result = worker_tasks._stage_speaker_turns(ctx)  # noqa: SLF001
    assert result.metadata == {"turn_count": 1, "speaker_count": 1}


def test_stage_speaker_turns_logs_flicker_speaker_reassignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-flicker-stage")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.artifacts.language_analysis_json_path.write_text(
        json.dumps(
            {
                "dominant_language": "en",
                "language": {"detected": "en", "confidence": 0.95},
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello", "language": "en"}
                ],
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps(
            [
                {"speaker": "S_MAIN", "start": 0.0, "end": 5.0},
                {"speaker": "S_FLICKER", "start": 1.0, "end": 1.2},
                {"speaker": "S_MAIN", "start": 5.0, "end": 12.0},
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_runtime_json_path.write_text(
        json.dumps({"mode": "fallback", "used_dummy_fallback": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_speaker_turns",
        lambda *_a, **_k: [
            {"speaker": "S_MAIN", "start": 0.0, "end": 12.0, "text": "hello"}
        ],
    )
    monkeypatch.setattr(
        worker_tasks,
        "smooth_speaker_turns",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not smooth")),
    )

    caplog.set_level("WARNING", logger="lan_app.worker_tasks")
    result = worker_tasks._stage_speaker_turns(ctx)  # noqa: SLF001

    assert result.status == "completed"
    assert any(
        "Diarization flicker speaker reassigned" in record.getMessage()
        and "S_FLICKER" in record.getMessage()
        for record in caplog.records
    )
    assert all(
        row["speaker"] != "S_FLICKER" for row in ctx.diarization_segments
    )


def test_stage_snippet_export_writes_manifest_metadata_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-snippet-stage", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False}),
        encoding="utf-8",
    )

    def _fake_export(request):
        request.snippets_dir.mkdir(parents=True, exist_ok=True)
        (request.snippets_dir / "S1").mkdir(parents=True, exist_ok=True)
        (request.snippets_dir / "S1" / "1.wav").write_bytes(b"clip")
        (request.snippets_dir.parent / "snippets_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "speakers": {
                        "S1": [
                            {"status": "accepted", "relative_path": "S1/1.wav"},
                            {"status": "rejected_overlap"},
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        return [request.snippets_dir / "S1" / "1.wav"]

    monkeypatch.setattr(worker_tasks, "export_speaker_snippets", _fake_export)

    result = worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001

    assert result.status == "completed"
    assert result.metadata == {
        "manifest_status": "partial",
        "accepted_snippets": 1,
        "speaker_count": 1,
        "warning_count": 1,
        "degraded_diarization": False,
        "noise_speakers": [],
    }
    manifest = json.loads(worker_tasks._snippets_manifest_path(ctx).read_text(encoding="utf-8"))  # noqa: SLF001
    assert manifest["manifest_status"] == "partial"
    assert manifest["accepted_snippets"] == 1
    assert manifest["warning_count"] == 1
    assert manifest["noise_speakers"] == []


def test_stage_snippet_export_skips_metadata_noise_when_manifest_value_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-snippet-bad-noise", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False}),
        encoding="utf-8",
    )

    def _fake_export(request):
        request.snippets_dir.mkdir(parents=True, exist_ok=True)
        (request.snippets_dir.parent / "snippets_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "speakers": {
                        "S1": [{"status": "accepted", "relative_path": "S1/1.wav"}],
                    },
                    "noise_speakers": "not-a-list",
                }
            ),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(worker_tasks, "export_speaker_snippets", _fake_export)
    monkeypatch.setattr(
        worker_tasks,
        "apply_noise_flags_to_manifest",
        lambda *_a, **_k: {"noise_speakers": [], "speaker_metrics": {}, "threshold": 0.3},
    )

    def _no_op_metadata(*_a, **_k):
        # _stage_snippet_export now calls update_diarization_metadata_with_noise
        # twice: once upfront to clear stale noise data (manifest doesn't exist
        # yet) and once after the snippet pass. Only the post-snippet call
        # should re-corrupt the manifest to exercise the defensive isinstance
        # guard for a non-list noise_speakers entry.
        manifest_path = worker_tasks._snippets_manifest_path(ctx)  # noqa: SLF001
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["noise_speakers"] = "still-not-a-list"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        worker_tasks, "update_diarization_metadata_with_noise", _no_op_metadata
    )

    result = worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001
    assert "noise_speakers" not in result.metadata


def test_stage_snippet_export_skips_noise_detection_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-snippet-no-noise", create_raw_audio=True)
    ctx.pipeline_settings.noise_detection_enabled = False
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False}),
        encoding="utf-8",
    )

    def _fake_export(request):
        request.snippets_dir.mkdir(parents=True, exist_ok=True)
        (request.snippets_dir.parent / "snippets_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "speakers": {
                        "S1": [
                            {"status": "accepted", "relative_path": "S1/1.wav"},
                        ]
                    },
                    "noise_speakers": "not-a-list",
                }
            ),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(worker_tasks, "export_speaker_snippets", _fake_export)
    monkeypatch.setattr(
        worker_tasks,
        "apply_noise_flags_to_manifest",
        lambda *_a, **_k: pytest.fail("noise detection should be disabled"),
    )

    result = worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001
    assert "noise_speakers" not in result.metadata


def test_stage_export_artifacts_no_speech_speakers_excludes_filtered(
    tmp_path: Path,
) -> None:
    """The no_speech transcript roster must reflect the filtered speaker set."""

    cfg, ctx = _new_ctx(
        tmp_path, "rec-export-nospeech-filtered", create_raw_audio=True
    )
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.pipeline_settings.llm_model = "stub-model"
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False, "noise_speakers": ["SPEAKER_NOISE"]}),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps(
            [
                {"speaker": "SPEAKER_REAL", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_NOISE", "start": 1.0, "end": 1.5},
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_NOISE",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "background noise hiss static hum buzz",
                }
            ]
        ),
        encoding="utf-8",
    )
    snippets_dir = ctx.artifacts.recording_artifacts.snippets_dir
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (snippets_dir.parent / "snippets_manifest.json").write_text(
        json.dumps({"version": 1, "speakers": {}, "manifest_status": "ok"}),
        encoding="utf-8",
    )
    derived = ctx.artifacts.derived_dir
    (derived / "language_analysis.json").write_text(
        json.dumps(
            {
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
                "language_distribution": {"en": 1.0},
                "language_spans": [],
                "segments": [
                    {
                        "text": "background noise hiss static hum buzz",
                        "start": 0.0,
                        "end": 1.0,
                    }
                ],
                "target_summary_language": "en",
                "review": {},
            }
        ),
        encoding="utf-8",
    )
    (derived / "asr_execution.json").write_text(
        json.dumps({"used_multilingual_path": False, "selected_mode": "single_language"}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.summary_json_path.write_text(
        json.dumps({"status": "ok", "summary": "ok"}),
        encoding="utf-8",
    )

    result = worker_tasks._stage_export_artifacts(ctx)  # noqa: SLF001
    assert result.metadata["output_status"] == "no_speech"
    transcript_payload = json.loads(
        ctx.artifacts.recording_artifacts.transcript_json_path.read_text(encoding="utf-8")
    )
    assert transcript_payload.get("speakers") == []
    assert transcript_payload.get("text") == ""
    # language_segments are dropped so refresh_recording_metrics can't rebuild
    # participant metrics from noise text when all turns were filtered.
    assert transcript_payload.get("segments") == []


def test_stage_snippet_export_clears_stale_noise_metadata_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed snippet export must wipe stale noise_speakers from prior runs."""

    _cfg_value, ctx = _new_ctx(tmp_path, "rec-snippet-stale", create_raw_audio=True)
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    metadata_path = ctx.artifacts.recording_artifacts.diarization_metadata_json_path
    metadata_path.write_text(
        json.dumps(
            {
                "degraded": False,
                "noise_speakers": ["SPEAKER_STALE"],
                "noise_speaker_metrics": {"SPEAKER_STALE": {"flagged": True}},
            }
        ),
        encoding="utf-8",
    )

    def _failing_export(_request):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(worker_tasks, "export_speaker_snippets", _failing_export)

    result = worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert result.metadata["manifest_status"] == "export_failed"
    persisted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted_metadata["noise_speakers"] == []
    assert persisted_metadata["noise_speaker_metrics"] == {}


def test_stage_export_artifacts_exclude_noise_handles_missing_or_empty_noise_list(
    tmp_path: Path,
) -> None:
    cfg, ctx = _new_ctx(
        tmp_path, "rec-export-exclude-no-noise", create_raw_audio=True
    )
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.pipeline_settings.llm_model = "stub-model"
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    # diarization_metadata: noise_speakers is None (not a list) for this branch.
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False, "noise_speakers": None}),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello team"}]
        ),
        encoding="utf-8",
    )
    snippets_dir = ctx.artifacts.recording_artifacts.snippets_dir
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (snippets_dir.parent / "snippets_manifest.json").write_text(
        json.dumps({"version": 1, "speakers": {}, "manifest_status": "ok"}),
        encoding="utf-8",
    )
    derived = ctx.artifacts.derived_dir
    (derived / "language_analysis.json").write_text(
        json.dumps(
            {
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
                "language_distribution": {"en": 1.0},
                "language_spans": [],
                "segments": [
                    {
                        "text": "hello team this is the real speaker speaking now",
                        "start": 0.0,
                        "end": 1.0,
                    }
                ],
                "target_summary_language": "en",
                "review": {},
            }
        ),
        encoding="utf-8",
    )
    (derived / "asr_execution.json").write_text(
        json.dumps({"used_multilingual_path": False, "selected_mode": "single_language"}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.summary_json_path.write_text(
        json.dumps({"status": "ok", "summary": "ok"}),
        encoding="utf-8",
    )

    result_none = worker_tasks._stage_export_artifacts(ctx)  # noqa: SLF001
    assert result_none.status == "completed"

    # Now re-run with empty list -> should also keep all turns
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False, "noise_speakers": []}),
        encoding="utf-8",
    )
    result_empty = worker_tasks._stage_export_artifacts(ctx)  # noqa: SLF001
    assert result_empty.status == "completed"
    transcript_payload = json.loads(
        ctx.artifacts.recording_artifacts.transcript_json_path.read_text(encoding="utf-8")
    )
    speaker_lines = transcript_payload.get("speaker_lines") or []
    assert any("S1" in line for line in speaker_lines)


def test_stage_export_artifacts_skips_filter_when_noise_detection_disabled(
    tmp_path: Path,
) -> None:
    """When noise detection is off, stale noise_speakers metadata must not filter turns."""

    cfg, ctx = _new_ctx(
        tmp_path, "rec-export-stale-noise", create_raw_audio=True
    )
    ctx.pipeline_settings.noise_detection_enabled = False
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.pipeline_settings.llm_model = "stub-model"
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps(
            {"degraded": False, "noise_speakers": ["SPEAKER_NOISE"]}
        ),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps(
            [
                {"speaker": "SPEAKER_REAL", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_NOISE", "start": 1.0, "end": 1.5},
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_REAL",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello team this is the real speaker speaking now",
                },
                {
                    "speaker": "SPEAKER_NOISE",
                    "start": 1.0,
                    "end": 1.5,
                    "text": "background noise hiss static hum buzz",
                },
            ]
        ),
        encoding="utf-8",
    )
    snippets_dir = ctx.artifacts.recording_artifacts.snippets_dir
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (snippets_dir.parent / "snippets_manifest.json").write_text(
        json.dumps({"version": 1, "speakers": {}, "manifest_status": "ok"}),
        encoding="utf-8",
    )
    derived = ctx.artifacts.derived_dir
    (derived / "language_analysis.json").write_text(
        json.dumps(
            {
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
                "language_distribution": {"en": 1.0},
                "language_spans": [],
                "segments": [
                    {
                        "text": "hello team this is the real speaker speaking now and also background noise hiss static hum buzz",
                        "start": 0.0,
                        "end": 1.5,
                    }
                ],
                "target_summary_language": "en",
                "review": {},
            }
        ),
        encoding="utf-8",
    )
    (derived / "asr_execution.json").write_text(
        json.dumps({"used_multilingual_path": False, "selected_mode": "single_language"}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.summary_json_path.write_text(
        json.dumps({"status": "ok", "summary": "ok"}),
        encoding="utf-8",
    )

    result = worker_tasks._stage_export_artifacts(ctx)  # noqa: SLF001
    assert result.status == "completed"
    persisted_turns = json.loads(
        ctx.artifacts.recording_artifacts.speaker_turns_json_path.read_text(encoding="utf-8")
    )
    assert any(str(turn.get("speaker")) == "SPEAKER_NOISE" for turn in persisted_turns)


def test_stage_export_artifacts_preserves_segments_when_noise_labels_stale(
    tmp_path: Path,
) -> None:
    """Stale noise_speakers that don't match current turns must not swap segments."""

    cfg, ctx = _new_ctx(
        tmp_path, "rec-export-stale-labels", create_raw_audio=True
    )
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.pipeline_settings.llm_model = "stub-model"
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps(
            {"degraded": False, "noise_speakers": ["SPEAKER_GHOST"]}
        ),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "SPEAKER_REAL", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_REAL",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello team this is the real speaker speaking now",
                }
            ]
        ),
        encoding="utf-8",
    )
    snippets_dir = ctx.artifacts.recording_artifacts.snippets_dir
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (snippets_dir.parent / "snippets_manifest.json").write_text(
        json.dumps({"version": 1, "speakers": {}, "manifest_status": "ok"}),
        encoding="utf-8",
    )
    derived = ctx.artifacts.derived_dir
    raw_segment = {
        "text": "hello team this is the real speaker speaking now",
        "start": 0.0,
        "end": 1.0,
        "language": "en",
    }
    (derived / "language_analysis.json").write_text(
        json.dumps(
            {
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
                "language_distribution": {"en": 1.0},
                "language_spans": [],
                "segments": [raw_segment],
                "target_summary_language": "en",
                "review": {},
            }
        ),
        encoding="utf-8",
    )
    (derived / "asr_execution.json").write_text(
        json.dumps({"used_multilingual_path": False, "selected_mode": "single_language"}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.summary_json_path.write_text(
        json.dumps({"status": "ok", "summary": "ok"}),
        encoding="utf-8",
    )

    result = worker_tasks._stage_export_artifacts(ctx)  # noqa: SLF001
    assert result.status == "completed"
    transcript_payload = json.loads(
        ctx.artifacts.recording_artifacts.transcript_json_path.read_text(encoding="utf-8")
    )
    # Stale noise label doesn't match any real speaker, so segments stay as
    # raw ASR output (language metadata preserved) rather than speaker turns.
    segments = transcript_payload.get("segments") or []
    assert any(seg.get("language") == "en" for seg in segments)
    # transcript.txt must also remain the ASR-derived clean_text, not a
    # speaker-turn-derived recomputation.
    transcript_txt = ctx.artifacts.recording_artifacts.transcript_txt_path.read_text(
        encoding="utf-8"
    )
    assert transcript_txt == ctx.clean_text


def test_stage_export_artifacts_filters_noise_speakers_from_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, ctx = _new_ctx(
        tmp_path, "rec-export-noise-filter", create_raw_audio=True
    )
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.pipeline_settings.llm_model = "stub-model"
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps(
            {
                "degraded": False,
                "noise_speakers": ["SPEAKER_NOISE"],
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps(
            [
                {"speaker": "SPEAKER_REAL", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_NOISE", "start": 1.0, "end": 1.5},
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_REAL",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello team this is the real speaker speaking now clearly",
                },
                {
                    "speaker": "SPEAKER_NOISE",
                    "start": 1.0,
                    "end": 1.5,
                    "text": "background noise hiss static hum buzz crackle",
                },
            ]
        ),
        encoding="utf-8",
    )
    snippets_dir = ctx.artifacts.recording_artifacts.snippets_dir
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (snippets_dir.parent / "snippets_manifest.json").write_text(
        json.dumps({"version": 1, "speakers": {}, "manifest_status": "ok"}),
        encoding="utf-8",
    )
    derived = ctx.artifacts.derived_dir
    (derived / "language_analysis.json").write_text(
        json.dumps(
            {
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
                "language_distribution": {"en": 1.0},
                "language_spans": [],
                "segments": [
                    {"text": "hello team this is the real speaker speaking now", "start": 0.0, "end": 1.0},
                    {"text": "background noise hiss static hum", "start": 1.0, "end": 1.5},
                ],
                "target_summary_language": "en",
                "review": {},
            }
        ),
        encoding="utf-8",
    )
    (derived / "asr_execution.json").write_text(
        json.dumps({"used_multilingual_path": False, "selected_mode": "single_language"}),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.summary_json_path.write_text(
        json.dumps({"status": "ok", "summary": "ok"}),
        encoding="utf-8",
    )

    result = worker_tasks._stage_export_artifacts(ctx)  # noqa: SLF001
    assert result.status == "completed"
    transcript_payload = json.loads(
        ctx.artifacts.recording_artifacts.transcript_json_path.read_text(encoding="utf-8")
    )
    speaker_lines = transcript_payload.get("speaker_lines") or []
    assert any("SPEAKER_REAL" in line for line in speaker_lines)
    assert all("SPEAKER_NOISE" not in line for line in speaker_lines)
    assert "SPEAKER_NOISE" not in (transcript_payload.get("speakers") or [])
    assert "SPEAKER_REAL" in (transcript_payload.get("speakers") or [])
    # Partial-filter case must also scrub transcript.json["segments"] so UI
    # fallbacks don't rebuild noise turns from raw ASR output.
    persisted_segments = transcript_payload.get("segments") or []
    assert any(
        str(seg.get("speaker")) == "SPEAKER_REAL" for seg in persisted_segments
    )
    assert all(
        str(seg.get("speaker")) != "SPEAKER_NOISE" for seg in persisted_segments
    )
    persisted_turns = json.loads(
        ctx.artifacts.recording_artifacts.speaker_turns_json_path.read_text(encoding="utf-8")
    )
    assert all(
        str(turn.get("speaker")) != "SPEAKER_NOISE" for turn in persisted_turns
    )
    assert any(str(turn.get("speaker")) == "SPEAKER_REAL" for turn in persisted_turns)
    transcript_text = ctx.artifacts.recording_artifacts.transcript_txt_path.read_text(
        encoding="utf-8"
    )
    assert "background noise hiss" not in transcript_text
    assert "real speaker speaking" in transcript_text
    assert transcript_payload["text"] == transcript_text


def test_snippet_manifest_helpers_cover_edge_cases(tmp_path: Path) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-snippet-helper")

    with pytest.raises(RuntimeError, match="Missing snippets manifest"):
        worker_tasks._finalize_snippets_manifest(ctx, manifest_status="ok")  # noqa: SLF001

    assert worker_tasks._snippet_manifest_counts({"speakers": []}) == {  # noqa: SLF001
        "accepted_snippets": 0,
        "speaker_count": 0,
        "warning_count": 0,
    }

    counts = worker_tasks._snippet_manifest_counts(  # noqa: SLF001
        {
            "speakers": {
                "S1": "bad",
                "S2": [{}, "bad", {"status": "accepted"}, {"status": "rejected_overlap"}],
            },
            "warnings": ["bad", {"code": "warn"}],
        }
    )
    assert counts == {"accepted_snippets": 1, "speaker_count": 1, "warning_count": 2}

    manifest_path = worker_tasks._snippets_manifest_path(ctx)  # noqa: SLF001
    manifest_path.write_text(json.dumps({"version": 1, "speakers": {}, "warnings": [{"code": "stale"}]}), encoding="utf-8")
    manifest = worker_tasks._finalize_snippets_manifest(  # noqa: SLF001
        ctx,
        manifest_status="",
        warnings=[
            {"code": "only_code"},
            {"message": "only message"},
            {"code": "", "message": ""},
            "bad",  # type: ignore[list-item]
        ],
    )
    assert manifest["manifest_status"] == "ok"
    assert manifest["warnings"] == [
        {"code": "only_code"},
        {"message": "only message"},
    ]
    assert "degraded_diarization" not in manifest

    manifest_path.write_text(json.dumps({"version": 1, "speakers": {}, "warnings": [{"code": "stale"}]}), encoding="utf-8")
    manifest = worker_tasks._finalize_snippets_manifest(ctx, manifest_status="clean", degraded_diarization=True)  # noqa: SLF001
    assert manifest["manifest_status"] == "clean"
    assert manifest["degraded_diarization"] is True
    assert "warnings" not in manifest


def test_stage_snippet_export_handles_empty_turns_and_export_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx_no_turns = _new_ctx(tmp_path, "rec-snippet-empty", create_raw_audio=True)
    ctx_no_turns.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx_no_turns.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx_no_turns.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx_no_turns.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx_no_turns.artifacts.diarization_segments_json_path.write_text(
        json.dumps([]),
        encoding="utf-8",
    )
    ctx_no_turns.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        "[]",
        encoding="utf-8",
    )
    ctx_no_turns.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": True}),
        encoding="utf-8",
    )

    empty_result = worker_tasks._stage_snippet_export(ctx_no_turns)  # noqa: SLF001
    assert empty_result.metadata["manifest_status"] == "no_usable_speech"
    assert empty_result.metadata["degraded_diarization"] is True
    empty_manifest = json.loads(worker_tasks._snippets_manifest_path(ctx_no_turns).read_text(encoding="utf-8"))  # noqa: SLF001
    assert empty_manifest["warnings"][0]["code"] == "no_speaker_turns"

    _cfg_value, ctx_fail = _new_ctx(tmp_path, "rec-snippet-fail", create_raw_audio=True)
    ctx_fail.precheck_result = PrecheckResult(10.0, 0.5, None)
    _write_pcm_wav(ctx_fail.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx_fail.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx_fail.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    ctx_fail.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx_fail.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    ctx_fail.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker_tasks,
        "export_speaker_snippets",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    failed_result = worker_tasks._stage_snippet_export(ctx_fail)  # noqa: SLF001
    assert failed_result.metadata["manifest_status"] == "export_failed"
    assert failed_result.metadata["warning_count"] == 1
    failed_manifest = json.loads(worker_tasks._snippets_manifest_path(ctx_fail).read_text(encoding="utf-8"))  # noqa: SLF001
    assert failed_manifest["warnings"][0]["message"] == "boom"


def test_stage_snippet_export_requires_audio_and_upstream_artifacts(tmp_path: Path) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-snippet-missing")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)

    with pytest.raises(RuntimeError, match="Missing sanitized audio"):
        worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001

    _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
    ctx.artifacts.audio_sanitize_json_path.write_text(
        json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Missing diarization segments artifact"):
        worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001

    ctx.artifacts.diarization_segments_json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing speaker turns artifact"):
        worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001

    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing diarization metadata artifact"):
        worker_tasks._stage_snippet_export(ctx)  # noqa: SLF001


def test_stage_snippet_export_reports_no_clean_and_degraded_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _prepare_ctx(recording_id: str, *, degraded: bool) -> worker_tasks._PipelineExecutionContext:
        _cfg_value, ctx = _new_ctx(tmp_path, recording_id, create_raw_audio=True)
        ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
        _write_pcm_wav(ctx.artifacts.sanitized_audio_path, duration_sec=0.2)
        ctx.artifacts.audio_sanitize_json_path.write_text(
            json.dumps({"output_path": str(ctx.artifacts.sanitized_audio_path)}),
            encoding="utf-8",
        )
        ctx.artifacts.diarization_segments_json_path.write_text(
            json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
            encoding="utf-8",
        )
        ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
            json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
            encoding="utf-8",
        )
        ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
            json.dumps({"degraded": degraded}),
            encoding="utf-8",
        )
        return ctx

    def _fake_export(request):
        request.snippets_dir.mkdir(parents=True, exist_ok=True)
        (request.snippets_dir.parent / "snippets_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "speakers": {"S1": [{"status": "rejected_overlap"}]},
                }
            ),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(worker_tasks, "export_speaker_snippets", _fake_export)

    clean_ctx = _prepare_ctx("rec-snippet-no-clean", degraded=False)
    clean_result = worker_tasks._stage_snippet_export(clean_ctx)  # noqa: SLF001
    assert clean_result.metadata["manifest_status"] == "no_clean_snippets"

    degraded_ctx = _prepare_ctx("rec-snippet-degraded", degraded=True)
    degraded_result = worker_tasks._stage_snippet_export(degraded_ctx)  # noqa: SLF001
    assert degraded_result.metadata["manifest_status"] == "degraded"


def test_stage_export_artifacts_uses_existing_snippet_manifest_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-export-stage")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.pipeline_settings.llm_model = "test-model"
    ctx.artifacts.diarization_segments_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.summary_json_path.write_text(
        json.dumps({"status": "ok", "topic": "Weekly Sync"}),
        encoding="utf-8",
    )
    worker_tasks._snippets_manifest_path(ctx).write_text(  # noqa: SLF001
        json.dumps(
            {
                "version": 1,
                "manifest_status": "ok",
                "accepted_snippets": 1,
                "speaker_count": 1,
                "warning_count": 0,
                "speakers": {"S1": [{"status": "accepted", "relative_path": "S1/1.wav"}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", ["Ada"]),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_language_analysis_artifact",
        lambda _ctx: {
            "language": {"detected": "en", "confidence": 0.95},
            "dominant_language": "en",
            "language_distribution": {"en": 100.0},
            "language_spans": [],
            "target_summary_language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "review": {},
        },
    )
    monkeypatch.setattr(worker_tasks, "_load_asr_execution", lambda _ctx: {})
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_require_llm_model", lambda _model: "test-model")
    monkeypatch.setattr(worker_tasks, "load_speaker_aliases", lambda _path: {})
    monkeypatch.setattr(worker_tasks, "export_speaker_snippets", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected export")))

    result = worker_tasks._stage_export_artifacts(ctx)  # noqa: SLF001

    assert result.status == "completed"
    assert result.metadata["snippets"] == 1


def test_stage_llm_extract_filters_noise_speakers_from_summary_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-llm-noise-filter")
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.artifacts.language_analysis_json_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello team this is the real speaker speaking now",
                    }
                ],
                "target_summary_language": "en",
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_REAL",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello team this is the real speaker speaking now",
                },
                {
                    "speaker": "SPEAKER_NOISE",
                    "start": 1.0,
                    "end": 1.5,
                    "text": "background noise hiss static hum buzz crackle",
                },
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False, "noise_speakers": ["SPEAKER_NOISE"]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _capture_prompts(turns, summary_lang, *, calendar_title=None, calendar_attendees=None):
        captured["turns"] = list(turns)
        return ("sys", "user")

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_require_llm_model", lambda _m: "test-model"
    )
    monkeypatch.setattr(worker_tasks, "LLMClient", lambda: object())
    monkeypatch.setattr(worker_tasks, "load_speaker_aliases", lambda _p: {})
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_sentiment_score", lambda _text: 3
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_use_chunked_llm", lambda *_a, **_k: False
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", []),
    )
    monkeypatch.setattr(worker_tasks, "build_structured_summary_prompts", _capture_prompts)

    async def _fake_generate(*_a, **_k):
        return {"content": json.dumps({"topic": "ok", "summary_bullets": ["x"]})}

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_generate_llm_message", _fake_generate
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_summary_payload",
        lambda **_k: {"status": "ok", "topic": "ok"},
    )
    monkeypatch.setattr(
        worker_tasks,
        "_set_recording_progress_best_effort",
        lambda *_a, **_k: None,
    )

    result = worker_tasks._stage_llm_extract(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert captured["turns"], "build_structured_summary_prompts must be called"
    assert all(
        str(turn.get("speaker")) != "SPEAKER_NOISE" for turn in captured["turns"]
    )
    assert any(
        str(turn.get("speaker")) == "SPEAKER_REAL" for turn in captured["turns"]
    )


def test_stage_llm_extract_returns_no_speech_when_filter_drops_all_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If exclusion filters every speaker, skip LLM and report no_speech."""

    _cfg_value, ctx = _new_ctx(tmp_path, "rec-llm-all-noise")
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.artifacts.language_analysis_json_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "background noise hiss static hum buzz",
                    }
                ],
                "target_summary_language": "en",
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_NOISE",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "background noise hiss static hum buzz",
                }
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False, "noise_speakers": ["SPEAKER_NOISE"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_require_llm_model", lambda _m: "test-model"
    )
    monkeypatch.setattr(worker_tasks, "LLMClient", lambda: object())
    monkeypatch.setattr(worker_tasks, "load_speaker_aliases", lambda _p: {})
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", []),
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_structured_summary_prompts",
        lambda *_a, **_k: pytest.fail("LLM should not be invoked when all turns filtered"),
    )

    result = worker_tasks._stage_llm_extract(ctx)  # noqa: SLF001
    assert result.status == "skipped"
    assert result.metadata["skip_reason"] == "no_speech"


def test_stage_llm_extract_preserves_inputs_when_noise_labels_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale noise_speakers that don't match any turn must not swap LLM inputs."""

    _cfg_value, ctx = _new_ctx(tmp_path, "rec-llm-stale")
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.artifacts.language_analysis_json_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello team this is the real speaker speaking now",
                    }
                ],
                "target_summary_language": "en",
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_REAL",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello team this is the real speaker speaking now",
                }
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False, "noise_speakers": ["SPEAKER_GHOST"]}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _capture_prompts(turns, summary_lang, *, calendar_title=None, calendar_attendees=None):
        captured["turns"] = list(turns)
        return ("sys", "user")

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_require_llm_model", lambda _m: "test-model"
    )
    monkeypatch.setattr(worker_tasks, "LLMClient", lambda: object())
    monkeypatch.setattr(worker_tasks, "load_speaker_aliases", lambda _p: {})
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_sentiment_score", lambda _t: 3
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_use_chunked_llm", lambda *_a, **_k: False
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", []),
    )
    monkeypatch.setattr(worker_tasks, "build_structured_summary_prompts", _capture_prompts)

    async def _fake_generate(*_a, **_k):
        return {"content": json.dumps({"topic": "ok", "summary_bullets": ["x"]})}

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_generate_llm_message", _fake_generate
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_summary_payload",
        lambda **_k: {"status": "ok", "topic": "ok"},
    )
    monkeypatch.setattr(
        worker_tasks,
        "_set_recording_progress_best_effort",
        lambda *_a, **_k: None,
    )

    result = worker_tasks._stage_llm_extract(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert captured["turns"] == ctx.speaker_turns


@pytest.mark.parametrize("noise_payload", [None, []])
def test_stage_llm_extract_skips_filter_when_noise_metadata_unusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    noise_payload: object,
) -> None:
    """Filter skip paths: noise_speakers missing or empty should not drop turns."""

    suffix = "none" if noise_payload is None else "empty"
    _cfg_value, ctx = _new_ctx(tmp_path, f"rec-llm-noise-skip-{suffix}")
    ctx.pipeline_settings.exclude_noise_speakers_from_transcript = True
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.artifacts.language_analysis_json_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello team this is the real speaker speaking now",
                    }
                ],
                "target_summary_language": "en",
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "S1",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello team this is the real speaker speaking now",
                }
            ]
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.diarization_metadata_json_path.write_text(
        json.dumps({"degraded": False, "noise_speakers": noise_payload}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _capture_prompts(turns, summary_lang, *, calendar_title=None, calendar_attendees=None):
        captured["turns"] = list(turns)
        return ("sys", "user")

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_require_llm_model", lambda _m: "test-model"
    )
    monkeypatch.setattr(worker_tasks, "LLMClient", lambda: object())
    monkeypatch.setattr(worker_tasks, "load_speaker_aliases", lambda _p: {})
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_sentiment_score", lambda _text: 3
    )
    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_use_chunked_llm", lambda *_a, **_k: False
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", []),
    )
    monkeypatch.setattr(worker_tasks, "build_structured_summary_prompts", _capture_prompts)

    async def _fake_generate(*_a, **_k):
        return {"content": json.dumps({"topic": "ok", "summary_bullets": ["x"]})}

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator, "_generate_llm_message", _fake_generate
    )
    monkeypatch.setattr(
        worker_tasks,
        "build_summary_payload",
        lambda **_k: {"status": "ok", "topic": "ok"},
    )
    monkeypatch.setattr(
        worker_tasks,
        "_set_recording_progress_best_effort",
        lambda *_a, **_k: None,
    )

    result = worker_tasks._stage_llm_extract(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert any(str(turn.get("speaker")) == "S1" for turn in captured["turns"])


def test_stage_llm_extract_chunked_summary_invokes_progress_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg_value, ctx = _new_ctx(tmp_path, "rec-llm-chunked")
    ctx.precheck_result = PrecheckResult(10.0, 0.5, None)
    ctx.artifacts.language_analysis_json_path.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "Discussed the work plan."}],
                "target_summary_language": "en",
            }
        ),
        encoding="utf-8",
    )
    ctx.artifacts.recording_artifacts.speaker_turns_json_path.write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "Discussed the work plan."}]),
        encoding="utf-8",
    )

    progress_updates: list[tuple[str, float]] = []
    chunked_kwargs: dict[str, object] = {}
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_require_llm_model", lambda _model: "test-model")
    monkeypatch.setattr(worker_tasks, "LLMClient", lambda: object())
    monkeypatch.setattr(worker_tasks, "load_speaker_aliases", lambda _path: {})
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_sentiment_score", lambda _text: 3)
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_speaker_turn_prompt_text", lambda *_a, **_k: "Prompt text")
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_use_chunked_llm", lambda *_a, **_k: True)
    monkeypatch.setattr(
        worker_tasks,
        "_load_calendar_summary_context",
        lambda *_a, **_k: ("Weekly Sync", ["Ada", "Bob"]),
    )
    monkeypatch.setattr(
        worker_tasks,
        "_set_recording_progress_best_effort",
        lambda _recording_id, stage, progress, settings: progress_updates.append((stage, progress)),
    )

    async def _fake_run_chunked_llm_summary(*, progress_callback, **_kwargs):
        chunked_kwargs.update(_kwargs)
        progress_callback("llm_chunk_1", 0.91)
        return {"status": "ok", "topic": "Weekly Sync"}

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_run_chunked_llm_summary",
        _fake_run_chunked_llm_summary,
    )

    result = worker_tasks._stage_llm_extract(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert ("llm_chunk_1", 0.91) in progress_updates
    assert chunked_kwargs["speaker_turns"] == [
        {"speaker": "S1", "start": 0.0, "end": 1.0, "text": "Discussed the work plan."}
    ]
    assert chunked_kwargs["aliases"] == {}
    assert chunked_kwargs["calendar_title"] == "Weekly Sync"
    assert chunked_kwargs["calendar_attendees"] == ["Ada", "Bob"]


def test_stage_metrics_updates_language_settings_only_when_payload_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-metrics-update")
    ctx.has_explicit_summary_target = True
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        worker_tasks,
        "refresh_recording_metrics",
        lambda *_a, **_k: {"participants": [], "meeting": {"total_interruptions": 0}},
    )
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_language_settings",
        lambda _recording_id, settings, **kwargs: calls.append(kwargs),
    )

    ctx.artifacts.recording_artifacts.transcript_json_path.write_text(
        json.dumps({"target_summary_language": "it"}),
        encoding="utf-8",
    )
    result = worker_tasks._stage_metrics(ctx)  # noqa: SLF001
    assert result.status == "completed"
    assert calls == [{"target_summary_language": "it"}]

    _cfg_value, ctx_no_update = _new_ctx(tmp_path, "rec-metrics-no-update")
    ctx_no_update.has_explicit_summary_target = False
    monkeypatch.setattr(
        worker_tasks,
        "set_recording_language_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected update")),
    )
    ctx_no_update.artifacts.recording_artifacts.transcript_json_path.write_text("{}", encoding="utf-8")
    result = worker_tasks._stage_metrics(ctx_no_update)  # noqa: SLF001
    assert result.status == "completed"


def test_terminal_state_and_legacy_pipeline_handle_review_and_missing_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-terminal-review")
    ctx.routing_payload = {"status_after_routing": RECORDING_STATUS_NEEDS_REVIEW}
    monkeypatch.setattr(
        worker_tasks,
        "_review_reason_from_routing",
        lambda **_kwargs: ("routing_review", "Routing requires review"),
    )

    outcome = worker_tasks._terminal_state_from_stage_artifacts(ctx=ctx)  # noqa: SLF001
    assert outcome.status == RECORDING_STATUS_NEEDS_REVIEW
    assert outcome.review_reason_code == "routing_review"

    cfg_missing, _ctx_unused = _new_ctx(tmp_path, "rec-legacy-missing-audio")
    legacy_outcome = worker_tasks._run_precheck_pipeline_legacy(  # noqa: SLF001
        recording_id="rec-legacy-missing-audio",
        settings=cfg_missing,
        log_path=worker_tasks._step_log_path("rec-legacy-missing-audio", JOB_TYPE_PRECHECK, cfg_missing),
    )
    assert legacy_outcome.status == RECORDING_STATUS_QUARANTINE
    assert legacy_outcome.quarantine_reason == "raw_audio_missing"


def test_run_precheck_pipeline_covers_all_valid_and_invalidated_stage_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    stage = pipeline_stages.PIPELINE_STAGE_DEFINITIONS[0]
    monkeypatch.setattr(worker_tasks, "PIPELINE_STAGE_DEFINITIONS", (stage,))

    create_recording("rec-pipeline-valid", source="test", source_filename="valid.wav", settings=cfg)
    mark_recording_pipeline_stage_skipped(
        "rec-pipeline-valid",
        stage_name=stage.name,
        metadata={"skip_reason": "already_done"},
        settings=cfg,
    )
    monkeypatch.setattr(
        worker_tasks,
        "_terminal_state_from_stage_artifacts",
        lambda **_kwargs: worker_tasks.PipelineTerminalState(status=RECORDING_STATUS_READY),
    )
    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-pipeline-valid",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-pipeline-valid", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == RECORDING_STATUS_READY

    create_recording("rec-pipeline-invalid", source="test", source_filename="invalid.wav", settings=cfg)
    mark_recording_pipeline_stage_completed(
        "rec-pipeline-invalid",
        stage_name=stage.name,
        metadata={"label": stage.label},
        settings=cfg,
    )
    monkeypatch.setitem(
        worker_tasks._PIPELINE_STAGE_RUNNERS,
        stage.name,
        lambda _ctx: worker_tasks._StageResult(status="completed"),
    )
    monkeypatch.setattr(
        worker_tasks,
        "mark_recording_pipeline_stage_completed",
        lambda *_a, **_k: {"duration_ms": "bad"},
    )
    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-pipeline-invalid",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-pipeline-invalid", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == RECORDING_STATUS_READY

    log_text = (
        cfg.recordings_root / "rec-pipeline-invalid" / "logs" / "step-precheck.log"
    ).read_text(encoding="utf-8")
    assert "stage invalidated: sanitize_audio" in log_text
    assert "stage completed: sanitize_audio" in log_text

    create_recording("rec-pipeline-none-row", source="test", source_filename="none.wav", settings=cfg)
    monkeypatch.setattr(
        worker_tasks,
        "mark_recording_pipeline_stage_completed",
        lambda *_a, **_k: None,
    )
    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-pipeline-none-row",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-pipeline-none-row", JOB_TYPE_PRECHECK, cfg),
    )
    assert outcome.status == RECORDING_STATUS_READY


def test_run_precheck_pipeline_stops_after_completed_stage_when_cancel_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-pipeline-stop-1",
        source="test",
        source_filename="stop.wav",
        settings=cfg,
    )
    stages = (
        pipeline_stages.PIPELINE_STAGE_DEFINITIONS[0],
        pipeline_stages.PIPELINE_STAGE_DEFINITIONS[1],
    )
    monkeypatch.setattr(worker_tasks, "PIPELINE_STAGE_DEFINITIONS", stages)
    stage2_called = {"value": False}

    def _stage_one(_ctx):
        set_recording_cancel_request(
            "rec-pipeline-stop-1",
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
            settings=cfg,
        )
        return worker_tasks._StageResult(status="completed")  # noqa: SLF001

    def _stage_two(_ctx):
        stage2_called["value"] = True
        return worker_tasks._StageResult(status="completed")  # noqa: SLF001

    monkeypatch.setitem(worker_tasks._PIPELINE_STAGE_RUNNERS, stages[0].name, _stage_one)
    monkeypatch.setitem(worker_tasks._PIPELINE_STAGE_RUNNERS, stages[1].name, _stage_two)

    outcome = worker_tasks._run_precheck_pipeline(  # noqa: SLF001
        recording_id="rec-pipeline-stop-1",
        settings=cfg,
        log_path=worker_tasks._step_log_path("rec-pipeline-stop-1", JOB_TYPE_PRECHECK, cfg),
    )

    assert outcome.status == RECORDING_STATUS_STOPPED
    assert stage2_called["value"] is False
    stage_rows = list_recording_pipeline_stages("rec-pipeline-stop-1", settings=cfg)
    assert [(row["stage_name"], row["status"]) for row in stage_rows] == [
        (stages[0].name, "completed")
    ]
    recording = worker_tasks.get_recording("rec-pipeline-stop-1", settings=cfg) or {}
    assert recording["cancel_reason_text"] == "Cancelled by user"
    log_text = (
        cfg.recordings_root / "rec-pipeline-stop-1" / "logs" / "step-precheck.log"
    ).read_text(encoding="utf-8")
    assert "cancelled_by_user stage=sanitize_audio checkpoint=after_stage" in log_text


def test_stage_llm_extract_stop_after_current_chunk_marks_completed_chunk_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, ctx = _new_ctx(tmp_path, "rec-llm-stop-1")
    ctx.precheck_result = PrecheckResult(
        duration_sec=45.0,
        speech_ratio=0.8,
        quarantine_reason=None,
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_language_analysis_artifact",
        lambda _ctx: {
            "segments": [{"text": "hello world today"}],
            "target_summary_language": "en",
        },
    )
    monkeypatch.setattr(
        worker_tasks,
        "_load_json_list",
        lambda _path: [{"speaker": "S1", "text": "hello world today", "start": 0.0, "end": 1.0}],
    )
    monkeypatch.setattr(worker_tasks, "_load_calendar_summary_context", lambda *_a, **_k: ("Weekly Sync", ["Ada"]))
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_require_llm_model", lambda _model: "test-model")
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_sentiment_score", lambda _text: 3)
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_speaker_turn_prompt_text", lambda *_a, **_k: "Prompt text")
    monkeypatch.setattr(worker_tasks.pipeline_orchestrator, "_use_chunked_llm", lambda *_a, **_k: True)
    monkeypatch.setattr(worker_tasks, "load_speaker_aliases", lambda _path: {})

    class _FakeLLM:
        async def generate(self, *_args, **_kwargs):
            return {"content": "{}"}

    monkeypatch.setattr(worker_tasks, "LLMClient", lambda: _FakeLLM())
    seen = {"second_chunk_started": False}

    async def _fake_run_chunked_llm_summary(*, llm, chunk_state_store, **_kwargs):
        chunk_state_store.mark_started(
            chunk_group="extract",
            chunk_index="1",
            chunk_total=2,
            metadata={"chunk_id": "1"},
        )
        await llm.generate(system_prompt="sys", user_prompt="user", model="test-model")
        set_recording_cancel_request(
            "rec-llm-stop-1",
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
            settings=cfg,
        )
        worker_tasks.set_recording_status(
            "rec-llm-stop-1",
            RECORDING_STATUS_STOPPING,
            settings=cfg,
        )
        chunk_state_store.mark_completed(
            chunk_group="extract",
            chunk_index="1",
            chunk_total=2,
            metadata={"chunk_id": "1"},
        )
        seen["second_chunk_started"] = True
        chunk_state_store.mark_started(
            chunk_group="extract",
            chunk_index="2",
            chunk_total=2,
            metadata={"chunk_id": "2"},
        )
        return {"status": "ok"}

    monkeypatch.setattr(
        worker_tasks.pipeline_orchestrator,
        "_run_chunked_llm_summary",
        _fake_run_chunked_llm_summary,
    )

    with pytest.raises(worker_tasks.RecordingStopRequested):
        worker_tasks._stage_llm_extract(ctx)  # noqa: SLF001

    assert seen["second_chunk_started"] is False
    chunk_rows = list_recording_llm_chunk_states("rec-llm-stop-1", settings=cfg)
    assert [(row["chunk_index"], row["status"]) for row in chunk_rows] == [("1", "completed")]
