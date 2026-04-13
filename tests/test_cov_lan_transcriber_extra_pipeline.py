from __future__ import annotations

import asyncio
import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from lan_app.config import AppSettings
from lan_app.db import (
    create_recording,
    init_db,
    list_recording_llm_chunk_states,
    upsert_recording_llm_chunk_state,
)
from lan_transcriber import llm_client
from lan_transcriber.pipeline_steps import language, orchestrator as pipeline, precheck, snippets, speaker_turns, summary_builder


def _settings(tmp_path: Path, **overrides: Any) -> pipeline.Settings:
    defaults = {
        "speaker_db": tmp_path / "db.yaml",
        "tmp_root": tmp_path / "tmp",
        "recordings_root": tmp_path / "recordings",
    }
    defaults.update(overrides)
    return pipeline.Settings(**defaults)


def _db_settings(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path / "data",
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


class _SqliteChunkStateStore:
    def __init__(self, recording_id: str, settings: AppSettings) -> None:
        self.recording_id = recording_id
        self.settings = settings

    def list_states(self, *, chunk_group: str) -> list[dict[str, Any]]:
        return list_recording_llm_chunk_states(
            self.recording_id,
            chunk_group=chunk_group,
            settings=self.settings,
        )

    def upsert_state(self, **kwargs: Any) -> dict[str, Any] | None:
        return upsert_recording_llm_chunk_state(
            self.recording_id,
            settings=self.settings,
            **kwargs,
        )

    def mark_started(self, **kwargs: Any) -> dict[str, Any] | None:
        from lan_app.db import mark_recording_llm_chunk_started

        return mark_recording_llm_chunk_started(
            self.recording_id,
            settings=self.settings,
            **kwargs,
        )

    def mark_completed(self, **kwargs: Any) -> dict[str, Any] | None:
        from lan_app.db import mark_recording_llm_chunk_completed

        return mark_recording_llm_chunk_completed(
            self.recording_id,
            settings=self.settings,
            **kwargs,
        )

    def mark_failed(self, **kwargs: Any) -> dict[str, Any] | None:
        from lan_app.db import mark_recording_llm_chunk_failed

        return mark_recording_llm_chunk_failed(
            self.recording_id,
            settings=self.settings,
            **kwargs,
        )

    def mark_split(self, **kwargs: Any) -> dict[str, Any] | None:
        from lan_app.db import mark_recording_llm_chunk_split

        return mark_recording_llm_chunk_split(
            self.recording_id,
            settings=self.settings,
            **kwargs,
        )

    def clear_states(self, *, chunk_group: str | None = None) -> int:
        from lan_app.db import clear_recording_llm_chunk_states

        return clear_recording_llm_chunk_states(
            self.recording_id,
            chunk_group=chunk_group,
            settings=self.settings,
        )


def _chunk_store(tmp_path: Path, recording_id: str) -> tuple[AppSettings, _SqliteChunkStateStore]:
    cfg = _db_settings(tmp_path)
    init_db(cfg)
    create_recording(
        recording_id,
        source="test",
        source_filename="audio.wav",
        settings=cfg,
    )
    return cfg, _SqliteChunkStateStore(recording_id, cfg)


def _annotation_from_segments(*segments: tuple[float, float, str]):
    rows = []
    segment_only = []
    for index, (start, end, speaker) in enumerate(segments, start=1):
        segment = SimpleNamespace(start=start, end=end)
        segment_only.append((segment,))
        if index % 2:
            rows.append((segment, speaker))
        else:
            rows.append((segment, f"track-{index}", speaker))
    return SimpleNamespace(
        itertracks=lambda yield_label=False: iter(rows if yield_label else segment_only)
    )


def _llm_chunk_turns(text: str = "single chunk transcript") -> list[dict[str, Any]]:
    return [
        {
            "speaker": "S1",
            "start": 0.0,
            "end": 1.0,
            "text": text,
        }
    ]


def test_pipeline_settings_reads_llm_model_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.delenv("LAN_LLM_MODEL", raising=False)
    cfg = _settings(tmp_path)
    assert cfg.llm_model == "env-model"


def test_pipeline_settings_allows_direct_llm_model_override(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, llm_model="direct-model")
    assert cfg.llm_model == "direct-model"


def test_pipeline_settings_read_llm_chunking_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_CHUNK_MAX_CHARS", "4096")
    monkeypatch.setenv("LLM_CHUNK_OVERLAP_CHARS", "256")
    monkeypatch.setenv("LLM_CHUNK_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("LLM_CHUNK_SPLIT_MIN_CHARS", "900")
    monkeypatch.setenv("LLM_CHUNK_SPLIT_MAX_DEPTH", "3")
    monkeypatch.setenv("LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS", "8192")
    monkeypatch.setenv("LLM_MERGE_MAX_TOKENS", "3072")

    cfg = _settings(tmp_path)
    assert cfg.llm_chunk_max_chars == 4096
    assert cfg.llm_chunk_overlap_chars == 256
    assert cfg.llm_chunk_timeout_seconds == 45.0
    assert cfg.llm_chunk_split_min_chars == 900
    assert cfg.llm_chunk_split_max_depth == 3
    assert cfg.llm_long_transcript_threshold_chars == 8192
    assert cfg.llm_merge_max_tokens == 3072


def test_pipeline_settings_read_diarization_quality_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LAN_DIARIZATION_PROFILE", "meeting")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_SPEAKERS", "3")
    monkeypatch.setenv("LAN_DIARIZATION_MAX_SPEAKERS", "5")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_DURATION_SECONDS", "11.5")
    monkeypatch.setenv("LAN_DIARIZATION_DIALOG_RETRY_MIN_TURNS", "4")
    monkeypatch.setenv("LAN_DIARIZATION_MERGE_GAP_SECONDS", "0.6")
    monkeypatch.setenv("LAN_DIARIZATION_MIN_TURN_SECONDS", "0.4")

    cfg = _settings(tmp_path)
    assert cfg.diarization_profile == "meeting"
    assert cfg.diarization_min_speakers == 3
    assert cfg.diarization_max_speakers == 5
    assert cfg.diarization_dialog_retry_min_duration_seconds == 11.5
    assert cfg.diarization_dialog_retry_min_turns == 4
    assert cfg.diarization_merge_gap_seconds == 0.6
    assert cfg.diarization_min_turn_seconds == 0.4


@pytest.mark.asyncio
async def test_orchestrator_llm_helper_supports_sync_generate_and_skips_blank_turns(
    tmp_path: Path,
) -> None:
    class _SyncLLM:
        def generate(self, **_kwargs: Any) -> str:
            return "plain"

    message = await pipeline._generate_llm_message(
        _SyncLLM(),
        system_prompt="sys",
        user_prompt="usr",
        model="m",
        response_format=None,
        max_tokens=256,
    )
    assert message == {"role": "assistant", "content": "plain"}

    prompt_text = pipeline._speaker_turn_prompt_text(
        [
            {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "hello"},
            {"start": 1.0, "end": 2.0, "speaker": "S2", "text": "   "},
        ],
        aliases={"S1": "Alex", "S2": "Priya"},
    )
    assert prompt_text == "[0.00-1.00] Alex: hello"
    assert pipeline._llm_timeout_message(None) == "timed out"


@pytest.mark.asyncio
async def test_orchestrator_retry_helper_skips_non_retryable_cases_and_tolerates_log_failures():
    pipeline._best_effort_step_log(
        lambda _message: (_ for _ in ()).throw(RuntimeError("log failed")),
        "ignored",
    )

    class _NoRetryDiariser:
        dialog_retry_min_turns = 4
        dialog_retry_min_duration_seconds = 15.0
        last_run_metadata = {
            "diarization_profile": "auto",
            "initial_profile": "meeting",
            "auto_profile_enabled": True,
            "effective_hints": "not-a-dict",
            "speaker_count_before_retry": 1,
        }

        async def retry_dialog(self, _audio_path: Path):
            raise AssertionError("retry_dialog should not run")

    diarization = object()
    result = await pipeline._maybe_retry_dialog_diarization(
        diariser=_NoRetryDiariser(),
        audio_path=Path("/tmp/audio.wav"),
        diarization=diarization,
        asr_segments=[{"text": "only one"}],
        precheck_result=precheck.PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
        step_log_callback=None,
    )

    assert result is diarization


@pytest.mark.asyncio
async def test_orchestrator_retry_helper_skips_forced_meeting_configs():
    class _MeetingDiariser:
        dialog_retry_min_turns = 4
        dialog_retry_min_duration_seconds = 15.0
        last_run_metadata = {
            "diarization_profile": "meeting",
            "initial_profile": "meeting",
            "auto_profile_enabled": False,
            "override_reason": "profile_forced_meeting",
            "initial_hints": {"min_speakers": 2, "max_speakers": 6},
            "effective_hints": {"min_speakers": 2, "max_speakers": 6},
            "speaker_count_before_retry": 1,
        }

        async def retry_dialog(self, _audio_path: Path):
            raise AssertionError("retry_dialog should not run")

    diarization = _annotation_from_segments((0.0, 30.0, "S1"))
    diariser = _MeetingDiariser()
    result = await pipeline._maybe_retry_dialog_diarization(
        diariser=diariser,
        audio_path=Path("/tmp/audio.wav"),
        diarization=diarization,
        asr_segments=[{"text": "first"}, {"text": "second"}, {"text": "third"}, {"text": "fourth"}],
        precheck_result=precheck.PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
        step_log_callback=None,
    )

    assert result is diarization
    assert diariser.last_run_metadata["selected_profile"] == "meeting"
    assert diariser.last_run_metadata["profile_selection"]["dialog_retry_attempted"] is False
    assert diariser.last_run_metadata["profile_selection"]["winner_reason"] == "profile_forced_meeting"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_profile", "override_reason"),
    [
        ("dialog", "profile_forced_dialog"),
        ("auto", "explicit_speaker_hints"),
    ],
)
async def test_orchestrator_retry_helper_preserves_retry_for_forced_dialog_configs(
    requested_profile: str,
    override_reason: str,
):
    class _RetryingDiariser:
        dialog_retry_min_turns = 4
        dialog_retry_min_duration_seconds = 15.0

        def __init__(self):
            self.retry_calls = 0
            self.last_run_metadata = {
                "diarization_profile": requested_profile,
                "initial_profile": "dialog",
                "auto_profile_enabled": False,
                "override_reason": override_reason,
                "initial_hints": {"min_speakers": 2, "max_speakers": 2},
                "effective_hints": {"min_speakers": 2, "max_speakers": 2},
                "speaker_count_before_retry": 1,
                "dialog_retry_used": False,
            }

        async def retry_dialog(self, _audio_path: Path):
            self.retry_calls += 1
            self.last_run_metadata.update(
                {
                    "retry_hints": {"min_speakers": 2, "max_speakers": 2},
                    "effective_hints": {"min_speakers": 2, "max_speakers": 2},
                    "dialog_retry_used": True,
                    "speaker_count_after_retry": 2,
                }
            )
            return _annotation_from_segments(
                (0.0, 1.0, "S1"),
                (1.0, 2.0, "S2"),
                (2.0, 3.0, "S1"),
                (3.0, 4.0, "S2"),
            )

    diarization = _annotation_from_segments((0.0, 30.0, "S1"))
    diariser = _RetryingDiariser()
    result = await pipeline._maybe_retry_dialog_diarization(
        diariser=diariser,
        audio_path=Path("/tmp/audio.wav"),
        diarization=diarization,
        asr_segments=[{"text": "first"}, {"text": "second"}, {"text": "third"}, {"text": "fourth"}],
        precheck_result=precheck.PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
        step_log_callback=None,
    )

    assert result is not diarization
    assert diariser.retry_calls == 1
    assert diariser.last_run_metadata["selected_profile"] == "dialog"
    assert diariser.last_run_metadata["dialog_retry_used"] is True
    assert diariser.last_run_metadata["effective_hints"] == {
        "min_speakers": 2,
        "max_speakers": 2,
    }
    assert diariser.last_run_metadata["profile_selection"]["dialog_retry_attempted"] is True
    assert diariser.last_run_metadata["profile_selection"]["selected_result"] == "dialog_retry"


@pytest.mark.asyncio
async def test_orchestrator_retry_helper_keeps_forced_dialog_profile_when_retry_floor_not_met():
    class _ForcedDialogDiariser:
        dialog_retry_min_turns = 4
        dialog_retry_min_duration_seconds = 15.0

        def __init__(self):
            self.retry_calls = 0
            self.last_run_metadata = {
                "diarization_profile": "dialog",
                "initial_profile": "dialog",
                "auto_profile_enabled": False,
                "override_reason": "profile_forced_dialog",
                "initial_hints": {"min_speakers": 2, "max_speakers": 2},
                "effective_hints": {"min_speakers": 2, "max_speakers": 2},
                "speaker_count_before_retry": 2,
                "dialog_retry_used": False,
            }

        async def retry_dialog(self, _audio_path: Path):
            self.retry_calls += 1
            raise AssertionError("retry_dialog should not run")

    diarization = _annotation_from_segments(
        (0.0, 1.0, "S1"),
        (1.0, 2.0, "S2"),
        (2.0, 3.0, "S1"),
        (3.0, 4.0, "S2"),
    )
    diariser = _ForcedDialogDiariser()
    result = await pipeline._maybe_retry_dialog_diarization(
        diariser=diariser,
        audio_path=Path("/tmp/audio.wav"),
        diarization=diarization,
        asr_segments=[{"text": "first"}, {"text": "second"}, {"text": "third"}, {"text": "fourth"}],
        precheck_result=precheck.PrecheckResult(
            duration_sec=4.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
        step_log_callback=None,
    )

    assert result is diarization
    assert diariser.retry_calls == 0
    assert diariser.last_run_metadata["selected_profile"] == "dialog"
    assert diariser.last_run_metadata["profile_selection"]["classification_reason"] == (
        "dialog_like_below_min_duration"
    )
    assert diariser.last_run_metadata["profile_selection"]["selected_result"] == "initial_pass"


@pytest.mark.asyncio
async def test_orchestrator_retry_helper_keeps_initial_diarization_when_retry_fails():
    segment = SimpleNamespace(start=0.0, end=30.0)

    class _RetryingDiariser:
        dialog_retry_min_turns = 4
        dialog_retry_min_duration_seconds = 15.0
        last_run_metadata = {
            "diarization_profile": "auto",
            "initial_profile": "meeting",
            "auto_profile_enabled": True,
            "initial_hints": {"min_speakers": 2, "max_speakers": 6},
            "effective_hints": {"min_speakers": 2, "max_speakers": 6},
            "speaker_count_before_retry": 1,
        }

        async def retry_dialog(self, _audio_path: Path):
            raise RuntimeError("retry boom")

    diarization = SimpleNamespace(
        itertracks=lambda yield_label=False: (
            iter([(segment, "S1")]) if yield_label else iter([(segment,)])
        )
    )
    step_messages: list[str] = []
    result = await pipeline._maybe_retry_dialog_diarization(
        diariser=_RetryingDiariser(),
        audio_path=Path("/tmp/audio.wav"),
        diarization=diarization,
        asr_segments=[
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
            {"text": "fourth"},
        ],
        precheck_result=precheck.PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
        step_log_callback=step_messages.append,
    )

    assert result is diarization
    assert step_messages == [
        "diarization auto-profile retry classification=single_speaker_long_recording min_speakers=2 max_speakers=2",
        "diarization dialog retry failed: retry boom",
    ]


@pytest.mark.asyncio
async def test_orchestrator_retry_helper_uses_default_retry_hints_when_missing():
    first_segment = SimpleNamespace(start=0.0, end=30.0)

    class _RetryingDiariser:
        dialog_retry_min_turns = 4
        dialog_retry_min_duration_seconds = 15.0
        last_run_metadata = {
            "diarization_profile": "auto",
            "initial_profile": "meeting",
            "auto_profile_enabled": True,
            "initial_hints": {"min_speakers": 2, "max_speakers": 6},
            "effective_hints": {"min_speakers": 2, "max_speakers": 6},
            "speaker_count_before_retry": 1,
        }

        async def retry_dialog(self, _audio_path: Path):
            retry_segments = [
                SimpleNamespace(start=0.0, end=1.0),
                SimpleNamespace(start=1.0, end=2.0),
                SimpleNamespace(start=2.0, end=3.0),
            ]
            return SimpleNamespace(
                itertracks=lambda yield_label=False: (
                    iter(
                        [
                            (retry_segments[0], "S1"),
                            (retry_segments[1], "S2"),
                            (retry_segments[2], "S1"),
                        ]
                    )
                    if yield_label
                    else iter([(retry_segments[0],), (retry_segments[1],), (retry_segments[2],)])
                )
            )

    diarization = SimpleNamespace(
        itertracks=lambda yield_label=False: (
            iter([(first_segment, "S1")]) if yield_label else iter([(first_segment,)])
        )
    )
    diariser = _RetryingDiariser()

    result = await pipeline._maybe_retry_dialog_diarization(
        diariser=diariser,
        audio_path=Path("/tmp/audio.wav"),
        diarization=diarization,
        asr_segments=[
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
            {"text": "fourth"},
        ],
        precheck_result=precheck.PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
        step_log_callback=None,
    )

    assert result is not diarization
    assert diariser.last_run_metadata["effective_hints"] == {
        "min_speakers": 2,
        "max_speakers": 2,
    }


@pytest.mark.asyncio
async def test_orchestrator_retry_helper_clears_retry_used_when_initial_pass_wins():
    first_segments = [
        SimpleNamespace(start=0.0, end=1.0),
        SimpleNamespace(start=1.0, end=2.0),
        SimpleNamespace(start=2.0, end=3.0),
        SimpleNamespace(start=3.0, end=4.0),
    ]

    def _annotation_from_rows(rows: list[tuple[SimpleNamespace, str]]):
        return SimpleNamespace(
            itertracks=lambda yield_label=False: (
                iter(rows) if yield_label else iter([(segment,) for segment, _speaker in rows])
            )
        )

    initial_diarization = _annotation_from_rows(
        [
            (first_segments[0], "S1"),
            (first_segments[1], "S2"),
            (first_segments[2], "S1"),
            (first_segments[3], "S2"),
        ]
    )

    class _RetryingDiariser:
        dialog_retry_min_turns = 4
        dialog_retry_min_duration_seconds = 15.0
        last_run_metadata = {
            "diarization_profile": "auto",
            "initial_profile": "meeting",
            "auto_profile_enabled": True,
            "initial_hints": {"min_speakers": 2, "max_speakers": 6},
            "effective_hints": {"min_speakers": 2, "max_speakers": 6},
            "speaker_count_before_retry": 2,
            "dialog_retry_used": False,
        }

        async def retry_dialog(self, _audio_path: Path):
            self.last_run_metadata.update(
                {
                    "retry_hints": {"min_speakers": 2, "max_speakers": 2},
                    "effective_hints": {"min_speakers": 2, "max_speakers": 2},
                    "dialog_retry_used": True,
                    "speaker_count_after_retry": 3,
                }
            )
            retry_segments = [
                SimpleNamespace(start=0.0, end=1.0),
                SimpleNamespace(start=1.0, end=2.0),
                SimpleNamespace(start=2.0, end=3.0),
                SimpleNamespace(start=3.0, end=3.3),
            ]
            return _annotation_from_rows(
                [
                    (retry_segments[0], "S1"),
                    (retry_segments[1], "S2"),
                    (retry_segments[2], "S1"),
                    (retry_segments[3], "S3"),
                ]
            )

    diariser = _RetryingDiariser()
    result = await pipeline._maybe_retry_dialog_diarization(
        diariser=diariser,
        audio_path=Path("/tmp/audio.wav"),
        diarization=initial_diarization,
        asr_segments=[
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
            {"text": "fourth"},
        ],
        precheck_result=precheck.PrecheckResult(
            duration_sec=30.0,
            speech_ratio=0.5,
            quarantine_reason=None,
        ),
        step_log_callback=None,
    )

    assert result is initial_diarization
    assert diariser.last_run_metadata["dialog_retry_used"] is False
    assert diariser.last_run_metadata["effective_hints"] == {
        "min_speakers": 2,
        "max_speakers": 6,
    }
    assert diariser.last_run_metadata["profile_selection"]["selected_result"] == "initial_pass"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_rejects_empty_chunk_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="produced no chunks"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="hello world",
            speaker_turns=_llm_chunk_turns("hello world"),
            aliases={},
            derived_dir=tmp_path / "derived",
            llm=object(),
            cfg=_settings(tmp_path, llm_model="model"),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_rejects_empty_compacted_transcript(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="compaction produced no usable content"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="legacy transcript",
            speaker_turns=[{"speaker": "S1", "start": 0.0, "end": 0.2, "text": "..."}],
            aliases={},
            derived_dir=tmp_path / "derived",
            llm=object(),
            cfg=_settings(tmp_path, llm_model="model"),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_writes_compact_artifacts_and_keeps_calendar_context(
    tmp_path: Path,
) -> None:
    prompts: list[dict[str, Any]] = []

    class _CompactLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            prompts.append(payload)
            if "chunk" in payload:
                return {
                    "content": json.dumps(
                        {
                            "topic_candidates": ["Weekly Sync"],
                            "summary_bullets": ["Reviewed blockers"],
                            "decisions": [],
                            "action_items": [],
                            "emotional_cues": ["Focused"],
                            "questions": {"total_count": 0, "types": {}, "extracted": []},
                        }
                    )
                }
            return {
                "content": json.dumps(
                    {
                        "topic": "Merged topic",
                        "summary_bullets": ["Merged summary bullet"],
                        "decisions": [],
                        "action_items": [],
                        "emotional_summary": "Focused.",
                        "questions": {"total_count": 0, "types": {}, "extracted": []},
                    }
                )
            }

    result = await pipeline._run_chunked_llm_summary(
        transcript_text="legacy transcript",
        speaker_turns=[
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 1.0,
                "text": "discussion item zero " * 4,
            },
            {
                "speaker": "SPEAKER_00",
                "start": 1.2,
                "end": 2.0,
                "text": "follow up details " * 3,
            },
            {
                "speaker": "SPEAKER_01",
                "start": 3.0,
                "end": 4.0,
                "text": "response and owner alignment " * 3,
            },
        ],
        aliases={"SPEAKER_00": "Alex", "SPEAKER_01": "Priya"},
        derived_dir=tmp_path / "derived",
        llm=_CompactLLM(),
        cfg=_settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=60,
            llm_chunk_overlap_chars=12,
        ),
        llm_model="model",
        target_summary_language="en",
        friendly=0,
        default_topic="Meeting summary",
        calendar_title="Weekly Sync",
        calendar_attendees=["Alex", "Priya"],
        progress_callback=None,
    )

    derived = tmp_path / "derived"
    plan_payload = json.loads((derived / "llm_chunks_plan.json").read_text(encoding="utf-8"))
    compact_payload = json.loads((derived / "llm_compact_transcript.json").read_text(encoding="utf-8"))
    compact_text = (derived / "llm_compact_transcript.txt").read_text(encoding="utf-8")

    assert result["topic"] == "Merged topic"
    assert plan_payload["source_chars"] > plan_payload["compact_chars"]
    assert plan_payload["compaction"]["speaker_mapping"]
    assert compact_payload["compact_turn_count"] == 2
    assert compact_text.startswith("Alex:")
    assert "[" not in compact_text
    assert prompts[0]["calendar"] == {"title": "Weekly Sync", "attendees": ["Alex", "Priya"]}
    assert prompts[0]["speaker_mapping"]
    assert prompts[0]["chunk"]["time_range"]["start_seconds"] == 0.0
    assert prompts[-1]["calendar"] == {"title": "Weekly Sync", "attendees": ["Alex", "Priya"]}


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_timeout_writes_error_artifact(tmp_path: Path) -> None:
    class _SlowLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            await asyncio.sleep(0.01)
            return {"content": "{}"}

    derived = tmp_path / "derived"
    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed \[llm_chunk_timeout\]: timed out after 0.001s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
            speaker_turns=_llm_chunk_turns(),
            aliases={},
            derived_dir=derived,
            llm=_SlowLLM(),
            cfg=_settings(
                tmp_path,
                llm_model="model",
                llm_chunk_max_chars=100,
                llm_chunk_timeout_seconds=0.001,
            ),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )

    payload = json.loads((derived / "llm_chunk_001_error.json").read_text(encoding="utf-8"))
    assert payload["reason_code"] == "llm_chunk_timeout"
    assert payload["error"] == "timed out after 0.001s"
    assert payload["status"] == "failed"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_asyncio_timeout_writes_error_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_asyncio_timeout(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        raise asyncio.TimeoutError()

    monkeypatch.setattr(pipeline, "_generate_llm_message", _raise_asyncio_timeout)

    derived = tmp_path / "derived"
    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed \[llm_chunk_timeout\]: timed out after 0.001s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
            speaker_turns=_llm_chunk_turns(),
            aliases={},
            derived_dir=derived,
            llm=object(),
            cfg=_settings(
                tmp_path,
                llm_model="model",
                llm_chunk_max_chars=100,
                llm_chunk_timeout_seconds=0.001,
            ),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )

    payload = json.loads((derived / "llm_chunk_001_error.json").read_text(encoding="utf-8"))
    assert payload["reason_code"] == "llm_chunk_timeout"
    assert payload["error"] == "timed out after 0.001s"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_timeout_sentinel_writes_error_artifact(
    tmp_path: Path,
) -> None:
    class _TimeoutSentinelLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"content": "**LLM timeout**", "role": "assistant"}

    derived = tmp_path / "derived"
    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed \[llm_chunk_request_timeout\]: request timed out"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
            speaker_turns=_llm_chunk_turns(),
            aliases={},
            derived_dir=derived,
            llm=_TimeoutSentinelLLM(),
            cfg=_settings(
                tmp_path,
                llm_model="model",
                llm_chunk_max_chars=100,
                llm_chunk_timeout_seconds=0.001,
            ),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )

    assert json.loads((derived / "llm_chunk_001_raw.json").read_text(encoding="utf-8")) == {
        "content": "**LLM timeout**",
        "role": "assistant",
    }
    payload = json.loads((derived / "llm_chunk_001_error.json").read_text(encoding="utf-8"))
    assert payload["reason_code"] == "llm_chunk_request_timeout"
    assert payload["error"] == "request timed out"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_merge_timeout_sentinel_fails(tmp_path: Path) -> None:
    class _MergeTimeoutSentinelLLM:
        timeout = 12.0

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if "chunk" in payload:
                return {
                    "content": json.dumps(
                        {
                            "summary_bullets": ["Chunk 1"],
                            "decisions": [],
                            "action_items": [],
                            "emotional_cues": ["Focused"],
                            "questions": {"total_count": 0, "types": {}, "extracted": []},
                        }
                    )
                }
            return {"content": "**LLM timeout**", "role": "assistant"}

    derived = tmp_path / "derived"
    with pytest.raises(RuntimeError, match=r"LLM merge failed \[llm_merge_request_timeout\]: request timed out after 12s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
            speaker_turns=_llm_chunk_turns(),
            aliases={},
            derived_dir=derived,
            llm=_MergeTimeoutSentinelLLM(),
            cfg=_settings(
                tmp_path,
                llm_model="model",
                llm_chunk_max_chars=100,
            ),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )

    assert json.loads((derived / "llm_merge_raw.json").read_text(encoding="utf-8")) == {
        "content": "**LLM timeout**",
        "role": "assistant",
    }
    assert json.loads((derived / "llm_merge_error.json").read_text(encoding="utf-8")) == {
        "reason_code": "llm_merge_request_timeout",
        "error": "request timed out after 12s",
        "chunk_count": 1,
    }


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_passes_merge_retry_budget(tmp_path: Path) -> None:
    merge_kwargs: dict[str, Any] = {}

    class _MergeBudgetLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if "chunk" in payload:
                return {
                    "content": json.dumps(
                        {
                            "summary_bullets": ["Chunk 1"],
                            "decisions": [],
                            "action_items": [],
                            "emotional_cues": ["Focused"],
                            "questions": {"total_count": 0, "types": {}, "extracted": []},
                        }
                    )
                }
            merge_kwargs.update(kwargs)
            return {
                "content": json.dumps(
                    {
                        "topic": "Merged topic",
                        "summary_bullets": ["Merged summary bullet"],
                        "decisions": [],
                        "action_items": [],
                        "emotional_summary": "Focused.",
                        "questions": {"total_count": 0, "types": {}, "extracted": []},
                    }
                )
            }

    await pipeline._run_chunked_llm_summary(
        transcript_text="single chunk transcript",
        speaker_turns=_llm_chunk_turns(),
        aliases={},
        derived_dir=tmp_path / "derived",
        llm=_MergeBudgetLLM(),
        cfg=_settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=100,
            llm_max_tokens=1024,
            llm_max_tokens_retry=2048,
            llm_merge_max_tokens=3072,
        ),
        llm_model="model",
        target_summary_language="en",
        friendly=0,
        default_topic="Meeting summary",
        calendar_title=None,
        calendar_attendees=[],
        progress_callback=None,
    )

    assert merge_kwargs["max_tokens"] == 3072
    assert merge_kwargs["max_tokens_retry"] == 3072


def _chunk_extract_message(label: str) -> dict[str, str]:
    return {
        "content": json.dumps(
            {
                "topic_candidates": [f"Topic {label}"],
                "summary_bullets": [f"Bullet {label}"],
                "decisions": [f"Decision {label}"],
                "action_items": [],
                "emotional_cues": [f"Cue {label}"],
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        )
    }


def _merge_message(topic: str = "Merged topic") -> dict[str, str]:
    return {
        "content": json.dumps(
            {
                "topic": topic,
                "summary_bullets": [f"{topic} summary"],
                "decisions": [],
                "action_items": [],
                "emotional_summary": "Focused.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        )
    }


def _planned_root_chunks() -> list[pipeline.TranscriptChunk]:
    return [
        pipeline.TranscriptChunk(
            index=1,
            total=2,
            text="A: first section " * 6,
            base_text="A: first section " * 6,
            start_seconds=0.0,
            end_seconds=10.0,
        ),
        pipeline.TranscriptChunk(
            index=2,
            total=2,
            text="B: second section " * 6,
            base_text="B: second section " * 6,
            start_seconds=10.0,
            end_seconds=20.0,
        ),
    ]


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_reuses_completed_chunk_state_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, store = _chunk_store(tmp_path, "rec-chunk-resume-1")
    call_log: list[str] = []
    failure_state = {"fail_chunk_2_once": True}

    class _RetryLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            chunk_meta = payload.get("chunk")
            if isinstance(chunk_meta, dict):
                chunk_id = str(chunk_meta.get("id"))
                call_log.append(chunk_id)
                if chunk_id == "2" and failure_state["fail_chunk_2_once"]:
                    failure_state["fail_chunk_2_once"] = False
                    return {"content": "not json"}
                return _chunk_extract_message(chunk_id)
            call_log.append("merge")
            return _merge_message()

    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks())

    kwargs = {
        "transcript_text": "resume transcript",
        "speaker_turns": _llm_chunk_turns("resume transcript"),
        "aliases": {},
        "derived_dir": tmp_path / "derived",
        "llm": _RetryLLM(),
        "cfg": _settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=80,
            llm_chunk_overlap_chars=0,
        ),
        "llm_model": "model",
        "target_summary_language": "en",
        "friendly": 0,
        "default_topic": "Meeting summary",
        "calendar_title": None,
        "calendar_attendees": [],
        "progress_callback": None,
        "chunk_state_store": store,
    }

    with pytest.raises(RuntimeError, match=r"LLM chunk 2/2 failed \[llm_chunk_parse_error\]: json_object_not_found"):
        await pipeline._run_chunked_llm_summary(**kwargs)

    rows = {
        row["chunk_index"]: row
        for row in list_recording_llm_chunk_states("rec-chunk-resume-1", settings=db_cfg)
    }
    assert rows["1"]["status"] == "completed"
    assert rows["2"]["status"] == "failed"
    assert call_log == ["1", "2"]

    result = await pipeline._run_chunked_llm_summary(**kwargs)

    assert result["topic"] == "Merged topic"
    assert call_log == ["1", "2", "2", "merge"]
    merge_input = json.loads((tmp_path / "derived" / "llm_merge_input.json").read_text(encoding="utf-8"))
    assert merge_input["chunk_count"] == 2
    assert merge_input["chunks"][0]["chunk_id"] == "1"
    assert merge_input["chunks"][1]["chunk_id"] == "2"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_invalid_completed_artifact_is_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, store = _chunk_store(tmp_path, "rec-chunk-artifact-1")
    call_log: list[str] = []

    class _ArtifactLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            chunk_meta = payload.get("chunk")
            if isinstance(chunk_meta, dict):
                chunk_id = str(chunk_meta.get("id"))
                call_log.append(chunk_id)
                return _chunk_extract_message(chunk_id)
            call_log.append("merge")
            return _merge_message("Artifact Merge")

    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks())

    kwargs = {
        "transcript_text": "artifact transcript",
        "speaker_turns": _llm_chunk_turns("artifact transcript"),
        "aliases": {},
        "derived_dir": tmp_path / "derived",
        "llm": _ArtifactLLM(),
        "cfg": _settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=80,
            llm_chunk_overlap_chars=0,
        ),
        "llm_model": "model",
        "target_summary_language": "en",
        "friendly": 0,
        "default_topic": "Meeting summary",
        "calendar_title": None,
        "calendar_attendees": [],
        "progress_callback": None,
        "chunk_state_store": store,
    }

    await pipeline._run_chunked_llm_summary(**kwargs)
    (tmp_path / "derived" / "llm_chunk_001_extract.json").unlink()

    result = await pipeline._run_chunked_llm_summary(**kwargs)

    assert result["topic"] == "Artifact Merge"
    assert call_log == ["1", "2", "merge", "1", "merge"]
    rows = {
        row["chunk_index"]: row
        for row in list_recording_llm_chunk_states("rec-chunk-artifact-1", settings=db_cfg)
    }
    assert rows["1"]["attempt"] == 2
    assert rows["2"]["attempt"] == 1


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_timeout_splits_and_persists_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, store = _chunk_store(tmp_path, "rec-chunk-split-1")
    seen_chunks: list[str] = []
    plan_chunks = [
        pipeline.TranscriptChunk(
            index=1,
            total=1,
            text=(
                "S1: alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha\n"
                "S1: beta beta beta beta beta beta beta beta beta beta beta beta beta beta"
            ),
            base_text=(
                "S1: alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha\n"
                "S1: beta beta beta beta beta beta beta beta beta beta beta beta beta beta"
            ),
            start_seconds=0.0,
            end_seconds=20.0,
        )
    ]

    class _SplitLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            chunk_meta = payload.get("chunk")
            if isinstance(chunk_meta, dict):
                chunk_id = str(chunk_meta.get("id"))
                seen_chunks.append(chunk_id)
                if chunk_id == "1":
                    raise asyncio.TimeoutError()
                return _chunk_extract_message(chunk_id)
            seen_chunks.append("merge")
            return _merge_message("Split Merge")

    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: plan_chunks)

    result = await pipeline._run_chunked_llm_summary(
        transcript_text="split transcript",
        speaker_turns=_llm_chunk_turns("split transcript"),
        aliases={},
        derived_dir=tmp_path / "derived",
        llm=_SplitLLM(),
        cfg=_settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=200,
            llm_chunk_overlap_chars=0,
            llm_chunk_split_min_chars=64,
            llm_chunk_split_max_depth=2,
        ),
        llm_model="model",
        target_summary_language="en",
        friendly=0,
        default_topic="Meeting summary",
        calendar_title=None,
        calendar_attendees=[],
        progress_callback=None,
        chunk_state_store=store,
    )

    assert result["topic"] == "Split Merge"
    assert seen_chunks[:3] == ["1", "1a", "1b"]
    rows = {
        row["chunk_index"]: row
        for row in list_recording_llm_chunk_states("rec-chunk-split-1", settings=db_cfg)
    }
    assert rows["1"]["status"] == "split"
    assert rows["1a"]["status"] == "completed"
    assert rows["1b"]["status"] == "completed"
    error_payload = json.loads((tmp_path / "derived" / "llm_chunk_001_error.json").read_text(encoding="utf-8"))
    assert error_payload["reason_code"] == "llm_chunk_timeout"
    assert error_payload["status"] == "split"
    assert error_payload["child_chunk_indexes"] == ["1a", "1b"]
    plan_payload = json.loads((tmp_path / "derived" / "llm_chunks_plan.json").read_text(encoding="utf-8"))
    assert plan_payload["split_chunks"][0]["child_chunk_indexes"] == ["1a", "1b"]


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_split_guard_fails_without_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, store = _chunk_store(tmp_path, "rec-chunk-split-guard-1")
    monkeypatch.setattr(
        pipeline,
        "plan_compact_transcript_chunks",
        lambda *_a, **_k: [
            pipeline.TranscriptChunk(
                index=1,
                total=1,
                text="S1: short short short short short short",
                base_text="S1: short short short short short short",
                start_seconds=0.0,
                end_seconds=5.0,
            )
        ],
    )

    class _GuardLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if isinstance(payload.get("chunk"), dict):
                raise asyncio.TimeoutError()
            raise AssertionError("merge should not run when the guard blocks splitting")

    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed \[llm_chunk_timeout\]: timed out after 120s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="guard transcript",
            speaker_turns=_llm_chunk_turns("guard transcript"),
            aliases={},
            derived_dir=tmp_path / "derived",
            llm=_GuardLLM(),
            cfg=_settings(
                tmp_path,
                llm_model="model",
                llm_chunk_max_chars=120,
                llm_chunk_overlap_chars=0,
                llm_chunk_split_min_chars=64,
                llm_chunk_split_max_depth=0,
            ),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
            chunk_state_store=store,
        )

    rows = list_recording_llm_chunk_states("rec-chunk-split-guard-1", settings=db_cfg)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert json.loads((tmp_path / "derived" / "llm_chunk_001_error.json").read_text(encoding="utf-8"))["reason_code"] == "llm_chunk_timeout"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_timeout_splits_without_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_chunks: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "plan_compact_transcript_chunks",
        lambda *_a, **_k: [
            pipeline.TranscriptChunk(
                index=1,
                total=1,
                text=(
                    "S1: alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha\n"
                    "S1: beta beta beta beta beta beta beta beta beta beta beta beta beta beta"
                ),
                base_text=(
                    "S1: alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha\n"
                    "S1: beta beta beta beta beta beta beta beta beta beta beta beta beta beta"
                ),
                start_seconds=0.0,
                end_seconds=20.0,
            )
        ],
    )

    class _SplitLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            chunk_meta = payload.get("chunk")
            if isinstance(chunk_meta, dict):
                chunk_id = str(chunk_meta.get("id"))
                seen_chunks.append(chunk_id)
                if chunk_id == "1":
                    raise asyncio.TimeoutError()
                return _chunk_extract_message(chunk_id)
            seen_chunks.append("merge")
            return _merge_message("Split Merge")

    result = await pipeline._run_chunked_llm_summary(
        transcript_text="split transcript",
        speaker_turns=_llm_chunk_turns("split transcript"),
        aliases={},
        derived_dir=tmp_path / "derived",
        llm=_SplitLLM(),
        cfg=_settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=200,
            llm_chunk_overlap_chars=0,
            llm_chunk_split_min_chars=64,
            llm_chunk_split_max_depth=2,
        ),
        llm_model="model",
        target_summary_language="en",
        friendly=0,
        default_topic="Meeting summary",
        calendar_title=None,
        calendar_attendees=[],
        progress_callback=None,
    )

    assert result["topic"] == "Split Merge"
    assert seen_chunks[:3] == ["1", "1a", "1b"]


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_split_tolerates_store_without_returned_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, _seed_store = _chunk_store(tmp_path, "rec-chunk-split-none-1")
    monkeypatch.setattr(
        pipeline,
        "plan_compact_transcript_chunks",
        lambda *_a, **_k: [
            pipeline.TranscriptChunk(
                index=1,
                total=1,
                text=(
                    "S1: alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha\n"
                    "S1: beta beta beta beta beta beta beta beta beta beta beta beta beta beta"
                ),
                base_text=(
                    "S1: alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha\n"
                    "S1: beta beta beta beta beta beta beta beta beta beta beta beta beta beta"
                ),
                start_seconds=0.0,
                end_seconds=20.0,
            )
        ],
    )

    class _SplitLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            chunk_meta = payload.get("chunk")
            if isinstance(chunk_meta, dict):
                chunk_id = str(chunk_meta.get("id"))
                if chunk_id == "1":
                    raise asyncio.TimeoutError()
                return _chunk_extract_message(chunk_id)
            return _merge_message("Split Merge")

    class _NoSplitRowStore(_SqliteChunkStateStore):
        def mark_split(self, **kwargs: Any) -> dict[str, Any] | None:
            super().mark_split(**kwargs)
            return None

    result = await pipeline._run_chunked_llm_summary(
        transcript_text="split transcript",
        speaker_turns=_llm_chunk_turns("split transcript"),
        aliases={},
        derived_dir=tmp_path / "derived",
        llm=_SplitLLM(),
        cfg=_settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=200,
            llm_chunk_overlap_chars=0,
            llm_chunk_split_min_chars=64,
            llm_chunk_split_max_depth=2,
        ),
        llm_model="model",
        target_summary_language="en",
        friendly=0,
        default_topic="Meeting summary",
        calendar_title=None,
        calendar_attendees=[],
        progress_callback=None,
        chunk_state_store=_NoSplitRowStore("rec-chunk-split-none-1", db_cfg),
    )

    assert result["topic"] == "Split Merge"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_merge_retry_reuses_completed_extracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db_cfg, store = _chunk_store(tmp_path, "rec-chunk-merge-1")
    seen_chunks: list[str] = []
    merge_attempts = {"count": 0}

    class _MergeRetryLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            chunk_meta = payload.get("chunk")
            if isinstance(chunk_meta, dict):
                chunk_id = str(chunk_meta.get("id"))
                seen_chunks.append(chunk_id)
                return _chunk_extract_message(chunk_id)
            merge_attempts["count"] += 1
            if merge_attempts["count"] == 1:
                return {"content": "**LLM timeout**", "role": "assistant"}
            return _merge_message("Merge Retry")

    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks())

    kwargs = {
        "transcript_text": "merge transcript",
        "speaker_turns": _llm_chunk_turns("merge transcript"),
        "aliases": {},
        "derived_dir": tmp_path / "derived",
        "llm": _MergeRetryLLM(),
        "cfg": _settings(
            tmp_path,
            llm_model="model",
            llm_chunk_max_chars=80,
            llm_chunk_overlap_chars=0,
        ),
        "llm_model": "model",
        "target_summary_language": "en",
        "friendly": 0,
        "default_topic": "Meeting summary",
        "calendar_title": None,
        "calendar_attendees": [],
        "progress_callback": None,
        "chunk_state_store": store,
    }

    with pytest.raises(RuntimeError, match=r"LLM merge failed \[llm_merge_request_timeout\]: request timed out"):
        await pipeline._run_chunked_llm_summary(**kwargs)

    result = await pipeline._run_chunked_llm_summary(**kwargs)

    assert result["topic"] == "Merge Retry"
    assert seen_chunks == ["1", "2"]
    assert merge_attempts["count"] == 2


@pytest.mark.asyncio
async def test_emit_step_log_handles_awaitable_and_callback_errors() -> None:
    messages: list[str] = []

    async def _async_logger(message: str) -> None:
        messages.append(message)

    def _sync_logger(message: str) -> str:
        messages.append(f"sync:{message}")
        return "ok"

    def _broken_logger(_message: str) -> None:
        raise OSError("ignore")

    await pipeline._emit_step_log(_async_logger, "hello")  # noqa: SLF001
    await pipeline._emit_step_log(_sync_logger, "world")  # noqa: SLF001
    await pipeline._emit_step_log(_broken_logger, "ignored")  # noqa: SLF001

    assert messages == ["hello", "sync:world"]


def test_chunk_state_store_protocol_stubs_are_callable() -> None:
    dummy = object()

    assert pipeline.ChunkStateStore.list_states(dummy, chunk_group="extract") is None
    assert pipeline.ChunkStateStore.upsert_state(
        dummy,
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
        status="planned",
    ) is None
    assert pipeline.ChunkStateStore.mark_started(
        dummy,
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
    ) is None
    assert pipeline.ChunkStateStore.mark_completed(
        dummy,
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
    ) is None
    assert pipeline.ChunkStateStore.mark_failed(
        dummy,
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
    ) is None
    assert pipeline.ChunkStateStore.mark_split(
        dummy,
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
    ) is None
    assert pipeline.ChunkStateStore.clear_states(dummy, chunk_group="extract") is None


def test_chunk_resume_helper_utilities_cover_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")

    assert pipeline._read_json_artifact(tmp_path / "missing.json") is None  # noqa: SLF001
    assert pipeline._read_json_artifact(broken) is None  # noqa: SLF001
    assert pipeline._read_json_artifact(list_payload) is None  # noqa: SLF001
    assert pipeline._chunk_request_timeout_seconds(object()) is None  # noqa: SLF001
    assert pipeline._chunk_artifact_token("alpha!") == "chunkalpha_"  # noqa: SLF001

    valid_row = {
        "chunk_index": "1",
        "parent_chunk_index": None,
        "metadata_json": {
            "transcript_hash": "hash",
            "text": "chunk text",
            "base_text": "chunk text",
            "order_path": [1],
            "depth": 0,
        },
    }
    assert pipeline._chunk_runtime_from_state_row(valid_row, transcript_hash="hash") is not None  # noqa: SLF001
    assert pipeline._chunk_runtime_from_state_row({}, transcript_hash="hash") is None  # noqa: SLF001
    assert pipeline._chunk_runtime_from_state_row(  # noqa: SLF001
        {"chunk_index": "1", "metadata_json": {"transcript_hash": "other"}},
        transcript_hash="hash",
    ) is None
    assert pipeline._chunk_runtime_from_state_row(  # noqa: SLF001
        {"chunk_index": "1", "metadata_json": {"transcript_hash": "hash", "text": "", "base_text": "x", "order_path": [1]}},
        transcript_hash="hash",
    ) is None
    assert pipeline._chunk_runtime_from_state_row(  # noqa: SLF001
        {"chunk_index": "1", "metadata_json": {"transcript_hash": "hash", "text": "x", "base_text": "x", "order_path": []}},
        transcript_hash="hash",
    ) is None
    assert pipeline._chunk_runtime_from_state_row(  # noqa: SLF001
        {"chunk_index": "1", "metadata_json": {"transcript_hash": "hash", "text": "x", "base_text": "x", "order_path": ["bad"]}},
        transcript_hash="hash",
    ) is None
    assert pipeline._chunk_runtime_from_state_row(  # noqa: SLF001
        {"chunk_index": "1", "metadata_json": {"transcript_hash": "hash", "text": "x", "base_text": "x", "order_path": [0]}},
        transcript_hash="hash",
    ) is None

    rows = pipeline._active_chunk_rows_by_id([{"chunk_index": ""}, {"chunk_index": "1"}])  # noqa: SLF001
    assert rows == {"1": {"chunk_index": "1"}}
    assert pipeline._load_active_chunk_runtime([{"status": "split", **valid_row}], transcript_hash="hash") is None  # noqa: SLF001
    assert pipeline._load_active_chunk_runtime(  # noqa: SLF001
        [{"status": "split", "chunk_index": "1", "metadata_json": {"transcript_hash": "other"}}],
        transcript_hash="hash",
    ) is None

    class _NoUpdateStore:
        def upsert_state(self, **_kwargs: Any) -> None:
            return None

    runtime = pipeline._ChunkRuntime(  # noqa: SLF001
        chunk_id="1",
        parent_chunk_id=None,
        depth=0,
        order_path=(1,),
        text="chunk text",
        base_text="chunk text",
        overlap_prefix="",
    )
    synced = pipeline._sync_chunk_rows(  # noqa: SLF001
        chunk_state_store=_NoUpdateStore(),
        active_chunks=[runtime],
        rows_by_id={"1": {"status": "planned"}},
        transcript_hash="hash",
    )
    assert synced == {"1": {"status": "planned"}}
    assert pipeline._best_chunk_split_index([pipeline.TranscriptChunk(index=1, total=1, text="x", base_text="x")]) == 0  # noqa: SLF001

    cfg = _settings(tmp_path, llm_model="model", llm_chunk_split_min_chars=64, llm_chunk_split_max_depth=1)
    deep_chunk = pipeline._ChunkRuntime(  # noqa: SLF001
        chunk_id="1",
        parent_chunk_id=None,
        depth=1,
        order_path=(1,),
        text="x" * 200,
        base_text="x" * 200,
        overlap_prefix="",
    )
    small_chunk = pipeline._ChunkRuntime(  # noqa: SLF001
        chunk_id="1",
        parent_chunk_id=None,
        depth=0,
        order_path=(1,),
        text="short",
        base_text="short",
        overlap_prefix="",
    )
    assert pipeline._split_runtime_chunk(deep_chunk, cfg=cfg) == []  # noqa: SLF001
    assert pipeline._split_runtime_chunk(small_chunk, cfg=cfg) == []  # noqa: SLF001

    monkeypatch.setattr(
        pipeline,
        "plan_transcript_chunks",
        lambda *_a, **_k: [pipeline.TranscriptChunk(index=1, total=1, text="x", base_text="x")],
    )
    split_candidate = pipeline._ChunkRuntime(  # noqa: SLF001
        chunk_id="1",
        parent_chunk_id=None,
        depth=0,
        order_path=(1,),
        text="x" * 200,
        base_text="x" * 200,
        overlap_prefix="",
    )
    assert pipeline._split_runtime_chunk(split_candidate, cfg=cfg) == []  # noqa: SLF001

    monkeypatch.setattr(
        pipeline,
        "plan_transcript_chunks",
        lambda *_a, **_k: [
            pipeline.TranscriptChunk(index=1, total=2, text="left", base_text="left"),
            pipeline.TranscriptChunk(index=2, total=2, text="right", base_text="right"),
        ],
    )
    monkeypatch.setattr(pipeline, "_best_chunk_split_index", lambda _parts: 0)
    assert pipeline._split_runtime_chunk(split_candidate, cfg=cfg) == []  # noqa: SLF001

    monkeypatch.setattr(pipeline, "_best_chunk_split_index", lambda _parts: 1)
    monkeypatch.setattr(
        pipeline,
        "plan_transcript_chunks",
        lambda *_a, **_k: [
            pipeline.TranscriptChunk(index=1, total=2, text="L" * 80, base_text="L" * 80),
            pipeline.TranscriptChunk(index=2, total=2, text="R" * 80, base_text="R" * 80),
        ],
    )
    split_children = pipeline._split_runtime_chunk(split_candidate, cfg=cfg)  # noqa: SLF001
    assert [child.chunk_id for child in split_children] == ["1a", "1b"]
    assert split_children[0].end_seconds is None
    assert split_children[1].start_seconds is None

    monkeypatch.setattr(
        pipeline,
        "plan_transcript_chunks",
        lambda *_a, **_k: [
            pipeline.TranscriptChunk(index=1, total=2, text="same", base_text="x" * 200),
            pipeline.TranscriptChunk(index=2, total=2, text="same", base_text="x" * 200),
        ],
    )
    assert pipeline._split_runtime_chunk(split_candidate, cfg=cfg) == []  # noqa: SLF001


def test_validated_completed_chunk_extract_rejects_missing_or_mismatched_artifacts(
    tmp_path: Path,
) -> None:
    chunk = pipeline._ChunkRuntime(  # noqa: SLF001
        chunk_id="1",
        parent_chunk_id=None,
        depth=0,
        order_path=(1,),
        text="chunk text",
        base_text="chunk text",
        overlap_prefix="",
    )

    extract_payload, error = pipeline._validated_completed_chunk_extract(  # noqa: SLF001
        derived_dir=tmp_path,
        chunk=chunk,
        position=1,
        total=1,
    )
    assert extract_payload is None
    assert error == "missing raw artifact llm_chunk_001_raw.json"

    raw_path, extract_path, _error_path = pipeline._chunk_artifact_paths(tmp_path, chunk.chunk_id)  # noqa: SLF001
    raw_path.write_text(json.dumps({"role": "assistant", "content": "{}"}), encoding="utf-8")
    extract_path.write_text(json.dumps({"chunk_id": "2"}), encoding="utf-8")

    extract_payload, error = pipeline._validated_completed_chunk_extract(  # noqa: SLF001
        derived_dir=tmp_path,
        chunk=chunk,
        position=1,
        total=1,
    )
    assert extract_payload is None
    assert error == "extract chunk_id mismatch for 1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc_factory", "expected_reason"),
    [
        (lambda: httpx.ReadTimeout("request timed out"), "llm_chunk_request_timeout"),
        (lambda: ConnectionError("network down"), "llm_chunk_connection_error"),
        (lambda: RuntimeError("boom"), "llm_chunk_runtime_error"),
    ],
)
async def test_run_chunked_llm_summary_handles_chunk_exception_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_factory,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks()[:1])

    class _ErrorLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            raise exc_factory()

    match = "request timed out" if expected_reason.endswith("request_timeout") else "network down" if expected_reason.endswith("connection_error") else "boom"
    with pytest.raises(RuntimeError, match=rf"LLM chunk 1/1 failed \[{expected_reason}\]: {match}"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="error transcript",
            speaker_turns=_llm_chunk_turns("error transcript"),
            aliases={},
            derived_dir=tmp_path / "derived",
            llm=_ErrorLLM(),
            cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_failure_tolerates_store_without_failed_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, _seed_store = _chunk_store(tmp_path, "rec-chunk-failed-none-1")
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks()[:1])

    class _FailingLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            raise RuntimeError("boom")

    class _NoFailedRowStore(_SqliteChunkStateStore):
        def mark_failed(self, **kwargs: Any) -> dict[str, Any] | None:
            super().mark_failed(**kwargs)
            return None

    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed \[llm_chunk_runtime_error\]: boom"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="error transcript",
            speaker_turns=_llm_chunk_turns("error transcript"),
            aliases={},
            derived_dir=tmp_path / "derived",
            llm=_FailingLLM(),
            cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
            chunk_state_store=_NoFailedRowStore("rec-chunk-failed-none-1", db_cfg),
        )


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_resets_stale_state_before_retrying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, store = _chunk_store(tmp_path, "rec-chunk-stale-1")
    upsert_recording_llm_chunk_state(
        "rec-chunk-stale-1",
        chunk_group="extract",
        chunk_index="1",
        chunk_total=1,
        status="completed",
        metadata={
            "transcript_hash": "stale",
            "text": "stale text",
            "base_text": "stale text",
            "order_path": [1],
        },
        settings=db_cfg,
    )
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks()[:1])

    class _FreshLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if isinstance(payload.get("chunk"), dict):
                return _chunk_extract_message("1")
            return _merge_message("Fresh Merge")

    result = await pipeline._run_chunked_llm_summary(
        transcript_text="fresh transcript",
        speaker_turns=_llm_chunk_turns("fresh transcript"),
        aliases={},
        derived_dir=tmp_path / "derived",
        llm=_FreshLLM(),
        cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
        llm_model="model",
        target_summary_language="en",
        friendly=0,
        default_topic="Meeting summary",
        calendar_title=None,
        calendar_attendees=[],
        progress_callback=None,
        chunk_state_store=store,
    )

    assert result["topic"] == "Fresh Merge"
    rows = list_recording_llm_chunk_states("rec-chunk-stale-1", settings=db_cfg)
    assert rows[0]["metadata_json"]["transcript_hash"] != "stale"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_invalidates_completed_chunk_when_store_upsert_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cfg, seed_store = _chunk_store(tmp_path, "rec-chunk-upsert-none-1")
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks()[:1])

    class _FreshLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if isinstance(payload.get("chunk"), dict):
                return _chunk_extract_message("1")
            return _merge_message("Fresh Merge")

    kwargs = {
        "transcript_text": "fresh transcript",
        "speaker_turns": _llm_chunk_turns("fresh transcript"),
        "aliases": {},
        "derived_dir": tmp_path / "derived",
        "llm": _FreshLLM(),
        "cfg": _settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
        "llm_model": "model",
        "target_summary_language": "en",
        "friendly": 0,
        "default_topic": "Meeting summary",
        "calendar_title": None,
        "calendar_attendees": [],
        "progress_callback": None,
        "chunk_state_store": seed_store,
    }

    first_result = await pipeline._run_chunked_llm_summary(**kwargs)
    assert first_result["topic"] == "Fresh Merge"
    (tmp_path / "derived" / "llm_chunk_001_raw.json").unlink()

    class _NoRowUpdateStore(_SqliteChunkStateStore):
        def upsert_state(self, **kwargs: Any) -> dict[str, Any] | None:
            super().upsert_state(**kwargs)
            return None

    store = _NoRowUpdateStore("rec-chunk-upsert-none-1", db_cfg)
    second_result = await pipeline._run_chunked_llm_summary(
        **{**kwargs, "chunk_state_store": store},
    )

    assert second_result["topic"] == "Fresh Merge"


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_merge_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks()[:1])

    async def _merge_timeout(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        if '"chunk"' in kwargs["user_prompt"]:
            return _chunk_extract_message("1")
        raise asyncio.TimeoutError()

    monkeypatch.setattr(pipeline, "_generate_llm_message", _merge_timeout)
    with pytest.raises(RuntimeError, match=r"LLM merge failed \[llm_merge_timeout\]: timed out after 120s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="merge timeout transcript",
            speaker_turns=_llm_chunk_turns("merge timeout transcript"),
            aliases={},
            derived_dir=tmp_path / "derived-timeout",
            llm=object(),
            cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )

    async def _merge_connection(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        if '"chunk"' in kwargs["user_prompt"]:
            return _chunk_extract_message("1")
        raise ConnectionError("merge offline")

    monkeypatch.setattr(pipeline, "_generate_llm_message", _merge_connection)
    with pytest.raises(RuntimeError, match=r"LLM merge failed \[llm_merge_connection_error\]: merge offline"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="merge connection transcript",
            speaker_turns=_llm_chunk_turns("merge connection transcript"),
            aliases={},
            derived_dir=tmp_path / "derived-connection",
            llm=object(),
            cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )

    monkeypatch.setattr(
        pipeline,
        "_validated_completed_chunk_extract",
        lambda **_kwargs: (None, "missing extract"),
    )

    class _MergeMissingExtractLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if isinstance(payload.get("chunk"), dict):
                return _chunk_extract_message("1")
            return _merge_message("Unused Merge")

    with pytest.raises(RuntimeError, match=r"LLM merge failed \[llm_merge_parse_error\]: missing extract"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="merge extract transcript",
            speaker_turns=_llm_chunk_turns("merge extract transcript"),
            aliases={},
            derived_dir=tmp_path / "derived-missing",
            llm=_MergeMissingExtractLLM(),
            cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc_factory", "expected_reason", "message"),
    [
        (lambda: httpx.ReadTimeout("merge request timed out"), "llm_merge_request_timeout", "request timed out"),
        (lambda: RuntimeError("merge boom"), "llm_merge_parse_error", "merge boom"),
    ],
)
async def test_run_chunked_llm_summary_handles_merge_exception_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_factory,
    expected_reason: str,
    message: str,
) -> None:
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks()[:1])

    class _MergeErrorLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if isinstance(payload.get("chunk"), dict):
                return _chunk_extract_message("1")
            raise exc_factory()

    with pytest.raises(RuntimeError, match=rf"LLM merge failed \[{expected_reason}\]: {message}"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="merge error transcript",
            speaker_turns=_llm_chunk_turns("merge error transcript"),
            aliases={},
            derived_dir=tmp_path / "derived",
            llm=_MergeErrorLLM(),
            cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_merge_parse_error_after_raw_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "plan_compact_transcript_chunks", lambda *_a, **_k: _planned_root_chunks()[:1])

    def _raise_merge_parse(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError("bad merge payload")

    monkeypatch.setattr(
        pipeline,
        "build_summary_payload",
        _raise_merge_parse,
    )

    class _MergeParseLLM:
        async def generate(self, **kwargs: Any) -> dict[str, str]:
            payload = json.loads(kwargs["user_prompt"])
            if isinstance(payload.get("chunk"), dict):
                return _chunk_extract_message("1")
            return _merge_message("Broken Merge")

    with pytest.raises(RuntimeError, match=r"LLM merge failed \[llm_merge_parse_error\]: bad merge payload"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="merge parse transcript",
            speaker_turns=_llm_chunk_turns("merge parse transcript"),
            aliases={},
            derived_dir=tmp_path / "derived-parse",
            llm=_MergeParseLLM(),
            cfg=_settings(tmp_path, llm_model="model", llm_chunk_max_chars=80, llm_chunk_overlap_chars=0),
            llm_model="model",
            target_summary_language="en",
            friendly=0,
            default_topic="Meeting summary",
            calendar_title=None,
            calendar_attendees=[],
            progress_callback=None,
        )


def _audio_file(tmp_path: Path, name: str = "audio.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00")
    return path


class _NoTracksDiariser:
    async def __call__(self, _audio_path: Path):
        return object()


class _FakeLLM:
    async def generate(self, **_kwargs: Any) -> dict[str, str]:
        return {
            "content": json.dumps(
                {
                    "topic": "T",
                    "summary_bullets": ["ok"],
                    "decisions": [],
                    "action_items": [],
                    "emotional_summary": "Neutral",
                    "questions": {"total_count": 0, "types": {}, "extracted": []},
                }
            )
        }


@pytest.mark.asyncio
async def test_run_pipeline_uses_sequential_scheduler_for_shared_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    step_log: list[str] = []

    class _TrackingDiariser:
        mode = "pyannote"
        last_run_metadata: dict[str, Any] = {}

        async def __call__(self, _audio_path: Path):
            events.append("diar")
            return _annotation_from_segments((0.0, 1.0, "S1"))

    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [{"start": 0.0, "end": 1.0, "text": "hello"}],
            {"language": "en", "language_probability": 0.9},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_language_aware_asr",
        lambda audio_path, *, override_lang, configured_mode, tmp_root, transcribe_fn, step_log_callback=None: (
            events.append("asr") or transcribe_fn(audio_path, override_lang) + ({
                "used_multilingual_path": False,
                "selected_mode": "single_language",
                "selection_reason": "test",
                "chunks": [],
            },)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_scheduler_plan",
        lambda _cfg, _diariser: SimpleNamespace(
            asr_device="cuda",
            diarization_device="cuda:0",
            effective_mode="sequential",
            requested_mode="auto",
            reason="auto_shared_or_single_device",
        ),
    )
    monkeypatch.setattr(pipeline, "_cleanup_cuda_memory", lambda device: events.append(f"cleanup:{device}"))
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "sequential.mp3"),
        cfg=_settings(
            tmp_path,
            llm_model="model",
            asr_device="cuda",
            diarization_device="cuda:0",
            gpu_scheduler_mode="auto",
        ),
        llm=_FakeLLM(),
        diariser=_TrackingDiariser(),
        recording_id="rec-sequential-scheduler",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
        step_log_callback=step_log.append,
    )

    assert events[:3] == ["asr", "cleanup:cuda", "diar"]
    assert any("mode=sequential" in message for message in step_log)
    assert any("cleared CUDA cache before diarization stage" in message for message in step_log)


@pytest.mark.asyncio
async def test_run_pipeline_uses_parallel_scheduler_only_for_safe_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _TrackingDiariser:
        mode = "pyannote"
        last_run_metadata: dict[str, Any] = {}

        async def __call__(self, _audio_path: Path):
            events.append("diar")
            return _annotation_from_segments((0.0, 1.0, "S1"))

    async def _fake_gather(*aws):
        events.append("gather")
        results = []
        for awaitable in aws:
            results.append(await awaitable)
        return tuple(results)

    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [{"start": 0.0, "end": 1.0, "text": "hello"}],
            {"language": "en", "language_probability": 0.9},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "run_language_aware_asr",
        lambda audio_path, *, override_lang, configured_mode, tmp_root, transcribe_fn, step_log_callback=None: (
            events.append("asr") or transcribe_fn(audio_path, override_lang) + ({
                "used_multilingual_path": False,
                "selected_mode": "single_language",
                "selection_reason": "test",
                "chunks": [],
            },)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_scheduler_plan",
        lambda _cfg, _diariser: SimpleNamespace(
            asr_device="cpu",
            diarization_device="cuda:1",
            effective_mode="parallel",
            requested_mode="auto",
            reason="auto_parallel_safe",
        ),
    )
    monkeypatch.setattr(pipeline.asyncio, "gather", _fake_gather)
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "parallel.mp3"),
        cfg=_settings(
            tmp_path,
            llm_model="model",
            asr_device="cpu",
            diarization_device="cuda:1",
            gpu_scheduler_mode="auto",
        ),
        llm=_FakeLLM(),
        diariser=_TrackingDiariser(),
        recording_id="rec-parallel-scheduler",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )

    assert "gather" in events
    assert "asr" in events
    assert "diar" in events


def test_timeout_seconds_and_retryable_status_paths() -> None:
    assert llm_client._timeout_seconds("bad", default=7.5) == 7.5
    assert llm_client._timeout_seconds("-1", default=7.5) == 7.5
    assert llm_client._int_setting("bad", default=1024, minimum=256) == 1024
    assert llm_client._int_setting("128", default=1024, minimum=256) == 1024
    assert llm_client._int_setting("512", default=1024, minimum=256) == 512
    assert llm_client._resolve_retry_max_tokens("2000", base_max_tokens=1000) == 2000
    assert llm_client._resolve_retry_max_tokens("800", base_max_tokens=1000) == 2000
    assert llm_client._resolve_retry_max_tokens(None, base_max_tokens=5000) == 5000
    assert (
        llm_client._resolve_retry_max_tokens(
            "5000",
            base_max_tokens=4096,
        )
        == 5000
    )
    assert llm_client._base_url_host("http://example.test:8000/path") == "example.test"
    assert llm_client._base_url_host("not-a-url") == "not-a-url"

    req = httpx.Request("POST", "http://example.test")
    retry_resp = httpx.Response(503, request=req)
    non_retry_resp = httpx.Response(404, request=req)
    assert llm_client._is_retryable_exception(
        httpx.HTTPStatusError("boom", request=req, response=retry_resp)
    )
    assert not llm_client._is_retryable_exception(
        httpx.HTTPStatusError("boom", request=req, response=non_retry_resp)
    )


def test_resolve_base_url_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    assert llm_client._resolve_base_url(" http://custom ") == "http://custom"

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LAN_ENV", "staging")
    with pytest.raises(ValueError, match="Missing required environment variable"):
        llm_client._resolve_base_url(None)


@pytest.mark.asyncio
async def test_post_chat_completion_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[str]:
            return ["bad-shape"]

    class _Client:
        def __init__(self, **_kwargs: Any):
            return None

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def post(self, *_args: Any, **_kwargs: Any) -> _Resp:
            return _Resp()

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _Client)
    client = llm_client.LLMClient(base_url="http://example.test", timeout=0.1)
    with pytest.raises(ValueError, match="JSON object"):
        await client._post_chat_completion(
            url="http://example.test/v1/chat/completions",
            payload={"messages": []},
            headers={},
        )


@pytest.mark.asyncio
async def test_load_mock_message_edge_cases(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("   ", encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=empty_path)._load_mock_message() == {
        "role": "assistant",
        "content": "",
    }

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json", encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=invalid_path)._load_mock_message() == {
        "role": "assistant",
        "content": "not-json",
    }

    string_path = tmp_path / "string.json"
    string_path.write_text(json.dumps("hello"), encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=string_path)._load_mock_message() == {
        "role": "assistant",
        "content": "hello",
    }

    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=list_path)._load_mock_message() == {
        "role": "assistant",
        "content": '["a", "b"]',
    }

    choices_non_dict_path = tmp_path / "choices-non-dict.json"
    choices_non_dict_path.write_text(
        json.dumps({"choices": ["x"], "content": "fallback"}),
        encoding="utf-8",
    )
    assert llm_client.LLMClient(mock_response_path=choices_non_dict_path)._load_mock_message() == {
        "role": "assistant",
        "content": "fallback",
    }

    choices_message_path = tmp_path / "choices-message.json"
    choices_message_path.write_text(
        json.dumps({"choices": [{"message": {"role": "assistant", "content": "from-choices"}}]}),
        encoding="utf-8",
    )
    assert llm_client.LLMClient(mock_response_path=choices_message_path)._load_mock_message() == {
        "role": "assistant",
        "content": "from-choices",
    }

    choices_bad_message_path = tmp_path / "choices-bad-message.json"
    choices_bad_message_path.write_text(
        json.dumps({"choices": [{"message": "not-a-dict"}], "content": "fallback-after-bad-message"}),
        encoding="utf-8",
    )
    assert llm_client.LLMClient(mock_response_path=choices_bad_message_path)._load_mock_message() == {
        "role": "assistant",
        "content": "fallback-after-bad-message",
    }


@pytest.mark.asyncio
async def test_generate_headers_and_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    client = llm_client.LLMClient(base_url="http://example.test", api_key="secret-key")

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        del attempt_number
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    result = await client.generate(
        "sys",
        "usr",
        model="x",
        response_format={"type": "json_object"},
    )
    assert result["content"] == "ok"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["max_tokens"] == client.max_tokens

    fallback_client = llm_client.LLMClient(base_url="http://example.test")

    async def _fallback_post(**_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": "bad-shape"}], "content": "fallback-content"}

    monkeypatch.setattr(fallback_client, "_post_chat_completion", _fallback_post)
    fallback = await fallback_client.generate("sys", "usr")
    assert fallback["content"] == "fallback-content"

    async def _fallback_non_dict_choice(**_kwargs: Any) -> dict[str, Any]:
        return {"choices": [123], "content": "fallback-non-dict"}

    monkeypatch.setattr(fallback_client, "_post_chat_completion", _fallback_non_dict_choice)
    fallback2 = await fallback_client.generate("sys", "usr")
    assert fallback2["content"] == "fallback-non-dict"

    async def _missing_content_post(**_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{}]}

    monkeypatch.setattr(fallback_client, "_post_chat_completion", _missing_content_post)
    with pytest.raises(ValueError, match="missing choices"):
        await fallback_client.generate("sys", "usr")


def test_guess_language_and_analysis_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    assert language._guess_language_from_text("") is None
    assert language._guess_language_from_text("!!! 123 ???") is None
    assert language._guess_language_from_text("hola señor, gracias") == "es"
    assert language._guess_language_from_text("hello team and thanks") == "en"
    assert language._guess_language_from_text("hello hola") is None
    assert language._guess_language_from_text("alpha beta gamma") is None

    assert (
        language.segment_language(
            {"text": "???", "language": None},
            detected_language=None,
            transcript_language_override=None,
        )
        == "unknown"
    )

    analysis = language.analyse_languages(
        [
            {"start": 0.0, "end": 0.1, "text": "???"},
            {"start": 5.0, "end": 1.0, "text": "backwards", "language": "en"},
            {"start": 6.0, "end": 12.0, "text": "hello team", "language": "en"},
        ],
        detected_language=None,
        transcript_language_override=None,
    )
    assert analysis.dominant_language == "en"
    backwards_span = next(span for span in analysis.spans if span["start"] == 5.0)
    assert backwards_span["end"] == 5.0

    unknown_only = language.analyse_languages(
        [{"start": 0.0, "end": 1.0, "text": "???"}],
        detected_language=None,
        transcript_language_override=None,
    )
    assert unknown_only.dominant_language == "unknown"

    monkeypatch.setattr(language, "_duration_weight", lambda *_args, **_kwargs: 0.0)
    zero_weight = language.analyse_languages(
        [{"start": 0.0, "end": 1.0, "text": "hello team"}],
        detected_language="en",
        transcript_language_override=None,
    )
    assert zero_weight.distribution == {}


def test_ffprobe_and_speech_ratio_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _audio_file(tmp_path, "a.wav")
    monkeypatch.setattr(precheck.shutil, "which", lambda _name: "/usr/bin/tool")

    def _run_raises(*_args: Any, **_kwargs: Any):
        raise RuntimeError("run failed")

    monkeypatch.setattr(precheck.subprocess, "run", _run_raises)
    assert precheck._audio_duration_from_ffprobe(audio) is None

    class _Proc:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(1, "12"))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(0, "   "))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(0, "not-a-float"))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(0, "-1.0"))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.wave, "open", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    assert precheck._speech_ratio_from_wave(audio) is None

    class _ProcStdoutNone:
        stdout = None
        returncode = 0

        def wait(self, timeout=None):
            del timeout
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(precheck.subprocess, "Popen", lambda *_a, **_k: _ProcStdoutNone())
    assert precheck._speech_ratio_from_ffmpeg(audio) is None

    class _StdoutEmpty:
        def read(self, _size: int) -> bytes:
            return b""

    class _ProcEmpty:
        def __init__(self) -> None:
            self.stdout = _StdoutEmpty()
            self.returncode = 0

        def wait(self, timeout=None):
            del timeout
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(precheck.subprocess, "Popen", lambda *_a, **_k: _ProcEmpty())
    assert precheck._speech_ratio_from_ffmpeg(audio) == 0.0

    monkeypatch.setattr(
        precheck.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("popen-failed")),
    )
    assert precheck._speech_ratio_from_ffmpeg(audio) is None


def test_snippets_edge_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "snips"
    target.mkdir(parents=True, exist_ok=True)
    (target / "stale.wav").write_bytes(b"old")
    snippets._clear_dir(target)
    assert list(target.iterdir()) == []

    # Force the defensive `clip_end <= clip_start` branch.
    monkeypatch.setattr(snippets, "min", lambda *_args: 0.0, raising=False)
    start, end = snippets._snippet_window(
        0.0,
        0.0,
        pad_seconds=0.0,
        max_duration_sec=0.0,
        duration_sec=None,
    )
    assert start == 0.0
    assert end == 0.0

    class _WaveNoFrames:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getframerate(self) -> int:
            return 16000

        def getnchannels(self) -> int:
            return 1

        def getsampwidth(self) -> int:
            return 2

        def getnframes(self) -> int:
            return 100

        def setpos(self, _value: int) -> None:
            return None

        def readframes(self, _value: int) -> bytes:
            return b""

    monkeypatch.setattr(snippets.wave, "open", lambda *_a, **_k: _WaveNoFrames())
    assert not snippets._extract_wav_snippet_with_wave(
        tmp_path / "in.wav",
        tmp_path / "out.wav",
        start_sec=0.0,
        end_sec=1.0,
    )

    monkeypatch.setattr(snippets.wave, "open", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    assert not snippets._extract_wav_snippet_with_wave(
        tmp_path / "in.wav",
        tmp_path / "out.wav",
        start_sec=0.0,
        end_sec=1.0,
    )


def test_export_speaker_snippets_overlap_and_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _audio_file(tmp_path, "src.wav")
    out_dir = tmp_path / "derived" / "snippets"
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: True)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: True)

    outputs = snippets.export_speaker_snippets(
        snippets.SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 5.0, "speaker": "S1"},
                {"start": 1.0, "end": 4.0, "speaker": "S1"},
                {"start": 9.0, "end": 9.0, "speaker": "S1"},
            ],
            snippets_dir=out_dir,
            duration_sec=10.0,
        )
    )
    assert len(outputs) == 2
    manifest = json.loads((tmp_path / "derived" / "snippets_manifest.json").read_text(encoding="utf-8"))
    assert [entry["status"] for entry in manifest["speakers"]["S1"]] == [
        "accepted",
        "accepted",
        "rejected_short",
    ]


def test_export_speaker_snippets_fallback_continue_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _audio_file(tmp_path, "src3.wav")
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: True)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: True)
    outputs = snippets.export_speaker_snippets(
        snippets.SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 2.0, "end": 2.0, "speaker": "S3"},
                {"start": 3.0, "end": 3.0, "speaker": "S3"},
            ],
            snippets_dir=tmp_path / "snips3",
            duration_sec=10.0,
        )
    )
    assert outputs == []
    manifest = json.loads((tmp_path / "snips3" / ".." / "snippets_manifest.json").resolve().read_text(encoding="utf-8"))
    assert [entry["status"] for entry in manifest["speakers"]["S3"]] == [
        "rejected_short",
        "rejected_short",
    ]


def test_export_speaker_snippets_stops_after_three_non_overlapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _audio_file(tmp_path, "src2.wav")
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: True)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: True)
    outputs = snippets.export_speaker_snippets(
        snippets.SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 6.0, "speaker": "S2"},
                {"start": 7.0, "end": 8.0, "speaker": "S2"},
                {"start": 9.0, "end": 10.0, "speaker": "S2"},
                {"start": 11.0, "end": 12.0, "speaker": "S2"},
            ],
            snippets_dir=tmp_path / "snips2",
            duration_sec=12.0,
        )
    )
    assert len(outputs) == 3
    manifest = json.loads((tmp_path / "snips2" / ".." / "snippets_manifest.json").resolve().read_text(encoding="utf-8"))
    assert manifest["speakers"]["S2"][-1]["status"] == "rejected_rank_limit"


def test_speaker_turn_helpers_cover_remaining_branches() -> None:
    assert speaker_turns._normalise_word({"word": " "}, 0.0, 1.0) is None
    assert speaker_turns._normalise_word({"word": "ok", "start": 2.0, "end": 1.0}, 0.0, 1.0)["end"] == 2.0

    normalised = speaker_turns.normalise_asr_segments(
        [
            {
                "start": 2.0,
                "end": 1.0,
                "text": "segment",
                "words": ["bad", {"word": ""}, {"word": "ok", "start": 1.0, "end": 0.5}],
            }
        ]
    )
    assert normalised[0]["end"] == 2.0
    assert normalised[0]["words"][0]["word"] == "ok"

    assert speaker_turns._diarization_segments(None) == []

    class _Diar:
        def itertracks(self, yield_label: bool = False):
            del yield_label
            yield "bad-item"
            yield (SimpleNamespace(start=4.0, end=1.0), "S9")

    diar_rows = speaker_turns._diarization_segments(_Diar())
    assert diar_rows[0]["end"] == 4.0

    assert speaker_turns._pick_speaker(0.0, 1.0, []) == "S1"
    assert (
        speaker_turns._pick_speaker(
            1.0,
            1.0,
            [{"start": 1.0, "end": 1.0, "speaker": "S2"}],
        )
        == "S2"
    )

    words = speaker_turns._words_from_segments(
        [
            {"start": 0.0, "end": 1.0, "text": "fallback", "words": "not-a-list"},
            {"start": 1.0, "end": 2.0, "words": ["bad", {"word": ""}, {"word": "x", "start": 2.0, "end": 1.0}]},
        ],
        default_language="en",
    )
    assert any(item["word"] == "fallback" for item in words)
    assert any(item["word"] == "x" and item["end"] == 2.0 for item in words)
    words_no_lang = speaker_turns._words_from_segments(
        [{"start": 0.0, "end": 1.0, "text": "nolanguage"}],
        default_language=None,
    )
    assert "language" not in words_no_lang[0]

    normalised_turns = speaker_turns._normalise_turns(
        [
            "bad",
            {"text": "   "},
            {"start": 2.0, "end": 1.0, "speaker": "S1", "text": "ok", "language": "EN-us"},
        ]
    )
    assert normalised_turns[0]["end"] == 2.0
    assert normalised_turns[0]["language"] == "en"

    stats = speaker_turns.count_interruptions(
        [
            {"start": 0.0, "end": 0.0, "speaker": "S1", "text": "zero"},
            {"start": 5.0, "end": 8.0, "speaker": "S2", "text": "base"},
            {"start": 4.0, "end": 6.0, "speaker": "S3", "text": "too-early"},
            {"start": 7.9, "end": 8.1, "speaker": "S4", "text": "tiny-overlap"},
        ],
        overlap_threshold=2.0,
    )
    assert stats["total"] == 0

    turns = speaker_turns.build_speaker_turns(
        [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"start": 0.0, "end": 0.4, "word": "hello"}],
                "language": "en",
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "team",
                "words": [{"start": 2.0, "end": 2.4, "word": "team"}],
                "language": "en",
            },
        ],
        [{"start": 0.0, "end": 4.0, "speaker": "S1"}],
        default_language=None,
        merge_gap_sec=1.0,
    )
    assert len(turns) == 2

    turns_no_language = speaker_turns.build_speaker_turns(
        [
            {"start": 0.0, "end": 0.5, "text": "one", "words": [{"start": 0.0, "end": 0.5, "word": "one"}]},
            {"start": 2.0, "end": 2.5, "text": "two", "words": [{"start": 2.0, "end": 2.5, "word": "two"}]},
        ],
        [{"start": 0.0, "end": 4.0, "speaker": "S1"}],
        default_language=None,
        merge_gap_sec=1.0,
    )
    assert len(turns_no_language) == 2

    equal_start_stats = speaker_turns.count_interruptions(
        [
            {"start": 1.0, "end": 3.0, "speaker": "S1", "text": "first"},
            {"start": 1.0, "end": 2.0, "speaker": "S2", "text": "second"},
        ],
        overlap_threshold=0.0,
    )
    assert equal_start_stats["total"] == 0


def test_summary_builder_helper_edge_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="task is required"):
        summary_builder.ActionItem(task="  ")

    item = summary_builder.ActionItem(task="x", owner=None, deadline=None, confidence=0.3)
    assert item.owner is None
    assert item.deadline is None

    q = summary_builder.Question(types="bad")
    assert set(q.types.keys()) == {"open", "yes_no", "clarification", "status", "decision_seeking"}

    resp = summary_builder.SummaryResponse(
        topic="T",
        summary_bullets=["one"],
        decisions=[],
        action_items=[],
        emotional_summary=[],
        questions={"total_count": 0, "types": {}, "extracted": []},
    )
    assert resp.emotional_summary == "Neutral and focused discussion."

    assert summary_builder._chunk_text_for_prompt("   ") == []
    chunks = summary_builder._chunk_text_for_prompt("x" * 25, max_chars=10)
    assert len(chunks) == 3

    limited_turns = summary_builder._normalise_prompt_speaker_turns(
        [{"start": 0, "end": 1, "speaker": "S1", "text": "a b c"}],
        max_turns=0,
    )
    assert limited_turns == []

    _sys_prompt, user_prompt = summary_builder.build_summary_prompts("hello", "en")
    payload = json.loads(user_prompt)
    assert payload["speaker_turns"][0]["text"] == "hello"
    _sys_prompt_empty, user_prompt_empty = summary_builder.build_summary_prompts("   ", "en")
    assert json.loads(user_prompt_empty)["speaker_turns"] == []

    assert summary_builder._extract_json_dict("{not-json}{not-json}") is None

    original_findall = summary_builder.re.findall
    summary_builder.re.findall = lambda *_args, **_kwargs: ['{"topic":"x"}']  # type: ignore[assignment]
    try:
        extracted = summary_builder._extract_json_dict("[1]")
    finally:
        summary_builder.re.findall = original_findall  # type: ignore[assignment]
    assert extracted == {"topic": "x"}

    many = summary_builder._normalise_action_items_fallback([{"task": f"t{i}"} for i in range(40)])
    assert len(many) == 30
    mixed = summary_builder._normalise_action_items_fallback(["task", "", {"task": " "}])
    assert mixed[0]["task"] == "task"
    scalar = summary_builder._normalise_action_items_fallback("single task")
    assert scalar[0]["task"] == "single task"

    class _FakeValidationError:
        def errors(self) -> list[dict[str, Any]]:
            return []

    assert summary_builder._validation_reason(_FakeValidationError()) == "validation_failed"

    payload_no_topic = summary_builder.build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "summary_bullets": ["one"],
                "decisions": [],
                "action_items": [],
                "emotional_summary": "ok",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="Default topic",
    )
    assert payload_no_topic["topic"] == "Default topic"

    payload_invalid_no_artifacts = summary_builder.build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "bad",
                "summary_bullets": [],
                "decisions": [],
                "action_items": [],
                "emotional_summary": "ok",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )
    assert payload_invalid_no_artifacts["parse_error"] is True

    payload_no_json_no_artifacts = summary_builder.build_summary_payload(
        raw_llm_content="plain text",
        model="m",
        target_summary_language="en",
        friendly=0,
    )
    assert payload_no_json_no_artifacts["parse_error_reason"] == "json_object_not_found"

    fallback_from_summary = summary_builder._fallback_payload(
        raw_llm_content="raw summary fallback",
        extracted={"summary": "- one\n- two"},
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="topic",
        parse_error_reason="reason",
    )
    assert fallback_from_summary["summary_bullets"] == ["one", "two"]
    fallback_from_bullets = summary_builder._fallback_payload(
        raw_llm_content="raw summary fallback",
        extracted={"summary_bullets": ["already"], "topic": "A"},
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="topic",
        parse_error_reason="reason",
    )
    assert fallback_from_bullets["summary_bullets"] == ["already"]

    fallback_from_raw = summary_builder._fallback_payload(
        raw_llm_content="- from raw",
        extracted={},
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="topic",
        parse_error_reason="reason",
    )
    assert fallback_from_raw["summary_bullets"] == ["from raw"]


def test_orchestrator_helpers_cover_remaining_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert pipeline._merge_similar(["abc", "xyz", "pqr"], threshold=0.9) == ["abc", "xyz", "pqr"]
    assert pipeline._merge_similar(["abc", "abd", "xyz"], threshold=0.6) == ["abc", "xyz"]

    ann = pipeline._fallback_diarization(None)
    tracks = list(ann.itertracks(yield_label=True))
    assert tracks[0][1] == "S1"

    cfg = _settings(tmp_path, asr_device="auto")
    original_import = builtins.__import__

    def _import_fail_torch(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "torch":
            raise ImportError("no torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_fail_torch)
    assert pipeline._select_asr_device(cfg) == "cpu"
    monkeypatch.setattr(builtins, "__import__", original_import)

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    fake_torch.version = SimpleNamespace(cuda="12.4")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert pipeline._select_asr_device(cfg) == "cuda"
    assert pipeline._select_diarization_device(_settings(tmp_path, diarization_device="cuda:1")) == "cuda:1"

    parallel_plan = pipeline._resolve_scheduler_plan(
        _settings(
            tmp_path,
            asr_device="cuda:0",
            diarization_device="cuda:1",
            gpu_scheduler_mode="auto",
        ),
        SimpleNamespace(mode="pyannote"),
    )
    assert parallel_plan.effective_mode == "parallel"

    monkeypatch.setitem(
        sys.modules,
        "torch",
        ModuleType("torch"),
    )
    sys.modules["torch"].cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)  # type: ignore[attr-defined]
    sys.modules["torch"].version = SimpleNamespace(cuda="12.4")  # type: ignore[attr-defined]
    forced_parallel_same_gpu = pipeline._resolve_scheduler_plan(
        _settings(
            tmp_path,
            asr_device="cuda",
            diarization_device="cuda:0",
            gpu_scheduler_mode="parallel",
        ),
        SimpleNamespace(mode="pyannote"),
    )
    assert forced_parallel_same_gpu.effective_mode == "sequential"

    configured = _settings(tmp_path, asr_compute_type="float32")
    assert pipeline._select_compute_type(configured, "cpu") == "float32"
    assert pipeline._select_compute_type(_settings(tmp_path), "cuda:0") == "float16"

    target_dir = tmp_path / "clear"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "file.txt").write_text("x", encoding="utf-8")
    pipeline._clear_dir(target_dir)
    assert list(target_dir.iterdir()) == []

    pipeline._ASR_MODEL_CACHE[("m", "cpu", "int8", "silero")] = object()  # noqa: SLF001
    cleaned: list[str] = []
    real_cleanup_cuda_memory = pipeline._cleanup_cuda_memory  # noqa: SLF001
    monkeypatch.setattr(pipeline, "_cleanup_cuda_memory", lambda device: cleaned.append(device))
    pipeline.clear_asr_model_cache()
    assert pipeline._ASR_MODEL_CACHE == {}  # noqa: SLF001
    assert cleaned == ["cuda"]
    monkeypatch.setattr(pipeline, "_cleanup_cuda_memory", real_cleanup_cuda_memory)

    logged: list[tuple[str, str, int, int]] = []
    monkeypatch.setattr(pipeline, "cuda_memory_info", lambda _device: (11, 22))
    monkeypatch.setattr(
        pipeline,
        "_logger",
        SimpleNamespace(
            info=lambda template, label, device, free_bytes, total_bytes: logged.append(
                (template, label, device, free_bytes, total_bytes)
            )
        ),
    )
    pipeline._log_cuda_memory_snapshot(label="ASR load", device="cuda:0")  # noqa: SLF001
    assert logged == [
        ("%s VRAM snapshot: device=%s free_bytes=%s total_bytes=%s", "ASR load", "cuda:0", 11, 22)
    ]

    empty_cache_calls: list[str] = []
    fake_torch.cuda.empty_cache = lambda: empty_cache_calls.append("empty")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    original_import = builtins.__import__

    def _import_fake_torch(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "torch":
            return fake_torch
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_fake_torch)
    pipeline._cleanup_cuda_memory("cpu")  # noqa: SLF001
    pipeline._cleanup_cuda_memory("cuda")  # noqa: SLF001
    assert empty_cache_calls == ["empty"]


def test_cleanup_cuda_memory_ignores_missing_cuda_and_empty_cache_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    torch_without_cuda = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_without_cuda)

    def _import_torch_without_cuda(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "torch":
            return torch_without_cuda
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_torch_without_cuda)
    pipeline._cleanup_cuda_memory("cuda")  # noqa: SLF001

    def _raise_empty_cache() -> None:
        raise RuntimeError("cache flush failed")

    torch_with_bad_empty_cache = ModuleType("torch")
    torch_with_bad_empty_cache.cuda = SimpleNamespace(empty_cache=_raise_empty_cache)
    monkeypatch.setitem(sys.modules, "torch", torch_with_bad_empty_cache)

    def _import_torch_with_bad_empty_cache(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "torch":
            return torch_with_bad_empty_cache
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_torch_with_bad_empty_cache)
    pipeline._cleanup_cuda_memory("cuda")  # noqa: SLF001


def test_sentiment_score_uses_truncation_and_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def _pipeline(task: str, **kwargs: Any):
        calls["task"] = task
        calls["factory_kwargs"] = dict(kwargs)

        def _infer(text: str, **infer_kwargs: Any):
            calls["text"] = text
            calls["infer_kwargs"] = dict(infer_kwargs)
            return [{"label": "positive", "score": 0.91}]

        return _infer

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    score = pipeline._sentiment_score("x" * 5000)
    assert score == 91
    assert calls["task"] == "sentiment-analysis"
    assert calls["factory_kwargs"] == {
        "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "device": -1,
    }
    assert calls["text"] == "x" * 4000
    assert calls["infer_kwargs"] == {"truncation": True, "max_length": 512}


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [
        ("negative", 0.2, 80),
        ("neutral", 0.7, 50),
    ],
)
def test_sentiment_score_negative_and_fallback_labels(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    value: float,
    expected: int,
) -> None:
    def _pipeline(_task: str, **_kwargs: Any):
        return lambda *_a, **_k: [{"label": label, "score": value}]

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert pipeline._sentiment_score("hello") == expected


def test_sentiment_score_returns_neutral_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []

    def _pipeline(_task: str, **_kwargs: Any):
        def _boom(*_args: Any, **_infer_kwargs: Any):
            raise RuntimeError("model failed")

        return _boom

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        pipeline._logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    assert pipeline._sentiment_score("hello") == 0.0
    assert warnings == ["Sentiment scoring failed (RuntimeError); using neutral score"]


@pytest.mark.asyncio
async def test_emit_progress_awaitable_and_error_branches() -> None:
    events: list[tuple[str, float]] = []

    async def _async_cb(stage: str, progress: float) -> None:
        events.append((stage, progress))

    await pipeline._emit_progress(_async_cb, stage="x", progress=0.5)
    assert events == [("x", 0.5)]

    def _bad_cb(_stage: str, _progress: float) -> None:
        raise RuntimeError("boom")

    await pipeline._emit_progress(_bad_cb, stage="y", progress=0.9)


def test_whisperx_asr_callback_and_retry_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_whisperx = ModuleType("whisperx")

    def _legacy(_path: str, **_kwargs: Any):
        return [{"start": 0.0, "end": 1.0, "text": "hi"}], {"language": "en"}

    fake_whisperx.transcribe = _legacy
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(pipeline, "ensure_ctranslate2_no_execstack", lambda: ["/tmp/lib.so"])
    monkeypatch.setattr(pipeline, "patch_pyannote_inference_ignore_use_auth_token", lambda: True)

    def _bad_step_log(_msg: str) -> None:
        raise RuntimeError("cannot log")

    segments, info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "a.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_enable_align=False, vad_method="pyannote"),
        step_log_callback=_bad_step_log,
    )
    assert segments and info["language"] == "en"

    calls: list[dict[str, Any]] = []

    def _fake_call(_fn: Any, *_args: Any, **kwargs: Any):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise TypeError("synthetic type error")
        return ([{"start": 0.0, "end": 1.0, "text": "ok"}], {"language": "en"})

    monkeypatch.setattr(pipeline, "call_with_supported_kwargs", _fake_call)
    monkeypatch.setattr(pipeline, "ensure_ctranslate2_no_execstack", lambda: [])
    monkeypatch.setattr(pipeline, "patch_pyannote_inference_ignore_use_auth_token", lambda: False)
    segments, info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "b.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_enable_align=False),
        step_log_callback=list.append,
    )
    assert segments and info["language"] == "en"
    assert "word_timestamps" in calls[0]
    assert "word_timestamps" not in calls[1]


def test_whisperx_asr_align_typeerror_and_exception_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.transcribe = None

    class _Model:
        vad_model = staticmethod(lambda _payload: [])

        def transcribe(self, audio: str, *, batch_size: int, vad_filter: bool, language: str | None):
            del audio, batch_size, vad_filter, language
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "a"}], "language": "en"}

    fake_whisperx.load_audio = lambda _p: "audio"
    fake_whisperx.load_model = lambda *_a, **_k: _Model()
    fake_whisperx.load_align_model = lambda **_k: ("model", {"lang": "en"})

    def _align_typeerror(*args: Any, **kwargs: Any):
        if kwargs:
            raise TypeError("unexpected keyword")
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "aligned"}]}

    fake_whisperx.align = _align_typeerror
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    segments, _info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "modern.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_device="cpu", asr_enable_align=True),
    )
    assert segments[0]["text"] == "aligned"

    def _align_error(*_args: Any, **_kwargs: Any):
        raise RuntimeError("align failed")

    fake_whisperx.align = _align_error
    segments, _info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "modern2.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_device="cpu", asr_enable_align=True),
    )
    assert segments[0]["text"] == "a"


@pytest.mark.asyncio
async def test_run_pipeline_logs_flicker_speaker_reassignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [
                {"start": 0.0, "end": 1.0, "text": "alpha beta gamma delta epsilon"},
                {"start": 1.0, "end": 2.0, "text": "zeta eta theta iota kappa"},
                {"start": 2.0, "end": 3.0, "text": "lambda mu nu xi omicron"},
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    class _FlickerDiariser:
        async def __call__(self, _audio_path: Path):
            return _annotation_from_segments(
                (0.0, 5.0, "S_MAIN"),
                (1.0, 1.2, "S_FLICKER"),
                (5.0, 12.0, "S_MAIN"),
            )

    cfg = _settings(tmp_path)
    caplog.set_level("WARNING", logger="lan_transcriber.pipeline_steps.orchestrator")
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "flicker.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_FlickerDiariser(),
        recording_id="rec-flicker",
        precheck=pipeline.PrecheckResult(
            duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None
        ),
    )
    assert result.summary.strip() == "- ok"
    assert any(
        "Diarization flicker speaker reassigned" in record.getMessage()
        and "S_FLICKER" in record.getMessage()
        for record in caplog.records
    )

    derived = cfg.recordings_root / "rec-flicker" / "derived"
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    assert all(row["speaker"] != "S_FLICKER" for row in diar_data)


@pytest.mark.asyncio
async def test_run_pipeline_backfills_detected_language_and_uses_fallback_diarization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [{"start": 0.0, "end": 1.0, "text": "hello team and thanks"}],
            {"language": "unknown", "language_probability": 0.2},
        ),
    )
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(
        pipeline,
        "export_speaker_snippets",
        lambda _req: [],
    )
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    cfg = _settings(tmp_path)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "pipeline.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_NoTracksDiariser(),
        recording_id="rec-fallback-diar",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )
    assert result.summary.strip() == "- ok"

    derived = cfg.recordings_root / "rec-fallback-diar" / "derived"
    transcript_data = json.loads((derived / "transcript.json").read_text(encoding="utf-8"))
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    assert transcript_data["language"]["detected"] == "en"
    assert diar_data and diar_data[0]["speaker"] == "S1"


@pytest.mark.asyncio
async def test_run_pipeline_restores_previous_whisperx_transcriber_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_transcriber = object()

    def _session_transcriber(
        _audio_path: Path,
        _override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert getattr(pipeline._whisperx_transcriber_state, "transcribe_audio") is _session_transcriber
        return (
            [{"start": 0.0, "end": 1.0, "text": "hello team and thanks"}],
            {"language": "en", "language_probability": 0.95},
        )

    def _fake_run_language_aware_asr(
        audio_path: Path,
        *,
        override_lang: str | None,
        configured_mode: str,
        tmp_root: Path,
        transcribe_fn,
        step_log_callback=None,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        del configured_mode, tmp_root, step_log_callback
        segments, info = transcribe_fn(audio_path, override_lang)
        return (
            segments,
            info,
            {
                "used_multilingual_path": False,
                "selected_mode": "single_language",
                "selection_reason": "test",
                "chunks": [],
            },
        )

    async def _fake_to_thread(fn, /, *args, **kwargs):
        pipeline._whisperx_transcriber_state.transcribe_audio = previous_transcriber
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(
        pipeline,
        "_build_whisperx_transcriber",
        lambda **_kwargs: _session_transcriber,
    )
    monkeypatch.setattr(pipeline, "run_language_aware_asr", _fake_run_language_aware_asr)
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    cfg = _settings(tmp_path)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "pipeline-session.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_NoTracksDiariser(),
        recording_id="rec-transcriber-session",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )

    assert result.summary.strip() == "- ok"
    assert getattr(pipeline._whisperx_transcriber_state, "transcribe_audio") is previous_transcriber
    delattr(pipeline._whisperx_transcriber_state, "transcribe_audio")


@pytest.mark.asyncio
async def test_run_pipeline_preserves_previous_session_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_transcriber = object()

    def _session_transcriber(
        _audio_path: Path,
        _override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return (
            [{"start": 0.0, "end": 1.0, "text": "hello team and thanks"}],
            {"language": "en", "language_probability": 0.95},
        )

    def _fake_run_language_aware_asr(
        audio_path: Path,
        *,
        override_lang: str | None,
        configured_mode: str,
        tmp_root: Path,
        transcribe_fn,
        step_log_callback=None,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        del configured_mode, tmp_root, step_log_callback
        delattr(pipeline._whisperx_transcriber_state, "use_session_transcriber")
        segments, info = transcribe_fn(audio_path, override_lang)
        return (
            segments,
            info,
            {
                "used_multilingual_path": False,
                "selected_mode": "single_language",
                "selection_reason": "test",
                "chunks": [],
            },
        )

    async def _fake_to_thread(fn, /, *args, **kwargs):
        pipeline._whisperx_transcriber_state.transcribe_audio = previous_transcriber
        pipeline._whisperx_transcriber_state.use_session_transcriber = True
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(
        pipeline,
        "_build_whisperx_transcriber",
        lambda **_kwargs: _session_transcriber,
    )
    monkeypatch.setattr(pipeline, "run_language_aware_asr", _fake_run_language_aware_asr)
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    cfg = _settings(tmp_path)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "pipeline-session-flag.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_NoTracksDiariser(),
        recording_id="rec-transcriber-session-flag",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )

    assert result.summary.strip() == "- ok"
    assert getattr(pipeline._whisperx_transcriber_state, "transcribe_audio") is previous_transcriber
    assert getattr(pipeline._whisperx_transcriber_state, "use_session_transcriber") is True
    delattr(pipeline._whisperx_transcriber_state, "transcribe_audio")
    delattr(pipeline._whisperx_transcriber_state, "use_session_transcriber")


@pytest.mark.asyncio
async def test_run_pipeline_tolerates_session_transcriber_cleanup_before_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _session_transcriber(
        _audio_path: Path,
        _override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return (
            [{"start": 0.0, "end": 1.0, "text": "hello team and thanks"}],
            {"language": "en", "language_probability": 0.95},
        )

    def _fake_run_language_aware_asr(
        audio_path: Path,
        *,
        override_lang: str | None,
        configured_mode: str,
        tmp_root: Path,
        transcribe_fn,
        step_log_callback=None,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
        del configured_mode, tmp_root, step_log_callback
        delattr(pipeline._whisperx_transcriber_state, "transcribe_audio")
        delattr(pipeline._whisperx_transcriber_state, "use_session_transcriber")
        segments, info = transcribe_fn(audio_path, override_lang)
        return (
            segments,
            info,
            {
                "used_multilingual_path": False,
                "selected_mode": "single_language",
                "selection_reason": "test",
                "chunks": [],
            },
        )

    monkeypatch.setattr(
        pipeline,
        "_build_whisperx_transcriber",
        lambda **_kwargs: _session_transcriber,
    )
    monkeypatch.setattr(pipeline, "run_language_aware_asr", _fake_run_language_aware_asr)
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    cfg = _settings(tmp_path)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "pipeline-session-cleanup.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_NoTracksDiariser(),
        recording_id="rec-transcriber-session-cleanup",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )

    assert result.summary.strip() == "- ok"
    assert not hasattr(pipeline._whisperx_transcriber_state, "transcribe_audio")
    assert not hasattr(pipeline._whisperx_transcriber_state, "use_session_transcriber")


@pytest.mark.asyncio
async def test_run_pipeline_skips_noise_detection_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [{"start": 0.0, "end": 1.0, "text": "hello team and thanks for joining"}],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})
    monkeypatch.setattr(
        pipeline,
        "apply_noise_flags_to_manifest",
        lambda *_a, **_k: pytest.fail("noise detection should be disabled"),
    )

    cfg = _settings(tmp_path, noise_detection_enabled=False)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "no-noise-detection.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_NoTracksDiariser(),
        recording_id="rec-no-noise-detection",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )
    assert result.summary.strip() == "- ok"


@pytest.mark.asyncio
async def test_run_pipeline_filters_noise_speakers_from_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [
                {"start": 0.0, "end": 1.0, "text": "hello team and thanks for joining"},
                {"start": 1.0, "end": 2.0, "text": "background noise hiss static hum"},
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})
    monkeypatch.setattr(
        pipeline,
        "apply_noise_flags_to_manifest",
        lambda *_a, **_k: {
            "noise_speakers": ["S2"],
            "speaker_metrics": {"S2": {"flagged": True}},
            "threshold": 0.3,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "update_diarization_metadata_with_noise",
        lambda *_a, **_k: None,
    )

    captured_prompts: dict[str, Any] = {}

    def _capture_prompts(turns, summary_lang, *, calendar_title=None, calendar_attendees=None):
        captured_prompts["turns"] = list(turns)
        return ("sys-prompt", "user-prompt")

    monkeypatch.setattr(pipeline, "build_structured_summary_prompts", _capture_prompts)

    class _NoiseDiariser:
        async def __call__(self, _audio_path: Path):
            return _annotation_from_segments(
                (0.0, 1.0, "S1"),
                (1.0, 2.0, "S2"),
            )

    cfg = _settings(tmp_path, exclude_noise_speakers_from_transcript=True)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "filter-noise.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_NoiseDiariser(),
        recording_id="rec-noise-filter",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )
    assert result.summary.strip() == "- ok"
    transcript = json.loads(
        (cfg.recordings_root / "rec-noise-filter" / "derived" / "transcript.json").read_text(
            encoding="utf-8"
        )
    )
    speaker_lines = transcript.get("speaker_lines") or []
    assert any("S1" in line for line in speaker_lines)
    assert all("S2" not in line for line in speaker_lines)
    assert "S1" in (transcript.get("speakers") or [])
    assert "S2" not in (transcript.get("speakers") or [])
    persisted_turns = json.loads(
        (cfg.recordings_root / "rec-noise-filter" / "derived" / "speaker_turns.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(str(turn.get("speaker")) != "S2" for turn in persisted_turns)
    assert any(str(turn.get("speaker")) == "S1" for turn in persisted_turns)
    assert all(seg.speaker != "S2" for seg in result.segments)
    assert any(seg.speaker == "S1" for seg in result.segments)
    metrics_payload = json.loads(
        (cfg.recordings_root / "rec-noise-filter" / "derived" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics_payload["speaker_turns"] == len(persisted_turns)
    transcript_txt = (
        cfg.recordings_root / "rec-noise-filter" / "derived" / "transcript.txt"
    ).read_text(encoding="utf-8")
    assert "background noise hiss" not in transcript_txt
    assert "hello team and thanks" in transcript_txt
    assert transcript["text"] == transcript_txt
    assert result.body == transcript_txt
    assert captured_prompts["turns"], "LLM prompt builder must be called"
    assert all(
        str(turn.get("speaker")) != "S2" for turn in captured_prompts["turns"]
    )
    assert any(
        str(turn.get("speaker")) == "S1" for turn in captured_prompts["turns"]
    )
    # transcript.json["segments"] must not carry noise text when the filter
    # drops a subset of speakers — UI fallbacks rebuild turns from there.
    assert any(
        str(seg.get("speaker")) == "S1" for seg in (transcript.get("segments") or [])
    )
    assert all(
        str(seg.get("speaker")) != "S2" for seg in (transcript.get("segments") or [])
    )


@pytest.mark.asyncio
async def test_run_pipeline_returns_no_speech_when_all_turns_flagged_as_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If exclusion filters every speaker, skip LLM and emit no_speech summary."""

    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [{"start": 0.0, "end": 1.0, "text": "background noise hiss static hum buzz"}],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    monkeypatch.setattr(pipeline, "export_speaker_snippets", lambda _req: [])
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})
    monkeypatch.setattr(
        pipeline,
        "apply_noise_flags_to_manifest",
        lambda *_a, **_k: {
            "noise_speakers": ["S1"],
            "speaker_metrics": {"S1": {"flagged": True}},
            "threshold": 0.3,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "update_diarization_metadata_with_noise",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_sentiment_score",
        lambda _t: pytest.fail("Sentiment must not run when all turns filtered"),
    )
    monkeypatch.setattr(
        pipeline,
        "build_structured_summary_prompts",
        lambda *_a, **_k: pytest.fail("LLM must not run when all turns filtered"),
    )

    class _SingleSpeakerDiariser:
        async def __call__(self, _audio_path: Path):
            return _annotation_from_segments((0.0, 1.0, "S1"))

    cfg = _settings(tmp_path, exclude_noise_speakers_from_transcript=True)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "all-noise.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_SingleSpeakerDiariser(),
        recording_id="rec-all-noise",
        precheck=pipeline.PrecheckResult(
            duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None
        ),
    )
    summary_payload = json.loads(
        (cfg.recordings_root / "rec-all-noise" / "derived" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_payload["status"] == "no_speech"
    assert result.body == ""
    metrics_payload = json.loads(
        (cfg.recordings_root / "rec-all-noise" / "derived" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics_payload["status"] == "no_speech"
    transcript_payload = json.loads(
        (cfg.recordings_root / "rec-all-noise" / "derived" / "transcript.json").read_text(
            encoding="utf-8"
        )
    )
    # language_segments are cleared when filter empties every turn so UI
    # fallbacks can't rebuild noise turns from transcript.json["segments"].
    assert transcript_payload.get("segments") == []
    assert transcript_payload.get("speakers") == []


@pytest.mark.asyncio
async def test_run_pipeline_fails_fast_when_llm_model_is_blank(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, llm_model="   ")
    with pytest.raises(RuntimeError, match="LLM_MODEL is required"):
        await pipeline.run_pipeline(
            audio_path=_audio_file(tmp_path, "missing-model.mp3"),
            cfg=cfg,
            llm=_FakeLLM(),
            diariser=_NoTracksDiariser(),
            recording_id="rec-missing-model",
            precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
        )


@pytest.mark.asyncio
async def test_protocol_stub_call_executes() -> None:
    out = await pipeline.Diariser.__call__(object(), Path("x"))
    assert out is None


def test_build_speaker_turns_false_final_current_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TruthyEmptyWords:
        def __bool__(self) -> bool:
            return True

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(speaker_turns, "_words_from_segments", lambda *_a, **_k: _TruthyEmptyWords())
    turns = speaker_turns.build_speaker_turns([], [], default_language=None)
    assert turns == []
