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
    monkeypatch.setenv("LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS", "8192")
    monkeypatch.setenv("LLM_MERGE_MAX_TOKENS", "3072")

    cfg = _settings(tmp_path)
    assert cfg.llm_chunk_max_chars == 4096
    assert cfg.llm_chunk_overlap_chars == 256
    assert cfg.llm_chunk_timeout_seconds == 45.0
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
    monkeypatch.setattr(pipeline, "plan_transcript_chunks", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="produced no chunks"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="hello world",
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
    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed: timed out after 0.001s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
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

    assert json.loads((derived / "llm_chunk_001_error.json").read_text(encoding="utf-8")) == {
        "error": "timed out after 0.001s"
    }


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_asyncio_timeout_writes_error_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_asyncio_timeout(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        raise asyncio.TimeoutError()

    monkeypatch.setattr(pipeline, "_generate_llm_message", _raise_asyncio_timeout)

    derived = tmp_path / "derived"
    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed: timed out after 0.001s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
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

    assert json.loads((derived / "llm_chunk_001_error.json").read_text(encoding="utf-8")) == {
        "error": "timed out after 0.001s"
    }


@pytest.mark.asyncio
async def test_run_chunked_llm_summary_timeout_sentinel_writes_error_artifact(
    tmp_path: Path,
) -> None:
    class _TimeoutSentinelLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"content": "**LLM timeout**", "role": "assistant"}

    derived = tmp_path / "derived"
    with pytest.raises(RuntimeError, match=r"LLM chunk 1/1 failed: timed out after 0.001s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
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
    assert json.loads((derived / "llm_chunk_001_error.json").read_text(encoding="utf-8")) == {
        "error": "timed out after 0.001s"
    }


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
    with pytest.raises(RuntimeError, match=r"LLM merge failed: timed out after 12s"):
        await pipeline._run_chunked_llm_summary(
            transcript_text="single chunk transcript",
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
    )
    assert len(turns) == 2

    turns_no_language = speaker_turns.build_speaker_turns(
        [
            {"start": 0.0, "end": 0.5, "text": "one", "words": [{"start": 0.0, "end": 0.5, "word": "one"}]},
            {"start": 2.0, "end": 2.5, "text": "two", "words": [{"start": 2.0, "end": 2.5, "word": "two"}]},
        ],
        [{"start": 0.0, "end": 4.0, "speaker": "S1"}],
        default_language=None,
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
