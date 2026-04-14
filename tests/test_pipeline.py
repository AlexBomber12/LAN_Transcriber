from __future__ import annotations

import builtins
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import wave

import httpx
import pytest
import respx

# stub heavy deps before importing pipeline
import sys

whisperx = ModuleType("whisperx")
whisperx.utils = SimpleNamespace(get_segments=lambda *_a, **_k: "hello")
whisperx.Transcriber = SimpleNamespace()  # placeholder for patching
whisperx.transcribe = lambda *a, **k: ([], {})
sys.modules["whisperx"] = whisperx

transformers = ModuleType("transformers")
transformers.pipeline = lambda *a, **k: lambda text: [
    {"label": "positive", "score": 0.9}
]
sys.modules["transformers"] = transformers

from lan_transcriber import llm_client, pipeline  # noqa: E402
from lan_transcriber.pipeline_steps.language import LanguageAnalysis  # noqa: E402
from lan_transcriber.pipeline_steps import precheck as precheck_step  # noqa: E402


def fake_audio(tmp_path: Path, name: str = "sample.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00")
    return path


def wav_audio(
    tmp_path: Path,
    *,
    name: str,
    duration_sec: float,
    speech: bool,
) -> Path:
    path = tmp_path / name
    rate = 16000
    samples = int(rate * duration_sec)
    frames = bytearray()
    for idx in range(samples):
        if speech:
            value = int(9000 * math.sin((2.0 * math.pi * 220.0 * idx) / rate))
        else:
            value = 0
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(bytes(frames))
    return path


def precheck_ok() -> pipeline.PrecheckResult:
    return pipeline.PrecheckResult(
        duration_sec=30.0,
        speech_ratio=0.5,
        quarantine_reason=None,
    )


def _annotation_from_segments(*segments: tuple[float, float, str]):
    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            for start, end, speaker in segments:
                window = SimpleNamespace(start=start, end=end)
                if yield_label:
                    yield window, speaker
                else:
                    yield (window,)

    return _Annotation()


class DummyDiariser:
    mode = "pyannote"

    async def __call__(self, audio_path: Path):
        return _annotation_from_segments((0.0, 1.0, "S1"))


class TwoSpeakerDiariser:
    mode = "pyannote"

    async def __call__(self, audio_path: Path):
        return _annotation_from_segments((0.0, 12.0, "S1"), (12.0, 24.0, "S2"))


class TwoSpeakerTripletDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label: bool = False):
                from types import SimpleNamespace

                if yield_label:
                    yield (SimpleNamespace(start=0.0, end=12.0), "track-1", "S1")
                    yield (SimpleNamespace(start=12.0, end=24.0), "track-2", "S2")
                else:
                    yield (SimpleNamespace(start=0.0, end=12.0),)
                    yield (SimpleNamespace(start=12.0, end=24.0),)

        return Ann()


class FailingDiariser:
    async def __call__(self, audio_path: Path):
        raise RuntimeError("diariser boom")


class FailingItertracksDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label: bool = False):
                raise RuntimeError("itertracks boom")

        return Ann()


class DialogRetryDiariser:
    def __init__(self) -> None:
        self.mode = "pyannote"
        self.profile = "auto"
        self.dialog_retry_min_turns = 3
        self.dialog_retry_min_duration_seconds = 10.0
        self.last_run_metadata = {
            "requested_profile": "auto",
            "diarization_profile": "auto",
            "initial_profile": "meeting",
            "selected_profile": None,
            "auto_profile_enabled": True,
            "override_reason": None,
            "initial_hints": {"min_speakers": 2, "max_speakers": 6},
            "retry_hints": None,
            "effective_hints": {"min_speakers": 2, "max_speakers": 6},
            "profile_selection": None,
            "dialog_retry_used": False,
            "speaker_count_before_retry": None,
            "speaker_count_after_retry": None,
        }

    async def __call__(self, audio_path: Path):
        self.last_run_metadata.update(
            {
                "dialog_retry_used": False,
                "speaker_count_before_retry": 1,
                "speaker_count_after_retry": 1,
            }
        )
        return _annotation_from_segments((0.0, 2.4, "S1"))

    async def retry_dialog(self, audio_path: Path):
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
            (1.0, 1.2, "S2"),
            (1.2, 2.4, "S1"),
        )


class EmptyTracksDiariser:
    def __init__(self) -> None:
        self.mode = "pyannote"
        self.profile = "auto"
        self.last_run_metadata = {
            "requested_profile": "auto",
            "diarization_profile": "auto",
            "initial_profile": "meeting",
            "selected_profile": None,
            "auto_profile_enabled": True,
            "override_reason": None,
            "initial_hints": {"min_speakers": 2, "max_speakers": 6},
            "retry_hints": None,
            "effective_hints": {"min_speakers": 2, "max_speakers": 6},
            "profile_selection": None,
            "dialog_retry_used": False,
            "speaker_count_before_retry": 0,
            "speaker_count_after_retry": 0,
        }

    async def __call__(self, audio_path: Path):
        return object()


@pytest.mark.asyncio
@respx.mock
async def test_tripled_dedup(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {"start": 0.0, "end": 1.0, "text": "hello world."},
                {"start": 1.0, "end": 2.0, "text": "hello world."},
                {"start": 2.0, "end": 3.0, "text": "hello world."},
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )

    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "- ok"}}]}
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.8}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    res = await pipeline.run_pipeline(
        fake_audio(tmp_path, "tripled.mp3"),
        cfg,
        llm_client.LLMClient(),
        DummyDiariser(),
        precheck=precheck_ok(),
    )

    assert res.body.strip() == "hello world."
    assert res.summary.strip() == "- ok"
    assert res.summary_path.name == "summary.json"
    assert res.body_path.name == "transcript.txt"
    assert res.body_path.read_text(encoding="utf-8") == "hello world."
    summary_data = json.loads(res.summary_path.read_text(encoding="utf-8"))
    assert summary_data["summary"] == "- ok"


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_emits_progress_stages_in_order(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello world team"}],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    events: list[tuple[str, float]] = []

    def _progress(stage: str, progress: float) -> None:
        events.append((stage, progress))

    await pipeline.run_pipeline(
        fake_audio(tmp_path, "progress.mp3"),
        cfg,
        llm_client.LLMClient(),
        DummyDiariser(),
        precheck=precheck_ok(),
        progress_callback=_progress,
    )

    assert events == [
        ("precheck", 0.05),
        ("stt", 0.10),
        ("stt", 0.35),
        ("diarize", 0.60),
        ("align", 0.68),
        ("language", 0.75),
        ("llm", 0.90),
        ("metrics", 0.98),
    ]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("mode", ["pyannote", " PyAnNoTe "])
async def test_pipeline_writes_diarization_metadata_and_smooths_retry_output(
    tmp_path: Path,
    mocker,
    mode: str,
):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "alpha",
                    "words": [{"start": 0.0, "end": 1.0, "word": "alpha"}],
                },
                {
                    "start": 1.0,
                    "end": 1.2,
                    "text": "noise",
                    "words": [{"start": 1.0, "end": 1.2, "word": "noise"}],
                },
                {
                    "start": 1.2,
                    "end": 2.4,
                    "text": "omega",
                    "words": [{"start": 1.2, "end": 2.4, "word": "omega"}],
                },
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        diarization_profile="auto",
        diarization_merge_gap_seconds=0.2,
        diarization_min_turn_seconds=0.5,
    )
    recording_id = "rec-diar-meta-1"

    diariser = DialogRetryDiariser()
    diariser.mode = mode

    await pipeline.run_pipeline(
        fake_audio(tmp_path, "dialog-retry.mp3"),
        cfg,
        llm_client.LLMClient(),
        diariser,
        recording_id=recording_id,
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / recording_id / "derived"
    speaker_turns = json.loads((derived / "speaker_turns.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (derived / "diarization_metadata.json").read_text(encoding="utf-8")
    )

    assert speaker_turns == [
        {"start": 0.0, "end": 2.4, "speaker": "S1", "text": "alpha noise omega", "language": "en"}
    ]
    assert metadata["diarization_profile"] == "auto"
    assert metadata["selected_profile"] == "dialog"
    assert metadata["selected_result"] == "dialog_retry"
    assert metadata["dialog_retry_attempted"] is True
    assert metadata["dialog_retry_used"] is True
    assert metadata["hints_applied"] == {"max_speakers": 2, "min_speakers": 2}
    assert metadata["initial_top_two_coverage"] == 1.0
    assert metadata["used_dummy_fallback"] is False
    assert metadata["smoothing_applied"] is True
    assert metadata["speaker_count_before_smoothing"] == 2
    assert metadata["speaker_count_after_smoothing"] == 1
    assert metadata["micro_turn_absorptions"] == 1


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_applies_speaker_merge_when_centroids_match(
    tmp_path: Path,
    mocker,
):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": "hello team this is the first speaker",
                    "words": [
                        {"start": 0.0, "end": 10.0, "word": "hello team this is the first speaker"}
                    ],
                },
                {
                    "start": 12.0,
                    "end": 22.0,
                    "text": "hello team this is the second speaker",
                    "words": [
                        {"start": 12.0, "end": 22.0, "word": "hello team this is the second speaker"}
                    ],
                },
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        speaker_merge_enabled=True,
        speaker_merge_similarity_threshold=0.5,
    )

    diariser = TwoSpeakerDiariser()
    # Inject a cached fake embedding model so the orchestrator skips the
    # pyannote.audio import path and calls our stub directly.
    diariser._lan_speaker_embedding_model = (  # type: ignore[attr-defined]
        lambda audio_path, start, end: [1.0, 0.0]
    )

    await pipeline.run_pipeline(
        fake_audio(tmp_path, "merge.mp3"),
        cfg,
        llm_client.LLMClient(),
        diariser,
        recording_id="rec-merge-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-merge-1" / "derived"
    metadata = json.loads(
        (derived / "diarization_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["speaker_merges"]
    # All segments must collapse to a single speaker after the merge.
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    assert len({row["speaker"] for row in diar_data}) == 1


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_writes_empty_speaker_merges_when_voices_differ(
    tmp_path: Path,
    mocker,
):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": "alpha beta gamma delta epsilon",
                    "words": [
                        {"start": 0.0, "end": 10.0, "word": "alpha beta gamma delta epsilon"}
                    ],
                },
                {
                    "start": 12.0,
                    "end": 22.0,
                    "text": "zeta eta theta iota kappa",
                    "words": [
                        {"start": 12.0, "end": 22.0, "word": "zeta eta theta iota kappa"}
                    ],
                },
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    def _embed(audio_path, start, end):
        # Two orthogonal vectors so the pair similarity is 0, well below the
        # threshold. This exercises the "embedding model ran, no merges" path.
        return [1.0, 0.0] if start < 11.0 else [0.0, 1.0]

    diariser = TwoSpeakerDiariser()
    diariser._lan_speaker_embedding_model = _embed  # type: ignore[attr-defined]

    await pipeline.run_pipeline(
        fake_audio(tmp_path, "merge-orthogonal.mp3"),
        cfg,
        llm_client.LLMClient(),
        diariser,
        recording_id="rec-merge-orthogonal-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-merge-orthogonal-1" / "derived"
    metadata = json.loads(
        (derived / "diarization_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["speaker_merges"] == {}
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    assert len({row["speaker"] for row in diar_data}) == 2


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_respects_speaker_merge_disabled_flag(
    tmp_path: Path,
    mocker,
):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": "alpha beta gamma delta epsilon",
                    "words": [
                        {"start": 0.0, "end": 10.0, "word": "alpha beta gamma delta epsilon"}
                    ],
                },
                {
                    "start": 12.0,
                    "end": 22.0,
                    "text": "zeta eta theta iota kappa",
                    "words": [
                        {"start": 12.0, "end": 22.0, "word": "zeta eta theta iota kappa"}
                    ],
                },
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        speaker_merge_enabled=False,
    )

    diariser = TwoSpeakerDiariser()
    # Even with identical centroids the disabled flag must short-circuit.
    diariser._lan_speaker_embedding_model = (  # type: ignore[attr-defined]
        lambda audio_path, start, end: [1.0, 0.0]
    )

    await pipeline.run_pipeline(
        fake_audio(tmp_path, "merge-disabled.mp3"),
        cfg,
        llm_client.LLMClient(),
        diariser,
        recording_id="rec-merge-disabled-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-merge-disabled-1" / "derived"
    metadata = json.loads(
        (derived / "diarization_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["speaker_merges"] == {}
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    assert len({row["speaker"] for row in diar_data}) == 2


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_marks_dummy_fallback_in_diarization_metadata(
    tmp_path: Path,
    mocker,
):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [{"start": 0.0, "end": 1.0, "word": "hello"}],
                }
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    recording_id = "rec-diar-fallback-1"

    await pipeline.run_pipeline(
        fake_audio(tmp_path, "dummy-fallback.mp3"),
        cfg,
        llm_client.LLMClient(),
        EmptyTracksDiariser(),
        recording_id=recording_id,
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / recording_id / "derived"
    metadata = json.loads(
        (derived / "diarization_metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["mode"] == "pyannote"
    assert metadata["degraded"] is True
    assert metadata["used_dummy_fallback"] is True
    assert metadata["smoothing_applied"] is False
    assert metadata["speaker_count_before_smoothing"] == 1
    assert metadata["speaker_count_after_smoothing"] == 1


@pytest.mark.asyncio
async def test_pipeline_short_transcript_keeps_single_pass_llm(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello world team"}],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    payloads: list[dict[str, object]] = []

    class _SinglePassLLM:
        async def generate(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            model: str | None = None,
            response_format: dict[str, object] | None = None,
        ):
            del system_prompt, model, response_format
            payloads.append(json.loads(user_prompt))
            return {
                "content": json.dumps(
                    {
                        "topic": "Short update",
                        "summary_bullets": ["Reviewed one short update."],
                        "decisions": [],
                        "action_items": [],
                        "emotional_summary": "Calm and focused.",
                        "questions": {"total_count": 0, "types": {}, "extracted": []},
                    }
                )
            }

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        llm_model="test-llm-model",
        llm_long_transcript_threshold_chars=1000,
        llm_chunk_max_chars=120,
        llm_chunk_overlap_chars=10,
    )
    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "short-single-pass.mp3"),
        cfg=cfg,
        llm=_SinglePassLLM(),
        diariser=DummyDiariser(),
        recording_id="rec-single-pass-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-single-pass-1" / "derived"
    assert len(payloads) == 1
    assert "speaker_turns" in payloads[0]
    assert not (derived / "llm_chunks_plan.json").exists()
    assert not (derived / "llm_merge_input.json").exists()


@pytest.mark.asyncio
async def test_pipeline_long_transcript_uses_chunked_llm_progress_and_artifacts(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": float(index),
                    "end": float(index) + 1.0,
                    "text": f"discussion item {index} " * 6,
                }
                for index in range(4)
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    events: list[tuple[str, float]] = []
    merge_payload: dict[str, object] = {}
    chunk_payloads: list[dict[str, object]] = []

    class _ChunkedLLM:
        async def generate(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            model: str | None = None,
            response_format: dict[str, object] | None = None,
        ):
            del system_prompt, model, response_format
            payload = json.loads(user_prompt)
            if "chunk" in payload:
                chunk_payloads.append(payload)
                index = int(payload["chunk"]["index"])
                return {
                    "content": json.dumps(
                        {
                            "topic_candidates": [f"Topic {index}"],
                            "summary_bullets": [f"Chunk bullet {index}"],
                            "decisions": [f"Decision {index}"] if index == 1 else [],
                            "action_items": (
                                [{"task": "Send notes", "owner": "Alex", "confidence": 0.8}]
                                if index == 2
                                else []
                            ),
                            "emotional_cues": ["Focused"],
                            "questions": (
                                {
                                    "total_count": 1,
                                    "types": {"status": 1},
                                    "extracted": ["Is QA done?"],
                                }
                                if index == 1
                                else {"total_count": 0, "types": {}, "extracted": []}
                            ),
                        }
                    )
                }
            merge_payload["payload"] = payload
            return {
                "content": json.dumps(
                    {
                        "topic": "Merged topic",
                        "summary_bullets": ["Merged summary bullet"],
                        "decisions": ["Decision 1"],
                        "action_items": [{"task": "Send notes", "owner": "Alex", "confidence": 0.8}],
                        "emotional_summary": "Focused and positive.",
                        "questions": {
                            "total_count": 1,
                            "types": {"status": 1},
                            "extracted": ["Is QA done?"],
                        },
                    }
                )
            }

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        llm_model="test-llm-model",
        llm_long_transcript_threshold_chars=120,
        llm_chunk_max_chars=90,
        llm_chunk_overlap_chars=20,
        llm_merge_max_tokens=1536,
    )

    def _progress(stage: str, progress: float) -> None:
        events.append((stage, progress))

    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "chunked.mp3"),
        cfg=cfg,
        llm=_ChunkedLLM(),
        diariser=DummyDiariser(),
        recording_id="rec-chunked-1",
        precheck=precheck_ok(),
        progress_callback=_progress,
    )

    derived = cfg.recordings_root / "rec-chunked-1" / "derived"
    plan_payload = json.loads((derived / "llm_chunks_plan.json").read_text(encoding="utf-8"))
    total_chunks = len(plan_payload["chunks"])
    compact_payload = json.loads((derived / "llm_compact_transcript.json").read_text(encoding="utf-8"))
    compact_text = (derived / "llm_compact_transcript.txt").read_text(encoding="utf-8")
    stage_names = [stage for stage, _progress in events]
    summary_payload = json.loads((derived / "summary.json").read_text(encoding="utf-8"))

    assert total_chunks > 1
    assert plan_payload["source_chars"] > plan_payload["compact_chars"]
    assert compact_payload["source_chars"] == plan_payload["source_chars"]
    assert compact_payload["compact_chars"] == plan_payload["compact_chars"]
    assert compact_payload["speaker_mapping"]
    assert "[" not in compact_text
    assert all(chunk["effective_chars"] >= chunk["base_chars"] for chunk in plan_payload["chunks"])
    assert all("time_range" in chunk for chunk in plan_payload["chunks"])
    assert stage_names[:6] == [
        "precheck",
        "stt",
        "stt",
        "diarize",
        "align",
        "language",
    ]
    assert stage_names[6] == f"llm_chunk_1_of_{total_chunks}"
    assert stage_names[6 + total_chunks] == "llm_merge"
    assert stage_names[-1] == "metrics"
    assert merge_payload["payload"]["merge_input"]["chunk_count"] == total_chunks
    assert chunk_payloads[0]["calendar"] == {
        "title": None,
        "attendees": [],
    }
    assert chunk_payloads[0]["chunk"]["time_range"]["start_seconds"] == 0.0
    assert "speaker_mapping" in chunk_payloads[0]
    assert "[" not in str(chunk_payloads[0]["transcript_chunk"])
    assert (derived / "llm_chunk_001_raw.json").exists()
    assert (derived / f"llm_chunk_{total_chunks:03d}_extract.json").exists()
    assert (derived / "llm_compact_transcript.txt").exists()
    assert (derived / "llm_compact_transcript.json").exists()
    assert (derived / "llm_merge_input.json").exists()
    assert (derived / "llm_merge_raw.json").exists()
    assert summary_payload["topic"] == "Merged topic"
    assert summary_payload["summary_bullets"] == ["Merged summary bullet"]


@pytest.mark.asyncio
async def test_pipeline_chunk_failure_is_explicit_and_marks_metrics_failed(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": float(index),
                    "end": float(index) + 1.0,
                    "text": f"discussion item {index} " * 6,
                }
                for index in range(4)
            ],
            {"language": "en", "language_probability": 0.95},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    class _FailingChunkLLM:
        async def generate(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            model: str | None = None,
            response_format: dict[str, object] | None = None,
        ):
            del system_prompt, model, response_format
            payload = json.loads(user_prompt)
            if "chunk" not in payload:
                raise AssertionError("merge should not run after a chunk failure")
            if int(payload["chunk"]["index"]) == 1:
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
            return {"content": "not json"}

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        llm_model="test-llm-model",
        llm_long_transcript_threshold_chars=120,
        llm_chunk_max_chars=90,
        llm_chunk_overlap_chars=20,
    )

    with pytest.raises(
        RuntimeError,
        match=r"LLM chunk 2/\d+ failed \[llm_chunk_parse_error\]: json_object_not_found",
    ):
        await pipeline.run_pipeline(
            audio_path=fake_audio(tmp_path, "chunked-fail.mp3"),
            cfg=cfg,
            llm=_FailingChunkLLM(),
            diariser=DummyDiariser(),
            recording_id="rec-chunked-fail-1",
            precheck=precheck_ok(),
        )

    derived = cfg.recordings_root / "rec-chunked-fail-1" / "derived"
    summary_data = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    metrics_data = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))
    error_payload = json.loads((derived / "llm_chunk_002_error.json").read_text(encoding="utf-8"))

    assert summary_data["status"] == "failed"
    assert metrics_data["status"] == "failed"
    assert "LLM chunk 2/" in metrics_data["error"]
    assert error_payload["error"] == "json_object_not_found"


@pytest.mark.asyncio
@respx.mock
async def test_alias_persist(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hi there friend"}],
            {"language": "en", "language_probability": 0.7},
        ),
    )

    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "- sum"}}]}
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.5}],
    )
    db = tmp_path / "db.yaml"
    db.write_text("S1: Alice\n", encoding="utf-8")
    cfg = pipeline.Settings(
        speaker_db=db,
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    res = await pipeline.run_pipeline(
        fake_audio(tmp_path, "one.mp3"),
        cfg,
        llm_client.LLMClient(),
        DummyDiariser(),
        precheck=precheck_ok(),
    )
    assert res.speakers == ["Alice"]
    assert res.segments[0].speaker == "S1"
    db.write_text("S1: Bob\n", encoding="utf-8")
    pipeline.refresh_aliases(res, db)
    assert res.speakers == ["Bob"]

    import yaml

    saved = yaml.safe_load(db.read_text(encoding="utf-8"))
    assert "S1" in saved


@pytest.mark.asyncio
@respx.mock
async def test_white_noise(tmp_path: Path, mocker):
    mocker.patch("whisperx.transcribe", return_value=([], {"language": "en"}))
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": ""}}]}
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.5}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    res = await pipeline.run_pipeline(
        fake_audio(tmp_path, "noise.mp3"),
        cfg,
        llm_client.LLMClient(),
        DummyDiariser(),
        precheck=precheck_ok(),
    )

    assert res.summary == "No speech detected"
    assert res.body == ""


@pytest.mark.asyncio
@respx.mock
async def test_no_talk(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "long silence indeed"}],
            {"language": "en"},
        ),
    )

    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": ""}}]}
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.0}],
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    with pytest.raises(
        llm_client.LLMEmptyContentError,
        match="LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS",
    ):
        await pipeline.run_pipeline(
            fake_audio(tmp_path, "notalk.mp3"),
            cfg,
            llm_client.LLMClient(),
            DummyDiariser(),
            precheck=precheck_ok(),
        )


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_preserves_zero_language_confidence(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello"}],
            {"language": "en", "language_probability": 0.0},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.5}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "zero-confidence.mp3"),
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-zero-conf",
        precheck=precheck_ok(),
    )

    transcript_path = cfg.recordings_root / "rec-zero-conf" / "derived" / "transcript.json"
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert payload["language"]["confidence"] == 0.0


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_writes_required_artifacts(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "hello team",
                    "words": [
                        {"start": 0.2, "end": 0.7, "word": "hello"},
                        {"start": 0.8, "end": 1.4, "word": "team"},
                    ],
                },
                {
                    "start": 12.0,
                    "end": 17.0,
                    "text": "status update",
                    "words": [
                        {"start": 12.1, "end": 12.6, "word": "status"},
                        {"start": 12.7, "end": 13.2, "word": "update"},
                    ],
                },
            ],
            {"language": "en", "language_probability": 0.98},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="pipeline.wav",
        duration_sec=24.0,
        speech=True,
    )
    result = await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-pipe-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-pipe-1" / "derived"
    transcript_data = json.loads((derived / "transcript.json").read_text(encoding="utf-8"))
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    speaker_turns = json.loads((derived / "speaker_turns.json").read_text(encoding="utf-8"))

    assert transcript_data["language"]["detected"] == "en"
    assert transcript_data["language"]["confidence"] == 0.98
    assert transcript_data["segments"][0]["words"][0]["word"] == "hello"
    assert diar_data[0]["speaker"] == "S1"
    assert diar_data[1]["speaker"] == "S2"
    assert speaker_turns[0]["speaker"] == "S1"
    assert speaker_turns[-1]["speaker"] == "S2"

    snippets = sorted((derived / "snippets").glob("*/*.wav"))
    assert len(snippets) >= 2
    assert all(path.stat().st_size > 44 for path in snippets)
    assert len(result.unknown_chunks) >= 2


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_writes_language_spans_for_mixed_language_segments(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "hello team and thanks",
                    "language": "en",
                },
                {
                    "start": 4.0,
                    "end": 12.0,
                    "text": "hola equipo y gracias por venir",
                    "language": "es",
                },
            ],
            {"language": "en", "language_probability": 0.91},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "mixed.mp3"),
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-mixed-lang-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-mixed-lang-1" / "derived"
    transcript_data = json.loads((derived / "transcript.json").read_text(encoding="utf-8"))
    assert transcript_data["dominant_language"] == "es"
    assert set(transcript_data["language_distribution"]).issuperset({"en", "es"})
    assert len(transcript_data["language_spans"]) >= 2
    assert transcript_data["language_spans"][0]["lang"] == "en"
    assert transcript_data["language_spans"][1]["lang"] == "es"


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_single_language_auto_keeps_single_asr_pass(tmp_path: Path, mocker):
    asr_languages: list[str | None] = []

    def _transcribe(_audio_path: str, **kwargs):
        asr_languages.append(kwargs.get("language"))
        return (
            [{"start": 0.0, "end": 6.0, "text": "hello team thanks for joining"}],
            {"language": "en", "language_probability": 0.96},
        )

    mocker.patch("whisperx.transcribe", side_effect=_transcribe)
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        asr_multilingual_mode="auto",
    )
    audio = wav_audio(
        tmp_path,
        name="single-language.wav",
        duration_sec=8.0,
        speech=True,
    )
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-single-language-1",
        precheck=precheck_ok(),
    )

    transcript_data = json.loads(
        (
            cfg.recordings_root
            / "rec-single-language-1"
            / "derived"
            / "transcript.json"
        ).read_text(encoding="utf-8")
    )
    assert asr_languages == ["auto"]
    assert transcript_data["multilingual_asr"]["used_multilingual_path"] is False
    assert transcript_data["multilingual_asr"]["selected_mode"] == "single_language"
    assert transcript_data["review"]["required"] is False


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_multilingual_asr_uses_chunk_language_hints(tmp_path: Path, mocker):
    asr_languages: list[str | None] = []

    def _transcribe(audio_path: str, **kwargs):
        language = kwargs.get("language")
        asr_languages.append(language)
        if audio_path == str(audio):
            return (
                [
                    {"start": 0.0, "end": 4.0, "text": "hello team and thanks"},
                    {"start": 4.0, "end": 8.0, "text": "hola equipo y gracias"},
                ],
                {"language": "en", "language_probability": 0.91},
            )
        if language == "en":
            return (
                [{"start": 0.0, "end": 4.0, "text": "hello team and thanks"}],
                {"language": "en", "language_probability": 0.98},
            )
        return (
            [{"start": 0.0, "end": 4.0, "text": "hola equipo y gracias"}],
            {"language": "es", "language_probability": 0.97},
        )

    mocker.patch("whisperx.transcribe", side_effect=_transcribe)
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        asr_multilingual_mode="auto",
    )
    audio = wav_audio(
        tmp_path,
        name="mixed-language.wav",
        duration_sec=8.0,
        speech=True,
    )
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-multilingual-hints-1",
        precheck=precheck_ok(),
    )

    transcript_data = json.loads(
        (
            cfg.recordings_root
            / "rec-multilingual-hints-1"
            / "derived"
            / "transcript.json"
        ).read_text(encoding="utf-8")
    )
    assert asr_languages == ["auto", "en", "es"]
    assert transcript_data["multilingual_asr"]["used_multilingual_path"] is True
    assert transcript_data["multilingual_asr"]["selected_mode"] == "multilingual"
    assert transcript_data["language_spans"][0]["lang"] == "en"
    assert transcript_data["language_spans"][1]["lang"] == "es"
    hinted_languages = {
        row["language_hint"]
        for row in transcript_data["segments"]
        if "language_hint" in row
    }
    assert hinted_languages == {"en", "es"}
    assert transcript_data["review"]["required"] is False


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_multilingual_asr_forwards_glossary_context_and_writes_artifact(
    tmp_path: Path,
    mocker,
):
    seen_calls: list[dict[str, object]] = []

    def _transcribe(audio_path: str, **kwargs):
        seen_calls.append(
            {
                "audio_path": audio_path,
                "language": kwargs.get("language"),
                "initial_prompt": kwargs.get("initial_prompt"),
                "hotwords": kwargs.get("hotwords"),
            }
        )
        if audio_path == str(audio):
            return (
                [
                    {"start": 0.0, "end": 4.0, "text": "hello team and thanks"},
                    {"start": 4.0, "end": 8.0, "text": "hola equipo y gracias"},
                ],
                {"language": "en", "language_probability": 0.91},
            )
        if kwargs.get("language") == "en":
            return (
                [{"start": 0.0, "end": 4.0, "text": "hello team and thanks"}],
                {"language": "en", "language_probability": 0.98},
            )
        return (
            [{"start": 0.0, "end": 4.0, "text": "hola equipo y gracias"}],
            {"language": "es", "language_probability": 0.97},
        )

    mocker.patch("whisperx.transcribe", side_effect=_transcribe)
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        asr_multilingual_mode="auto",
    )
    audio = wav_audio(
        tmp_path,
        name="mixed-language-glossary.wav",
        duration_sec=8.0,
        speech=True,
    )
    glossary_payload = {
        "version": 1,
        "recording_id": "rec-multilingual-glossary-1",
        "entries": [
            {
                "canonical_text": "Sander",
                "aliases": ["Sandia"],
                "kind": "person",
                "sources": ["correction"],
                "terms": ["Sander", "Sandia"],
                "term_count": 2,
            }
        ],
        "terms": ["Sander", "Sandia"],
        "entry_count": 1,
        "term_count": 2,
        "truncated": False,
        "initial_prompt": "Glossary: Sander; Sandia",
        "hotwords": "Sander, Sandia",
    }
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-multilingual-glossary-1",
        precheck=precheck_ok(),
        asr_glossary=glossary_payload,
    )

    assert [call["language"] for call in seen_calls] == ["auto", "en", "es"]
    assert all(
        call["initial_prompt"] == "Glossary: Sander; Sandia"
        and call["hotwords"] == "Sander, Sandia"
        for call in seen_calls
    )
    artifact = json.loads(
        (
            cfg.recordings_root
            / "rec-multilingual-glossary-1"
            / "derived"
            / "asr_glossary.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["recording_id"] == "rec-multilingual-glossary-1"
    assert artifact["terms"] == ["Sander", "Sandia"]


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_omits_glossary_artifact_when_runtime_drops_all_glossary_hints(
    tmp_path: Path,
    mocker,
):
    def _transcribe(audio_path: str, **kwargs):
        assert audio_path.endswith(".wav")
        if "initial_prompt" in kwargs:
            raise TypeError("FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'initial_prompt'")
        if "hotwords" in kwargs:
            raise TypeError("FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'hotwords'")
        if "word_timestamps" in kwargs:
            raise TypeError("unexpected keyword argument 'word_timestamps'")
        return (
            [{"start": 0.0, "end": 1.0, "text": "hello team"}],
            {"language": "en", "language_probability": 0.98},
        )

    mocker.patch("whisperx.transcribe", side_effect=_transcribe)
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        asr_multilingual_mode="force_single_language",
    )
    audio = wav_audio(
        tmp_path,
        name="runtime-drop-glossary.wav",
        duration_sec=8.0,
        speech=True,
    )
    step_log: list[str] = []
    glossary_payload = {
        "version": 1,
        "recording_id": "rec-runtime-drop-glossary",
        "entries": [
            {
                "canonical_text": "Sander",
                "aliases": ["Sandia"],
                "kind": "person",
                "sources": ["correction"],
                "terms": ["Sander", "Sandia"],
                "term_count": 2,
            }
        ],
        "terms": ["Sander", "Sandia"],
        "entry_count": 1,
        "term_count": 2,
        "truncated": False,
        "initial_prompt": "Glossary: Sander; Sandia",
        "hotwords": "Sander, Sandia",
    }
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-runtime-drop-glossary",
        precheck=precheck_ok(),
        asr_glossary=glossary_payload,
        step_log_callback=step_log.append,
    )

    assert any("glossary context unsupported" in line for line in step_log)
    artifact = (
        cfg.recordings_root
        / "rec-runtime-drop-glossary"
        / "derived"
        / "asr_glossary.json"
    )
    assert not artifact.exists()


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_multilingual_model_path_reuses_single_whisperx_model_load(
    tmp_path: Path,
    mocker,
    monkeypatch,
):
    fake_whisperx = ModuleType("whisperx")
    fake_asr = ModuleType("whisperx.asr")
    load_model_calls = 0
    asr_languages: list[str | None] = []

    class _FakeModel:
        vad_model = staticmethod(lambda _payload: [])

        def transcribe(
            self,
            audio_input: str,
            *,
            batch_size: int,
            vad_filter: bool,
            language: str | None,
        ) -> dict[str, object]:
            del audio_input
            assert batch_size == 16
            assert vad_filter is True
            asr_languages.append(language)
            if language is None:
                return {
                    "segments": [
                        {"start": 0.0, "end": 4.0, "text": "hello team and thanks"},
                        {"start": 4.0, "end": 8.0, "text": "hola equipo y gracias"},
                    ],
                    "language": "en",
                }
            if language == "en":
                return {
                    "segments": [
                        {"start": 0.0, "end": 4.0, "text": "hello team and thanks"}
                    ],
                    "language": "en",
                }
            return {
                "segments": [
                    {"start": 0.0, "end": 4.0, "text": "hola equipo y gracias"}
                ],
                "language": "es",
            }

    def _load_audio(path: str) -> str:
        return path

    def _load_model(
        model_name: str,
        device: str,
        compute_type: str = "int8",
        vad_method: str = "silero",
    ) -> _FakeModel:
        nonlocal load_model_calls
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        assert vad_method == "silero"
        load_model_calls += 1
        return _FakeModel()

    fake_whisperx.transcribe = None
    fake_whisperx.load_audio = _load_audio
    fake_whisperx.asr = fake_asr
    fake_asr.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)
    monkeypatch.setattr(
        "lan_transcriber.pipeline_steps.orchestrator.ensure_ctranslate2_no_execstack",
        lambda: [],
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        asr_device="cpu",
        asr_enable_align=False,
        asr_multilingual_mode="auto",
    )
    audio = wav_audio(
        tmp_path,
        name="mixed-model-path.wav",
        duration_sec=8.0,
        speech=True,
    )
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-multilingual-model-reuse-1",
        precheck=precheck_ok(),
    )

    transcript_data = json.loads(
        (
            cfg.recordings_root
            / "rec-multilingual-model-reuse-1"
            / "derived"
            / "transcript.json"
        ).read_text(encoding="utf-8")
    )
    assert load_model_calls == 1
    assert asr_languages == [None, "en", "es"]
    assert transcript_data["multilingual_asr"]["used_multilingual_path"] is True
    assert transcript_data["segments"][0]["language_hint"] == "en"
    assert transcript_data["segments"][1]["language_hint"] == "es"


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_multilingual_uncertainty_sets_review_metadata(tmp_path: Path, mocker):
    def _transcribe(audio_path: str, **kwargs):
        language = kwargs.get("language")
        if audio_path == str(audio):
            return (
                [
                    {"start": 0.0, "end": 4.0, "text": "hello team and thanks"},
                    {"start": 4.0, "end": 8.0, "text": "hola equipo y gracias"},
                ],
                {"language": "en", "language_probability": 0.91},
            )
        if language == "en":
            return (
                [{"start": 0.0, "end": 4.0, "text": "hello team and thanks"}],
                {"language": "en", "language_probability": 0.98},
            )
        return (
            [{"start": 0.0, "end": 4.0, "text": "hello team and thanks"}],
            {"language": "en", "language_probability": 0.99},
        )

    mocker.patch("whisperx.transcribe", side_effect=_transcribe)
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        asr_multilingual_mode="auto",
    )
    audio = wav_audio(
        tmp_path,
        name="mixed-uncertain.wav",
        duration_sec=8.0,
        speech=True,
    )
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-multilingual-review-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-multilingual-review-1" / "derived"
    transcript_data = json.loads((derived / "transcript.json").read_text(encoding="utf-8"))
    metrics_data = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))
    assert transcript_data["review"]["required"] is True
    assert transcript_data["review"]["reason_code"] == "multilingual_uncertain"
    assert transcript_data["multilingual_asr"]["chunks"][1]["conflict"] is True
    assert metrics_data["review_required"] is True


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_multilingual_backfills_detected_language_without_distribution_confidence(
    tmp_path: Path,
    mocker,
):
    mocker.patch(
        "lan_transcriber.pipeline_steps.orchestrator.run_language_aware_asr",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello", "language": "en"}],
            {"language": "unknown"},
            {"used_multilingual_path": True, "selected_mode": "multilingual"},
        ),
    )
    mocker.patch(
        "lan_transcriber.pipeline_steps.orchestrator.analyse_languages",
        return_value=LanguageAnalysis(
            segments=[{"start": 0.0, "end": 1.0, "text": "hello", "language": "en"}],
            dominant_language="en",
            distribution={},
            spans=[],
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="multilingual-backfill.wav",
        duration_sec=8.0,
        speech=True,
    )
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-multilingual-backfill-1",
        precheck=precheck_ok(),
    )

    transcript_data = json.loads(
        (
            cfg.recordings_root
            / "rec-multilingual-backfill-1"
            / "derived"
            / "transcript.json"
        ).read_text(encoding="utf-8")
    )
    assert transcript_data["language"]["detected"] == "en"
    assert transcript_data["language"]["confidence"] is None


def test_segment_language_prefers_detected_over_text_guess():
    segment = {"text": "the and to of in"}
    resolved = pipeline._segment_language(
        segment,
        detected_language="fr",
        transcript_language_override=None,
    )
    assert resolved == "fr"


@pytest.mark.asyncio
async def test_pipeline_summary_language_override_changes_prompt(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello team today."}],
            {"language": "en", "language_probability": 0.9},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )

    captured: dict[str, str] = {}

    async def _fake_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ):
        captured["system"] = system_prompt
        return {"content": "- resumen"}

    mocker.patch.object(llm_client.LLMClient, "generate", _fake_generate)

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "summary-lang.mp3"),
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-summary-lang-1",
        precheck=precheck_ok(),
        target_summary_language="es",
    )

    assert "in Spanish." in captured["system"]
    summary_data = json.loads(
        (cfg.recordings_root / "rec-summary-lang-1" / "derived" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_data["target_summary_language"] == "es"


def test_build_summary_payload_prefers_parsed_summary_field():
    payload = pipeline.build_summary_payload(
        raw_llm_content='{"topic":"T","summary":"- one\\n- two","decisions":[],"action_items":[],"emotional_summary":"ok","questions":{"total_count":0,"types":{},"extracted":[]}}',
        model="m",
        target_summary_language="en",
        friendly=0,
    )
    assert payload["summary_bullets"] == ["one", "two"]
    assert payload["summary"] == "- one\n- two"


def test_build_summary_payload_normalises_non_finite_action_item_confidence():
    payload = pipeline.build_summary_payload(
        raw_llm_content='{"topic":"T","summary_bullets":["one"],"decisions":[],"action_items":[{"task":"Do thing","confidence":NaN}],"emotional_summary":"ok","questions":{"total_count":0,"types":{},"extracted":[]}}',
        model="m",
        target_summary_language="en",
        friendly=0,
    )
    assert payload["action_items"][0]["confidence"] == 0.5


def test_build_summary_payload_normalises_non_finite_question_counts():
    payload = pipeline.build_summary_payload(
        raw_llm_content='{"topic":"T","summary_bullets":["one"],"decisions":[],"action_items":[],"emotional_summary":"ok","questions":{"total_count":NaN,"types":{"open":NaN},"extracted":[]}}',
        model="m",
        target_summary_language="en",
        friendly=0,
    )
    assert payload["questions"]["total_count"] == 0
    assert payload["questions"]["types"]["open"] == 0


def test_build_structured_summary_prompts_preserves_long_turn_text():
    long_text = " ".join(f"word{i}" for i in range(300))
    expected = " ".join(long_text.split())
    _system_prompt, user_prompt = pipeline.build_structured_summary_prompts(
        [
            {
                "start": 0.0,
                "end": 120.0,
                "speaker": "S1",
                "text": long_text,
                "language": "en",
            }
        ],
        "en",
    )
    payload = json.loads(user_prompt)
    turns = payload["speaker_turns"]
    assert len(turns) > 1
    reconstructed = " ".join(str(turn["text"]) for turn in turns)
    assert reconstructed == expected


def test_build_structured_summary_prompts_keeps_turns_beyond_legacy_cap():
    speaker_turns = [
        {"start": float(i), "end": float(i) + 0.5, "speaker": "S1", "text": f"turn {i}"}
        for i in range(350)
    ]
    _system_prompt, user_prompt = pipeline.build_structured_summary_prompts(speaker_turns, "en")
    payload = json.loads(user_prompt)
    turns = payload["speaker_turns"]
    assert len(turns) == len(speaker_turns)
    assert turns[-1]["text"] == "turn 349"
    assert payload["required_schema"]["tone_score"] == "integer [0,100], where 100 = very positive/friendly"


@pytest.mark.asyncio
async def test_pipeline_writes_structured_summary_payload(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello team, we ship on Friday."}],
            {"language": "en", "language_probability": 0.9},
        ),
    )

    async def _fake_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ):
        return {
            "content": json.dumps(
                {
                    "topic": "Weekly release sync",
                    "summary_bullets": ["Team confirmed release scope for Friday."],
                    "decisions": ["Release window is Friday 16:00 UTC."],
                    "action_items": [
                        {
                            "task": "Send release notes",
                            "owner": "Alex",
                            "deadline": "2026-02-23",
                            "confidence": 0.92,
                        }
                    ],
                    "tone_score": 74,
                    "emotional_summary": "Focused and optimistic.",
                    "questions": {
                        "total_count": 1,
                        "types": {
                            "open": 0,
                            "yes_no": 0,
                            "clarification": 0,
                            "status": 1,
                            "decision_seeking": 0,
                        },
                        "extracted": ["Is QA complete?"],
                    },
                }
            )
        }

    mocker.patch.object(llm_client.LLMClient, "generate", _fake_generate)

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "structured.mp3"),
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-structured-1",
        precheck=precheck_ok(),
    )

    summary_data = json.loads(
        (cfg.recordings_root / "rec-structured-1" / "derived" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_data["topic"] == "Weekly release sync"
    assert summary_data["decisions"] == ["Release window is Friday 16:00 UTC."]
    assert summary_data["action_items"][0]["owner"] == "Alex"
    assert summary_data["action_items"][0]["confidence"] == 0.92
    assert summary_data["tone_score"] == 74
    assert summary_data["friendly"] == 74
    assert summary_data["questions"]["types"]["status"] == 1
    assert summary_data["summary_bullets"] == ["Team confirmed release scope for Friday."]


@pytest.mark.asyncio
async def test_pipeline_prompt_includes_calendar_context(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "status update for roadmap"}],
            {"language": "en", "language_probability": 0.8},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )

    captured: dict[str, str] = {}

    async def _fake_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ):
        captured["user_prompt"] = user_prompt
        return {"content": '{"summary_bullets":["ok"],"decisions":[],"action_items":[],"questions":{"total_count":0,"types":{},"extracted":[]},"topic":"A","emotional_summary":"Neutral."}'}

    mocker.patch.object(llm_client.LLMClient, "generate", _fake_generate)

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "calendar.mp3"),
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-calendar-1",
        precheck=precheck_ok(),
        calendar_title="Roadmap Review",
        calendar_attendees=["Alex", "Priya"],
    )

    prompt_payload = json.loads(captured["user_prompt"])
    assert prompt_payload["calendar"]["title"] == "Roadmap Review"
    assert prompt_payload["calendar"]["attendees"] == ["Alex", "Priya"]


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_transcript_language_override_is_used_for_asr(tmp_path: Path, mocker):
    captured: dict[str, object] = {}

    def _fake_transcribe(*_args, **kwargs):
        captured["language"] = kwargs.get("language")
        return (
            [{"start": 0.0, "end": 1.0, "text": "hola equipo hoy."}],
            {"language": "es", "language_probability": 0.9},
        )

    mocker.patch("whisperx.transcribe", side_effect=_fake_transcribe)
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- resumen"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    await pipeline.run_pipeline(
        audio_path=fake_audio(tmp_path, "override-asr.mp3"),
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-override-asr-1",
        precheck=precheck_ok(),
        transcript_language_override="es",
    )

    assert captured["language"] == "es"


@pytest.mark.asyncio
@respx.mock
async def test_pipeline_accepts_triplet_itertracks_without_mode_and_exports_clean_snippets(
    tmp_path: Path,
    mocker,
):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "hello team",
                    "words": [
                        {"start": 0.2, "end": 0.7, "word": "hello"},
                        {"start": 0.8, "end": 1.4, "word": "team"},
                    ],
                },
                {
                    "start": 12.0,
                    "end": 17.0,
                    "text": "status update",
                    "words": [
                        {"start": 12.1, "end": 12.6, "word": "status"},
                        {"start": 12.7, "end": 13.2, "word": "update"},
                    ],
                },
            ],
            {"language": "en", "language_probability": 0.98},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.6}],
    )
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "- summary"}}]},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="triplet.wav",
        duration_sec=24.0,
        speech=True,
    )
    await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerTripletDiariser(),
        recording_id="rec-triplet-1",
        precheck=precheck_ok(),
    )

    derived = cfg.recordings_root / "rec-triplet-1" / "derived"
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    metadata = json.loads((derived / "diarization_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((derived / "snippets_manifest.json").read_text(encoding="utf-8"))

    assert [row["speaker"] for row in diar_data] == ["S1", "S2"]
    assert metadata["mode"] == "unknown"
    assert metadata["degraded"] is False
    assert [entry["status"] for entry in manifest["speakers"]["S1"]] == ["accepted"]
    assert [entry["status"] for entry in manifest["speakers"]["S2"]] == ["accepted"]


@pytest.mark.asyncio
async def test_pipeline_quarantine_clears_stale_snippets(tmp_path: Path):
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="quarantine.wav",
        duration_sec=24.0,
        speech=True,
    )
    snippets_root = cfg.recordings_root / "rec-quarantine-1" / "derived" / "snippets"
    stale_snippet = snippets_root / "S1" / "old.wav"
    stale_snippet.parent.mkdir(parents=True, exist_ok=True)
    stale_snippet.write_bytes(b"stale")

    result = await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=TwoSpeakerDiariser(),
        recording_id="rec-quarantine-1",
        precheck=pipeline.PrecheckResult(
            duration_sec=5.0,
            speech_ratio=0.0,
            quarantine_reason="duration_lt_20s",
        ),
    )

    assert result.summary == "Quarantined"
    assert snippets_root.exists()
    assert list(snippets_root.iterdir()) == []


@pytest.mark.asyncio
async def test_pipeline_quarantine_skips_whisperx_import(tmp_path: Path, monkeypatch):
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="quarantine-no-whisperx.wav",
        duration_sec=24.0,
        speech=True,
    )

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "whisperx":
            raise AssertionError("whisperx should not be imported for quarantined runs")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "whisperx", raising=False)
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    result = await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-quarantine-no-whisperx",
        precheck=pipeline.PrecheckResult(
            duration_sec=5.0,
            speech_ratio=0.0,
            quarantine_reason="duration_lt_20s",
        ),
    )

    assert result.summary == "Quarantined"


@pytest.mark.asyncio
async def test_pipeline_without_glossary_removes_stale_glossary_artifact(tmp_path: Path) -> None:
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        llm_model="test-model",
    )
    audio = fake_audio(tmp_path, name="stale-glossary.mp3")
    artifact = (
        cfg.recordings_root
        / "rec-stale-glossary"
        / "derived"
        / "asr_glossary.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({"version": 1, "recording_id": "rec-stale-glossary", "terms": ["stale"]}),
        encoding="utf-8",
    )

    result = await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-stale-glossary",
        precheck=pipeline.PrecheckResult(
            duration_sec=5.0,
            speech_ratio=0.0,
            quarantine_reason="duration_lt_20s",
        ),
    )

    assert result.summary == "Quarantined"
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_pipeline_quarantine_does_not_write_glossary_artifact_when_glossary_is_present(
    tmp_path: Path,
) -> None:
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        llm_model="test-model",
    )
    audio = fake_audio(tmp_path, name="quarantine-glossary.mp3")
    artifact = (
        cfg.recordings_root
        / "rec-quarantine-glossary"
        / "derived"
        / "asr_glossary.json"
    )

    result = await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-quarantine-glossary",
        precheck=pipeline.PrecheckResult(
            duration_sec=5.0,
            speech_ratio=0.0,
            quarantine_reason="duration_lt_20s",
        ),
        asr_glossary={
            "version": 1,
            "recording_id": "rec-quarantine-glossary",
            "terms": ["Sander", "Sandia"],
            "initial_prompt": "Glossary: Sander; Sandia",
            "hotwords": "Sander, Sandia",
        },
    )

    assert result.summary == "Quarantined"
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_pipeline_no_speech_clears_stale_snippets(tmp_path: Path, mocker):
    mocker.patch("whisperx.transcribe", return_value=([], {"language": "en"}))

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="no-speech.wav",
        duration_sec=24.0,
        speech=False,
    )
    snippets_root = cfg.recordings_root / "rec-no-speech-1" / "derived" / "snippets"
    stale_snippet = snippets_root / "S1" / "old.wav"
    stale_snippet.parent.mkdir(parents=True, exist_ok=True)
    stale_snippet.write_bytes(b"stale")

    result = await pipeline.run_pipeline(
        audio_path=audio,
        cfg=cfg,
        llm=llm_client.LLMClient(),
        diariser=DummyDiariser(),
        recording_id="rec-no-speech-1",
        precheck=precheck_ok(),
    )

    assert result.summary == "No speech detected"
    assert snippets_root.exists()
    assert list(snippets_root.iterdir()) == []


@pytest.mark.asyncio
async def test_pipeline_error_marks_metrics_failed(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello world today"}],
            {"language": "en", "language_probability": 0.9},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
    )

    class _FailingLLM:
        async def generate(self, **_kwargs):
            raise RuntimeError("llm boom")

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="llm-fail.wav",
        duration_sec=24.0,
        speech=True,
    )

    with pytest.raises(RuntimeError, match="llm boom"):
        await pipeline.run_pipeline(
            audio_path=audio,
            cfg=cfg,
            llm=_FailingLLM(),
            diariser=DummyDiariser(),
            recording_id="rec-llm-fail-1",
            precheck=precheck_ok(),
        )

    derived = cfg.recordings_root / "rec-llm-fail-1" / "derived"
    summary_data = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    metrics_data = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))

    assert summary_data["status"] == "failed"
    assert metrics_data["status"] == "failed"
    assert metrics_data["error"] == "llm boom"


@pytest.mark.asyncio
async def test_pipeline_snippet_export_error_marks_metrics_failed(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello world today"}],
            {"language": "en", "language_probability": 0.9},
        ),
    )
    mocker.patch(
        "lan_transcriber.pipeline_steps.orchestrator.export_speaker_snippets",
        side_effect=RuntimeError("snippet boom"),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="snippet-fail.wav",
        duration_sec=24.0,
        speech=True,
    )

    with pytest.raises(RuntimeError, match="snippet boom"):
        await pipeline.run_pipeline(
            audio_path=audio,
            cfg=cfg,
            llm=llm_client.LLMClient(),
            diariser=DummyDiariser(),
            recording_id="rec-snippet-fail-1",
            precheck=precheck_ok(),
        )

    derived = cfg.recordings_root / "rec-snippet-fail-1" / "derived"
    summary_data = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    metrics_data = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))

    assert summary_data["status"] == "failed"
    assert summary_data["friendly"] == 0
    assert metrics_data["status"] == "failed"
    assert metrics_data["error"] == "snippet boom"


@pytest.mark.asyncio
async def test_pipeline_pre_llm_error_marks_metrics_failed(tmp_path: Path, mocker):
    mocker.patch("whisperx.transcribe", side_effect=RuntimeError("asr boom"))

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="asr-fail.wav",
        duration_sec=24.0,
        speech=True,
    )

    with pytest.raises(RuntimeError, match="asr boom"):
        await pipeline.run_pipeline(
            audio_path=audio,
            cfg=cfg,
            llm=llm_client.LLMClient(),
            diariser=DummyDiariser(),
            recording_id="rec-asr-fail-1",
            precheck=precheck_ok(),
        )

    derived = cfg.recordings_root / "rec-asr-fail-1" / "derived"
    summary_data = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    metrics_data = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))

    assert summary_data["status"] == "failed"
    assert metrics_data["status"] == "failed"
    assert metrics_data["error"] == "asr boom"


@pytest.mark.asyncio
async def test_pipeline_diariser_error_marks_metrics_failed(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello world today"}],
            {"language": "en", "language_probability": 0.9},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="diariser-fail.wav",
        duration_sec=24.0,
        speech=True,
    )

    with pytest.raises(RuntimeError, match="diariser boom"):
        await pipeline.run_pipeline(
            audio_path=audio,
            cfg=cfg,
            llm=llm_client.LLMClient(),
            diariser=FailingDiariser(),
            recording_id="rec-diariser-fail-1",
            precheck=precheck_ok(),
        )

    derived = cfg.recordings_root / "rec-diariser-fail-1" / "derived"
    summary_data = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    metrics_data = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))

    assert summary_data["status"] == "failed"
    assert metrics_data["status"] == "failed"
    assert metrics_data["error"] == "diariser boom"


@pytest.mark.asyncio
async def test_pipeline_itertracks_error_marks_metrics_failed(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello world today"}],
            {"language": "en", "language_probability": 0.9},
        ),
    )

    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = wav_audio(
        tmp_path,
        name="itertracks-fail.wav",
        duration_sec=24.0,
        speech=True,
    )

    with pytest.raises(RuntimeError, match="itertracks boom"):
        await pipeline.run_pipeline(
            audio_path=audio,
            cfg=cfg,
            llm=llm_client.LLMClient(),
            diariser=FailingItertracksDiariser(),
            recording_id="rec-itertracks-fail-1",
            precheck=precheck_ok(),
        )

    derived = cfg.recordings_root / "rec-itertracks-fail-1" / "derived"
    summary_data = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    metrics_data = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))

    assert summary_data["status"] == "failed"
    assert metrics_data["status"] == "failed"
    assert metrics_data["error"] == "itertracks boom"


def test_run_precheck_quarantine_rules(tmp_path: Path):
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )

    short_audio = wav_audio(
        tmp_path,
        name="short.wav",
        duration_sec=5.0,
        speech=True,
    )
    short_result = pipeline.run_precheck(short_audio, cfg)
    assert short_result.quarantine_reason == "duration_lt_20s"
    assert short_result.duration_sec is not None and short_result.duration_sec < 20.0

    silent_audio = wav_audio(
        tmp_path,
        name="silent.wav",
        duration_sec=30.0,
        speech=False,
    )
    silent_result = pipeline.run_precheck(silent_audio, cfg)
    assert silent_result.quarantine_reason == "speech_ratio_lt_0.10"
    assert silent_result.speech_ratio is not None and silent_result.speech_ratio < 0.10

    voiced_audio = wav_audio(
        tmp_path,
        name="voiced.wav",
        duration_sec=30.0,
        speech=True,
    )
    voiced_result = pipeline.run_precheck(voiced_audio, cfg)
    assert voiced_result.quarantine_reason is None
    assert voiced_result.speech_ratio is not None and voiced_result.speech_ratio > 0.10


def test_run_precheck_quarantines_when_metrics_unavailable(tmp_path: Path, monkeypatch):
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    audio = tmp_path / "probe-fail.mp3"
    audio.write_bytes(b"not-real-audio")

    monkeypatch.setattr(precheck_step, "_audio_duration_from_wave", lambda _path: None)
    monkeypatch.setattr(precheck_step, "_audio_duration_from_ffprobe", lambda _path: None)
    monkeypatch.setattr(precheck_step, "_speech_ratio_from_wave", lambda _path: None)
    monkeypatch.setattr(precheck_step, "_speech_ratio_from_ffmpeg", lambda _path: None)

    result = pipeline.run_precheck(audio, cfg)
    assert result.quarantine_reason == "precheck_metrics_unavailable"
    assert result.duration_sec is None
    assert result.speech_ratio is None


def test_speech_ratio_from_ffmpeg_uses_unbounded_wait(tmp_path: Path, monkeypatch):
    audio = tmp_path / "ffmpeg-input.mp3"
    audio.write_bytes(b"audio")

    monkeypatch.setattr(precheck_step.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Stdout:
        def __init__(self):
            self._chunks = [b"\x00" * 960, b"\x00" * 960, b""]

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    class _Proc:
        def __init__(self):
            self.stdout = _Stdout()
            self.returncode = 0
            self.wait_timeout = "unset"

        def wait(self, timeout=None):
            self.wait_timeout = timeout
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    proc = _Proc()
    monkeypatch.setattr(precheck_step.subprocess, "Popen", lambda *_args, **_kwargs: proc)

    ratio = precheck_step._speech_ratio_from_ffmpeg(audio)
    assert ratio == 0.0
    assert proc.wait_timeout is None
