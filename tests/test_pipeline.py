from __future__ import annotations

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


class DummyDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label: bool = False):
                from types import SimpleNamespace

                if yield_label:
                    yield SimpleNamespace(start=0.0, end=1.0), "S1"
                else:
                    yield (SimpleNamespace(start=0.0, end=1.0),)

        return Ann()


class TwoSpeakerDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label: bool = False):
                from types import SimpleNamespace

                if yield_label:
                    yield SimpleNamespace(start=0.0, end=12.0), "S1"
                    yield SimpleNamespace(start=12.0, end=24.0), "S2"
                else:
                    yield (SimpleNamespace(start=0.0, end=12.0),)
                    yield (SimpleNamespace(start=12.0, end=24.0),)

        return Ann()


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

    respx.post("http://llm:8000/v1/chat/completions").mock(
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
async def test_alias_persist(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hi there friend"}],
            {"language": "en", "language_probability": 0.7},
        ),
    )

    respx.post("http://llm:8000/v1/chat/completions").mock(
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
    respx.post("http://llm:8000/v1/chat/completions").mock(
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

    respx.post("http://llm:8000/v1/chat/completions").mock(
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
    res = await pipeline.run_pipeline(
        fake_audio(tmp_path, "notalk.mp3"),
        cfg,
        llm_client.LLMClient(),
        DummyDiariser(),
        precheck=precheck_ok(),
    )

    assert res.friendly == 0
    assert res.summary.strip() == ""


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
    respx.post("http://llm:8000/v1/chat/completions").mock(
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
    respx.post("http://llm:8000/v1/chat/completions").mock(
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
async def test_pipeline_accepts_pyannote_triplet_itertracks(tmp_path: Path, mocker):
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
    respx.post("http://llm:8000/v1/chat/completions").mock(
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
    assert [row["speaker"] for row in diar_data] == ["S1", "S2"]


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
