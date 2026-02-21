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


class FailingDiariser:
    async def __call__(self, audio_path: Path):
        raise RuntimeError("diariser boom")


class FailingItertracksDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label: bool = False):
                raise RuntimeError("itertracks boom")

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
    assert res.summary.strip() == "- No summary available."


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


@pytest.mark.asyncio
async def test_pipeline_writes_structured_summary_payload(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=(
            [{"start": 0.0, "end": 1.0, "text": "hello team, we ship on Friday."}],
            {"language": "en", "language_probability": 0.9},
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.7}],
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
    respx.post("http://llm:8000/v1/chat/completions").mock(
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

    monkeypatch.setattr(pipeline, "_audio_duration_from_wave", lambda _path: None)
    monkeypatch.setattr(pipeline, "_audio_duration_from_ffprobe", lambda _path: None)
    monkeypatch.setattr(pipeline, "_speech_ratio_from_wave", lambda _path: None)
    monkeypatch.setattr(pipeline, "_speech_ratio_from_ffmpeg", lambda _path: None)

    result = pipeline.run_precheck(audio, cfg)
    assert result.quarantine_reason == "precheck_metrics_unavailable"
    assert result.duration_sec is None
    assert result.speech_ratio is None


def test_speech_ratio_from_ffmpeg_uses_unbounded_wait(tmp_path: Path, monkeypatch):
    audio = tmp_path / "ffmpeg-input.mp3"
    audio.write_bytes(b"audio")

    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

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
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *_args, **_kwargs: proc)

    ratio = pipeline._speech_ratio_from_ffmpeg(audio)
    assert ratio == 0.0
    assert proc.wait_timeout is None
