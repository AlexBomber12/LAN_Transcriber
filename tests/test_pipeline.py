import json
from pathlib import Path
from types import SimpleNamespace, ModuleType

import httpx
import pytest
import respx

# stub heavy deps before importing pipeline
import sys

whisperx = ModuleType("whisperx")
whisperx.utils = SimpleNamespace(get_segments=lambda *_a, **_k: "hello")
sys.modules["whisperx"] = whisperx

transformers = ModuleType("transformers")
transformers.pipeline = lambda *a, **k: lambda text: [
    {"label": "positive", "score": 0.9}
]
sys.modules["transformers"] = transformers

from lan_transcriber import pipeline, llm_client  # noqa: E402


class DummyDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label=False):
                from types import SimpleNamespace

                yield SimpleNamespace(start=0.0, end=1.0), "S1"

        return Ann()


@pytest.mark.asyncio
@respx.mock
async def test_run_pipeline(tmp_path: Path, monkeypatch):
    whisperx.transcribe = lambda *a, **k: (
        [
            {"start": 0.0, "end": 1.0, "text": "hello world."},
            {"start": 1.0, "end": 2.0, "text": "hello world."},
            {"start": 2.0, "end": 3.0, "text": "hello world."},
        ],
        {},
    )

    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "- bullet"}}]}
        )
    )

    monkeypatch.setattr(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.8}],
    )

    cfg = pipeline.Settings(speaker_db=tmp_path / "db.json", tmp_root=tmp_path)
    res = await pipeline.run_pipeline(
        tmp_path / "f.wav", cfg, llm_client.LLMClient(), DummyDiariser()
    )

    assert res.summary == "- bullet"
    assert "bullet" in res.summary
    assert res.body.strip() == "hello world."
    assert 0 <= res.friendly <= 100
    assert len(res.speakers) == 1


@pytest.mark.asyncio
@respx.mock
async def test_alias_persist(tmp_path: Path, monkeypatch):
    whisperx.transcribe = lambda *a, **k: (
        [{"start": 0.0, "end": 1.0, "text": "hi"}],
        {},
    )

    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "- sum"}}]}
        )
    )
    monkeypatch.setattr(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.5}],
    )

    db = tmp_path / "db.json"
    db.write_text(json.dumps({"S1": "Alice"}))
    cfg = pipeline.Settings(speaker_db=db, tmp_root=tmp_path)
    res = await pipeline.run_pipeline(
        tmp_path / "f.wav", cfg, llm_client.LLMClient(), DummyDiariser()
    )
    assert res.speakers == ["Alice"]
    saved = json.loads(db.read_text())
    assert "S1" in saved


@pytest.mark.asyncio
@respx.mock
async def test_empty_asr(tmp_path: Path, monkeypatch):
    whisperx.transcribe = lambda *a, **k: ([], {})
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "- bullet"}}]}
        )
    )
    monkeypatch.setattr(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.5}],
    )
    cfg = pipeline.Settings(speaker_db=tmp_path / "db.json", tmp_root=tmp_path)
    res = await pipeline.run_pipeline(
        tmp_path / "noise.wav", cfg, llm_client.LLMClient(), DummyDiariser()
    )
    assert res.summary == "No speech detected"
    assert res.body == ""
