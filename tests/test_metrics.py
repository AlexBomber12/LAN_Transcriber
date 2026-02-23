import asyncio
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import respx
import pytest
import sys

from lan_transcriber import pipeline, llm_client, metrics

# stub heavy deps
whisperx = ModuleType("whisperx")
whisperx.utils = SimpleNamespace(get_segments=lambda *_a, **_k: "hello")
whisperx.transcribe = lambda *a, **k: ([{"start":0.0,"end":1.0,"text":"hi"}], {})
sys.modules["whisperx"] = whisperx

transformers = ModuleType("transformers")
transformers.pipeline = lambda *a, **k: lambda text: [{"label": "positive", "score": 0.5}]
sys.modules["transformers"] = transformers

class DummyDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label=False):
                from types import SimpleNamespace
                yield SimpleNamespace(start=0.0, end=1.0), "S1"
        return Ann()

def mp3(tmp: Path) -> Path:
    p = tmp / "a.mp3"
    p.write_bytes(b"\x00")
    return p


@pytest.mark.asyncio
@respx.mock
async def test_metrics_file(tmp_path: Path, monkeypatch):
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "db.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
    )
    await pipeline.run_pipeline(mp3(tmp_path), cfg, llm_client.LLMClient(), DummyDiariser())

    path = tmp_path / "metrics.snap"

    async def fast(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(metrics.asyncio, "sleep", fast)
    with pytest.raises(asyncio.CancelledError):
        await metrics.write_metrics_snapshot(path)

    data = path.read_text()
    assert "p95_latency_seconds" in data
    assert "error_rate_total" in data
