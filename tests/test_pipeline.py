from pathlib import Path
from types import ModuleType, SimpleNamespace

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

from lan_transcriber import pipeline, llm_client  # noqa: E402


FIX = Path(__file__).with_suffix("").parent / "fixtures"


def mp3(name: str) -> Path:
    """Return a temporary audio path for ``name``."""
    p = FIX / name
    p.write_bytes(b"\x00")
    return p


class DummyDiariser:
    async def __call__(self, audio_path: Path):
        class Ann:
            def itertracks(self, yield_label=False):
                from types import SimpleNamespace

                yield SimpleNamespace(start=0.0, end=1.0), "S1"

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
            {},
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

    cfg = pipeline.Settings(speaker_db=tmp_path / "db.yaml", tmp_root=tmp_path)
    res = await pipeline.run_pipeline(
        mp3("3_tripled.mp3"), cfg, llm_client.LLMClient(), DummyDiariser()
    )

    assert res.body.strip() == "hello world."
    assert res.summary.strip() == "- ok"


@pytest.mark.asyncio
@respx.mock
async def test_alias_persist(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=([{"start": 0.0, "end": 1.0, "text": "hi there friend"}], {}),
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
    db.write_text("S1: Alice\n")
    cfg = pipeline.Settings(speaker_db=db, tmp_root=tmp_path)
    res = await pipeline.run_pipeline(
        mp3("1_EN.mp3"), cfg, llm_client.LLMClient(), DummyDiariser()
    )
    assert res.speakers == ["Alice"]
    import yaml

    saved = yaml.safe_load(db.read_text())
    assert "S1" in saved


@pytest.mark.asyncio
@respx.mock
async def test_white_noise(tmp_path: Path, mocker):
    mocker.patch("whisperx.transcribe", return_value=([], {}))

    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": ""}}]}
        ),
    )
    mocker.patch(
        "transformers.pipeline",
        lambda *a, **k: lambda text: [{"label": "positive", "score": 0.5}],
    )

    cfg = pipeline.Settings(speaker_db=tmp_path / "db.yaml", tmp_root=tmp_path)
    res = await pipeline.run_pipeline(
        mp3("4_white_noise.mp3"), cfg, llm_client.LLMClient(), DummyDiariser()
    )

    assert res.summary == "No speech detected"
    assert res.body == ""


@pytest.mark.asyncio
@respx.mock
async def test_no_talk(tmp_path: Path, mocker):
    mocker.patch(
        "whisperx.transcribe",
        return_value=([{"start": 0.0, "end": 1.0, "text": "long silence indeed"}], {}),
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

    cfg = pipeline.Settings(speaker_db=tmp_path / "db.yaml", tmp_root=tmp_path)
    res = await pipeline.run_pipeline(
        mp3("5_no_talk.mp3"), cfg, llm_client.LLMClient(), DummyDiariser()
    )

    assert res.friendly == 0
    assert res.summary.strip() == ""
