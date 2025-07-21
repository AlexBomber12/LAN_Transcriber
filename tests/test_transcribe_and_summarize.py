from pathlib import Path
import asyncio

import httpx
import respx
import sys
import pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import types

sys.modules.setdefault("whisperx", types.ModuleType("whisperx"))
sys.modules["whisperx"].utils = types.SimpleNamespace(get_segments=lambda *_a, **_k: "")
os.environ.setdefault("CI", "true")

import web_transcribe  # noqa: E402


@respx.mock
def test_transcribe_and_summarize(tmp_path: Path):
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "summary"}}]},
        )
    )
    sum_path, full_path = asyncio.run(web_transcribe.transcribe_and_summarize("hello"))
    assert sum_path.exists()
    assert full_path.exists()
    assert sum_path.read_text() == "summary"
    assert full_path.read_text() == "hello"
