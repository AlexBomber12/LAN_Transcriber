import asyncio
import httpx
import respx
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lan_transcriber import llm_client


@respx.mock
def test_generate():
    route = respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "the-result"}}]},
        )
    )
    result = asyncio.run(llm_client.generate("s", "u", model="m"))
    assert route.called
    assert result == "the-result"
