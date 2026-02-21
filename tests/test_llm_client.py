import asyncio
import httpx
import respx
import pytest
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lan_transcriber import llm_client


def test_retry_predicate_includes_timeout_exceptions():
    assert llm_client._is_retryable_exception(httpx.ReadTimeout("read timeout"))
    assert llm_client._is_retryable_exception(TimeoutError())


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
    assert result["content"] == "the-result"


@pytest.mark.asyncio
async def test_timeout(monkeypatch):
    async def slow_post(*_a, **_k):
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx.AsyncClient, "post", slow_post)
    client = llm_client.LLMClient(timeout=0.05)
    res = await client.generate("s", "u")
    assert res["content"] == "**LLM timeout**"
    assert llm_client.llm_timeouts_total._value.get() >= 1


@pytest.mark.asyncio
async def test_mock_response_path(tmp_path):
    mock_path = tmp_path / "mock-response.json"
    mock_path.write_text(
        '{"topic":"Weekly sync","summary_bullets":["Launched v2"],"decisions":[],"action_items":[],"emotional_summary":"Positive","questions":{"total_count":0,"types":{"open":0,"yes_no":0,"clarification":0,"status":0,"decision_seeking":0},"extracted":[]}}',
        encoding="utf-8",
    )
    client = llm_client.LLMClient(mock_response_path=mock_path)
    res = await client.generate("system", "user")
    assert "Weekly sync" in res["content"]


@pytest.mark.asyncio
async def test_mock_response_path_with_content_field(tmp_path):
    mock_path = tmp_path / "mock-content.json"
    mock_path.write_text(
        '{"role":"assistant","content":"from-mock"}',
        encoding="utf-8",
    )
    client = llm_client.LLMClient(mock_response_path=mock_path)
    res = await client.generate("system", "user")
    assert res["content"] == "from-mock"


@respx.mock
def test_generate_with_content_only_response():
    route = respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"content": "fallback-content"})
    )
    result = asyncio.run(llm_client.generate("s", "u", model="m"))
    assert route.called
    assert result["content"] == "fallback-content"
