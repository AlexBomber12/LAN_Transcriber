import asyncio
import json
import pathlib
import sys
from typing import Any

import httpx
import pytest
import respx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lan_transcriber import llm_client


def test_retry_predicate_includes_timeout_exceptions() -> None:
    assert llm_client._is_retryable_exception(httpx.ReadTimeout("read timeout"))
    assert llm_client._is_retryable_exception(TimeoutError())


def test_base_url_host_returns_input_when_url_parser_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_url_error(_value: str):
        raise ValueError("bad-url")

    monkeypatch.setattr(llm_client.httpx, "URL", _raise_url_error)
    assert llm_client._base_url_host("http://invalid") == "http://invalid"


@pytest.mark.asyncio
@respx.mock
async def test_generate_payload_includes_max_tokens() -> None:
    route = respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "the-result"},
                    }
                ]
            },
        )
    )
    client = llm_client.LLMClient(
        base_url="http://127.0.0.1:8000",
        max_tokens=1536,
        max_tokens_retry=3072,
    )

    result = await client.generate("s", "u", model="m")
    assert route.called
    assert result["content"] == "the-result"
    payload = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert payload["max_tokens"] == 1536


@pytest.mark.asyncio
async def test_generate_retries_once_on_finish_reason_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    client = llm_client.LLMClient(
        base_url="http://example.test",
        max_tokens=600,
        max_tokens_retry=1200,
    )

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        del url, headers
        attempts.append(payload["max_tokens"])
        if len(attempts) == 1:
            return {
                "id": "req-1",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "partial"},
                    }
                ],
            }
        return {
            "id": "req-2",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "final"},
                }
            ],
        }

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    result = await client.generate("sys", "usr", model="m")
    assert result["content"] == "final"
    assert attempts == [600, 1200]


@pytest.mark.asyncio
async def test_generate_retries_once_on_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    client = llm_client.LLMClient(
        base_url="http://example.test",
        max_tokens=700,
        max_tokens_retry=1400,
    )

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        del url, headers
        attempts.append(payload["max_tokens"])
        if len(attempts) == 1:
            return {
                "id": "req-empty-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "   "},
                    }
                ],
            }
        return {
            "id": "req-empty-2",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "usable"},
                }
            ],
        }

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    result = await client.generate("sys", "usr", model="m")
    assert result["content"] == "usable"
    assert attempts == [700, 1400]


@pytest.mark.asyncio
async def test_generate_raises_truncated_error_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    client = llm_client.LLMClient(
        base_url="http://example.test",
        max_tokens=800,
        max_tokens_retry=1600,
    )

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        del url, headers
        attempts.append(payload["max_tokens"])
        return {
            "id": f"req-length-{len(attempts)}",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "partial"},
                }
            ],
        }

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    with pytest.raises(llm_client.LLMTruncatedResponseError, match="Increase LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS"):
        await client.generate("sys", "usr", model="m")
    assert attempts == [800, 1600]


@pytest.mark.asyncio
async def test_generate_raises_empty_content_error_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = llm_client.LLMClient(
        base_url="http://example.test",
        max_tokens=900,
        max_tokens_retry=1800,
    )
    responses = [
        {
            "id": "req-empty-first",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": ""},
                }
            ],
        },
        {
            "id": "req-empty-second",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": " \n\t"},
                }
            ],
        },
    ]

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        del url, payload, headers
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    with pytest.raises(llm_client.LLMEmptyContentError, match="reasoning-only output") as exc_info:
        await client.generate("sys", "usr", model="m")
    err = exc_info.value
    assert err.finish_reason == "stop"
    assert err.request_id == "req-empty-second"
    assert err.raw_response["id"] == "req-empty-second"
    assert "host=example.test" in str(err)


@pytest.mark.asyncio
async def test_generate_returns_timeout_when_retry_attempt_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = llm_client.LLMClient(
        base_url="http://example.test",
        max_tokens=900,
        max_tokens_retry=1800,
    )
    calls = 0

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        nonlocal calls
        del url, payload, headers
        calls += 1
        if calls == 1:
            return {
                "id": "req-retry-timeout",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "partial"},
                    }
                ],
            }
        raise TimeoutError("retry-timeout")

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    result = await client.generate("sys", "usr", model="m")
    assert result["content"] == "**LLM timeout**"
    assert calls == 2


@pytest.mark.asyncio
async def test_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_post(*_a: Any, **_k: Any) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx.AsyncClient, "post", slow_post)
    client = llm_client.LLMClient(timeout=0.05)
    res = await client.generate("s", "u")
    assert res["content"] == "**LLM timeout**"
    assert llm_client.llm_timeouts_total._value.get() >= 1


@pytest.mark.asyncio
async def test_mock_response_path(tmp_path: pathlib.Path) -> None:
    mock_path = tmp_path / "mock-response.json"
    mock_path.write_text(
        '{"topic":"Weekly sync","summary_bullets":["Launched v2"],"decisions":[],"action_items":[],"emotional_summary":"Positive","questions":{"total_count":0,"types":{"open":0,"yes_no":0,"clarification":0,"status":0,"decision_seeking":0},"extracted":[]}}',
        encoding="utf-8",
    )
    client = llm_client.LLMClient(mock_response_path=mock_path)
    res = await client.generate("system", "user")
    assert "Weekly sync" in res["content"]


@pytest.mark.asyncio
async def test_mock_response_path_with_content_field(tmp_path: pathlib.Path) -> None:
    mock_path = tmp_path / "mock-content.json"
    mock_path.write_text(
        '{"role":"assistant","content":"from-mock"}',
        encoding="utf-8",
    )
    client = llm_client.LLMClient(mock_response_path=mock_path)
    res = await client.generate("system", "user")
    assert res["content"] == "from-mock"


@respx.mock
def test_generate_with_content_only_response() -> None:
    route = respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"content": "fallback-content"})
    )
    result = asyncio.run(llm_client.generate("s", "u", model="m"))
    assert route.called
    assert result["content"] == "fallback-content"
