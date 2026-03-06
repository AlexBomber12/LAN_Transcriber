import asyncio
import json
import logging
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


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://host:11434", "http://host:11434/v1/chat/completions"),
        ("http://host:11434/", "http://host:11434/v1/chat/completions"),
        ("http://host:11434/v1", "http://host:11434/v1/chat/completions"),
        ("http://host:11434/v1/", "http://host:11434/v1/chat/completions"),
        (
            "http://host:11434/v1/chat/completions",
            "http://host:11434/v1/chat/completions",
        ),
    ],
)
def test_build_chat_completions_url(base_url: str, expected: str) -> None:
    assert llm_client.build_chat_completions_url(base_url) == expected


def test_error_body_snippet_uses_json_error_object() -> None:
    response = httpx.Response(404, json={"error": {"message": "not found"}})
    assert (
        llm_client._error_body_snippet(response)
        == '{"message": "not found"}'
    )


def test_error_body_snippet_uses_empty_placeholder_when_body_missing() -> None:
    response = httpx.Response(
        404,
        content="",
        headers={"content-type": "text/plain"},
    )
    assert llm_client._error_body_snippet(response) == "<empty>"


def test_error_body_snippet_keeps_raw_text_when_error_field_is_null() -> None:
    response = httpx.Response(404, json={"error": None})
    assert llm_client._error_body_snippet(response) == '{"error":null}'


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
@respx.mock
async def test_generate_omits_model_when_not_configured() -> None:
    route = respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "ok"},
                    }
                ]
            },
        )
    )
    client = llm_client.LLMClient(
        base_url="http://127.0.0.1:8000",
        max_tokens=512,
        max_tokens_retry=1024,
    )
    client.default_model = None

    result = await client.generate("s", "u", model=None)
    assert result["content"] == "ok"
    payload = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert payload["max_tokens"] == 512
    assert "model" not in payload


def test_retry_max_tokens_defaults_to_scaled_base_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_MAX_TOKENS_RETRY", raising=False)
    client = llm_client.LLMClient(
        base_url="http://127.0.0.1:8000",
        max_tokens=4096,
        max_tokens_retry=None,
    )
    assert client.max_tokens_retry == 4096


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
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        del url, headers, attempt_number
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
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        del url, headers, attempt_number
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
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        del url, headers, attempt_number
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
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        del url, payload, headers, attempt_number
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
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        nonlocal calls
        del url, payload, headers, attempt_number
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


@pytest.mark.asyncio
async def test_generate_allows_per_call_max_token_override(monkeypatch: pytest.MonkeyPatch) -> None:
    client = llm_client.LLMClient(
        base_url="http://example.test",
        max_tokens=1024,
        max_tokens_retry=2048,
    )
    seen_max_tokens: list[int] = []

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        del url, headers
        seen_max_tokens.append(int(payload["max_tokens"]))
        if attempt_number == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": ""},
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)

    result = await client.generate("sys", "usr", max_tokens=1536)

    assert result["content"] == "ok"
    assert seen_max_tokens == [1536, 3072]


@respx.mock
def test_generate_with_content_only_response() -> None:
    route = respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"content": "fallback-content"})
    )
    result = asyncio.run(llm_client.generate("s", "u", model="m"))
    assert route.called
    assert result["content"] == "fallback-content"


@pytest.mark.asyncio
@respx.mock
async def test_generate_http_404_json_error_includes_metadata() -> None:
    route = respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(404, json={"error": "x" * 2105})
    )
    client = llm_client.LLMClient(
        base_url="http://127.0.0.1:8000/v1",
        max_tokens=512,
        max_tokens_retry=1024,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.generate("system-secret", "user-secret", model="m")
    assert route.called
    message = str(exc_info.value)
    assert "status_code=404" in message
    assert "llm_url=http://127.0.0.1:8000/v1/chat/completions" in message
    assert "model=m" in message
    assert "max_tokens=512" in message
    assert "...<truncated>" in message
    assert "system-secret" not in message
    assert "user-secret" not in message


@pytest.mark.asyncio
@respx.mock
async def test_generate_http_404_non_json_includes_raw_text() -> None:
    route = respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            404,
            content="plain-text-error",
            headers={"content-type": "text/plain"},
        )
    )
    client = llm_client.LLMClient(
        base_url="http://127.0.0.1:8000/",
        max_tokens=512,
        max_tokens_retry=1024,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.generate("s", "u", model="m")
    assert route.called
    message = str(exc_info.value)
    assert "status_code=404" in message
    assert "body=plain-text-error" in message


@pytest.mark.asyncio
@respx.mock
async def test_generate_debug_log_uses_safe_request_metadata(caplog: pytest.LogCaptureFixture) -> None:
    respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "ok"},
                    }
                ]
            },
        )
    )
    client = llm_client.LLMClient(
        base_url="http://127.0.0.1:8000",
        max_tokens=600,
        max_tokens_retry=1200,
        timeout=12.0,
    )

    with caplog.at_level(logging.DEBUG, logger=llm_client.__name__):
        await client.generate(
            "system-super-secret",
            "user-ultra-secret",
            model="safe-model",
            response_format={"type": "json_object"},
        )
    debug_messages = "\n".join(
        record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG
    )
    assert "llm_url=http://127.0.0.1:8000/v1/chat/completions" in debug_messages
    assert "model=safe-model" in debug_messages
    assert "max_tokens=600" in debug_messages
    assert "timeout_seconds=12.0" in debug_messages
    assert "attempt=1" in debug_messages
    assert "response_format=True" in debug_messages
    assert "system-super-secret" not in debug_messages
    assert "user-ultra-secret" not in debug_messages
