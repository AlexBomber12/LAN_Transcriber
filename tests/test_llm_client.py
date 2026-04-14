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


def test_post_chat_completion_reuses_shared_http_client_across_asyncio_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list["_FakeClient"] = []

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.is_closed = False
            self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
            created_clients.append(self)

        async def post(
            self,
            url: str,
            *,
            json: dict[str, Any],
            headers: dict[str, str],
        ) -> _FakeResponse:
            self.calls.append((url, json, headers))
            return _FakeResponse()

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _FakeClient)
    client = llm_client.LLMClient(base_url="http://example.test", timeout=0.1)

    async def _post(max_tokens: int, headers: dict[str, str]) -> dict[str, Any]:
        return await client._post_chat_completion(
            url="http://example.test/v1/chat/completions",
            payload={"messages": [], "max_tokens": max_tokens},
            headers=headers,
        )

    asyncio.run(llm_client.LLMClient.close())
    first = asyncio.run(_post(111, {"Authorization": "Bearer secret"}))
    second = asyncio.run(_post(222, {}))

    assert first["choices"][0]["message"]["content"] == "ok"
    assert second["choices"][0]["message"]["content"] == "ok"
    assert len(created_clients) == 1
    assert created_clients[0].calls[0][1]["max_tokens"] == 111
    assert created_clients[0].calls[1][1]["max_tokens"] == 222
    timeout = created_clients[0].kwargs["timeout"]
    assert timeout.connect == pytest.approx(0.1)
    assert timeout.read == pytest.approx(0.1)
    assert timeout.write == pytest.approx(0.1)
    assert timeout.pool == pytest.approx(0.1)

    asyncio.run(llm_client.LLMClient.close())
    assert created_clients[0].is_closed is True
    replacement = asyncio.run(client._get_client())
    assert len(created_clients) == 2
    assert replacement is created_clients[1]
    assert created_clients[1].kwargs["limits"] == httpx.Limits(
        max_connections=5,
        max_keepalive_connections=2,
    )
    asyncio.run(llm_client.LLMClient.close())


@pytest.mark.asyncio
async def test_close_tolerates_client_without_async_close() -> None:
    class _ClientWithoutClose:
        is_closed = False

    llm_client.LLMClient._http_clients = {("factory", 5.0): _ClientWithoutClose()}  # noqa: SLF001

    await llm_client.LLMClient.close()

    assert len(llm_client.LLMClient._http_clients) == 0  # noqa: SLF001


@pytest.mark.asyncio
async def test_close_skips_duplicate_and_already_closed_clients() -> None:
    close_calls: list[str] = []

    class _TrackedClient:
        def __init__(self, name: str, *, is_closed: bool) -> None:
            self.name = name
            self.is_closed = is_closed

        async def aclose(self) -> None:
            close_calls.append(self.name)
            self.is_closed = True

    shared = _TrackedClient("shared", is_closed=False)
    closed = _TrackedClient("closed", is_closed=True)
    llm_client.LLMClient._http_clients = {  # noqa: SLF001
        ("factory-a", 1.0): shared,
        ("factory-b", 1.0): shared,
        ("factory-c", 2.0): closed,
    }

    await llm_client.LLMClient.close()

    assert close_calls == ["shared"]
    assert len(llm_client.LLMClient._http_clients) == 0  # noqa: SLF001


def test_post_chat_completion_reuses_shared_http_client_across_event_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list["_FakeClient"] = []

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.is_closed = False
            created_clients.append(self)

        async def post(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
            return _FakeResponse()

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _FakeClient)
    client = llm_client.LLMClient(base_url="http://example.test", timeout=0.1)

    async def _post(max_tokens: int) -> None:
        await client._post_chat_completion(
            url="http://example.test/v1/chat/completions",
            payload={"messages": [], "max_tokens": max_tokens},
            headers={},
        )

    asyncio.run(llm_client.LLMClient.close())
    asyncio.run(_post(111))
    asyncio.run(_post(222))

    assert len(created_clients) == 1
    assert created_clients[0].is_closed is False
    asyncio.run(llm_client.LLMClient.close())


def test_running_loop_returns_none_without_active_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime_error() -> None:
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(llm_client.asyncio, "get_running_loop", _raise_runtime_error)

    assert llm_client.LLMClient._running_loop() is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_run_on_http_runtime_awaits_inline_when_already_on_http_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()

    async def _inline() -> str:
        return "inline"

    monkeypatch.setattr(
        llm_client.LLMClient,
        "_ensure_http_runtime",
        classmethod(lambda cls: loop),
    )
    monkeypatch.setattr(
        llm_client.LLMClient,
        "_running_loop",
        staticmethod(lambda: loop),
    )

    result = await llm_client.LLMClient._run_on_http_runtime(_inline())  # noqa: SLF001

    assert result == "inline"


@pytest.mark.asyncio
async def test_post_chat_completion_uses_separate_clients_per_timeout_without_closing_active_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list["_FakeClient"] = []

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.is_closed = False
            created_clients.append(self)

        async def post(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
            return _FakeResponse()

        async def aclose(self) -> None:
            self.is_closed = True

    await llm_client.LLMClient.close()
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _FakeClient)
    fast_client = llm_client.LLMClient(base_url="http://example.test", timeout=0.1)
    slow_client = llm_client.LLMClient(base_url="http://example.test", timeout=321.0)

    await fast_client._post_chat_completion(
        url="http://example.test/v1/chat/completions",
        payload={"messages": [], "max_tokens": 111},
        headers={},
    )
    await slow_client._post_chat_completion(
        url="http://example.test/v1/chat/completions",
        payload={"messages": [], "max_tokens": 222},
        headers={},
    )

    assert len(created_clients) == 2
    assert created_clients[0].is_closed is False
    assert created_clients[1].kwargs["timeout"].read == pytest.approx(321.0)
    await llm_client.LLMClient.close()
    assert created_clients[0].is_closed is True
    assert created_clients[1].is_closed is True


@pytest.mark.asyncio
async def test_close_skips_join_and_cleanup_reset_when_runtime_state_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_calls: list[str] = []

    class _Loop:
        def __init__(self, name: str) -> None:
            self.name = name
            self.stopped = False

        def is_closed(self) -> bool:
            return False

        def stop(self) -> None:
            self.stopped = True

        def call_soon_threadsafe(self, callback: Any) -> None:
            stop_calls.append(self.name)
            callback()

    class _Thread:
        def __init__(self, name: str) -> None:
            self.name = name

        def is_alive(self) -> bool:
            return True

    class _TrackedClient:
        def __init__(self) -> None:
            self.is_closed = False

        async def aclose(self) -> None:
            self.is_closed = True

    loop = _Loop("active")
    replacement_loop = _Loop("replacement")
    thread = _Thread("active")
    replacement_thread = _Thread("replacement")
    client = _TrackedClient()
    original_clients = llm_client.LLMClient._http_clients  # noqa: SLF001
    original_loop = llm_client.LLMClient._http_loop  # noqa: SLF001
    original_thread = llm_client.LLMClient._http_thread  # noqa: SLF001

    try:
        llm_client.LLMClient._http_clients = {("factory", 1.0): client}  # noqa: SLF001
        llm_client.LLMClient._http_loop = loop  # noqa: SLF001
        llm_client.LLMClient._http_thread = thread  # noqa: SLF001

        async def _fake_run_on_http_runtime(
            cls: type[llm_client.LLMClient],
            coroutine: Any,
        ) -> None:
            await coroutine
            cls._http_loop = replacement_loop  # noqa: SLF001
            cls._http_thread = replacement_thread  # noqa: SLF001

        monkeypatch.setattr(
            llm_client.LLMClient,
            "_run_on_http_runtime",
            classmethod(_fake_run_on_http_runtime),
        )
        monkeypatch.setattr(llm_client.threading, "current_thread", lambda: thread)

        await llm_client.LLMClient.close()

        assert client.is_closed is True
        assert stop_calls == ["active"]
        assert loop.stopped is True
        assert llm_client.LLMClient._http_loop is replacement_loop  # noqa: SLF001
        assert llm_client.LLMClient._http_thread is replacement_thread  # noqa: SLF001
    finally:
        llm_client.LLMClient._http_clients = original_clients  # noqa: SLF001
        llm_client.LLMClient._http_loop = original_loop  # noqa: SLF001
        llm_client.LLMClient._http_thread = original_thread  # noqa: SLF001


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
async def test_first_attempt_timeout_returns_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the first_attempt is None early return (line 438)."""
    client = llm_client.LLMClient(base_url="http://localhost:1")

    async def _fake_post(**_k: Any) -> dict[str, Any]:
        raise TimeoutError("first-attempt-timeout")

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    res = await client.generate("sys", "usr", model="m")
    assert res["content"] == "**LLM timeout**"
    assert res["role"] == "assistant"


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
async def test_generate_preserves_configured_retry_budget_when_only_max_tokens_is_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert seen_max_tokens == [1536, 2048]


@pytest.mark.asyncio
async def test_generate_allows_per_call_retry_max_token_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = await client.generate("sys", "usr", max_tokens=1536, max_tokens_retry=3072)

    assert result["content"] == "ok"
    assert seen_max_tokens == [1536, 3072]


@pytest.mark.asyncio
async def test_generate_retry_budget_never_drops_below_overridden_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = await client.generate("sys", "usr", max_tokens=3072)

    assert result["content"] == "ok"
    assert seen_max_tokens == [3072, 3072]


@pytest.mark.asyncio
async def test_generate_respects_explicit_retry_budget_equal_to_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = await client.generate("sys", "usr", max_tokens=3072, max_tokens_retry=3072)

    assert result["content"] == "ok"
    assert seen_max_tokens == [3072, 3072]


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


def test_worker_main_closes_shared_http_client_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import types

    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=type("Redis", (), {})))
    monkeypatch.setitem(sys.modules, "rq", types.SimpleNamespace(Worker=type("Worker", (), {})))
    worker_module = importlib.import_module("lan_app.worker")

    calls: dict[str, object] = {}
    settings = type(
        "Settings",
        (),
        {
            "redis_url": "redis://unit",
            "rq_queue_name": "audio",
            "rq_worker_burst": False,
            "data_root": pathlib.Path("/tmp/worker-llm-close"),
        },
    )()

    monkeypatch.setattr(worker_module, "AppSettings", lambda: settings)
    monkeypatch.setattr(worker_module, "init_db", lambda cfg: calls.setdefault("init_db", cfg))
    monkeypatch.setattr(worker_module, "write_worker_status", lambda *_args: None)
    monkeypatch.setattr(worker_module, "start_heartbeat_thread", lambda *_args: (None, None))
    monkeypatch.setattr(worker_module.Redis, "from_url", lambda _url: object(), raising=False)
    monkeypatch.setattr(worker_module, "_install_signal_handlers", lambda _worker: None)

    class _FakeWorker:
        def __init__(self, queues, *, connection):
            calls["queues"] = queues
            calls["connection"] = connection

        def work(self, *, with_scheduler, burst):
            calls["work"] = (with_scheduler, burst)
            raise RuntimeError("boom")

    async def _fake_close(_cls) -> None:
        calls["close_count"] = int(calls.get("close_count", 0)) + 1

    original_asyncio_run = asyncio.run

    def _run_coroutine(coro):
        calls["asyncio_run_called"] = True
        return original_asyncio_run(coro)

    monkeypatch.setattr(worker_module, "Worker", _FakeWorker)
    monkeypatch.setattr(worker_module.asyncio, "run", _run_coroutine)
    monkeypatch.setattr(worker_module.LLMClient, "close", classmethod(_fake_close))

    with pytest.raises(RuntimeError, match="boom"):
        worker_module.main()

    assert calls["init_db"] is settings
    assert calls["queues"] == ["audio"]
    assert calls["work"] == (False, False)
    assert calls["asyncio_run_called"] is True
    assert calls["close_count"] == 1


def test_worker_main_logs_close_failure_without_masking_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import types

    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=type("Redis", (), {})))
    monkeypatch.setitem(sys.modules, "rq", types.SimpleNamespace(Worker=type("Worker", (), {})))
    worker_module = importlib.import_module("lan_app.worker")

    calls: dict[str, object] = {}
    settings = type(
        "Settings",
        (),
        {
            "redis_url": "redis://unit",
            "rq_queue_name": "audio",
            "rq_worker_burst": False,
            "data_root": pathlib.Path("/tmp/worker-llm-close-warning"),
        },
    )()

    monkeypatch.setattr(worker_module, "AppSettings", lambda: settings)
    monkeypatch.setattr(worker_module, "init_db", lambda *_args: None)
    monkeypatch.setattr(worker_module, "write_worker_status", lambda *_args: None)
    monkeypatch.setattr(worker_module, "start_heartbeat_thread", lambda *_args: (None, None))
    monkeypatch.setattr(worker_module.Redis, "from_url", lambda _url: object(), raising=False)
    monkeypatch.setattr(worker_module, "_install_signal_handlers", lambda _worker: None)
    monkeypatch.setattr(
        worker_module._logger,
        "warning",
        lambda message, **kwargs: calls.update({"warning": message, "warning_kwargs": kwargs}),
    )

    class _FakeWorker:
        def __init__(self, queues, *, connection):
            calls["queues"] = queues
            calls["connection"] = connection

        def work(self, *, with_scheduler, burst):
            calls["work"] = (with_scheduler, burst)
            raise RuntimeError("worker-boom")

    def _run_coroutine(coro):
        coro.close()
        raise RuntimeError("close-boom")

    monkeypatch.setattr(worker_module, "Worker", _FakeWorker)
    monkeypatch.setattr(worker_module.asyncio, "run", _run_coroutine)

    with pytest.raises(RuntimeError, match="worker-boom"):
        worker_module.main()

    assert calls["warning"] == "Failed to close shared LLM HTTP client"
    assert calls["warning_kwargs"] == {"exc_info": True}
