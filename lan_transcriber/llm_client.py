"""HTTP client for talking to an external LLM service."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .metrics import llm_timeouts_total

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429}
_DEV_DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000"
_MIN_LLM_MAX_TOKENS = 256
_DEFAULT_LLM_MAX_TOKENS = 1024
_MAX_LLM_MAX_TOKENS = 4096
_logger = logging.getLogger(__name__)


def _timeout_seconds(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        timeout = float(value)
    except ValueError:
        return default
    return timeout if timeout > 0 else default


def _int_setting(
    value: int | str | None,
    *,
    default: int,
    minimum: int,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _resolve_retry_max_tokens(
    value: int | str | None,
    *,
    base_max_tokens: int,
) -> int:
    default_retry = min(base_max_tokens * 2, _MAX_LLM_MAX_TOKENS)
    parsed = _int_setting(
        value,
        default=default_retry,
        minimum=_MIN_LLM_MAX_TOKENS,
    )
    if parsed <= base_max_tokens and base_max_tokens < _MAX_LLM_MAX_TOKENS:
        return min(_MAX_LLM_MAX_TOKENS, max(default_retry, base_max_tokens + 1))
    return parsed


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.TimeoutException,
            TimeoutError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in _RETRYABLE_STATUS_CODES or status >= 500
    return False


def _resolve_base_url(base_url: str | None) -> str:
    configured = (base_url or os.getenv("LLM_BASE_URL") or "").strip()
    if configured:
        return configured

    lan_env = (os.getenv("LAN_ENV") or "dev").strip().lower() or "dev"
    if lan_env in {"staging", "prod"}:
        raise ValueError(
            f"Missing required environment variable for LAN_ENV={lan_env}: LLM_BASE_URL"
        )

    _logger.warning(
        "LLM_BASE_URL is not set in LAN_ENV=%s; defaulting to %s",
        lan_env,
        _DEV_DEFAULT_LLM_BASE_URL,
    )
    return _DEV_DEFAULT_LLM_BASE_URL


def _base_url_host(base_url: str) -> str:
    try:
        parsed = httpx.URL(base_url)
    except Exception:
        return base_url
    host = parsed.host
    return str(host) if host else base_url


class LLMEmptyContentError(ValueError):
    """Raised when the LLM reply has no usable assistant content."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        max_tokens: int,
        finish_reason: str | None,
        request_id: str | None,
        raw_response: Dict[str, Any],
    ) -> None:
        self.host = host
        self.model = model
        self.max_tokens = max_tokens
        self.finish_reason = finish_reason
        self.request_id = request_id
        self.raw_response = raw_response
        reason = finish_reason or "unknown"
        request = request_id or "unknown"
        super().__init__(
            "LLM returned empty message.content "
            f"(host={host}, model={model}, max_tokens={max_tokens}, "
            f"finish_reason={reason}, request_id={request}). "
            "Ollama may have returned reasoning-only output; increase "
            "LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS."
        )


class LLMTruncatedResponseError(ValueError):
    """Raised when the LLM reply keeps truncating after the explicit retry."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        max_tokens: int,
        request_id: str | None,
        raw_response: Dict[str, Any],
    ) -> None:
        self.host = host
        self.model = model
        self.max_tokens = max_tokens
        self.request_id = request_id
        self.raw_response = raw_response
        request = request_id or "unknown"
        super().__init__(
            "LLM response is truncated with finish_reason=length "
            f"(host={host}, model={model}, max_tokens={max_tokens}, request_id={request}). "
            "Increase LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS."
        )


class LLMClient:
    """Simple asynchronous client for the language model API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        mock_response_path: str | Path | None = None,
        max_tokens: int | None = None,
        max_tokens_retry: int | None = None,
    ) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.base_url_host = _base_url_host(self.base_url)
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.default_model = os.getenv("LLM_MODEL")
        self.timeout = (
            timeout
            if timeout is not None
            else _timeout_seconds(os.getenv("LLM_TIMEOUT_SECONDS"), default=30.0)
        )
        self.max_tokens = _int_setting(
            max_tokens if max_tokens is not None else os.getenv("LLM_MAX_TOKENS"),
            default=_DEFAULT_LLM_MAX_TOKENS,
            minimum=_MIN_LLM_MAX_TOKENS,
        )
        self.max_tokens_retry = _resolve_retry_max_tokens(
            (
                max_tokens_retry
                if max_tokens_retry is not None
                else os.getenv("LLM_MAX_TOKENS_RETRY")
            ),
            base_max_tokens=self.max_tokens,
        )
        configured_mock_path = mock_response_path or os.getenv("LLM_MOCK_RESPONSE_PATH")
        self.mock_response_path = Path(configured_mock_path) if configured_mock_path else None

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def _post_chat_completion(
        self,
        *,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with anyio.fail_after(self.timeout):
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("LLM response must be a JSON object")
            return data

    def _load_mock_message(self) -> Dict[str, str] | None:
        if self.mock_response_path is None:
            return None
        text = self.mock_response_path.read_text(encoding="utf-8").strip()
        if not text:
            return {"role": "assistant", "content": ""}
        try:
            payload = json.loads(text)
        except ValueError:
            return {"role": "assistant", "content": text}

        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict):
                        return {
                            "role": str(message.get("role") or "assistant"),
                            "content": str(message.get("content") or ""),
                        }
            if "content" in payload:
                return {
                    "role": str(payload.get("role") or "assistant"),
                    "content": str(payload.get("content") or ""),
                }
        if isinstance(payload, str):
            return {"role": "assistant", "content": payload}
        return {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}

    @staticmethod
    def _request_id(data: Dict[str, Any]) -> str | None:
        request_id = data.get("id") or data.get("request_id") or data.get("requestId")
        return str(request_id) if request_id is not None else None

    def _extract_message(
        self,
        data: Dict[str, Any],
    ) -> tuple[str, str, str | None, str | None]:
        request_id = self._request_id(data)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                finish_reason = first.get("finish_reason")
                finish_reason_text = (
                    str(finish_reason) if finish_reason is not None else None
                )
                message = first.get("message")
                if isinstance(message, dict):
                    return (
                        str(message.get("role") or "assistant"),
                        str(message.get("content") or ""),
                        finish_reason_text,
                        request_id,
                    )
        if "content" in data:
            return ("assistant", str(data.get("content") or ""), None, request_id)
        raise ValueError("LLM response missing choices[0].message.content")

    @staticmethod
    def _should_retry_response(*, finish_reason: str | None, content: str) -> bool:
        return finish_reason == "length" or not content.strip()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        """Send a chat completion request and return the assistant message."""

        mock_message = self._load_mock_message()
        if mock_message is not None:
            return mock_message

        model_name = model or self.default_model
        model_label = model_name or "<unset>"

        url = f"{self.base_url}/v1/chat/completions"
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {"messages": messages}
        if model_name is not None:
            payload["model"] = model_name
        if response_format is not None:
            payload["response_format"] = response_format

        async def _run_attempt(
            *,
            attempt_number: int,
            max_tokens: int,
        ) -> tuple[Dict[str, Any], str, str, str | None, str | None] | None:
            payload["max_tokens"] = max_tokens
            _logger.debug(
                "LLM request attempt=%s model=%s max_tokens=%s timeout_seconds=%s",
                attempt_number,
                model_label,
                max_tokens,
                self.timeout,
            )
            try:
                data = await self._post_chat_completion(
                    url=url,
                    payload=payload,
                    headers=headers,
                )
            except (TimeoutError, httpx.TimeoutException):
                llm_timeouts_total.inc()
                return None
            role, content, finish_reason, request_id = self._extract_message(data)
            return data, role, content, finish_reason, request_id

        first_attempt = await _run_attempt(
            attempt_number=1,
            max_tokens=self.max_tokens,
        )
        if first_attempt is None:
            return {"content": "**LLM timeout**", "role": "assistant"}
        data, role, content, finish_reason, request_id = first_attempt
        max_tokens_used = self.max_tokens

        if self._should_retry_response(finish_reason=finish_reason, content=content):
            _logger.info(
                "Retrying LLM request after attempt=%s (model=%s max_tokens=%s finish_reason=%s empty_content=%s)",
                1,
                model_label,
                self.max_tokens,
                finish_reason or "unknown",
                not content.strip(),
            )
            second_attempt = await _run_attempt(
                attempt_number=2,
                max_tokens=self.max_tokens_retry,
            )
            if second_attempt is None:
                return {"content": "**LLM timeout**", "role": "assistant"}
            data, role, content, finish_reason, request_id = second_attempt
            max_tokens_used = self.max_tokens_retry

        if finish_reason == "length":
            _logger.debug(
                "LLM raw response with finish_reason=length: %s",
                data,
            )
            raise LLMTruncatedResponseError(
                host=self.base_url_host,
                model=model_label,
                max_tokens=max_tokens_used,
                request_id=request_id,
                raw_response=data,
            )
        if not content.strip():
            _logger.debug("LLM raw response with empty message.content: %s", data)
            raise LLMEmptyContentError(
                host=self.base_url_host,
                model=model_label,
                max_tokens=max_tokens_used,
                finish_reason=finish_reason,
                request_id=request_id,
                raw_response=data,
            )
        return {"role": role, "content": content}


_default_client = LLMClient()


async def generate(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    response_format: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """Backwards-compatible helper that proxies to :class:`LLMClient`."""

    return await _default_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        response_format=response_format,
    )


__all__ = [
    "LLMClient",
    "LLMEmptyContentError",
    "LLMTruncatedResponseError",
    "generate",
    "llm_timeouts_total",
]
