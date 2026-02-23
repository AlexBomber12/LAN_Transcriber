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
_logger = logging.getLogger(__name__)


def _timeout_seconds(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        timeout = float(value)
    except ValueError:
        return default
    return timeout if timeout > 0 else default


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


class LLMClient:
    """Simple asynchronous client for the language model API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        mock_response_path: str | Path | None = None,
    ) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.default_model = os.getenv("LLM_MODEL")
        self.timeout = (
            timeout
            if timeout is not None
            else _timeout_seconds(os.getenv("LLM_TIMEOUT_SECONDS"), default=30.0)
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

        model = model or self.default_model

        url = f"{self.base_url}/v1/chat/completions"
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {"messages": messages}
        if model is not None:
            payload["model"] = model
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            data = await self._post_chat_completion(url=url, payload=payload, headers=headers)
        except (TimeoutError, httpx.TimeoutException):
            llm_timeouts_total.inc()
            return {"content": "**LLM timeout**", "role": "assistant"}

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return {
                        "role": str(message.get("role") or "assistant"),
                        "content": str(message.get("content") or ""),
                    }

        if "content" in data:
            return {"role": "assistant", "content": str(data.get("content") or "")}
        raise ValueError("LLM response missing choices[0].message.content")


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


__all__ = ["LLMClient", "generate", "llm_timeouts_total"]
