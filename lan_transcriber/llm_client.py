"""HTTP client for talking to an external LLM service."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, wait_exponential, stop_after_attempt


class LLMClient:
    """Simple asynchronous client for the language model API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://llm:8000")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.timeout = timeout

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3)
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant message."""

        model = model or os.getenv("LLM_MODEL")

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

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


_default_client = LLMClient()


async def generate(
    system_prompt: str, user_prompt: str, model: Optional[str] = None
) -> str:
    """Backwards-compatible helper that proxies to :class:`LLMClient`."""

    return await _default_client.generate(system_prompt, user_prompt, model)


__all__ = ["LLMClient", "generate"]
