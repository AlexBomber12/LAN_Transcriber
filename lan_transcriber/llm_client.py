import os
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, wait_exponential, stop_after_attempt

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8000")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_TIMEOUT = 30


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
async def generate(
    system_prompt: str, user_prompt: str, model: Optional[str] = os.getenv("LLM_MODEL")
) -> str:
    url = f"{LLM_BASE_URL}/v1/chat/completions"
    headers: Dict[str, str] = {}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload: Dict[str, Any] = {"messages": messages}
    if model is not None:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]
