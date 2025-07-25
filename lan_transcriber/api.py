from __future__ import annotations

import asyncio
from typing import List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .aliases import load_aliases, save_aliases, ALIAS_PATH
from .pipeline import refresh_aliases
from .models import TranscriptResult

app = FastAPI()
_subscribers: List[asyncio.Queue[str]] = []
_current_result: TranscriptResult | None = None


class AliasUpdate(BaseModel):
    alias: str


@app.post("/alias/{speaker_id}")
async def update_alias(speaker_id: str, upd: AliasUpdate):
    aliases = load_aliases(ALIAS_PATH)
    aliases[speaker_id] = upd.alias
    save_aliases(aliases, ALIAS_PATH)
    if _current_result is not None:
        refresh_aliases(_current_result, ALIAS_PATH)
    for q in list(_subscribers):
        q.put_nowait("updated")
    return {"speaker": speaker_id, "alias": upd.alias}


@app.get("/events")
async def events():
    q: asyncio.Queue[str] = asyncio.Queue()
    _subscribers.append(q)

    async def gen():
        try:
            while True:
                await q.get()
                yield f"event: speaker_alias_updated\ndata: updated\n\n"
        finally:
            _subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


def set_current_result(result: TranscriptResult | None) -> None:
    global _current_result
    _current_result = result


__all__ = ["app", "set_current_result"]
