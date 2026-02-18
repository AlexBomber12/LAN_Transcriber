from __future__ import annotations

import asyncio
from typing import List

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from lan_transcriber import aliases
from lan_transcriber.metrics import write_metrics_snapshot
from lan_transcriber.models import TranscriptResult
from lan_transcriber.pipeline import refresh_aliases

from .config import AppSettings

app = FastAPI()
ALIAS_PATH = aliases.ALIAS_PATH
_subscribers: List[asyncio.Queue[str]] = []
_current_result: TranscriptResult | None = None
_settings = AppSettings()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Simple health check used by monitoring."""
    return {"status": "ok"}


@app.on_event("startup")
async def _start_metrics() -> None:
    _settings.metrics_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(write_metrics_snapshot(_settings.metrics_snapshot_path))


class AliasUpdate(BaseModel):
    alias: str


@app.post("/alias/{speaker_id}")
async def update_alias(speaker_id: str, upd: AliasUpdate):
    path = aliases.ALIAS_PATH
    known = aliases.load_aliases(path)
    known[speaker_id] = upd.alias
    aliases.save_aliases(known, path)
    if _current_result is not None:
        refresh_aliases(_current_result, path)
    for queue in list(_subscribers):
        queue.put_nowait("updated")
    return {"speaker": speaker_id, "alias": upd.alias}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/events")
async def events():
    queue: asyncio.Queue[str] = asyncio.Queue()
    _subscribers.append(queue)

    async def gen():
        try:
            while True:
                await queue.get()
                yield "event: speaker_alias_updated\ndata: updated\n\n"
        finally:
            _subscribers.remove(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


def set_current_result(result: TranscriptResult | None) -> None:
    global _current_result
    _current_result = result


__all__ = ["ALIAS_PATH", "app", "set_current_result", "healthz"]
