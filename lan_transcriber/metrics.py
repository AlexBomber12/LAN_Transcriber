from __future__ import annotations

import asyncio
from pathlib import Path

from prometheus_client import Histogram, Counter, generate_latest

p95_latency_seconds = Histogram(
    "p95_latency_seconds",
    "Pipeline request latency seconds",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

error_rate_total = Counter("error_rate_total", "Unhandled pipeline exceptions")

llm_timeouts_total = Counter("llm_timeouts_total", "Total LLM timeouts")


async def write_metrics_snapshot(path: Path) -> None:
    """Periodically write metrics to ``path``."""
    while True:
        path.write_bytes(generate_latest())
        await asyncio.sleep(60)


__all__ = [
    "p95_latency_seconds",
    "error_rate_total",
    "llm_timeouts_total",
    "write_metrics_snapshot",
]
