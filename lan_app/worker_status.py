from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.gpu_policy import collect_cuda_runtime_facts

WORKER_STATUS_FILENAME = "worker_status.json"
WORKER_STATUS_STALE_AFTER = timedelta(minutes=10)
WORKER_HEARTBEAT_INTERVAL_SECONDS = 60.0


def worker_status_path(data_root: Path) -> Path:
    return Path(data_root) / WORKER_STATUS_FILENAME


def write_worker_status(
    data_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    facts = collect_cuda_runtime_facts()
    heartbeat = (now or datetime.now(tz=timezone.utc)).isoformat()
    payload: dict[str, Any] = {
        "gpu_available": bool(facts.is_available),
        "device_count": int(facts.device_count),
        "visible_devices": facts.visible_devices,
        "torch_cuda_version": facts.torch_cuda_version,
        "last_heartbeat": heartbeat,
    }
    path = worker_status_path(data_root)
    try:
        atomic_write_json(path, payload)
    except OSError:
        return payload
    return payload


def read_worker_status(data_root: Path) -> dict[str, Any] | None:
    path = worker_status_path(data_root)
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def is_worker_status_fresh(
    status: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    raw = str(status.get("last_heartbeat") or "").strip()
    if not raw:
        return False
    try:
        heartbeat = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    current = now or datetime.now(tz=timezone.utc)
    return (current - heartbeat) <= WORKER_STATUS_STALE_AFTER


def run_heartbeat_loop(
    data_root: Path,
    stop_event: threading.Event,
    *,
    interval: float = WORKER_HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    while True:
        write_worker_status(data_root)
        if stop_event.wait(interval):
            return


def start_heartbeat_thread(
    data_root: Path,
    *,
    interval: float = WORKER_HEARTBEAT_INTERVAL_SECONDS,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_heartbeat_loop,
        args=(data_root, stop_event),
        kwargs={"interval": interval},
        daemon=True,
        name="worker-heartbeat",
    )
    thread.start()
    return stop_event, thread


__all__ = [
    "WORKER_HEARTBEAT_INTERVAL_SECONDS",
    "WORKER_STATUS_FILENAME",
    "WORKER_STATUS_STALE_AFTER",
    "is_worker_status_fresh",
    "read_worker_status",
    "run_heartbeat_loop",
    "start_heartbeat_thread",
    "worker_status_path",
    "write_worker_status",
]
