from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lan_app import worker_status
from lan_transcriber.gpu_policy import CudaRuntimeFacts


def _gpu_facts(**overrides) -> CudaRuntimeFacts:
    base = {
        "is_available": True,
        "device_count": 1,
        "visible_devices": "0",
        "torch_cuda_version": "12.6",
    }
    base.update(overrides)
    return CudaRuntimeFacts(**base)


def test_worker_status_path(tmp_path: Path) -> None:
    assert worker_status.worker_status_path(tmp_path) == tmp_path / "worker_status.json"


def test_write_worker_status_writes_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(worker_status, "collect_cuda_runtime_facts", _gpu_facts)
    fixed_now = datetime(2026, 4, 12, 9, 30, tzinfo=timezone.utc)
    payload = worker_status.write_worker_status(tmp_path, now=fixed_now)

    assert payload["gpu_available"] is True
    assert payload["device_count"] == 1
    assert payload["visible_devices"] == "0"
    assert payload["torch_cuda_version"] == "12.6"
    assert payload["last_heartbeat"] == fixed_now.isoformat()

    on_disk = json.loads((tmp_path / "worker_status.json").read_text(encoding="utf-8"))
    assert on_disk == payload


def test_write_worker_status_tolerates_write_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(worker_status, "collect_cuda_runtime_facts", _gpu_facts)

    def _boom(_path, _data):
        raise OSError("disk full")

    monkeypatch.setattr(worker_status, "atomic_write_json", _boom)
    payload = worker_status.write_worker_status(tmp_path)
    assert payload["gpu_available"] is True
    assert not (tmp_path / "worker_status.json").exists()


def test_write_worker_status_defaults_to_current_utc(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(worker_status, "collect_cuda_runtime_facts", _gpu_facts)
    before = datetime.now(tz=timezone.utc)
    payload = worker_status.write_worker_status(tmp_path)
    after = datetime.now(tz=timezone.utc)
    heartbeat = datetime.fromisoformat(payload["last_heartbeat"])
    assert before <= heartbeat <= after


def test_read_worker_status_missing_file(tmp_path: Path) -> None:
    assert worker_status.read_worker_status(tmp_path) is None


def test_read_worker_status_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "worker_status.json").write_text("{broken", encoding="utf-8")
    assert worker_status.read_worker_status(tmp_path) is None


def test_read_worker_status_non_dict(tmp_path: Path) -> None:
    (tmp_path / "worker_status.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert worker_status.read_worker_status(tmp_path) is None


def test_read_worker_status_returns_dict(tmp_path: Path) -> None:
    payload = {"gpu_available": True, "last_heartbeat": "2026-04-12T09:00:00+00:00"}
    (tmp_path / "worker_status.json").write_text(json.dumps(payload), encoding="utf-8")
    assert worker_status.read_worker_status(tmp_path) == payload


def test_read_worker_status_os_error(tmp_path: Path, monkeypatch) -> None:
    def _boom(self, **kwargs):
        raise OSError("perm denied")

    monkeypatch.setattr(Path, "read_text", _boom)
    assert worker_status.read_worker_status(tmp_path) is None


def test_is_worker_status_fresh_variants() -> None:
    now = datetime(2026, 4, 12, 10, 0, tzinfo=timezone.utc)

    assert worker_status.is_worker_status_fresh({}, now=now) is False
    assert worker_status.is_worker_status_fresh({"last_heartbeat": ""}, now=now) is False
    assert worker_status.is_worker_status_fresh({"last_heartbeat": "not-a-date"}, now=now) is False

    fresh_ts = (now - timedelta(minutes=1)).isoformat()
    assert worker_status.is_worker_status_fresh({"last_heartbeat": fresh_ts}, now=now) is True

    stale_ts = (now - timedelta(hours=1)).isoformat()
    assert worker_status.is_worker_status_fresh({"last_heartbeat": stale_ts}, now=now) is False

    naive_fresh = (now.replace(tzinfo=None) - timedelta(minutes=2)).isoformat()
    assert worker_status.is_worker_status_fresh({"last_heartbeat": naive_fresh}, now=now) is True

    z_suffixed = (now - timedelta(minutes=3)).isoformat().replace("+00:00", "Z")
    assert worker_status.is_worker_status_fresh({"last_heartbeat": z_suffixed}, now=now) is True

    future_ts = (now + timedelta(minutes=5)).isoformat()
    assert worker_status.is_worker_status_fresh({"last_heartbeat": future_ts}, now=now) is False


def test_is_worker_status_fresh_defaults_now(monkeypatch) -> None:
    heartbeat = datetime.now(tz=timezone.utc).isoformat()
    assert worker_status.is_worker_status_fresh({"last_heartbeat": heartbeat}) is True


def test_run_heartbeat_loop_writes_and_exits(tmp_path: Path, monkeypatch) -> None:
    writes: list[Path] = []

    def _fake_write(path):
        writes.append(Path(path))

    monkeypatch.setattr(worker_status, "write_worker_status", _fake_write)

    event = threading.Event()
    event.set()

    worker_status.run_heartbeat_loop(tmp_path, event, interval=0.01)
    assert writes == [tmp_path]


def test_run_heartbeat_loop_runs_multiple_iterations(tmp_path: Path, monkeypatch) -> None:
    writes: list[Path] = []
    event = threading.Event()

    def _fake_write(path):
        writes.append(Path(path))
        if len(writes) >= 2:
            event.set()

    monkeypatch.setattr(worker_status, "write_worker_status", _fake_write)
    worker_status.run_heartbeat_loop(tmp_path, event, interval=0.001)
    assert len(writes) == 2


def test_start_heartbeat_thread_creates_daemon(monkeypatch, tmp_path: Path) -> None:
    created: dict[str, object] = {}

    class _FakeThread:
        def __init__(self, *, target, args, kwargs, daemon, name):
            created["target"] = target
            created["args"] = args
            created["kwargs"] = kwargs
            created["daemon"] = daemon
            created["name"] = name

        def start(self) -> None:
            created["started"] = True

    monkeypatch.setattr(worker_status.threading, "Thread", _FakeThread)
    event, thread = worker_status.start_heartbeat_thread(tmp_path, interval=5.0)

    assert isinstance(event, threading.Event)
    assert created["started"] is True
    assert created["daemon"] is True
    assert created["name"] == "worker-heartbeat"
    assert created["target"] is worker_status.run_heartbeat_loop
    assert created["args"] == (tmp_path, event)
    assert created["kwargs"] == {"interval": 5.0}
    assert isinstance(thread, _FakeThread)


def test_start_heartbeat_thread_default_interval(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeThread:
        def __init__(self, *, target, args, kwargs, daemon, name):
            captured["kwargs"] = kwargs
            del target, args, daemon, name

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(worker_status.threading, "Thread", _FakeThread)
    worker_status.start_heartbeat_thread(tmp_path)
    assert captured["kwargs"] == {"interval": worker_status.WORKER_HEARTBEAT_INTERVAL_SECONDS}
    assert captured["started"] is True


def test_worker_status_module_exports() -> None:
    assert "write_worker_status" in worker_status.__all__
    assert "read_worker_status" in worker_status.__all__
    assert "is_worker_status_fresh" in worker_status.__all__
    assert "start_heartbeat_thread" in worker_status.__all__


@pytest.mark.parametrize(
    "device_count",
    [0, 2],
)
def test_write_worker_status_captures_device_count(
    tmp_path: Path, monkeypatch, device_count: int
) -> None:
    monkeypatch.setattr(
        worker_status,
        "collect_cuda_runtime_facts",
        lambda: _gpu_facts(device_count=device_count, is_available=device_count > 0),
    )
    payload = worker_status.write_worker_status(tmp_path)
    assert payload["device_count"] == device_count
    assert payload["gpu_available"] is (device_count > 0)
