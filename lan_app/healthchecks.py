from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from redis import Redis
from rq import Worker as RQWorker

from .config import AppSettings
from .db import connect, init_db

_WORKER_HEARTBEAT_STALE_SECONDS = 180


def _ok(component: str, detail: str = "ok") -> dict[str, Any]:
    return {"component": component, "ok": True, "detail": detail}


def _fail(component: str, detail: str) -> dict[str, Any]:
    return {"component": component, "ok": False, "detail": detail}


def check_app_health() -> dict[str, Any]:
    return _ok("app")


def check_db_health(settings: AppSettings | None = None) -> dict[str, Any]:
    cfg = settings or AppSettings()
    try:
        init_db(cfg)
        with connect(cfg) as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as exc:
        return _fail("db", str(exc))
    return _ok("db")


def check_redis_health(settings: AppSettings | None = None) -> dict[str, Any]:
    cfg = settings or AppSettings()
    try:
        redis = Redis.from_url(cfg.redis_url)
        redis.ping()
    except Exception as exc:
        return _fail("redis", str(exc))
    return _ok("redis")


def _worker_queue_names(worker: object) -> set[str]:
    queue_names = getattr(worker, "queue_names", None)
    if callable(queue_names):
        try:
            names = queue_names()
        except TypeError:
            names = queue_names  # pragma: no cover
        if isinstance(names, (list, tuple, set)):
            return {str(name) for name in names}

    queues = getattr(worker, "queues", None)
    if isinstance(queues, (list, tuple, set)):
        out: set[str] = set()
        for queue in queues:
            out.add(str(getattr(queue, "name", queue)))
        return out
    return set()


def check_worker_health(settings: AppSettings | None = None) -> dict[str, Any]:
    cfg = settings or AppSettings()
    redis_health = check_redis_health(cfg)
    if not redis_health["ok"]:
        return _fail("worker", f"redis unavailable: {redis_health['detail']}")

    try:
        connection = Redis.from_url(cfg.redis_url)
        workers = RQWorker.all(connection=connection)
    except Exception as exc:
        return _fail("worker", str(exc))

    matching = []
    for worker in workers:
        if cfg.rq_queue_name in _worker_queue_names(worker):
            matching.append(worker)

    if not matching:
        return _fail("worker", f"no worker registered for queue '{cfg.rq_queue_name}'")

    now = datetime.now(tz=timezone.utc)
    for worker in matching:
        heartbeat = getattr(worker, "last_heartbeat", None)
        if heartbeat is None:
            continue
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=timezone.utc)
        age = (now - heartbeat).total_seconds()
        if age <= _WORKER_HEARTBEAT_STALE_SECONDS:
            return _ok("worker", f"worker '{worker.name}' heartbeat {int(age)}s ago")

    return _fail("worker", "worker heartbeat is stale")


def collect_health_checks(settings: AppSettings | None = None) -> dict[str, dict[str, Any]]:
    cfg = settings or AppSettings()
    return {
        "app": check_app_health(),
        "db": check_db_health(cfg),
        "redis": check_redis_health(cfg),
        "worker": check_worker_health(cfg),
    }


def check_health_component(
    component: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    if component == "app":
        return check_app_health()
    if component == "db":
        return check_db_health(cfg)
    if component == "redis":
        return check_redis_health(cfg)
    if component == "worker":
        return check_worker_health(cfg)
    raise ValueError(f"unsupported component: {component}")


def main(argv: list[str] | None = None) -> int:
    args = argv or []
    target = args[0] if args else "all"
    if target not in {"all", "app", "db", "redis", "worker"}:
        print(f"unsupported target: {target}")
        return 2

    if target == "all":
        checks = collect_health_checks()
        print(json.dumps(checks, ensure_ascii=True))
        return 0 if all(item["ok"] for item in checks.values()) else 1

    payload = check_health_component(target)
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
