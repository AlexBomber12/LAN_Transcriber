from __future__ import annotations

from redis import Redis

from .config import AppSettings

INGEST_LOCK_KEY = "lan:ingest:lock"


def _ingest_lock_ttl(settings: AppSettings) -> int:
    return max(1, int(settings.ingest_lock_ttl_seconds))


def _redis_client(settings: AppSettings) -> Redis:
    return Redis.from_url(settings.redis_url)


def try_acquire_ingest_lock(
    settings: AppSettings,
    *,
    redis_client: Redis | None = None,
    key: str = INGEST_LOCK_KEY,
) -> tuple[bool, int]:
    client = redis_client or _redis_client(settings)
    ttl_seconds = _ingest_lock_ttl(settings)
    acquired = bool(client.set(key, "1", nx=True, ex=ttl_seconds))
    if acquired:
        return True, ttl_seconds

    retry_after = client.ttl(key)
    if retry_after is None or int(retry_after) <= 0:
        return False, ttl_seconds
    return False, int(retry_after)


def release_ingest_lock(
    settings: AppSettings,
    *,
    redis_client: Redis | None = None,
    key: str = INGEST_LOCK_KEY,
) -> None:
    client = redis_client or _redis_client(settings)
    client.delete(key)


__all__ = [
    "INGEST_LOCK_KEY",
    "release_ingest_lock",
    "try_acquire_ingest_lock",
]
