from __future__ import annotations

from uuid import uuid4

from redis import Redis

from .config import AppSettings

INGEST_LOCK_KEY = "lan:ingest:lock"
_RELEASE_IF_VALUE_MATCHES_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
end
return 0
"""


def _ingest_lock_ttl(settings: AppSettings) -> int:
    return max(1, int(settings.ingest_lock_ttl_seconds))


def _redis_client(settings: AppSettings) -> Redis:
    return Redis.from_url(settings.redis_url)


def try_acquire_ingest_lock(
    settings: AppSettings,
    *,
    redis_client: Redis | None = None,
    key: str = INGEST_LOCK_KEY,
) -> tuple[bool, int, str | None]:
    client = redis_client or _redis_client(settings)
    ttl_seconds = _ingest_lock_ttl(settings)
    lock_token = uuid4().hex
    acquired = bool(client.set(key, lock_token, nx=True, ex=ttl_seconds))
    if acquired:
        return True, ttl_seconds, lock_token

    retry_after = client.ttl(key)
    if retry_after is None or int(retry_after) <= 0:
        return False, ttl_seconds, None
    return False, int(retry_after), None


def release_ingest_lock(
    settings: AppSettings,
    *,
    token: str | None,
    redis_client: Redis | None = None,
    key: str = INGEST_LOCK_KEY,
) -> bool:
    if token is None:
        return False
    client = redis_client or _redis_client(settings)
    result = client.eval(_RELEASE_IF_VALUE_MATCHES_SCRIPT, 1, key, token)
    return int(result) == 1


__all__ = [
    "INGEST_LOCK_KEY",
    "release_ingest_lock",
    "try_acquire_ingest_lock",
]
