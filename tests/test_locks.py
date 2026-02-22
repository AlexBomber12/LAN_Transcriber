from __future__ import annotations

from pathlib import Path

from lan_app.config import AppSettings
from lan_app.locks import release_ingest_lock, try_acquire_ingest_lock


class _FakeRedis:
    def __init__(self):
        self._store: dict[str, int] = {}

    def set(self, key: str, _value: str, *, nx: bool, ex: int) -> bool:
        if nx and key in self._store:
            return False
        self._store[key] = ex
        return True

    def ttl(self, key: str) -> int:
        return self._store.get(key, -2)

    def delete(self, key: str) -> int:
        return int(self._store.pop(key, None) is not None)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.ingest_lock_ttl_seconds = 123
    return cfg


def test_try_acquire_ingest_lock_returns_true_when_free(tmp_path: Path):
    cfg = _cfg(tmp_path)
    redis = _FakeRedis()

    acquired, retry_after = try_acquire_ingest_lock(cfg, redis_client=redis)

    assert acquired is True
    assert retry_after == 123


def test_try_acquire_ingest_lock_returns_retry_after_when_held(tmp_path: Path):
    cfg = _cfg(tmp_path)
    redis = _FakeRedis()
    redis.set("lan:ingest:lock", "1", nx=True, ex=31)

    acquired, retry_after = try_acquire_ingest_lock(cfg, redis_client=redis)

    assert acquired is False
    assert retry_after == 31


def test_release_ingest_lock_deletes_key(tmp_path: Path):
    cfg = _cfg(tmp_path)
    redis = _FakeRedis()
    redis.set("lan:ingest:lock", "1", nx=True, ex=31)

    release_ingest_lock(cfg, redis_client=redis)

    acquired, retry_after = try_acquire_ingest_lock(cfg, redis_client=redis)
    assert acquired is True
    assert retry_after == 123
