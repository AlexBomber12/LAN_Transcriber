from __future__ import annotations

from pathlib import Path

from lan_app.config import AppSettings
from lan_app.locks import release_ingest_lock, try_acquire_ingest_lock


class _FakeRedis:
    def __init__(self):
        self._store: dict[str, tuple[str, int]] = {}

    def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        if nx and key in self._store:
            return False
        self._store[key] = (str(value), int(ex))
        return True

    def ttl(self, key: str) -> int:
        row = self._store.get(key)
        if row is None:
            return -2
        return row[1]

    def get(self, key: str) -> str | None:
        row = self._store.get(key)
        if row is None:
            return None
        return row[0]

    def delete(self, key: str) -> int:
        return int(self._store.pop(key, None) is not None)

    def eval(self, _script: str, _numkeys: int, key: str, token: str) -> int:
        current = self.get(key)
        if current == token:
            self.delete(key)
            return 1
        return 0


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

    acquired, retry_after, token = try_acquire_ingest_lock(cfg, redis_client=redis)

    assert acquired is True
    assert retry_after == 123
    assert token is not None


def test_try_acquire_ingest_lock_returns_retry_after_when_held(tmp_path: Path):
    cfg = _cfg(tmp_path)
    redis = _FakeRedis()
    redis.set("lan:ingest:lock", "owned", nx=True, ex=31)

    acquired, retry_after, token = try_acquire_ingest_lock(cfg, redis_client=redis)

    assert acquired is False
    assert retry_after == 31
    assert token is None


def test_release_ingest_lock_deletes_key(tmp_path: Path):
    cfg = _cfg(tmp_path)
    redis = _FakeRedis()
    acquired, _retry_after, token = try_acquire_ingest_lock(cfg, redis_client=redis)
    assert acquired is True
    assert token is not None

    released = release_ingest_lock(cfg, token=token, redis_client=redis)
    assert released is True

    acquired, retry_after, _token = try_acquire_ingest_lock(cfg, redis_client=redis)
    assert acquired is True
    assert retry_after == 123


def test_release_ingest_lock_does_not_delete_new_owner_lock(tmp_path: Path):
    cfg = _cfg(tmp_path)
    redis = _FakeRedis()
    acquired, _retry_after, token = try_acquire_ingest_lock(cfg, redis_client=redis)
    assert acquired is True
    assert token is not None
    redis._store["lan:ingest:lock"] = ("other-owner", 31)

    released = release_ingest_lock(cfg, token=token, redis_client=redis)

    assert released is False
    assert redis.get("lan:ingest:lock") == "other-owner"
