from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import runpy
import signal
import sys
from types import SimpleNamespace

from fastapi import Response
import pytest
from starlette.requests import Request

import lan_app.config as config_module
import lan_app.db as db_module
from lan_app import auth, db_init, healthchecks, hf_repo, uploads, worker, workers


def _make_request(
    *,
    method: str = "GET",
    path: str = "/",
    scheme: str = "http",
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> Request:
    encoded_headers: list[tuple[bytes, bytes]] = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    if cookies:
        cookie_value = "; ".join(f"{k}={v}" for k, v in cookies.items())
        encoded_headers.append((b"cookie", cookie_value.encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": scheme,
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": b"",
        "headers": encoded_headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    return Request(scope)


def test_healthchecks_ok_fail_and_db_redis_checks(monkeypatch):
    assert healthchecks._ok("app") == {
        "component": "app",
        "ok": True,
        "detail": "ok",
    }
    assert healthchecks._fail("db", "down") == {
        "component": "db",
        "ok": False,
        "detail": "down",
    }
    assert healthchecks.check_app_health() == {
        "component": "app",
        "ok": True,
        "detail": "ok",
    }

    cfg = SimpleNamespace(redis_url="redis://unit", rq_queue_name="audio")
    db_calls: list[object] = []

    class _Conn:
        def execute(self, query: str):
            assert query == "SELECT 1"
            return self

        def fetchone(self):
            return (1,)

    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(healthchecks, "init_db", lambda settings: db_calls.append(settings))
    monkeypatch.setattr(healthchecks, "connect", lambda _settings: _Ctx())
    assert healthchecks.check_db_health(cfg) == {
        "component": "db",
        "ok": True,
        "detail": "ok",
    }
    assert db_calls == [cfg]

    def _db_boom(_settings):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(healthchecks, "connect", _db_boom)
    failed_db = healthchecks.check_db_health(cfg)
    assert failed_db["component"] == "db"
    assert failed_db["ok"] is False
    assert "db unavailable" in failed_db["detail"]

    class _RedisOK:
        def __init__(self):
            self.pinged = False

        def ping(self):
            self.pinged = True

    redis_client = _RedisOK()
    monkeypatch.setattr(healthchecks.Redis, "from_url", lambda url: redis_client)
    assert healthchecks.check_redis_health(cfg) == {
        "component": "redis",
        "ok": True,
        "detail": "ok",
    }
    assert redis_client.pinged is True

    class _RedisFail:
        def ping(self):
            raise RuntimeError("redis unavailable")

    monkeypatch.setattr(healthchecks.Redis, "from_url", lambda url: _RedisFail())
    failed_redis = healthchecks.check_redis_health(cfg)
    assert failed_redis["component"] == "redis"
    assert failed_redis["ok"] is False
    assert "redis unavailable" in failed_redis["detail"]


def test_worker_queue_name_extraction_variants():
    class _WithCallableQueueNames:
        def queue_names(self):
            return ["alpha", "beta"]

    assert healthchecks._worker_queue_names(_WithCallableQueueNames()) == {"alpha", "beta"}

    class _CallableNeedsArg:
        queues = [SimpleNamespace(name="gamma"), "delta"]

        def queue_names(self, _required):
            return ["ignored"]

    assert healthchecks._worker_queue_names(_CallableNeedsArg()) == {"gamma", "delta"}

    queue_holder = SimpleNamespace(queues=[SimpleNamespace(name="q1"), "q2"])
    assert healthchecks._worker_queue_names(queue_holder) == {"q1", "q2"}

    class _CallableWrongType:
        def queue_names(self):
            return "not-a-sequence"

    assert healthchecks._worker_queue_names(_CallableWrongType()) == set()
    assert healthchecks._worker_queue_names(SimpleNamespace(queues="queue")) == set()


def test_check_worker_health_paths(monkeypatch):
    cfg = SimpleNamespace(redis_url="redis://unit", rq_queue_name="audio")

    monkeypatch.setattr(
        healthchecks,
        "check_redis_health",
        lambda _settings: {"component": "redis", "ok": False, "detail": "down"},
    )
    unavailable = healthchecks.check_worker_health(cfg)
    assert unavailable["ok"] is False
    assert "redis unavailable: down" in unavailable["detail"]

    monkeypatch.setattr(
        healthchecks,
        "check_redis_health",
        lambda _settings: {"component": "redis", "ok": True, "detail": "ok"},
    )
    monkeypatch.setattr(healthchecks.Redis, "from_url", lambda _url: object())

    def _all_boom(*, connection):
        raise RuntimeError("rq failure")

    monkeypatch.setattr(healthchecks.RQWorker, "all", _all_boom)
    failed_lookup = healthchecks.check_worker_health(cfg)
    assert failed_lookup["ok"] is False
    assert "rq failure" in failed_lookup["detail"]

    class _OtherQueueWorker:
        name = "worker-1"
        last_heartbeat = datetime.now(tz=timezone.utc)

        def queue_names(self):
            return ["other"]

    monkeypatch.setattr(
        healthchecks.RQWorker,
        "all",
        lambda *, connection: [_OtherQueueWorker()],
    )
    missing = healthchecks.check_worker_health(cfg)
    assert missing["ok"] is False
    assert "no worker registered" in missing["detail"]

    fixed_now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    class _FixedDateTime:
        @staticmethod
        def now(*, tz):
            assert tz == timezone.utc
            return fixed_now

    monkeypatch.setattr(healthchecks, "datetime", _FixedDateTime)

    stale_with_no_heartbeat = SimpleNamespace(
        name="worker-none",
        queue_names=lambda: ["audio"],
        last_heartbeat=None,
    )
    stale_with_old_heartbeat = SimpleNamespace(
        name="worker-stale",
        queue_names=lambda: ["audio"],
        last_heartbeat=fixed_now - timedelta(seconds=400),
    )
    monkeypatch.setattr(
        healthchecks.RQWorker,
        "all",
        lambda *, connection: [stale_with_no_heartbeat, stale_with_old_heartbeat],
    )
    stale = healthchecks.check_worker_health(cfg)
    assert stale["ok"] is False
    assert stale["detail"] == "worker heartbeat is stale"

    fresh_naive = SimpleNamespace(
        name="worker-fresh",
        queue_names=lambda: ["audio"],
        last_heartbeat=(fixed_now - timedelta(seconds=10)).replace(tzinfo=None),
    )
    monkeypatch.setattr(healthchecks.RQWorker, "all", lambda *, connection: [fresh_naive])
    healthy = healthchecks.check_worker_health(cfg)
    assert healthy["ok"] is True
    assert "worker-fresh" in healthy["detail"]
    assert "10s ago" in healthy["detail"]


def test_collect_check_component_and_main(monkeypatch, capsys):
    cfg = SimpleNamespace()
    calls: list[str] = []

    monkeypatch.setattr(
        healthchecks,
        "check_app_health",
        lambda: calls.append("app") or {"component": "app", "ok": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        healthchecks,
        "check_db_health",
        lambda _settings: calls.append("db") or {"component": "db", "ok": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        healthchecks,
        "check_redis_health",
        lambda _settings: calls.append("redis")
        or {"component": "redis", "ok": True, "detail": "ok"},
    )
    monkeypatch.setattr(
        healthchecks,
        "check_worker_health",
        lambda _settings: calls.append("worker")
        or {"component": "worker", "ok": True, "detail": "ok"},
    )

    checks = healthchecks.collect_health_checks(cfg)
    assert list(checks.keys()) == ["app", "db", "redis", "worker"]
    assert calls == ["app", "db", "redis", "worker"]

    assert healthchecks.check_health_component("app", settings=cfg)["component"] == "app"
    assert healthchecks.check_health_component("db", settings=cfg)["component"] == "db"
    assert healthchecks.check_health_component("redis", settings=cfg)["component"] == "redis"
    assert healthchecks.check_health_component("worker", settings=cfg)["component"] == "worker"
    with pytest.raises(ValueError, match="unsupported component"):
        healthchecks.check_health_component("unknown", settings=cfg)

    assert healthchecks.main(["unknown"]) == 2
    assert "unsupported target: unknown" in capsys.readouterr().out

    monkeypatch.setattr(
        healthchecks,
        "collect_health_checks",
        lambda: {
            "app": {"ok": True},
            "db": {"ok": True},
        },
    )
    assert healthchecks.main(["all"]) == 0
    assert '"app"' in capsys.readouterr().out

    monkeypatch.setattr(
        healthchecks,
        "collect_health_checks",
        lambda: {
            "app": {"ok": True},
            "db": {"ok": False},
        },
    )
    assert healthchecks.main(["all"]) == 1

    monkeypatch.setattr(
        healthchecks,
        "check_health_component",
        lambda component: {"component": component, "ok": False, "detail": "bad"},
    )
    assert healthchecks.main(["db"]) == 1
    monkeypatch.setattr(
        healthchecks,
        "check_health_component",
        lambda component: {"component": component, "ok": True, "detail": "ok"},
    )
    assert healthchecks.main(["worker"]) == 0


def test_healthchecks_module_main_guard(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["healthchecks.py", "unsupported"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("lan_app.healthchecks", run_name="__main__")
    assert excinfo.value.code == 2


def test_auth_token_paths_and_request_authentication():
    assert auth._normalize_token(None) is None
    assert auth._normalize_token("   ") is None
    assert auth._normalize_token(" token ") == "token"

    enabled_settings = SimpleNamespace(api_bearer_token=" secret-token ")
    disabled_settings = SimpleNamespace(api_bearer_token="   ")
    assert auth.expected_bearer_token(enabled_settings) == "secret-token"
    assert auth.auth_enabled(enabled_settings) is True
    assert auth.expected_bearer_token(disabled_settings) is None
    assert auth.auth_enabled(disabled_settings) is False

    no_header_request = _make_request()
    assert auth._token_from_authorization_header(no_header_request) is None
    assert auth._token_from_cookie(no_header_request) is None

    wrong_scheme = _make_request(headers={"Authorization": "Basic abc"})
    assert auth._token_from_authorization_header(wrong_scheme) is None

    bearer_without_token = _make_request(headers={"Authorization": "Bearer   "})
    assert auth._token_from_authorization_header(bearer_without_token) is None

    valid_bearer = _make_request(headers={"Authorization": "Bearer secret-token"})
    assert auth._token_from_authorization_header(valid_bearer) == "secret-token"

    cookie_only = _make_request(cookies={auth.AUTH_COOKIE_NAME: " secret-token "})
    assert auth._token_from_cookie(cookie_only) == "secret-token"

    assert auth.request_is_authenticated(no_header_request, disabled_settings) is True
    assert auth.request_is_authenticated(no_header_request, enabled_settings) is False
    assert auth.request_is_authenticated(valid_bearer, enabled_settings) is True

    wrong_header_right_cookie = _make_request(
        headers={"Authorization": "Bearer wrong"},
        cookies={auth.AUTH_COOKIE_NAME: "secret-token"},
    )
    assert auth.request_is_authenticated(wrong_header_right_cookie, enabled_settings) is True


def test_auth_request_policy_cookie_flags_and_redirect_safety():
    assert auth.request_requires_auth(_make_request(method="GET", path="/api/uploads")) is False
    assert auth.request_requires_auth(_make_request(method="POST", path="/ui/login")) is False
    assert auth.request_requires_auth(_make_request(method="DELETE", path="/api/uploads")) is True

    forwarded_https = _make_request(
        scheme="http",
        headers={"x-forwarded-proto": "https, http"},
    )
    forwarded_http = _make_request(
        scheme="https",
        headers={"x-forwarded-proto": "http"},
    )
    plain_https = _make_request(scheme="https")
    assert auth.cookie_secure_flag(forwarded_https) is True
    assert auth.cookie_secure_flag(forwarded_http) is False
    assert auth.cookie_secure_flag(plain_https) is True

    response = Response()
    auth.set_auth_cookie(response, "token123", secure=True)
    set_cookie = response.headers.get("set-cookie", "")
    assert auth.AUTH_COOKIE_NAME in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie

    auth.clear_auth_cookie(response)
    cleared_cookie = response.headers.get("set-cookie", "")
    assert auth.AUTH_COOKIE_NAME in cleared_cookie

    assert auth.safe_next_path(None) == "/ui"
    assert auth.safe_next_path("relative") == "/ui"
    assert auth.safe_next_path("//evil.example") == "/ui"
    assert auth.safe_next_path("/ui/recordings") == "/ui/recordings"
    assert auth.safe_next_path("", default="/home") == "/home"


def test_worker_signal_handlers_and_main(monkeypatch):
    registered_handlers: dict[signal.Signals, object] = {}
    log_messages: list[str] = []

    def _capture_signal(sig_num, callback):
        registered_handlers[sig_num] = callback
        return None

    monkeypatch.setattr(worker.signal, "signal", _capture_signal)
    monkeypatch.setattr(worker._logger, "info", lambda msg, *args: log_messages.append(msg % args))

    class _DummyWorker:
        def __init__(self):
            self.stop_requests: list[int] = []

        def request_stop(self, signum, frame):
            self.stop_requests.append(signum)

    dummy_worker = _DummyWorker()
    worker._install_signal_handlers(dummy_worker)
    assert signal.SIGTERM in registered_handlers
    assert signal.SIGINT in registered_handlers

    registered_handlers[signal.SIGTERM](signal.SIGTERM, None)
    registered_handlers[signal.SIGINT](999, None)
    assert dummy_worker.stop_requests == [signal.SIGTERM, 999]
    assert any("SIGTERM" in message for message in log_messages)
    assert any("999" in message for message in log_messages)

    calls: dict[str, object] = {}
    settings = SimpleNamespace(
        redis_url="redis://unit",
        rq_queue_name="audio",
        rq_worker_burst=True,
    )
    monkeypatch.setattr(worker, "AppSettings", lambda: settings)
    monkeypatch.setattr(worker, "init_db", lambda cfg: calls.setdefault("init_db", cfg))
    connection = object()

    def _redis_from_url(url):
        calls["redis_url"] = url
        return connection

    monkeypatch.setattr(worker.Redis, "from_url", _redis_from_url)

    class _FakeRQWorker:
        def __init__(self, queues, *, connection):
            calls["queues"] = queues
            calls["connection"] = connection

        def work(self, *, with_scheduler, burst):
            calls["work"] = (with_scheduler, burst)

    monkeypatch.setattr(worker, "Worker", _FakeRQWorker)
    monkeypatch.setattr(worker, "_install_signal_handlers", lambda w: calls.setdefault("handler_worker", w))
    worker.main()

    assert calls["init_db"] is settings
    assert calls["redis_url"] == "redis://unit"
    assert calls["queues"] == ["audio"]
    assert calls["connection"] is connection
    assert calls["work"] == (False, True)
    assert calls["handler_worker"] is not None


def test_worker_module_main_guard(monkeypatch):
    calls: dict[str, object] = {}
    settings = SimpleNamespace(
        redis_url="redis://guard",
        rq_queue_name="guard-queue",
        rq_worker_burst=False,
    )
    monkeypatch.setattr(config_module, "AppSettings", lambda: settings)
    monkeypatch.setattr(db_module, "init_db", lambda cfg: calls.setdefault("init_db", cfg))
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)

    import redis
    import rq

    monkeypatch.setattr(redis.Redis, "from_url", lambda _url: object())

    class _GuardWorker:
        def __init__(self, queues, *, connection):
            calls["queues"] = queues

        def request_stop(self, signum, frame):
            return None

        def work(self, *, with_scheduler, burst):
            calls["work"] = (with_scheduler, burst)

    monkeypatch.setattr(rq, "Worker", _GuardWorker)
    monkeypatch.setattr(sys, "argv", ["worker.py"])
    runpy.run_module("lan_app.worker", run_name="__main__")

    assert calls["init_db"] is settings
    assert calls["queues"] == ["guard-queue"]
    assert calls["work"] == (False, False)


@pytest.mark.asyncio
async def test_process_recording_defaults(monkeypatch, tmp_path: Path):
    default_settings = object()
    default_llm = object()
    diariser = object()
    result = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(workers, "Settings", lambda: default_settings)
    monkeypatch.setattr(workers, "LLMClient", lambda: default_llm)

    async def _fake_run_pipeline(**kwargs):
        calls.update(kwargs)
        return result

    monkeypatch.setattr(workers, "run_pipeline", _fake_run_pipeline)
    audio_path = tmp_path / "audio.wav"

    output = await workers.process_recording(audio_path=audio_path, diariser=diariser, recording_id="rec-1")
    assert output is result
    assert calls == {
        "audio_path": audio_path,
        "cfg": default_settings,
        "llm": default_llm,
        "diariser": diariser,
        "recording_id": "rec-1",
    }


@pytest.mark.asyncio
async def test_process_recording_uses_passed_cfg_and_llm(monkeypatch, tmp_path: Path):
    provided_settings = object()
    provided_llm = object()
    diariser = object()
    result = object()

    monkeypatch.setattr(
        workers,
        "Settings",
        lambda: (_ for _ in ()).throw(AssertionError("Settings() should not be called")),
    )
    monkeypatch.setattr(
        workers,
        "LLMClient",
        lambda: (_ for _ in ()).throw(AssertionError("LLMClient() should not be called")),
    )

    async def _fake_run_pipeline(**kwargs):
        assert kwargs["cfg"] is provided_settings
        assert kwargs["llm"] is provided_llm
        assert kwargs["recording_id"] is None
        return result

    monkeypatch.setattr(workers, "run_pipeline", _fake_run_pipeline)
    audio_path = tmp_path / "audio.wav"

    output = await workers.process_recording(
        audio_path=audio_path,
        diariser=diariser,
        cfg=provided_settings,
        llm_client=provided_llm,
    )
    assert output is result


def test_db_init_main_and_guard(monkeypatch, capsys):
    settings = SimpleNamespace()
    db_path = Path("/tmp/unit-test-db.sqlite")
    monkeypatch.setattr(db_init, "AppSettings", lambda: settings)
    monkeypatch.setattr(db_init, "init_db", lambda cfg: db_path)
    db_init.main()
    assert capsys.readouterr().out.strip() == f"Database ready at {db_path}"

    monkeypatch.setattr(config_module, "AppSettings", lambda: settings)
    monkeypatch.setattr(db_module, "init_db", lambda cfg: db_path)
    runpy.run_module("lan_app.db_init", run_name="__main__")


def test_hf_repo_split_repo_id_and_revision():
    assert hf_repo.split_repo_id_and_revision("org/repo") == ("org/repo", None)
    assert hf_repo.split_repo_id_and_revision(" org/repo @ main ") == ("org/repo", "main")
    assert hf_repo.split_repo_id_and_revision("org/repo@   ") == ("org/repo", None)


def test_upload_filename_and_timestamp_helpers(monkeypatch):
    assert uploads.suffix_from_name("voice.WAV") == ".wav"
    assert uploads.suffix_from_name("voice") == ".mp3"

    assert uploads.safe_filename("") == "upload"
    assert uploads.safe_filename("../weird*.mp3") == "weird_.mp3"
    assert uploads.safe_filename("....") == "upload"

    assert uploads.parse_plaud_captured_at("Meeting 2024-04-05 06_07_08.mp3") == "2024-04-05T06:07:08Z"
    assert uploads.parse_plaud_captured_at("Meeting 2024-99-05 06_07_08.mp3") is None
    assert uploads.parse_plaud_captured_at("no timestamp here.mp3") is None
    assert uploads.infer_captured_at("Meeting 2024-04-05 06_07_08.mp3") == "2024-04-05T06:07:08Z"

    fixed_now = datetime(2026, 2, 3, 4, 5, 6, 789123, tzinfo=timezone.utc)

    class _FixedDateTime:
        @staticmethod
        def now(*, tz):
            assert tz == timezone.utc
            return fixed_now

    monkeypatch.setattr(uploads, "datetime", _FixedDateTime)
    assert uploads.infer_captured_at("plain_name.mp3") == "2026-02-03T04:05:06Z"


def test_write_upload_to_path_success_and_error_paths(tmp_path: Path):
    class _Upload:
        def __init__(self, file_obj):
            self.file = file_obj

    destination = tmp_path / "a" / "b" / "audio.bin"
    upload = _Upload(io.BytesIO(b"hello"))
    bytes_written = uploads.write_upload_to_path(upload, destination, max_bytes=10)
    assert bytes_written == 5
    assert destination.read_bytes() == b"hello"

    class _SeekFailFile(io.BytesIO):
        def seek(self, offset, whence=0):
            raise RuntimeError("seek not supported")

    destination_seek_fail = tmp_path / "seek-fail.bin"
    seek_fail_upload = _Upload(_SeekFailFile(b"data"))
    assert uploads.write_upload_to_path(seek_fail_upload, destination_seek_fail, max_bytes=None) == 4
    assert destination_seek_fail.read_bytes() == b"data"

    destination_too_large = tmp_path / "too-large.bin"
    too_large_upload = _Upload(io.BytesIO(b"abcdef"))
    with pytest.raises(ValueError, match="max upload size exceeded"):
        uploads.write_upload_to_path(too_large_upload, destination_too_large, max_bytes=3)
    assert destination_too_large.exists() is False

    class _ReadBoomFile:
        def seek(self, offset, whence=0):
            return 0

        def read(self, _size):
            raise OSError("read failed")

    destination_read_boom = tmp_path / "read-boom.bin"
    read_boom_upload = _Upload(_ReadBoomFile())
    with pytest.raises(OSError, match="read failed"):
        uploads.write_upload_to_path(read_boom_upload, destination_read_boom, max_bytes=None)
    assert destination_read_boom.exists() is False
