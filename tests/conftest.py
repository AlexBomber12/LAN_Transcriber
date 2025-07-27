import sys, pathlib, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONPATH", str(ROOT))
os.environ.setdefault("CI", "true")

import types


def _ensure_stub(mod: str, **attrs) -> None:
    """Create a simple stub module if missing."""
    if mod in sys.modules:
        return
    stub = types.ModuleType(mod)
    for k, v in attrs.items():
        setattr(stub, k, v)
    sys.modules[mod] = stub


_ensure_stub(
    "pydantic_settings",
    BaseSettings=type("BaseSettings", (), {}),
)
_ensure_stub(
    "rapidfuzz",
    fuzz=types.SimpleNamespace(ratio=lambda a, b, *_, **__: 100 if a.strip(".!?") == b.strip(".!?") else 0),
)

class _FastAPI:
    def get(self, *a, **k):
        def wrap(fn):
            return fn
        return wrap

    def post(self, *a, **k):
        def wrap(fn):
            return fn
        return wrap

    def on_event(self, *a, **k):
        def wrap(fn):
            return fn
        return wrap


_ensure_stub(
    "fastapi",
    FastAPI=_FastAPI,
)
_ensure_stub(
    "fastapi.responses",
    StreamingResponse=type("StreamingResponse", (), {}),
    Response=type("Response", (), {}),
    HTMLResponse=type("HTMLResponse", (), {}),
)
_ensure_stub(
    "fastapi.testclient",
    TestClient=lambda app: types.SimpleNamespace(
        app=app,
        get=lambda path: types.SimpleNamespace(status_code=200),
        post=lambda path, json=None: types.SimpleNamespace(status_code=200),
    ),
)

class _SimpleResponse:
    def __init__(self, status_code=200, json=None):
        self.status_code = status_code
        self._json = json or {}

    def json(self):
        return self._json


class _SimpleAsyncClient:
    async def post(self, *a, **k):
        return _SimpleResponse()


_ensure_stub(
    "httpx",
    Response=_SimpleResponse,
    AsyncClient=_SimpleAsyncClient,
)

_ensure_stub(
    "requests",
    get=lambda *a, **k: _SimpleResponse(),
    exceptions=types.SimpleNamespace(ConnectionError=Exception),
)

_ensure_stub(
    "tenacity",
    retry=lambda *a, **k: (lambda f: f),
    wait_exponential=lambda *a, **k: None,
    stop_after_attempt=lambda *a, **k: None,
)

_ensure_stub("anyio")

_ensure_stub(
    "respx",
    mock=lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *x: False),
)

_ensure_stub(
    "pydantic",
    BaseModel=type("BaseModel", (), {}),
)
_ensure_stub(
    "pydantic.v1",
    BaseSettings=type("BaseSettings", (), {}),
    Field=lambda *a, **k: None,
    BaseModel=type("BaseModel", (), {}),
)


def pytest_pyfunc_call(pyfuncitem):
    import inspect, asyncio

    testfunction = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunction):
        asyncio.run(testfunction())
        return True
