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
    fuzz=types.SimpleNamespace(ratio=lambda *_a, **_k: 100),
)
