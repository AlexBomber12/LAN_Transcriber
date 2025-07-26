from fastapi.testclient import TestClient
from web_transcribe import app


def test_root_returns_200():
    """The UI must serve HTML at “/” without crashing."""
    r = TestClient(app).get("/")
    assert r.status_code == 200, r.text
    assert "<html" in r.text.lower()


def test_openapi_ok():
    """FastAPI should build its schema (catches bool-schema bug)."""
    r = TestClient(app).get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["openapi"].startswith("3.")
