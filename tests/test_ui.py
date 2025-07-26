from fastapi.testclient import TestClient
from web_transcribe import app


def test_root_ok():
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 200

