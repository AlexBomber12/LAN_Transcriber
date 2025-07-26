from fastapi.testclient import TestClient
from web_transcribe import app


def test_root_200():
    client = TestClient(app)
    assert client.get("/").status_code == 200
