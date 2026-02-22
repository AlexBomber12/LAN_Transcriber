from fastapi.testclient import TestClient
from lan_app.api import app


def test_root_ok():
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 200
