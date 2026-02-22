from fastapi.testclient import TestClient
from lan_app.api import app


def test_root_200():
    client = TestClient(app)
    assert client.get("/").status_code == 200
