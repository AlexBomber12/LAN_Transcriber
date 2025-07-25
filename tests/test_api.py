from pathlib import Path
from fastapi.testclient import TestClient
from lan_transcriber import api, aliases


def test_update_alias(tmp_path: Path, monkeypatch):
    path = tmp_path / "db.yaml"
    aliases.save_aliases({}, path)
    monkeypatch.setattr(api, "ALIAS_PATH", path)
    monkeypatch.setattr(aliases, "ALIAS_PATH", path)
    client = TestClient(api.app)
    resp = client.post("/alias/S1", json={"alias": "Dave"})
    assert resp.status_code == 200
    assert aliases.load_aliases(path)["S1"] == "Dave"
