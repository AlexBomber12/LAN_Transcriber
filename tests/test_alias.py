from pathlib import Path
from lan_transcriber import aliases


def test_alias_roundtrip(tmp_path: Path):
    path = tmp_path / "db.yaml"
    aliases.save_aliases({"S1": "Alice"}, path)
    data = aliases.load_aliases(path)
    assert data == {"S1": "Alice"}


def test_update_alias(tmp_path: Path):
    path = tmp_path / "db.yaml"
    aliases.save_aliases({"S1": "Bob"}, path)
    data = aliases.load_aliases(path)
    assert data["S1"] == "Bob"
    data["S1"] = "Carol"
    aliases.save_aliases(data, path)
    again = aliases.load_aliases(path)
    assert again == {"S1": "Carol"}


def test_alias_persistence(tmp_path: Path):
    from lan_transcriber.aliases import save_aliases, load_aliases

    path = tmp_path / "bank.yaml"
    save_aliases({"S0": "Bob"}, path)
    assert load_aliases(path)["S0"] == "Bob"
