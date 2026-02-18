from pathlib import Path

from lan_transcriber import runtime_paths


def test_default_data_root_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LAN_DATA_ROOT", str(tmp_path))
    assert runtime_paths.default_data_root() == tmp_path
    assert runtime_paths.default_alias_path() == tmp_path / "db" / "speaker_bank.yaml"
    assert runtime_paths.default_recordings_root() == tmp_path / "recordings"


def test_default_data_root_fallback(monkeypatch) -> None:
    monkeypatch.delenv("LAN_DATA_ROOT", raising=False)
    monkeypatch.setattr(runtime_paths.os, "access", lambda *_args, **_kwargs: False)
    assert runtime_paths.default_data_root() == Path("data")
