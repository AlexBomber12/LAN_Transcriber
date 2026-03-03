from __future__ import annotations

from pathlib import Path

import yaml


def _load_compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_worker_gpu_exposed_in_compose_files() -> None:
    root = Path(__file__).resolve().parent.parent
    base = _load_compose(root / "docker-compose.yml")
    dev = _load_compose(root / "docker-compose.dev.yml")

    assert base["services"]["worker"]["gpus"] == "all"
    assert dev["services"]["worker"]["gpus"] == "all"
