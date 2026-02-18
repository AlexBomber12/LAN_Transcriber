import importlib
import sys
import pathlib
import pytest


@pytest.mark.parametrize("mod", ["web_transcribe", "lan_app.api"])
def test_imports(mod):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    assert importlib.import_module(mod)
