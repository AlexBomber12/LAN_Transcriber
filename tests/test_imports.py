import importlib, sys, pathlib
import pytest

@pytest.mark.parametrize("mod", ["web_transcribe"])
def test_imports(mod):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    assert importlib.import_module(mod)
