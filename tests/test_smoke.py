import importlib
import pytest

@pytest.mark.parametrize("name", ["web_transcribe", "ollama_python", "transformers"])
def test_import(name):
    assert importlib.import_module(name) is not None
