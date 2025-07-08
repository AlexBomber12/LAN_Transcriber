import importlib, sys, pathlib, pkg_resources  # noqa: F401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
assert importlib.import_module("web_transcribe")
