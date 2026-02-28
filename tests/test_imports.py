import importlib
import sys
import pathlib
import pytest


def _pop_module_tree(module_name: str) -> dict[str, object]:
    removed: dict[str, object] = {}
    for key in list(sys.modules):
        if key == module_name or key.startswith(f"{module_name}."):
            removed[key] = sys.modules.pop(key)
    return removed


@pytest.mark.parametrize(
    "mod",
    [
        "web_transcribe",
        "lan_app.api",
        "lan_app.worker",
        "lan_app.worker_tasks",
        "lan_app.ui_routes",
        "lan_app.db",
        "lan_app.exporter",
        "lan_transcriber.pipeline",
        "lan_transcriber.pipeline_steps.orchestrator",
        "lan_transcriber.pipeline_steps.precheck",
        "lan_transcriber.pipeline_steps.summary_builder",
    ],
)
def test_imports(mod):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    previous_modules = _pop_module_tree(mod)
    previous_whisperx_modules = _pop_module_tree("whisperx")
    try:
        importlib.invalidate_caches()
        assert importlib.import_module(mod)
        assert "whisperx" not in sys.modules
    finally:
        sys.modules.update(previous_modules)
        sys.modules.update(previous_whisperx_modules)


def test_orchestrator_import_does_not_import_whisperx():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    previous_orchestrator_modules = _pop_module_tree("lan_transcriber.pipeline_steps.orchestrator")
    previous_whisperx_modules = _pop_module_tree("whisperx")
    try:
        importlib.invalidate_caches()
        assert importlib.import_module("lan_transcriber.pipeline_steps.orchestrator")
        assert "whisperx" not in sys.modules
    finally:
        sys.modules.update(previous_orchestrator_modules)
        sys.modules.update(previous_whisperx_modules)
