import importlib
import sys
import pathlib
import pytest


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
    previous_whisperx = sys.modules.pop("whisperx", None)
    try:
        assert importlib.import_module(mod)
        assert "whisperx" not in sys.modules
    finally:
        if previous_whisperx is not None:
            sys.modules["whisperx"] = previous_whisperx


def test_orchestrator_import_does_not_import_whisperx():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    previous_whisperx = sys.modules.pop("whisperx", None)
    previous_orchestrator = sys.modules.pop("lan_transcriber.pipeline_steps.orchestrator", None)
    try:
        assert importlib.import_module("lan_transcriber.pipeline_steps.orchestrator")
        assert "whisperx" not in sys.modules
    finally:
        if previous_orchestrator is not None:
            sys.modules["lan_transcriber.pipeline_steps.orchestrator"] = previous_orchestrator
        if previous_whisperx is not None:
            sys.modules["whisperx"] = previous_whisperx
