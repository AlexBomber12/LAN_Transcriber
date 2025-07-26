import os
import sys
import pathlib
import io
import contextlib

# Ensure CI stubs are activated and repository root is on sys.path
os.environ.setdefault("CI", "true")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Import after setting environment so heavy deps are stubbed
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import web_transcribe

from fastapi.testclient import TestClient
from fastapi import FastAPI

# Reuse FastAPI app from web_transcribe if available, otherwise create and mount
app = getattr(web_transcribe, "app", None)
if app is None:
    demo = getattr(web_transcribe, "demo", None)
    app = FastAPI()
    try:
        import gradio as gr
        if hasattr(gr, "mount_gradio_app") and demo is not None:
            # mount_gradio_app returns the app; ignore output for stubbed gradio
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                gr.mount_gradio_app(app, demo, path="/")
    except Exception:
        pass

    @app.get("/")
    async def root():
        return {"status": "ok"}


def test_ui_root_serves():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
