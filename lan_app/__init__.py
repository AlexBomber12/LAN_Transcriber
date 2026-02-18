"""Application-layer package for LAN Transcriber."""

from .api import app as api_app
from .ui import app as ui_app

__all__ = ["api_app", "ui_app"]
