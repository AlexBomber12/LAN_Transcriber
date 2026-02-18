"""Compatibility wrapper for application API objects."""

from lan_app.api import ALIAS_PATH, app, healthz, set_current_result

__all__ = ["ALIAS_PATH", "app", "set_current_result", "healthz"]
