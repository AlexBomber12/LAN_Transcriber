from __future__ import annotations

import inspect
from typing import Any

_PATCH_SENTINEL = "_lan_ignore_use_auth_token"


def patch_pyannote_inference_ignore_use_auth_token() -> bool:
    """Patch pyannote Inference.__init__ to accept and ignore use_auth_token.

    Returns True only when a patch was applied in this call.
    """

    try:
        from pyannote.audio import Inference
    except Exception:
        return False

    init = getattr(Inference, "__init__", None)
    if init is None:
        return False
    if getattr(init, _PATCH_SENTINEL, False):
        return False

    try:
        signature = inspect.signature(init)
    except (TypeError, ValueError):
        return False

    if "use_auth_token" in signature.parameters:
        return False

    original_init = init

    def patched_init(self: Any, *args: Any, use_auth_token: Any = None, **kwargs: Any) -> Any:
        del use_auth_token
        return original_init(self, *args, **kwargs)

    setattr(patched_init, _PATCH_SENTINEL, True)
    Inference.__init__ = patched_init  # type: ignore[assignment]
    return True
