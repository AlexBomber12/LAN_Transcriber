from __future__ import annotations

import inspect
from typing import Any, Callable


def filter_kwargs_for_callable(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keyword arguments accepted by ``fn``.

    If ``fn`` accepts ``**kwargs`` or its signature cannot be inspected,
    the input kwargs are returned unchanged.
    """

    if not kwargs:
        return {}

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke ``fn`` with kwargs filtered to signature-supported keys."""

    filtered = filter_kwargs_for_callable(fn, kwargs)
    return fn(*args, **filtered)
