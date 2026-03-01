from __future__ import annotations

import inspect
import re
from typing import Any, Callable

_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument '([^']+)'")


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


def _extract_unexpected_kwarg_name(msg: str) -> str | None:
    match = _UNEXPECTED_KWARG_RE.search(msg)
    if match is None:
        return None
    return match.group(1)


def call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke ``fn`` while dropping unsupported kwargs when possible."""

    filtered = filter_kwargs_for_callable(fn, kwargs)
    max_retries = len(filtered)

    for _ in range(max_retries + 1):
        try:
            return fn(*args, **filtered)
        except TypeError as exc:
            unexpected_kwarg = _extract_unexpected_kwarg_name(str(exc))
            if unexpected_kwarg is None or unexpected_kwarg not in filtered:
                raise
            filtered = dict(filtered)
            filtered.pop(unexpected_kwarg, None)

    return fn(*args, **filtered)
