from __future__ import annotations

import inspect
import re
import threading
from typing import Any, Callable

_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument '([^']+)'")
_LAST_CALL_DETAILS = threading.local()


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


def clear_last_supported_kwargs_call_details() -> None:
    _LAST_CALL_DETAILS.value = None


def last_supported_kwargs_call_details() -> tuple[dict[str, Any] | None, tuple[str, ...] | None]:
    raw = getattr(_LAST_CALL_DETAILS, "value", None)
    if not isinstance(raw, tuple) or len(raw) != 2:
        return None, None
    filtered, dropped = raw
    if not isinstance(filtered, dict) or not isinstance(dropped, tuple):
        return None, None
    return dict(filtered), dropped


def call_with_supported_kwargs_details(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any], tuple[str, ...]]:
    """Invoke ``fn`` and report the kwargs that were ultimately dropped."""

    clear_last_supported_kwargs_call_details()
    filtered = filter_kwargs_for_callable(fn, kwargs)
    dropped = [key for key in kwargs if key not in filtered]
    max_retries = len(filtered)

    for _ in range(max_retries + 1):
        try:
            result = fn(*args, **filtered)
            details = (dict(filtered), tuple(dropped))
            _LAST_CALL_DETAILS.value = details
            return result, details[0], details[1]
        except TypeError as exc:
            unexpected_kwarg = _extract_unexpected_kwarg_name(str(exc))
            if unexpected_kwarg is None or unexpected_kwarg not in filtered:
                raise
            filtered = dict(filtered)
            filtered.pop(unexpected_kwarg, None)
            dropped.append(unexpected_kwarg)

    result = fn(*args, **filtered)
    details = (dict(filtered), tuple(dropped))
    _LAST_CALL_DETAILS.value = details
    return result, details[0], details[1]


def call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke ``fn`` while dropping unsupported kwargs when possible."""

    result, _filtered, _dropped = call_with_supported_kwargs_details(fn, *args, **kwargs)
    return result
