from __future__ import annotations

import contextlib
import importlib
import pickle
import re
from collections.abc import Iterator

_OMEGACONF_BASE_FQNS = (
    "omegaconf.listconfig.ListConfig",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.base.ContainerMetadata",
)

_UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global:\s+(?:GLOBAL\s+)?([A-Za-z_][\w.]*)")


def _import_omegaconf_symbol(fqn: str) -> object | None:
    if not fqn.startswith("omegaconf."):
        return None
    module_name, _, symbol_name = fqn.rpartition(".")
    if not module_name or not symbol_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, symbol_name, None)


def _collect_omegaconf_symbols(extra_fqns: list[str] | None) -> list[object]:
    fqns = list(_OMEGACONF_BASE_FQNS)
    for candidate in extra_fqns or []:
        if candidate in fqns:
            continue
        fqns.append(candidate)

    symbols: list[object] = []
    for fqn in fqns:
        symbol = _import_omegaconf_symbol(fqn)
        if symbol is not None:
            symbols.append(symbol)
    return symbols


@contextlib.contextmanager
def omegaconf_safe_globals_for_torch_load(extra_fqns: list[str] | None = None) -> Iterator[None]:
    """Best-effort OmegaConf allowlisting for torch weights-only unpickling."""

    try:
        import torch
    except Exception:
        yield
        return

    serialization = getattr(torch, "serialization", None)
    if serialization is None:
        yield
        return

    symbols = _collect_omegaconf_symbols(extra_fqns)
    if not symbols:
        yield
        return

    safe_globals = getattr(serialization, "safe_globals", None)
    if callable(safe_globals):
        context = None
        try:
            context = safe_globals(symbols)
        except Exception:
            context = None
        if context is not None:
            with context:
                yield
            return

    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    try:
        add_safe_globals(symbols)
    except Exception:
        pass
    yield


def parse_unsupported_global_fqn(message: str) -> str | None:
    match = _UNSUPPORTED_GLOBAL_RE.search(message or "")
    if match is None:
        return None
    fqn = match.group(1)
    if not fqn.startswith("omegaconf."):
        return None
    return fqn


def unsupported_global_omegaconf_fqn_from_error(error: BaseException) -> str | None:
    if not isinstance(error, (pickle.UnpicklingError, RuntimeError)):
        return None
    return parse_unsupported_global_fqn(str(error))


__all__ = [
    "omegaconf_safe_globals_for_torch_load",
    "parse_unsupported_global_fqn",
    "unsupported_global_omegaconf_fqn_from_error",
]
