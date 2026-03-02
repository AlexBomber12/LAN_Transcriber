from __future__ import annotations

import contextlib
import importlib
import logging
import pickle
import re
from collections.abc import Iterator

_log = logging.getLogger(__name__)

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


def _collect_omegaconf_symbols(extra_fqns: list[str] | None) -> list[tuple[str, object]]:
    """Return ``(fqn, class)`` pairs for all importable OmegaConf symbols."""
    fqns = list(_OMEGACONF_BASE_FQNS)
    for candidate in extra_fqns or []:
        if candidate in fqns:
            continue
        fqns.append(candidate)

    pairs: list[tuple[str, object]] = []
    for fqn in fqns:
        symbol = _import_omegaconf_symbol(fqn)
        if symbol is not None:
            pairs.append((fqn, symbol))
    return pairs


def _try_api(
    api: object,
    symbols_list: list[object],
    symbols_dict: dict[str, object],
) -> object | None:
    """Try *api*(list); on ``TypeError`` fall back to *api*(dict).

    Returns whatever *api* returns (e.g. a context-manager) or ``None``
    if both forms fail.  Logs exactly one warning on total failure.
    """
    try:
        return api(symbols_list)  # type: ignore[operator]
    except TypeError:
        pass
    except Exception as exc:
        _log.warning(
            "torch safe-globals %s(list) failed: %s: %s",
            getattr(api, "__name__", api),
            type(exc).__name__,
            exc,
        )
        return None
    # List form raised TypeError – try dict form.
    try:
        return api(symbols_dict)  # type: ignore[operator]
    except Exception as exc:
        _log.warning(
            "torch safe-globals %s rejected both list and dict forms: %s: %s",
            getattr(api, "__name__", api),
            type(exc).__name__,
            exc,
        )
        return None


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

    pairs = _collect_omegaconf_symbols(extra_fqns)
    if not pairs:
        yield
        return

    symbols_list = [symbol for _, symbol in pairs]
    symbols_dict = {fqn: symbol for fqn, symbol in pairs}

    safe_globals = getattr(serialization, "safe_globals", None)
    if callable(safe_globals):
        context = _try_api(safe_globals, symbols_list, symbols_dict)
        if context is not None:
            with context:
                yield
            return

    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        _try_api(add_safe_globals, symbols_list, symbols_dict)
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
