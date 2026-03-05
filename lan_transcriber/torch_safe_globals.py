from __future__ import annotations

import contextlib
import importlib
import logging
import pickle
import re
from collections.abc import Callable, Iterator

_log = logging.getLogger(__name__)

_OMEGACONF_BASE_FQNS = (
    "omegaconf.listconfig.ListConfig",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.base.ContainerMetadata",
)
_DIARIZATION_BASE_FQNS = (
    "torch.torch_version.TorchVersion",
    "omegaconf.listconfig.ListConfig",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.base.ContainerMetadata",
    "pyannote.audio.core.task.Specifications",
)
_DIARIZATION_TRUSTED_PREFIXES = ("omegaconf.", "pyannote.")
_DIARIZATION_TRUSTED_EXACT = {"torch.torch_version.TorchVersion"}

_UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global:\s+(?:GLOBAL\s+)?([A-Za-z_][\w.]*)")


def _import_symbol(fqn: str) -> object | None:
    module_name, _, symbol_name = fqn.rpartition(".")
    if not module_name or not symbol_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, symbol_name, None)


def _collect_symbols(
    base_fqns: tuple[str, ...],
    extra_fqns: list[str] | None,
    *,
    trusted_fqn: Callable[[str], bool],
) -> list[tuple[str, object]]:
    fqns = list(base_fqns)
    for candidate in extra_fqns or []:
        if not trusted_fqn(candidate):
            continue
        if candidate in fqns:
            continue
        fqns.append(candidate)

    pairs: list[tuple[str, object]] = []
    for fqn in fqns:
        if not trusted_fqn(fqn):
            continue
        symbol = _import_symbol(fqn)
        if symbol is not None:
            pairs.append((fqn, symbol))
    return pairs


def _collect_omegaconf_symbols(extra_fqns: list[str] | None) -> list[tuple[str, object]]:
    """Return ``(fqn, class)`` pairs for all importable OmegaConf symbols."""
    return _collect_symbols(
        _OMEGACONF_BASE_FQNS,
        extra_fqns,
        trusted_fqn=lambda fqn: fqn.startswith("omegaconf."),
    )


def is_trusted_diarization_global_fqn(fqn: str) -> bool:
    return fqn in _DIARIZATION_TRUSTED_EXACT or fqn.startswith(_DIARIZATION_TRUSTED_PREFIXES)


def _collect_diarization_symbols(extra_fqns: list[str] | None) -> list[tuple[str, object]]:
    return _collect_symbols(
        _DIARIZATION_BASE_FQNS,
        extra_fqns,
        trusted_fqn=is_trusted_diarization_global_fqn,
    )


def import_trusted_diarization_symbol(fqn: str) -> object | None:
    if not is_trusted_diarization_global_fqn(fqn):
        return None
    return _import_symbol(fqn)


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
def _torch_safe_globals_for_symbols(pairs: list[tuple[str, object]]) -> Iterator[None]:
    try:
        import torch
    except Exception:
        yield
        return

    serialization = getattr(torch, "serialization", None)
    if serialization is None:
        yield
        return

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


@contextlib.contextmanager
def omegaconf_safe_globals_for_torch_load(extra_fqns: list[str] | None = None) -> Iterator[None]:
    """Best-effort OmegaConf allowlisting for torch weights-only unpickling."""

    pairs = _collect_omegaconf_symbols(extra_fqns)
    with _torch_safe_globals_for_symbols(pairs):
        yield


@contextlib.contextmanager
def diarization_safe_globals_for_torch_load(extra_fqns: list[str] | None = None) -> Iterator[None]:
    """Best-effort trusted allowlisting for pyannote diarization checkpoint loading."""

    pairs = _collect_diarization_symbols(extra_fqns)
    with _torch_safe_globals_for_symbols(pairs):
        yield


def _parse_any_unsupported_global_fqn(message: str) -> str | None:
    match = _UNSUPPORTED_GLOBAL_RE.search(message or "")
    if match is None:
        return None
    return match.group(1)


def parse_unsupported_global_fqn(message: str) -> str | None:
    fqn = _parse_any_unsupported_global_fqn(message)
    if fqn is None:
        return None
    if not fqn.startswith("omegaconf."):
        return None
    return fqn


def unsupported_global_omegaconf_fqn_from_error(error: BaseException) -> str | None:
    if not isinstance(error, (pickle.UnpicklingError, RuntimeError)):
        return None
    return parse_unsupported_global_fqn(str(error))


def parse_diarization_unsupported_global_fqn(message: str) -> str | None:
    fqn = _parse_any_unsupported_global_fqn(message)
    if fqn is None:
        return None
    if not is_trusted_diarization_global_fqn(fqn):
        return None
    return fqn


def unsupported_global_diarization_fqn_from_error(error: BaseException) -> str | None:
    if not isinstance(error, (pickle.UnpicklingError, RuntimeError)):
        return None
    return parse_diarization_unsupported_global_fqn(str(error))


__all__ = [
    "diarization_safe_globals_for_torch_load",
    "import_trusted_diarization_symbol",
    "is_trusted_diarization_global_fqn",
    "omegaconf_safe_globals_for_torch_load",
    "parse_diarization_unsupported_global_fqn",
    "parse_unsupported_global_fqn",
    "unsupported_global_diarization_fqn_from_error",
    "unsupported_global_omegaconf_fqn_from_error",
]
