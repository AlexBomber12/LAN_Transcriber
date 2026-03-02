from __future__ import annotations

_ALLOWLIST_APPLIED = False


def allowlist_omegaconf_for_weights_only() -> None:
    """Allowlist OmegaConf container types for torch weights-only unpickling."""

    global _ALLOWLIST_APPLIED
    if _ALLOWLIST_APPLIED:
        return

    try:
        import torch
    except Exception:
        return

    serialization = getattr(torch, "serialization", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if not callable(add_safe_globals):
        return

    try:
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
    except Exception:
        return

    add_safe_globals([ListConfig, DictConfig])
    _ALLOWLIST_APPLIED = True


__all__ = ["allowlist_omegaconf_for_weights_only"]
