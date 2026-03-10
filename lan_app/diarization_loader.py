from __future__ import annotations

import logging
import os
import re
from typing import Any

from lan_transcriber.gpu_policy import (
    collect_cuda_runtime_facts,
    cuda_memory_info,
    is_gpu_oom_error,
    is_gpu_device,
    normalize_device,
    normalize_scheduler_mode,
    resolve_effective_device,
)
from lan_transcriber.torch_safe_globals import (
    diarization_safe_globals_for_torch_load,
    import_trusted_diarization_symbol,
    unsupported_global_diarization_fqn_from_error,
)

from .hf_repo import split_repo_id_and_revision

DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
_MAX_SAFE_GLOBAL_ATTEMPTS = 3

_REPO_HINT_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\b")
_LOG = logging.getLogger(__name__)


def resolve_diarization_model_id(model_id: str | None = None) -> str:
    if model_id is not None:
        normalized = model_id.strip()
    else:
        normalized = os.getenv("LAN_DIARIZATION_MODEL_ID", "").strip()
    return normalized or DEFAULT_DIARIZATION_MODEL_ID


def resolve_hf_token(token: str | None = None) -> str | None:
    if token is not None and token.strip():
        return token.strip()
    env_token = os.getenv("HF_TOKEN", "").strip()
    if env_token:
        return env_token
    fallback = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    return fallback or None


def _status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def classify_pipeline_load_error(exc: Exception) -> str:
    exc_name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "revisionnotfound" in exc_name or "revision not found" in message:
        return "revision_not_found"
    status_code = _status_code_from_exception(exc)
    if status_code in {401, 403}:
        return "gated_access"
    if "gated" in message or "unauthorized" in message or "forbidden" in message:
        return "gated_access"
    return "other"


def extract_repo_hints(exc: Exception) -> list[str]:
    return sorted(set(_REPO_HINT_RE.findall(str(exc))))


def _candidate_load_inputs(repo_id: str, revision: str | None) -> list[tuple[str, dict[str, str]]]:
    if revision:
        return [
            (repo_id, {"revision": revision}),
            (f"{repo_id}@{revision}", {}),
            (repo_id, {}),
        ]
    return [(repo_id, {})]


def _from_pretrained_with_safe_globals(
    from_pretrained: Any,
    candidate: str,
    kwargs: dict[str, Any],
) -> Any:
    extra_fqns: list[str] = []
    last_error: Exception | None = None
    for _ in range(_MAX_SAFE_GLOBAL_ATTEMPTS):
        with diarization_safe_globals_for_torch_load(extra_fqns=extra_fqns):
            try:
                return from_pretrained(candidate, **kwargs)
            except Exception as exc:
                last_error = exc
        retry_fqn = unsupported_global_diarization_fqn_from_error(last_error)
        if retry_fqn is None or retry_fqn in extra_fqns:
            raise last_error
        if import_trusted_diarization_symbol(retry_fqn) is None:
            raise last_error
        extra_fqns.append(retry_fqn)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to load diarization pipeline.")


def _log_cuda_memory_snapshot(*, label: str, device: str) -> None:
    memory_info = cuda_memory_info(device)
    if memory_info is None:
        return
    free_bytes, total_bytes = memory_info
    _LOG.info(
        "%s VRAM snapshot: device=%s free_bytes=%s total_bytes=%s",
        label,
        device,
        free_bytes,
        total_bytes,
    )


def _move_pipeline_to_best_device(
    model: Any,
    *,
    requested_device: str | None,
    scheduler_mode: str | None,
) -> str:
    normalized_request = normalize_device(requested_device)
    normalized_scheduler_mode = normalize_scheduler_mode(scheduler_mode)
    if normalized_request == "cpu":
        return "cpu"

    cuda_facts = collect_cuda_runtime_facts()
    try:
        import torch
    except Exception:
        if normalized_request != "auto":
            raise RuntimeError(
                f"Requested diarization device {normalized_request} but torch is unavailable."
            )
        return "cpu"

    effective_device = resolve_effective_device(
        normalized_request,
        cuda_facts=cuda_facts,
        label="diarization device",
    )
    if not is_gpu_device(effective_device):
        return "cpu"

    _log_cuda_memory_snapshot(label="Pyannote move", device=effective_device)

    try:
        model.to(torch.device(effective_device))
    except Exception as exc:
        if normalized_request == "auto" and normalized_scheduler_mode == "auto":
            _LOG.warning(
                "Failed to move pyannote diarization pipeline to %s; continuing on CPU: %s",
                effective_device,
                exc,
            )
            return "cpu"
        if normalized_request == "auto" and is_gpu_oom_error(exc):
            _LOG.warning(
                "Pyannote diarization GPU OOM on %s; retrying on CPU in auto mode: %s",
                effective_device,
                exc,
            )
            return "cpu"
        raise RuntimeError(
            f"Failed to move pyannote diarization pipeline to {effective_device}: {exc}"
        ) from exc
    return effective_device


def load_pyannote_pipeline(
    *,
    model_id: str | None = None,
    token: str | None = None,
    device: str | None = None,
    scheduler_mode: str | None = None,
) -> Any:
    resolved_model_id = resolve_diarization_model_id(model_id)
    repo_id, revision = split_repo_id_and_revision(resolved_model_id)
    if not repo_id:
        raise ValueError("LAN_DIARIZATION_MODEL_ID cannot be empty.")

    from pyannote.audio import Pipeline  # type: ignore

    resolved_token = resolve_hf_token(token)
    last_error: Exception | None = None
    for candidate, candidate_kwargs in _candidate_load_inputs(repo_id, revision):
        kwargs: dict[str, Any] = dict(candidate_kwargs)
        if resolved_token:
            kwargs["token"] = resolved_token
        requested_device = device if device is not None else os.getenv(
            "LAN_DIARIZATION_DEVICE",
            "auto",
        )
        if normalize_device(requested_device) != "cpu":
            resolved_device = resolve_effective_device(
                requested_device,
                cuda_facts=collect_cuda_runtime_facts(),
                label="diarization device",
            )
            if is_gpu_device(resolved_device):
                _log_cuda_memory_snapshot(label="Pyannote load", device=resolved_device)
        try:
            model = _from_pretrained_with_safe_globals(
                Pipeline.from_pretrained,
                candidate,
                kwargs,
            )
        except TypeError as exc:
            message = str(exc).lower()
            if (
                resolved_token
                and "unexpected keyword argument" in message
                and "token" in message
            ):
                kwargs.pop("token", None)
                try:
                    model = _from_pretrained_with_safe_globals(
                        Pipeline.from_pretrained,
                        candidate,
                        kwargs,
                    )
                except Exception as retry_exc:
                    last_error = retry_exc
                    continue
            else:
                last_error = exc
                continue
        except Exception as exc:
            last_error = exc
            continue

        if model is None or not callable(model):
            raise TypeError("Loaded diarization pipeline must be callable.")
        device_name = _move_pipeline_to_best_device(
            model,
            requested_device=requested_device,
            scheduler_mode=scheduler_mode or os.getenv("LAN_GPU_SCHEDULER_MODE", "auto"),
        )
        setattr(model, "_lan_effective_device", device_name)
        _LOG.info("Pyannote diarization device: %s", device_name)
        return model

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to load diarization pipeline.")


__all__ = [
    "DEFAULT_DIARIZATION_MODEL_ID",
    "classify_pipeline_load_error",
    "extract_repo_hints",
    "load_pyannote_pipeline",
    "resolve_diarization_model_id",
    "resolve_hf_token",
]
