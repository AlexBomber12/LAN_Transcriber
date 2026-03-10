from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

_DEVICE_RE = re.compile(r"^(auto|cpu|cuda(?::(?P<index>\d+))?)$")
_GPU_OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cuda failed with error out of memory",
    "cublas_status_alloc_failed",
    "cuda malloc",
    "hip out of memory",
    "outofmemoryerror",
)


@dataclass(frozen=True)
class CudaRuntimeFacts:
    is_available: bool
    device_count: int
    visible_devices: str | None
    torch_cuda_version: str | None


@dataclass(frozen=True)
class SchedulerDecision:
    requested_mode: str
    effective_mode: str
    asr_device: str
    diarization_device: str
    reason: str
    cuda_facts: CudaRuntimeFacts


def normalize_device(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower()
    if not normalized:
        return "auto"
    if normalized == "gpu":
        normalized = "cuda"
    if _DEVICE_RE.fullmatch(normalized) is None:
        raise ValueError(
            "Device must be one of auto, cpu, cuda, or cuda:<index>."
        )
    return normalized


def normalize_scheduler_mode(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower()
    if not normalized:
        return "auto"
    if normalized not in {"auto", "sequential", "parallel"}:
        raise ValueError("Scheduler mode must be one of auto, sequential, parallel.")
    return normalized


def collect_cuda_runtime_facts() -> CudaRuntimeFacts:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip() or None
    try:
        import torch
    except Exception:
        return CudaRuntimeFacts(
            is_available=False,
            device_count=0,
            visible_devices=visible_devices,
            torch_cuda_version=None,
        )

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return CudaRuntimeFacts(
            is_available=False,
            device_count=0,
            visible_devices=visible_devices,
            torch_cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
        )

    try:
        is_available = bool(cuda.is_available())
    except Exception:
        is_available = False
    try:
        device_count = int(cuda.device_count()) if is_available else 0
    except Exception:
        device_count = 0
    return CudaRuntimeFacts(
        is_available=is_available,
        device_count=max(device_count, 0),
        visible_devices=visible_devices,
        torch_cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
    )


def is_gpu_device(device: str | None) -> bool:
    return normalize_device(device) not in {"auto", "cpu"}


def device_index(device: str | None) -> int | None:
    normalized = normalize_device(device)
    if normalized in {"auto", "cpu"}:
        return None
    if normalized == "cuda":
        return 0
    return int(normalized.split(":", 1)[1])


def resolve_effective_device(
    requested_device: str | None,
    *,
    cuda_facts: CudaRuntimeFacts | None = None,
    label: str = "device",
) -> str:
    normalized = normalize_device(requested_device)
    facts = cuda_facts or collect_cuda_runtime_facts()
    if normalized == "auto":
        return "cuda" if facts.is_available else "cpu"
    if normalized == "cpu":
        return "cpu"
    if not facts.is_available:
        raise RuntimeError(
            f"Requested {label} {normalized} but CUDA is unavailable."
        )
    gpu_index = device_index(normalized)
    if (
        gpu_index is not None
        and gpu_index >= facts.device_count
    ):
        raise RuntimeError(
            f"Requested {label} {normalized} but only "
            f"{facts.device_count} visible CUDA device(s) are available."
        )
    return normalized


def devices_share_gpu(left: str | None, right: str | None) -> bool:
    if not is_gpu_device(left) or not is_gpu_device(right):
        return False
    return device_index(left) == device_index(right)


def parallel_devices_safe(
    asr_device: str | None,
    diarization_device: str | None,
) -> bool:
    asr_gpu = is_gpu_device(asr_device)
    diar_gpu = is_gpu_device(diarization_device)
    if asr_gpu and diar_gpu:
        return not devices_share_gpu(asr_device, diarization_device)
    if asr_gpu != diar_gpu:
        return True
    return False


def resolve_scheduler_decision(
    requested_mode: str | None,
    *,
    asr_device: str | None,
    diarization_device: str | None,
    diarization_is_heavy: bool,
    cuda_facts: CudaRuntimeFacts | None = None,
) -> SchedulerDecision:
    facts = cuda_facts or collect_cuda_runtime_facts()
    normalized_mode = normalize_scheduler_mode(requested_mode)
    effective_asr = resolve_effective_device(
        asr_device,
        cuda_facts=facts,
        label="ASR device",
    )
    effective_diarization = resolve_effective_device(
        diarization_device,
        cuda_facts=facts,
        label="diarization device",
    )
    safe_parallel = diarization_is_heavy and parallel_devices_safe(
        effective_asr,
        effective_diarization,
    )

    if normalized_mode == "sequential":
        return SchedulerDecision(
            requested_mode=normalized_mode,
            effective_mode="sequential",
            asr_device=effective_asr,
            diarization_device=effective_diarization,
            reason="forced_sequential",
            cuda_facts=facts,
        )
    if normalized_mode == "parallel":
        return SchedulerDecision(
            requested_mode=normalized_mode,
            effective_mode="parallel" if safe_parallel else "sequential",
            asr_device=effective_asr,
            diarization_device=effective_diarization,
            reason="forced_parallel" if safe_parallel else "parallel_not_safe",
            cuda_facts=facts,
        )
    return SchedulerDecision(
        requested_mode=normalized_mode,
        effective_mode="parallel" if safe_parallel else "sequential",
        asr_device=effective_asr,
        diarization_device=effective_diarization,
        reason="auto_parallel_safe" if safe_parallel else "auto_shared_or_single_device",
        cuda_facts=facts,
    )


def cuda_memory_info(device: str | None) -> tuple[int, int] | None:
    if not is_gpu_device(device):
        return None
    try:
        import torch
    except Exception:
        return None
    cuda = getattr(torch, "cuda", None)
    mem_get_info = getattr(cuda, "mem_get_info", None)
    if cuda is None or mem_get_info is None:
        return None
    target_index = device_index(device)
    target: Any = target_index if target_index is not None else 0
    try:
        free_bytes, total_bytes = mem_get_info(target)
    except Exception:
        return None
    try:
        return int(free_bytes), int(total_bytes)
    except (TypeError, ValueError):
        return None


def is_gpu_oom_error(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in _GPU_OOM_MARKERS)


__all__ = [
    "CudaRuntimeFacts",
    "SchedulerDecision",
    "collect_cuda_runtime_facts",
    "cuda_memory_info",
    "device_index",
    "devices_share_gpu",
    "is_gpu_device",
    "is_gpu_oom_error",
    "normalize_device",
    "normalize_scheduler_mode",
    "parallel_devices_safe",
    "resolve_effective_device",
    "resolve_scheduler_decision",
]
