from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from lan_transcriber import gpu_policy


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    device_count: int = 0,
    mem_info: tuple[int, int] | None = None,
    mem_error: Exception | None = None,
) -> None:
    torch_mod = ModuleType("torch")

    def _mem_get_info(_device: int):
        if mem_error is not None:
            raise mem_error
        if mem_info is None:
            raise AttributeError("mem_get_info unavailable")
        return mem_info

    torch_mod.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
        mem_get_info=_mem_get_info,
    )
    torch_mod.version = SimpleNamespace(cuda="12.4")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


def test_normalizers_cover_valid_and_invalid_inputs() -> None:
    assert gpu_policy.normalize_device(None) == "auto"
    assert gpu_policy.normalize_device("   ") == "auto"
    assert gpu_policy.normalize_device(" GPU ") == "cuda"
    assert gpu_policy.normalize_device("cuda:1") == "cuda:1"
    with pytest.raises(ValueError, match="Device must be one of"):
        gpu_policy.normalize_device("metal")

    assert gpu_policy.normalize_scheduler_mode(None) == "auto"
    assert gpu_policy.normalize_scheduler_mode("   ") == "auto"
    assert gpu_policy.normalize_scheduler_mode(" parallel ") == "parallel"
    with pytest.raises(ValueError, match="Scheduler mode must be one of"):
        gpu_policy.normalize_scheduler_mode("burst")


def test_collect_cuda_runtime_facts_handles_import_failure_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    sys.modules.pop("torch", None)
    facts_without_torch = gpu_policy.collect_cuda_runtime_facts()
    assert facts_without_torch == gpu_policy.CudaRuntimeFacts(
        is_available=False,
        device_count=0,
        visible_devices="0,1",
        torch_cuda_version=None,
    )

    _install_fake_torch(monkeypatch, cuda_available=True, device_count=2)
    facts_with_torch = gpu_policy.collect_cuda_runtime_facts()
    assert facts_with_torch == gpu_policy.CudaRuntimeFacts(
        is_available=True,
        device_count=2,
        visible_devices="0,1",
        torch_cuda_version="12.4",
    )


def test_collect_cuda_runtime_facts_handles_missing_cuda_and_probe_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    torch_without_cuda = ModuleType("torch")
    torch_without_cuda.version = SimpleNamespace(cuda="12.4")
    monkeypatch.setitem(sys.modules, "torch", torch_without_cuda)
    assert gpu_policy.collect_cuda_runtime_facts() == gpu_policy.CudaRuntimeFacts(
        is_available=False,
        device_count=0,
        visible_devices="2",
        torch_cuda_version="12.4",
    )

    def _raise_is_available() -> bool:
        raise RuntimeError("probe failed")

    torch_probe_failure = ModuleType("torch")
    torch_probe_failure.cuda = SimpleNamespace(
        is_available=_raise_is_available,
        device_count=lambda: 9,
    )
    torch_probe_failure.version = SimpleNamespace(cuda="12.5")
    monkeypatch.setitem(sys.modules, "torch", torch_probe_failure)
    assert gpu_policy.collect_cuda_runtime_facts() == gpu_policy.CudaRuntimeFacts(
        is_available=False,
        device_count=0,
        visible_devices="2",
        torch_cuda_version="12.5",
    )


def test_device_resolution_and_parallel_safety_cover_edge_cases() -> None:
    cpu_facts = gpu_policy.CudaRuntimeFacts(
        is_available=False,
        device_count=0,
        visible_devices=None,
        torch_cuda_version=None,
    )
    zero_visible_gpu_facts = gpu_policy.CudaRuntimeFacts(
        is_available=True,
        device_count=0,
        visible_devices=None,
        torch_cuda_version="12.4",
    )
    gpu_facts = gpu_policy.CudaRuntimeFacts(
        is_available=True,
        device_count=2,
        visible_devices="0,1",
        torch_cuda_version="12.4",
    )

    assert gpu_policy.resolve_effective_device("auto", cuda_facts=cpu_facts) == "cpu"
    assert gpu_policy.resolve_effective_device("auto", cuda_facts=gpu_facts) == "cuda"
    assert gpu_policy.resolve_effective_device("cuda:1", cuda_facts=gpu_facts) == "cuda:1"
    assert gpu_policy.device_index("auto") is None
    assert gpu_policy.device_index("cpu") is None

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        gpu_policy.resolve_effective_device("cuda", cuda_facts=cpu_facts)
    with pytest.raises(RuntimeError, match="only 0 visible CUDA device"):
        gpu_policy.resolve_effective_device("cuda", cuda_facts=zero_visible_gpu_facts)
    with pytest.raises(RuntimeError, match="only 2 visible CUDA device"):
        gpu_policy.resolve_effective_device("cuda:2", cuda_facts=gpu_facts)

    assert gpu_policy.devices_share_gpu("cuda", "cuda:0")
    assert not gpu_policy.devices_share_gpu("cuda:0", "cuda:1")
    assert not gpu_policy.devices_share_gpu("cpu", "cuda")
    assert gpu_policy.parallel_devices_safe("cpu", "cuda")
    assert gpu_policy.parallel_devices_safe("cuda:0", "cuda:1")
    assert not gpu_policy.parallel_devices_safe("cpu", "cpu")
    assert not gpu_policy.parallel_devices_safe("cuda", "cuda:0")


def test_scheduler_decision_resolves_sequential_and_parallel_modes() -> None:
    single_gpu = gpu_policy.CudaRuntimeFacts(
        is_available=True,
        device_count=1,
        visible_devices="0",
        torch_cuda_version="12.4",
    )
    dual_gpu = gpu_policy.CudaRuntimeFacts(
        is_available=True,
        device_count=2,
        visible_devices="0,1",
        torch_cuda_version="12.4",
    )

    auto_same_gpu = gpu_policy.resolve_scheduler_decision(
        "auto",
        asr_device="auto",
        diarization_device="auto",
        diarization_is_heavy=True,
        cuda_facts=single_gpu,
    )
    assert auto_same_gpu.effective_mode == "sequential"
    assert auto_same_gpu.reason == "auto_shared_or_single_device"

    auto_safe_parallel = gpu_policy.resolve_scheduler_decision(
        "auto",
        asr_device="cuda:0",
        diarization_device="cuda:1",
        diarization_is_heavy=True,
        cuda_facts=dual_gpu,
    )
    assert auto_safe_parallel.effective_mode == "parallel"
    assert auto_safe_parallel.reason == "auto_parallel_safe"

    forced_parallel_unsafe = gpu_policy.resolve_scheduler_decision(
        "parallel",
        asr_device="cuda",
        diarization_device="cuda:0",
        diarization_is_heavy=True,
        cuda_facts=single_gpu,
    )
    assert forced_parallel_unsafe.effective_mode == "sequential"
    assert forced_parallel_unsafe.reason == "parallel_not_safe"

    forced_sequential = gpu_policy.resolve_scheduler_decision(
        "sequential",
        asr_device="cpu",
        diarization_device="cuda:1",
        diarization_is_heavy=True,
        cuda_facts=dual_gpu,
    )
    assert forced_sequential.effective_mode == "sequential"
    assert forced_sequential.reason == "forced_sequential"

    auto_non_heavy = gpu_policy.resolve_scheduler_decision(
        "auto",
        asr_device="cpu",
        diarization_device="cuda",
        diarization_is_heavy=False,
        cuda_facts=single_gpu,
    )
    assert auto_non_heavy.effective_mode == "sequential"


def test_cuda_memory_info_and_gpu_oom_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert gpu_policy.cuda_memory_info("cpu") is None

    _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        device_count=1,
        mem_info=(123, 456),
    )
    assert gpu_policy.cuda_memory_info("cuda") == (123, 456)

    _install_fake_torch(
        monkeypatch,
        cuda_available=True,
        device_count=1,
        mem_error=RuntimeError("broken"),
    )
    assert gpu_policy.cuda_memory_info("cuda:0") is None

    torch_without_mem_get_info = ModuleType("torch")
    torch_without_mem_get_info.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    torch_without_mem_get_info.version = SimpleNamespace(cuda="12.4")
    monkeypatch.setitem(sys.modules, "torch", torch_without_mem_get_info)
    assert gpu_policy.cuda_memory_info("cuda") is None

    torch_bad_mem_info = ModuleType("torch")
    torch_bad_mem_info.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda _device: ("oops", object()),
    )
    torch_bad_mem_info.version = SimpleNamespace(cuda="12.4")
    monkeypatch.setitem(sys.modules, "torch", torch_bad_mem_info)
    assert gpu_policy.cuda_memory_info("cuda") is None

    assert gpu_policy.is_gpu_oom_error(RuntimeError("CUDA out of memory"))
    assert gpu_policy.is_gpu_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert not gpu_policy.is_gpu_oom_error(RuntimeError("network timeout"))
