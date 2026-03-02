from __future__ import annotations

import contextlib
import importlib
import pickle
import sys
from types import ModuleType, SimpleNamespace

import lan_transcriber.torch_safe_globals as torch_safe_globals


def _reload_module() -> ModuleType:
    return importlib.reload(torch_safe_globals)


def _install_fake_omegaconf(monkeypatch) -> tuple[type, type, type]:
    fake_omegaconf = ModuleType("omegaconf")
    fake_listconfig = ModuleType("omegaconf.listconfig")
    fake_dictconfig = ModuleType("omegaconf.dictconfig")
    fake_base = ModuleType("omegaconf.base")

    class ListConfig:
        pass

    class DictConfig:
        pass

    class ContainerMetadata:
        pass

    fake_listconfig.ListConfig = ListConfig
    fake_dictconfig.DictConfig = DictConfig
    fake_base.ContainerMetadata = ContainerMetadata
    fake_omegaconf.listconfig = fake_listconfig
    fake_omegaconf.dictconfig = fake_dictconfig
    fake_omegaconf.base = fake_base

    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)
    monkeypatch.setitem(sys.modules, "omegaconf.listconfig", fake_listconfig)
    monkeypatch.setitem(sys.modules, "omegaconf.dictconfig", fake_dictconfig)
    monkeypatch.setitem(sys.modules, "omegaconf.base", fake_base)
    return ListConfig, DictConfig, ContainerMetadata


def test_context_uses_torch_safe_globals_when_available(monkeypatch) -> None:
    module = _reload_module()
    calls: list[list[object]] = []
    add_calls: list[list[object]] = []
    entered: list[str] = []

    @contextlib.contextmanager
    def _safe_globals(items: list[object]):
        calls.append(list(items))
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    def _add_safe_globals(items: list[object]) -> None:
        add_calls.append(list(items))

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(safe_globals=_safe_globals, add_safe_globals=_add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config, container_metadata = _install_fake_omegaconf(monkeypatch)

    with module.omegaconf_safe_globals_for_torch_load():
        entered.append("body")

    assert calls == [[list_config, dict_config, container_metadata]]
    assert add_calls == []
    assert entered == ["enter", "body", "exit"]


def test_context_falls_back_to_add_safe_globals_and_supports_extra_fqns(monkeypatch) -> None:
    module = _reload_module()
    calls: list[list[object]] = []

    fake_custom = ModuleType("omegaconf.custom")

    class CustomNode:
        pass

    fake_custom.CustomNode = CustomNode
    monkeypatch.setitem(sys.modules, "omegaconf.custom", fake_custom)

    def _add_safe_globals(items: list[object]) -> None:
        calls.append(list(items))

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(add_safe_globals=_add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config, container_metadata = _install_fake_omegaconf(monkeypatch)

    with module.omegaconf_safe_globals_for_torch_load(
        extra_fqns=[
            "not.allowed.Symbol",
            "omegaconf.listconfig.ListConfig",
            "omegaconf.custom.CustomNode",
        ]
    ):
        pass

    assert calls == [[list_config, dict_config, container_metadata, CustomNode]]


def test_context_handles_missing_torch_serialization_or_omegaconf(monkeypatch) -> None:
    module = _reload_module()

    monkeypatch.setitem(sys.modules, "torch", None)
    with module.omegaconf_safe_globals_for_torch_load():
        pass

    fake_torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with module.omegaconf_safe_globals_for_torch_load():
        pass

    fake_torch.serialization = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "omegaconf", None)
    monkeypatch.setitem(sys.modules, "omegaconf.listconfig", None)
    monkeypatch.setitem(sys.modules, "omegaconf.dictconfig", None)
    monkeypatch.setitem(sys.modules, "omegaconf.base", None)
    with module.omegaconf_safe_globals_for_torch_load():
        pass


def test_context_handles_safe_globals_factory_and_add_safe_globals_errors(monkeypatch) -> None:
    module = _reload_module()
    add_calls: list[list[object]] = []

    def _broken_safe_globals(_items: list[object]):
        raise RuntimeError("no safe globals")

    def _add_safe_globals(items: list[object]) -> None:
        add_calls.append(list(items))
        raise RuntimeError("cannot add")

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(
        safe_globals=_broken_safe_globals,
        add_safe_globals=_add_safe_globals,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _install_fake_omegaconf(monkeypatch)

    with module.omegaconf_safe_globals_for_torch_load(extra_fqns=["omegaconf."]):
        pass

    assert len(add_calls) == 1


def test_parse_unsupported_global_fqn_accepts_only_omegaconf() -> None:
    module = _reload_module()

    assert (
        module.parse_unsupported_global_fqn(
            "Weights only load failed. Unsupported global: GLOBAL omegaconf.base.ContainerMetadata"
        )
        == "omegaconf.base.ContainerMetadata"
    )
    assert module.parse_unsupported_global_fqn("Unsupported global: GLOBAL builtins.list") is None
    assert module.parse_unsupported_global_fqn("no unsupported global here") is None


def test_unsupported_global_omegaconf_fqn_from_error_filters_error_types() -> None:
    module = _reload_module()

    err = pickle.UnpicklingError(
        "Weights only load failed. Unsupported global: GLOBAL omegaconf.listconfig.ListConfig"
    )
    assert module.unsupported_global_omegaconf_fqn_from_error(err) == "omegaconf.listconfig.ListConfig"

    runtime_err = RuntimeError(
        "Weights only load failed. Unsupported global: GLOBAL omegaconf.dictconfig.DictConfig"
    )
    assert module.unsupported_global_omegaconf_fqn_from_error(runtime_err) == "omegaconf.dictconfig.DictConfig"

    assert module.unsupported_global_omegaconf_fqn_from_error(ValueError("Unsupported global: GLOBAL omegaconf.x")) is None
