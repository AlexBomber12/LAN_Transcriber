from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import lan_transcriber.torch_safe_globals as torch_safe_globals


def _reload_module() -> ModuleType:
    return importlib.reload(torch_safe_globals)


def _install_fake_omegaconf(monkeypatch) -> tuple[type, type]:
    fake_omegaconf = ModuleType("omegaconf")
    fake_listconfig = ModuleType("omegaconf.listconfig")
    fake_dictconfig = ModuleType("omegaconf.dictconfig")

    class ListConfig:
        pass

    class DictConfig:
        pass

    fake_listconfig.ListConfig = ListConfig
    fake_dictconfig.DictConfig = DictConfig
    fake_omegaconf.listconfig = fake_listconfig
    fake_omegaconf.dictconfig = fake_dictconfig

    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)
    monkeypatch.setitem(sys.modules, "omegaconf.listconfig", fake_listconfig)
    monkeypatch.setitem(sys.modules, "omegaconf.dictconfig", fake_dictconfig)
    return ListConfig, DictConfig


def test_allowlist_adds_omegaconf_types_once(monkeypatch) -> None:
    module = _reload_module()
    calls: list[list[type]] = []

    def _add_safe_globals(items: list[type]) -> None:
        calls.append(list(items))

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(add_safe_globals=_add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config = _install_fake_omegaconf(monkeypatch)

    module.allowlist_omegaconf_for_weights_only()
    module.allowlist_omegaconf_for_weights_only()

    assert calls == [[list_config, dict_config]]


def test_allowlist_returns_when_torch_is_missing(monkeypatch) -> None:
    module = _reload_module()
    monkeypatch.setitem(sys.modules, "torch", None)

    module.allowlist_omegaconf_for_weights_only()


def test_allowlist_returns_when_add_safe_globals_is_missing(monkeypatch) -> None:
    module = _reload_module()
    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module.allowlist_omegaconf_for_weights_only()


def test_allowlist_returns_when_omegaconf_is_missing(monkeypatch) -> None:
    module = _reload_module()
    calls: list[list[type]] = []

    def _add_safe_globals(items: list[type]) -> None:
        calls.append(list(items))

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(add_safe_globals=_add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "omegaconf", None)
    monkeypatch.setitem(sys.modules, "omegaconf.listconfig", None)
    monkeypatch.setitem(sys.modules, "omegaconf.dictconfig", None)

    module.allowlist_omegaconf_for_weights_only()

    assert calls == []
