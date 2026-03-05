from __future__ import annotations

import contextlib
import importlib
import logging
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


# --- (a) safe_globals(list) works ---


def test_context_uses_torch_safe_globals_when_available(monkeypatch) -> None:
    module = _reload_module()
    calls: list[list[object]] = []
    add_calls: list[list[object]] = []
    entered: list[str] = []

    @contextlib.contextmanager
    def _safe_globals(items):
        calls.append(list(items) if isinstance(items, list) else [items])
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    def _add_safe_globals(items) -> None:
        add_calls.append(list(items) if isinstance(items, list) else [items])

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(safe_globals=_safe_globals, add_safe_globals=_add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config, container_metadata = _install_fake_omegaconf(monkeypatch)

    with module.omegaconf_safe_globals_for_torch_load():
        entered.append("body")

    assert calls == [[list_config, dict_config, container_metadata]]
    assert add_calls == []
    assert entered == ["enter", "body", "exit"]


# --- (b) safe_globals(list) TypeError, safe_globals(dict) works ---


def test_safe_globals_list_type_error_falls_back_to_dict(monkeypatch) -> None:
    module = _reload_module()
    dict_calls: list[dict[str, object]] = []
    entered: list[str] = []

    def _safe_globals(items):
        """Reject list, accept dict — TypeError at *call* time (like real torch)."""
        if isinstance(items, list):
            raise TypeError("list form unsupported")

        @contextlib.contextmanager
        def _cm():
            dict_calls.append(dict(items))
            entered.append("enter")
            try:
                yield
            finally:
                entered.append("exit")

        return _cm()

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(safe_globals=_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config, container_metadata = _install_fake_omegaconf(monkeypatch)

    with module.omegaconf_safe_globals_for_torch_load():
        entered.append("body")

    assert len(dict_calls) == 1
    assert dict_calls[0] == {
        "omegaconf.listconfig.ListConfig": list_config,
        "omegaconf.dictconfig.DictConfig": dict_config,
        "omegaconf.base.ContainerMetadata": container_metadata,
    }
    assert entered == ["enter", "body", "exit"]


# --- (c) add_safe_globals(list) works ---


def test_context_falls_back_to_add_safe_globals_and_supports_extra_fqns(monkeypatch) -> None:
    module = _reload_module()
    calls: list[list[object]] = []

    fake_custom = ModuleType("omegaconf.custom")

    class CustomNode:
        pass

    fake_custom.CustomNode = CustomNode
    monkeypatch.setitem(sys.modules, "omegaconf.custom", fake_custom)

    def _add_safe_globals(items) -> None:
        calls.append(list(items) if isinstance(items, list) else [items])

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


# --- (d) add_safe_globals(list) TypeError, add_safe_globals(dict) works ---


def test_add_safe_globals_list_type_error_falls_back_to_dict(monkeypatch) -> None:
    module = _reload_module()
    dict_calls: list[dict[str, object]] = []

    def _add_safe_globals(items) -> None:
        if isinstance(items, list):
            raise TypeError("list form unsupported")
        dict_calls.append(dict(items))

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(add_safe_globals=_add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config, container_metadata = _install_fake_omegaconf(monkeypatch)

    with module.omegaconf_safe_globals_for_torch_load():
        pass

    assert len(dict_calls) == 1
    assert dict_calls[0] == {
        "omegaconf.listconfig.ListConfig": list_config,
        "omegaconf.dictconfig.DictConfig": dict_config,
        "omegaconf.base.ContainerMetadata": container_metadata,
    }


# --- (e) torch missing -> no crash ---


def test_context_handles_missing_torch(monkeypatch) -> None:
    module = _reload_module()

    monkeypatch.setitem(sys.modules, "torch", None)
    with module.omegaconf_safe_globals_for_torch_load():
        pass


# --- (e/f) torch.serialization missing + omegaconf missing -> no crash ---


def test_context_handles_missing_torch_serialization_or_omegaconf(monkeypatch) -> None:
    module = _reload_module()

    # torch with no serialization attribute
    fake_torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with module.omegaconf_safe_globals_for_torch_load():
        pass

    # serialization exists but no symbols importable (omegaconf blocked)
    fake_torch.serialization = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "omegaconf", None)
    monkeypatch.setitem(sys.modules, "omegaconf.listconfig", None)
    monkeypatch.setitem(sys.modules, "omegaconf.dictconfig", None)
    monkeypatch.setitem(sys.modules, "omegaconf.base", None)
    with module.omegaconf_safe_globals_for_torch_load():
        pass


# --- safe_globals and add_safe_globals both error (non-TypeError) ---


def test_context_handles_safe_globals_factory_and_add_safe_globals_errors(monkeypatch, caplog) -> None:
    module = _reload_module()
    add_calls: list[list[object]] = []

    def _broken_safe_globals(_items):
        raise RuntimeError("no safe globals")

    def _add_safe_globals(items) -> None:
        add_calls.append(list(items) if isinstance(items, list) else [items])
        raise RuntimeError("cannot add")

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(
        safe_globals=_broken_safe_globals,
        add_safe_globals=_add_safe_globals,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _install_fake_omegaconf(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="lan_transcriber.torch_safe_globals"):
        with module.omegaconf_safe_globals_for_torch_load(extra_fqns=["omegaconf."]):
            pass

    assert len(add_calls) == 1
    # Warnings should be logged for both failures
    assert any("safe-globals" in r.message for r in caplog.records)


# --- safe_globals(list) and safe_globals(dict) both TypeError ---


def test_safe_globals_both_forms_type_error_falls_through_to_add(monkeypatch, caplog) -> None:
    module = _reload_module()
    add_calls: list[list[object]] = []

    def _broken_safe_globals(_items):
        raise TypeError("unsupported arg type")

    def _add_safe_globals(items) -> None:
        add_calls.append(list(items) if isinstance(items, list) else [items])

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(
        safe_globals=_broken_safe_globals,
        add_safe_globals=_add_safe_globals,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config, container_metadata = _install_fake_omegaconf(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="lan_transcriber.torch_safe_globals"):
        with module.omegaconf_safe_globals_for_torch_load():
            pass

    # safe_globals failed for both forms; fell through to add_safe_globals
    assert add_calls == [[list_config, dict_config, container_metadata]]
    assert any("rejected both" in r.message for r in caplog.records)


# --- add_safe_globals is None (serialization exists but has neither API) ---


def test_context_handles_no_add_safe_globals(monkeypatch) -> None:
    module = _reload_module()

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _install_fake_omegaconf(monkeypatch)

    with module.omegaconf_safe_globals_for_torch_load():
        pass


# --- (g) parser extracts OmegaConf FQN and rejects non-omegaconf FQN ---


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
    assert module.parse_unsupported_global_fqn("") is None
    assert module.parse_unsupported_global_fqn("Unsupported global: omegaconf.dictconfig.DictConfig") == "omegaconf.dictconfig.DictConfig"


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


# --- (h) auto-retry path succeeds on second attempt after allowlisting specific FQN ---


def test_auto_retry_pattern_with_extra_fqn(monkeypatch) -> None:
    """Simulate orchestrator retry: first load fails with unknown FQN, second succeeds."""
    module = _reload_module()
    call_log: list[list[object]] = []

    @contextlib.contextmanager
    def _safe_globals(items):
        call_log.append(list(items) if isinstance(items, list) else sorted(items.keys()) if isinstance(items, dict) else [items])
        yield

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(safe_globals=_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    list_config, dict_config, container_metadata = _install_fake_omegaconf(monkeypatch)

    # Add an extra omegaconf symbol that we'll "discover" via error parsing
    fake_nodes = ModuleType("omegaconf.nodes")

    class ValueNode:
        pass

    fake_nodes.ValueNode = ValueNode
    monkeypatch.setitem(sys.modules, "omegaconf.nodes", fake_nodes)

    # First attempt: standard allowlist
    with module.omegaconf_safe_globals_for_torch_load():
        pass

    assert call_log == [[list_config, dict_config, container_metadata]]

    # Simulate error from load
    error = pickle.UnpicklingError(
        "Unsupported global: GLOBAL omegaconf.nodes.ValueNode"
    )
    retry_fqn = module.unsupported_global_omegaconf_fqn_from_error(error)
    assert retry_fqn == "omegaconf.nodes.ValueNode"

    # Second attempt with extra FQN
    call_log.clear()
    with module.omegaconf_safe_globals_for_torch_load(extra_fqns=[retry_fqn]):
        pass

    assert call_log == [[list_config, dict_config, container_metadata, ValueNode]]


# --- _try_api: add_safe_globals both forms TypeError (log warning, no crash) ---


def test_add_safe_globals_both_forms_type_error_logs_warning(monkeypatch, caplog) -> None:
    module = _reload_module()

    def _add_safe_globals(_items) -> None:
        raise TypeError("nope")

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(add_safe_globals=_add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _install_fake_omegaconf(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="lan_transcriber.torch_safe_globals"):
        with module.omegaconf_safe_globals_for_torch_load():
            pass

    assert any("rejected both" in r.message for r in caplog.records)


def _install_fake_diarization_symbols(monkeypatch, module) -> dict[str, object]:
    class TorchVersion:
        pass

    class ListConfig:
        pass

    class DictConfig:
        pass

    class ContainerMetadata:
        pass

    class Specifications:
        pass

    symbols: dict[str, object] = {
        "torch.torch_version.TorchVersion": TorchVersion,
        "omegaconf.listconfig.ListConfig": ListConfig,
        "omegaconf.dictconfig.DictConfig": DictConfig,
        "omegaconf.base.ContainerMetadata": ContainerMetadata,
        "pyannote.audio.core.task.Specifications": Specifications,
        "pyannote.audio.core.task.ExtraSafe": type("ExtraSafe", (), {}),
    }
    monkeypatch.setattr(module, "_import_symbol", lambda fqn: symbols.get(fqn))
    return symbols


def test_diarization_context_safe_globals_dict_fallback(monkeypatch) -> None:
    module = _reload_module()
    dict_calls: list[dict[str, object]] = []
    entered: list[str] = []

    def _safe_globals(items):
        if isinstance(items, list):
            raise TypeError("list form unsupported")

        @contextlib.contextmanager
        def _cm():
            dict_calls.append(dict(items))
            entered.append("enter")
            try:
                yield
            finally:
                entered.append("exit")

        return _cm()

    fake_torch = ModuleType("torch")
    fake_torch.serialization = SimpleNamespace(safe_globals=_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    symbols = _install_fake_diarization_symbols(monkeypatch, module)

    with module.diarization_safe_globals_for_torch_load(
        extra_fqns=["pyannote.audio.core.task.ExtraSafe", "builtins.eval"]
    ):
        entered.append("body")

    assert len(dict_calls) == 1
    assert dict_calls[0] == {
        "torch.torch_version.TorchVersion": symbols["torch.torch_version.TorchVersion"],
        "omegaconf.listconfig.ListConfig": symbols["omegaconf.listconfig.ListConfig"],
        "omegaconf.dictconfig.DictConfig": symbols["omegaconf.dictconfig.DictConfig"],
        "omegaconf.base.ContainerMetadata": symbols["omegaconf.base.ContainerMetadata"],
        "pyannote.audio.core.task.Specifications": symbols["pyannote.audio.core.task.Specifications"],
        "pyannote.audio.core.task.ExtraSafe": symbols["pyannote.audio.core.task.ExtraSafe"],
    }
    assert entered == ["enter", "body", "exit"]


def test_diarization_parser_and_import_filter() -> None:
    module = _reload_module()

    assert module.is_trusted_diarization_global_fqn("pyannote.audio.core.task.Specifications")
    assert module.is_trusted_diarization_global_fqn("omegaconf.base.ContainerMetadata")
    assert module.is_trusted_diarization_global_fqn("torch.torch_version.TorchVersion")
    assert not module.is_trusted_diarization_global_fqn("builtins.eval")

    assert (
        module.parse_diarization_unsupported_global_fqn(
            "Unsupported global: GLOBAL torch.torch_version.TorchVersion"
        )
        == "torch.torch_version.TorchVersion"
    )
    assert (
        module.parse_diarization_unsupported_global_fqn(
            "Unsupported global: GLOBAL pyannote.audio.core.task.Specifications"
        )
        == "pyannote.audio.core.task.Specifications"
    )
    assert module.parse_diarization_unsupported_global_fqn("Unsupported global: GLOBAL builtins.eval") is None
    assert (
        module.unsupported_global_diarization_fqn_from_error(
            pickle.UnpicklingError("Unsupported global: GLOBAL omegaconf.listconfig.ListConfig")
        )
        == "omegaconf.listconfig.ListConfig"
    )
    assert module.unsupported_global_diarization_fqn_from_error(ValueError("Unsupported global: GLOBAL omegaconf.listconfig.ListConfig")) is None


def test_collect_symbols_skips_untrusted_base_entries(monkeypatch) -> None:
    module = _reload_module()
    imported: list[str] = []
    monkeypatch.setattr(
        module,
        "_import_symbol",
        lambda fqn: imported.append(fqn) or object(),
    )
    pairs = module._collect_symbols(
        ("builtins.eval",),
        None,
        trusted_fqn=lambda fqn: fqn.startswith("omegaconf."),
    )
    assert pairs == []
    assert imported == []


def test_import_trusted_diarization_symbol_only_imports_trusted(monkeypatch) -> None:
    module = _reload_module()
    imported: list[str] = []
    monkeypatch.setattr(
        module,
        "_import_symbol",
        lambda fqn: imported.append(fqn) or "ok",
    )
    assert module.import_trusted_diarization_symbol("builtins.eval") is None
    assert module.import_trusted_diarization_symbol("pyannote.audio.core.task.Specifications") == "ok"
    assert imported == ["pyannote.audio.core.task.Specifications"]
