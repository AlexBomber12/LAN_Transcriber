from __future__ import annotations

import builtins
import contextlib
import logging
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest

from lan_app import diarization_loader


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, impl):
    pyannote_audio = ModuleType("pyannote.audio")

    class _Pipeline:
        @staticmethod
        def from_pretrained(name: str, **kwargs):
            return impl(name, **kwargs)

    pyannote_audio.Pipeline = _Pipeline  # type: ignore[attr-defined]
    pyannote_pkg = ModuleType("pyannote")
    pyannote_pkg.audio = pyannote_audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool):
    torch_mod = ModuleType("torch")
    torch_mod.cuda = SimpleNamespace(is_available=lambda: cuda_available)  # type: ignore[attr-defined]
    torch_mod.device = lambda name: f"device:{name}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


def test_resolve_diarization_model_id_prefers_explicit_then_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LAN_DIARIZATION_MODEL_ID", "env/repo@main")
    assert diarization_loader.resolve_diarization_model_id(" explicit/repo@v1 ") == "explicit/repo@v1"
    assert diarization_loader.resolve_diarization_model_id() == "env/repo@main"

    monkeypatch.setenv("LAN_DIARIZATION_MODEL_ID", "   ")
    assert (
        diarization_loader.resolve_diarization_model_id()
        == diarization_loader.DEFAULT_DIARIZATION_MODEL_ID
    )


def test_resolve_hf_token_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf-env")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-env")
    assert diarization_loader.resolve_hf_token(" explicit ") == "explicit"
    assert diarization_loader.resolve_hf_token() == "hf-env"

    monkeypatch.setenv("HF_TOKEN", "  ")
    assert diarization_loader.resolve_hf_token() == "hub-env"

    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert diarization_loader.resolve_hf_token() is None


def test_classify_pipeline_load_error_paths():
    class _RevisionNotFoundError(RuntimeError):
        pass

    class _Unauthorized(RuntimeError):
        status_code = 401

    class _ForbiddenByResponse(RuntimeError):
        response = SimpleNamespace(status_code=403)

    assert (
        diarization_loader.classify_pipeline_load_error(_RevisionNotFoundError("x"))
        == "revision_not_found"
    )
    assert diarization_loader.classify_pipeline_load_error(_Unauthorized("x")) == "gated_access"
    assert (
        diarization_loader.classify_pipeline_load_error(_ForbiddenByResponse("x"))
        == "gated_access"
    )
    assert (
        diarization_loader.classify_pipeline_load_error(RuntimeError("gated model access denied"))
        == "gated_access"
    )
    assert diarization_loader.classify_pipeline_load_error(RuntimeError("boom")) == "other"


def test_extract_repo_hints_returns_sorted_unique_list():
    exc = RuntimeError("Need access for pyannote/segmentation and pyannote/speaker-diarization and pyannote/segmentation")
    assert diarization_loader.extract_repo_hints(exc) == [
        "pyannote/segmentation",
        "pyannote/speaker-diarization",
    ]


def test_load_pyannote_pipeline_handles_token_keyword_mismatch(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict[str, object]]] = []

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        if "token" in kwargs:
            raise TypeError("got an unexpected keyword argument 'token'")
        return lambda payload: payload

    _install_fake_pipeline(monkeypatch, _impl)
    model = diarization_loader.load_pyannote_pipeline(
        model_id="repo/test@rev1",
        token="token-1",
    )
    assert callable(model)
    assert calls == [
        ("repo/test", {"revision": "rev1", "token": "token-1"}),
        ("repo/test", {"revision": "rev1"}),
    ]


def test_load_pyannote_pipeline_continues_candidates_after_tokenless_retry_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, dict[str, object]]] = []

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        if len(calls) == 1:
            raise TypeError("got an unexpected keyword argument 'token'")
        if len(calls) == 2:
            raise RuntimeError("revision failed")
        return lambda payload: payload

    _install_fake_pipeline(monkeypatch, _impl)
    model = diarization_loader.load_pyannote_pipeline(
        model_id="repo/test@rev1",
        token="token-1",
    )
    assert callable(model)
    assert calls == [
        ("repo/test", {"revision": "rev1", "token": "token-1"}),
        ("repo/test", {"revision": "rev1"}),
        ("repo/test@rev1", {"token": "token-1"}),
    ]


def test_load_pyannote_pipeline_uses_ordered_candidates_and_raises_last_error(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, dict[str, object]]] = []

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        if len(calls) == 1:
            raise RuntimeError("first")
        if len(calls) == 2:
            raise RuntimeError("second")
        raise RuntimeError("third")

    _install_fake_pipeline(monkeypatch, _impl)

    with pytest.raises(RuntimeError, match="third"):
        diarization_loader.load_pyannote_pipeline(model_id="repo/ordered@bad")

    assert calls == [
        ("repo/ordered", {"revision": "bad"}),
        ("repo/ordered@bad", {}),
        ("repo/ordered", {}),
    ]


def test_load_pyannote_pipeline_preserves_non_token_type_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, dict[str, object]]] = []

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        raise TypeError("from_pretrained() missing required positional argument")

    _install_fake_pipeline(monkeypatch, _impl)
    with pytest.raises(TypeError, match="missing required positional argument"):
        diarization_loader.load_pyannote_pipeline(model_id="repo/type-error@rev")
    assert calls == [
        ("repo/type-error", {"revision": "rev"}),
        ("repo/type-error@rev", {}),
        ("repo/type-error", {}),
    ]


def test_load_pyannote_pipeline_supports_unversioned_model_and_rejects_non_callable(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, dict[str, object]]] = []

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        return object()

    _install_fake_pipeline(monkeypatch, _impl)
    with pytest.raises(TypeError, match="must be callable"):
        diarization_loader.load_pyannote_pipeline(model_id="repo/plain")
    assert calls == [("repo/plain", {})]


def test_load_pyannote_pipeline_uses_env_model_override(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict[str, object]]] = []

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        return lambda payload: payload

    _install_fake_pipeline(monkeypatch, _impl)
    monkeypatch.setenv("LAN_DIARIZATION_MODEL_ID", "repo/from-env@rev-env")
    model = diarization_loader.load_pyannote_pipeline()
    assert callable(model)
    assert calls == [("repo/from-env", {"revision": "rev-env"})]


def test_load_pyannote_pipeline_rejects_empty_model_and_import_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(ValueError, match="cannot be empty"):
        diarization_loader.load_pyannote_pipeline(model_id="@main")

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyannote.audio":
            err = ModuleNotFoundError("No module named 'pyannote'")
            err.name = "pyannote"
            raise err
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("pyannote", None)
    sys.modules.pop("pyannote.audio", None)
    with pytest.raises(ModuleNotFoundError, match="pyannote"):
        diarization_loader.load_pyannote_pipeline(model_id="repo/import@1")


def test_load_pyannote_pipeline_moves_to_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    moved_to: list[object] = []

    class _Model:
        def __call__(self, payload):
            return payload

        def to(self, device):
            moved_to.append(device)

    _install_fake_pipeline(monkeypatch, lambda *_a, **_k: _Model())
    _install_fake_torch(monkeypatch, cuda_available=True)

    with caplog.at_level(logging.INFO, logger=diarization_loader.__name__):
        model = diarization_loader.load_pyannote_pipeline(model_id="repo/cuda")

    assert callable(model)
    assert moved_to == ["device:cuda"]
    assert "Pyannote diarization device: cuda" in caplog.text


def test_load_pyannote_pipeline_stays_on_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    moved_to: list[object] = []

    class _Model:
        def __call__(self, payload):
            return payload

        def to(self, device):
            moved_to.append(device)

    _install_fake_pipeline(monkeypatch, lambda *_a, **_k: _Model())
    _install_fake_torch(monkeypatch, cuda_available=False)

    with caplog.at_level(logging.INFO, logger=diarization_loader.__name__):
        model = diarization_loader.load_pyannote_pipeline(model_id="repo/cpu")

    assert callable(model)
    assert moved_to == []
    assert "Pyannote diarization device: cpu" in caplog.text


def test_load_pyannote_pipeline_falls_back_to_cpu_when_cuda_move_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    class _Model:
        def __call__(self, payload):
            return payload

        def to(self, _device):
            raise RuntimeError("cuda move failed")

    _install_fake_pipeline(monkeypatch, lambda *_a, **_k: _Model())
    _install_fake_torch(monkeypatch, cuda_available=True)

    with caplog.at_level(logging.INFO, logger=diarization_loader.__name__):
        model = diarization_loader.load_pyannote_pipeline(model_id="repo/fallback")

    assert callable(model)
    assert (
        "Failed to move pyannote diarization pipeline to CUDA; continuing on CPU"
        in caplog.text
    )
    assert "Pyannote diarization device: cpu" in caplog.text


def test_load_pyannote_pipeline_stays_on_cpu_when_torch_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    class _Model:
        def __call__(self, payload):
            return payload

        def to(self, _device):
            raise AssertionError("to() should not be called without torch")

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            err = ModuleNotFoundError("No module named 'torch'")
            err.name = "torch"
            raise err
        return real_import(name, globals, locals, fromlist, level)

    _install_fake_pipeline(monkeypatch, lambda *_a, **_k: _Model())
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    sys.modules.pop("torch", None)

    with caplog.at_level(logging.INFO, logger=diarization_loader.__name__):
        model = diarization_loader.load_pyannote_pipeline(model_id="repo/no-torch")

    assert callable(model)
    assert "Pyannote diarization device: cpu" in caplog.text


def test_load_pyannote_pipeline_raises_runtime_error_when_candidates_are_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_pipeline(monkeypatch, lambda *_a, **_k: lambda payload: payload)
    monkeypatch.setattr(diarization_loader, "_candidate_load_inputs", lambda *_a, **_k: [])
    with pytest.raises(RuntimeError, match="Unable to load diarization pipeline"):
        diarization_loader.load_pyannote_pipeline(model_id="repo/empty@main")


def test_load_pyannote_pipeline_retries_trusted_unsupported_global(
    monkeypatch: pytest.MonkeyPatch,
):
    retry_fqn = "pyannote.audio.core.task.Specifications"
    calls: list[tuple[str, dict[str, object]]] = []
    safe_globals_calls: list[list[str]] = []
    imported: list[str] = []

    @contextlib.contextmanager
    def _fake_safe_globals(extra_fqns: list[str] | None = None):
        safe_globals_calls.append(list(extra_fqns or []))
        yield

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        if len(calls) == 1:
            raise pickle.UnpicklingError(
                "Weights only load failed. Unsupported global: GLOBAL "
                f"{retry_fqn}"
            )
        return lambda payload: payload

    _install_fake_pipeline(monkeypatch, _impl)
    monkeypatch.setattr(
        diarization_loader,
        "diarization_safe_globals_for_torch_load",
        _fake_safe_globals,
    )
    monkeypatch.setattr(
        diarization_loader,
        "import_trusted_diarization_symbol",
        lambda fqn: imported.append(fqn) or object(),
    )

    model = diarization_loader.load_pyannote_pipeline(model_id="repo/retry")
    assert callable(model)
    assert calls == [
        ("repo/retry", {}),
        ("repo/retry", {}),
    ]
    assert safe_globals_calls == [
        [],
        [retry_fqn],
    ]
    assert imported == [retry_fqn]


def test_load_pyannote_pipeline_rejects_untrusted_unsupported_global(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, dict[str, object]]] = []
    imported: list[str] = []

    @contextlib.contextmanager
    def _fake_safe_globals(extra_fqns: list[str] | None = None):
        del extra_fqns
        yield

    def _impl(name: str, **kwargs):
        calls.append((name, dict(kwargs)))
        raise RuntimeError("Unsupported global: GLOBAL builtins.eval")

    _install_fake_pipeline(monkeypatch, _impl)
    monkeypatch.setattr(
        diarization_loader,
        "diarization_safe_globals_for_torch_load",
        _fake_safe_globals,
    )
    monkeypatch.setattr(
        diarization_loader,
        "import_trusted_diarization_symbol",
        lambda fqn: imported.append(fqn) or object(),
    )

    with pytest.raises(RuntimeError, match="builtins.eval"):
        diarization_loader.load_pyannote_pipeline(model_id="repo/unsafe")
    assert calls == [("repo/unsafe", {})]
    assert imported == []


def test_from_pretrained_with_safe_globals_raises_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    retry_fqn = "pyannote.audio.core.task.Specifications"

    @contextlib.contextmanager
    def _fake_safe_globals(extra_fqns: list[str] | None = None):
        del extra_fqns
        yield

    def _always_unsupported(_name: str, **_kwargs):
        raise pickle.UnpicklingError(f"Unsupported global: GLOBAL {retry_fqn}")

    monkeypatch.setattr(
        diarization_loader,
        "diarization_safe_globals_for_torch_load",
        _fake_safe_globals,
    )
    monkeypatch.setattr(
        diarization_loader,
        "unsupported_global_diarization_fqn_from_error",
        lambda _exc: retry_fqn,
    )
    monkeypatch.setattr(
        diarization_loader,
        "import_trusted_diarization_symbol",
        lambda _fqn: None,
    )

    with pytest.raises(pickle.UnpicklingError, match=retry_fqn):
        diarization_loader._from_pretrained_with_safe_globals(
            _always_unsupported,
            "repo/test",
            {},
        )


def test_from_pretrained_with_safe_globals_raises_last_error_after_retry_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    retry_fqns = iter(
        [
            "pyannote.audio.core.task.Specifications",
            "omegaconf.base.ContainerMetadata",
            "torch.torch_version.TorchVersion",
        ]
    )

    @contextlib.contextmanager
    def _fake_safe_globals(extra_fqns: list[str] | None = None):
        del extra_fqns
        yield

    def _always_fail(_name: str, **_kwargs):
        calls.append("x")
        raise RuntimeError("unsupported global loop")

    monkeypatch.setattr(
        diarization_loader,
        "diarization_safe_globals_for_torch_load",
        _fake_safe_globals,
    )
    monkeypatch.setattr(
        diarization_loader,
        "unsupported_global_diarization_fqn_from_error",
        lambda _exc: next(retry_fqns),
    )
    monkeypatch.setattr(
        diarization_loader,
        "import_trusted_diarization_symbol",
        lambda _fqn: object(),
    )

    with pytest.raises(RuntimeError, match="unsupported global loop"):
        diarization_loader._from_pretrained_with_safe_globals(
            _always_fail,
            "repo/test",
            {},
        )
    assert len(calls) == diarization_loader._MAX_SAFE_GLOBAL_ATTEMPTS


def test_from_pretrained_with_safe_globals_raises_runtime_error_when_attempts_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(diarization_loader, "_MAX_SAFE_GLOBAL_ATTEMPTS", 0)

    with pytest.raises(RuntimeError, match="Unable to load diarization pipeline"):
        diarization_loader._from_pretrained_with_safe_globals(
            lambda *_a, **_k: object(),
            "repo/test",
            {},
        )
