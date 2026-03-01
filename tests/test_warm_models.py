from __future__ import annotations

import runpy
import sys

import pytest

from lan_app import diarization_loader
from lan_app.tools import warm_models


def test_warm_models_rejects_unsupported_models(capsys):
    rc = warm_models.main(["--models", "stt"])
    out = capsys.readouterr().out
    assert rc == warm_models.EXIT_OTHER_ERROR
    assert "Unsupported --models value(s): stt" in out


def test_warm_models_requires_token(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(warm_models, "resolve_hf_token", lambda *_a, **_k: None)
    rc = warm_models.main(["--models", "diarization"])
    out = capsys.readouterr().out
    assert rc == warm_models.EXIT_MISSING_TOKEN
    assert "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN" in out


def test_warm_models_success_path(monkeypatch: pytest.MonkeyPatch, capsys):
    seen: dict[str, object] = {}
    monkeypatch.setattr(warm_models, "resolve_hf_token", lambda *_a, **_k: "hf-token")
    monkeypatch.setattr(
        warm_models,
        "resolve_diarization_model_id",
        lambda *_a, **_k: "repo/test@main",
    )
    monkeypatch.setattr(
        warm_models,
        "load_pyannote_pipeline",
        lambda **kwargs: seen.setdefault("kwargs", kwargs) or (lambda payload: payload),
    )

    rc = warm_models.main([])
    out = capsys.readouterr().out
    assert rc == warm_models.EXIT_SUCCESS
    assert "Warmup complete for 'repo/test@main'." in out
    assert seen["kwargs"] == {"model_id": "repo/test@main", "token": "hf-token"}


def test_warm_models_gated_access_message_uses_repo_hints(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    monkeypatch.setattr(warm_models, "resolve_hf_token", lambda *_a, **_k: "hf-token")
    monkeypatch.setattr(
        warm_models,
        "resolve_diarization_model_id",
        lambda *_a, **_k: "pyannote/speaker-diarization@3.1",
    )

    def _raise(**_kwargs):
        raise RuntimeError("403")

    monkeypatch.setattr(warm_models, "load_pyannote_pipeline", _raise)
    monkeypatch.setattr(
        warm_models,
        "classify_pipeline_load_error",
        lambda _exc: "gated_access",
    )
    monkeypatch.setattr(
        warm_models,
        "extract_repo_hints",
        lambda _exc: ["pyannote/speaker-diarization", "pyannote/segmentation"],
    )

    rc = warm_models.main([])
    out = capsys.readouterr().out
    assert rc == warm_models.EXIT_GATED_ACCESS
    assert "Accept terms for: pyannote/speaker-diarization, pyannote/segmentation" in out


def test_warm_models_revision_and_other_errors(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(warm_models, "resolve_hf_token", lambda *_a, **_k: "hf-token")
    monkeypatch.setattr(
        warm_models,
        "resolve_diarization_model_id",
        lambda *_a, **_k: "repo/test@oops",
    )

    def _raise_revision(**_kwargs):
        raise RuntimeError("revision error")

    monkeypatch.setattr(warm_models, "load_pyannote_pipeline", _raise_revision)
    monkeypatch.setattr(
        warm_models,
        "classify_pipeline_load_error",
        lambda _exc: "revision_not_found",
    )
    rc_revision = warm_models.main([])
    out_revision = capsys.readouterr().out
    assert rc_revision == warm_models.EXIT_REVISION_NOT_FOUND
    assert "Revision not found for 'repo/test@oops'" in out_revision

    def _raise_other(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(warm_models, "load_pyannote_pipeline", _raise_other)
    monkeypatch.setattr(warm_models, "classify_pipeline_load_error", lambda _exc: "other")
    rc_other = warm_models.main([])
    out_other = capsys.readouterr().out
    assert rc_other == warm_models.EXIT_OTHER_ERROR
    assert "Warmup failed: RuntimeError: boom" in out_other


def test_warm_models_module_main_guard(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["warm_models.py"])
    monkeypatch.setattr(diarization_loader, "resolve_hf_token", lambda *_a, **_k: "hf-token")
    monkeypatch.setattr(
        diarization_loader,
        "resolve_diarization_model_id",
        lambda *_a, **_k: "repo/main@ok",
    )
    monkeypatch.setattr(
        diarization_loader,
        "load_pyannote_pipeline",
        lambda **_kwargs: (lambda payload: payload),
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("lan_app.tools.warm_models", run_name="__main__")
    assert exc_info.value.code == warm_models.EXIT_SUCCESS
