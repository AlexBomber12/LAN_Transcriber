from __future__ import annotations

import os
import re
from typing import Any

from .hf_repo import split_repo_id_and_revision

DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization@3.1"

_REPO_HINT_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\b")


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


def load_pyannote_pipeline(*, model_id: str | None = None, token: str | None = None) -> Any:
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
        try:
            model = Pipeline.from_pretrained(candidate, **kwargs)
        except TypeError as exc:
            message = str(exc).lower()
            if (
                resolved_token
                and "unexpected keyword argument" in message
                and "token" in message
            ):
                kwargs.pop("token", None)
                try:
                    model = Pipeline.from_pretrained(candidate, **kwargs)
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
