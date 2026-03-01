from __future__ import annotations

import argparse
from typing import Sequence

from lan_app.diarization_loader import (
    classify_pipeline_load_error,
    extract_repo_hints,
    load_pyannote_pipeline,
    resolve_diarization_model_id,
    resolve_hf_token,
)

EXIT_SUCCESS = 0
EXIT_MISSING_TOKEN = 2
EXIT_GATED_ACCESS = 3
EXIT_REVISION_NOT_FOUND = 4
EXIT_OTHER_ERROR = 5


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warmup and validate model caches.")
    parser.add_argument(
        "--models",
        default="diarization",
        help="Comma-separated models to warm. Supported value: diarization.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalize_models(raw_value: str) -> list[str]:
    values = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    return values or ["diarization"]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected_models = _normalize_models(args.models)
    unsupported = [name for name in selected_models if name != "diarization"]
    if unsupported:
        values = ", ".join(unsupported)
        print(f"Unsupported --models value(s): {values}. Use: diarization.")
        return EXIT_OTHER_ERROR

    token = resolve_hf_token()
    if token is None:
        print("Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
        return EXIT_MISSING_TOKEN

    model_id = resolve_diarization_model_id()
    repo_id = model_id.split("@", 1)[0].strip() or model_id
    try:
        load_pyannote_pipeline(model_id=model_id, token=token)
    except Exception as exc:
        classification = classify_pipeline_load_error(exc)
        if classification == "revision_not_found":
            print(
                "Revision not found for "
                f"'{model_id}'. Check LAN_DIARIZATION_MODEL_ID and retry."
            )
            return EXIT_REVISION_NOT_FOUND
        if classification == "gated_access":
            repos = extract_repo_hints(exc) or [repo_id]
            print(
                "Gated access is not granted. Accept terms for: "
                f"{', '.join(repos)}"
            )
            return EXIT_GATED_ACCESS
        print(f"Warmup failed: {type(exc).__name__}: {exc}")
        return EXIT_OTHER_ERROR

    print(f"Warmup complete for '{model_id}'.")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
