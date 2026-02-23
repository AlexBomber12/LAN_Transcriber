from __future__ import annotations

import json
from pathlib import Path

from lan_transcriber.pipeline_steps.artifacts import LlmDebugArtifacts, write_json_artifact, write_llm_debug_artifacts


def test_write_json_artifact_replaces_content_with_valid_json(tmp_path: Path):
    path = tmp_path / "derived" / "metrics.json"

    write_json_artifact(path, {"status": "running", "value": 1})
    write_json_artifact(path, {"status": "ok", "value": 2})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"status": "ok", "value": 2}


def test_write_llm_debug_artifacts_creates_expected_files(tmp_path: Path):
    spec = LlmDebugArtifacts(
        derived_dir=tmp_path / "derived",
        raw_output="raw",
        extracted_payload={"topic": "T"},
        validation_error={"reason": "bad_schema"},
    )

    write_llm_debug_artifacts(spec)

    assert (spec.derived_dir / "llm_raw.txt").read_text(encoding="utf-8") == "raw"
    assert json.loads((spec.derived_dir / "llm_extract.json").read_text(encoding="utf-8")) == {"topic": "T"}
    assert json.loads((spec.derived_dir / "llm_validation_error.json").read_text(encoding="utf-8")) == {
        "reason": "bad_schema"
    }
