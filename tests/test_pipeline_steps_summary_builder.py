from __future__ import annotations

import json
from pathlib import Path

from lan_transcriber.pipeline_steps.summary_builder import build_summary_payload


def test_build_summary_payload_valid_json_uses_schema():
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Sync",
                "summary_bullets": ["Discussed rollout"],
                "decisions": ["Ship Friday"],
                "action_items": [{"task": "Send recap", "owner": "Alex", "deadline": "2026-03-01", "confidence": 0.9}],
                "emotional_summary": "Focused and positive.",
                "questions": {"total_count": 1, "types": {"status": 1}, "extracted": ["Is QA done?"]},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=50,
    )

    assert payload["topic"] == "Sync"
    assert payload["action_items"][0]["owner"] == "Alex"
    assert payload.get("parse_error") is None


def test_build_summary_payload_invalid_json_writes_debug_artifacts_and_sets_parse_error(tmp_path: Path):
    derived = tmp_path / "derived"
    raw = json.dumps(
        {
            "topic": "Bad payload",
            "summary_bullets": 123,
            "decisions": ["Keep timeline"],
            "action_items": [{"task": "Send minutes", "owner": "Mina"}],
            "emotional_summary": "Neutral",
            "questions": {"total_count": 2, "types": {"open": 2}, "extracted": ["Who owns this?"]},
        }
    )

    payload = build_summary_payload(
        raw_llm_content=raw,
        model="m",
        target_summary_language="en",
        friendly=20,
        derived_dir=derived,
    )

    assert payload["parse_error"] is True
    assert payload["action_items"][0]["task"] == "Send minutes"
    assert payload["questions"]["total_count"] >= 1
    assert (derived / "llm_raw.txt").exists()
    assert (derived / "llm_extract.json").exists()
    assert (derived / "llm_validation_error.json").exists()


def test_build_summary_payload_no_json_object_sets_parse_error(tmp_path: Path):
    derived = tmp_path / "derived"
    payload = build_summary_payload(
        raw_llm_content="- plain text summary",
        model="m",
        target_summary_language="en",
        friendly=0,
        derived_dir=derived,
    )

    assert payload["parse_error"] is True
    assert payload["parse_error_reason"] == "json_object_not_found"
    assert json.loads((derived / "llm_extract.json").read_text(encoding="utf-8")) == {}
