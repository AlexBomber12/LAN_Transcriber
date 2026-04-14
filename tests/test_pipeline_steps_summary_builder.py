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
                "tone_score": 72,
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
    assert payload["friendly"] == 72
    assert payload["tone_score"] == 72
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


def test_build_summary_payload_empty_content_uses_default_summary_bullet() -> None:
    payload = build_summary_payload(
        raw_llm_content="   ",
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["parse_error"] is True
    assert payload["summary_bullets"] == ["No summary available."]
    assert payload["summary"] == "- No summary available."


def test_build_summary_payload_uses_friendly_fallback_when_tone_score_missing() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Fallback",
                "summary_bullets": ["Still valid"],
                "decisions": [],
                "action_items": [],
                "emotional_summary": "Neutral.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=33,
    )

    assert payload["friendly"] == 33
    assert payload["tone_score"] == 33


def test_build_summary_payload_uses_friendly_fallback_when_tone_score_blank() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Fallback",
                "summary_bullets": ["Still valid"],
                "decisions": [],
                "action_items": [],
                "tone_score": "",
                "emotional_summary": "Neutral.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=41,
    )

    assert payload["friendly"] == 41
    assert payload["tone_score"] == 41


def test_build_summary_payload_uses_friendly_fallback_when_tone_score_whitespace() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Fallback",
                "summary_bullets": ["Still valid"],
                "decisions": [],
                "action_items": [],
                "tone_score": "   ",
                "emotional_summary": "Neutral.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=44,
    )

    assert payload["friendly"] == 44
    assert payload["tone_score"] == 44


def test_build_summary_payload_uses_friendly_fallback_when_tone_score_non_numeric() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Fallback",
                "summary_bullets": ["Still valid"],
                "decisions": [],
                "action_items": [],
                "tone_score": "N/A",
                "friendly": 46,
                "emotional_summary": "Neutral.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["friendly"] == 46
    assert payload["tone_score"] == 46


def test_build_summary_payload_uses_friendly_fallback_when_tone_score_non_finite() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Fallback",
                "summary_bullets": ["Still valid"],
                "decisions": [],
                "action_items": [],
                "tone_score": "inf",
                "friendly": 47,
                "emotional_summary": "Neutral.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["friendly"] == 47
    assert payload["tone_score"] == 47


def test_build_summary_payload_uses_legacy_friendly_field_when_tone_score_missing() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Legacy",
                "summary_bullets": ["Old provider shape"],
                "decisions": [],
                "action_items": [],
                "friendly": 67,
                "emotional_summary": "Positive.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["friendly"] == 67
    assert payload["tone_score"] == 67


def test_build_summary_payload_parse_error_uses_legacy_friendly_field_when_tone_score_missing() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Legacy fallback",
                "summary_bullets": 123,
                "decisions": [],
                "action_items": [],
                "friendly": 58,
                "emotional_summary": "Positive.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["parse_error"] is True
    assert payload["friendly"] == 58
    assert payload["tone_score"] == 58


def test_build_summary_payload_parse_error_uses_legacy_friendly_field_when_tone_score_whitespace() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Legacy fallback",
                "summary_bullets": 123,
                "decisions": [],
                "action_items": [],
                "friendly": 61,
                "tone_score": " \t ",
                "emotional_summary": "Positive.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["parse_error"] is True
    assert payload["friendly"] == 61
    assert payload["tone_score"] == 61


def test_build_summary_payload_parse_error_uses_legacy_friendly_field_when_tone_score_non_numeric() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Legacy fallback",
                "summary_bullets": 123,
                "decisions": [],
                "action_items": [],
                "friendly": 63,
                "tone_score": "N/A",
                "emotional_summary": "Positive.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["parse_error"] is True
    assert payload["friendly"] == 63
    assert payload["tone_score"] == 63


def test_build_summary_payload_parse_error_prefers_tone_score_when_present() -> None:
    payload = build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "Tone wins",
                "summary_bullets": 123,
                "decisions": [],
                "action_items": [],
                "friendly": 12,
                "tone_score": 77,
                "emotional_summary": "Positive.",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )

    assert payload["parse_error"] is True
    assert payload["friendly"] == 77
    assert payload["tone_score"] == 77
