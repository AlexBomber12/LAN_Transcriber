from __future__ import annotations

import json
from pathlib import Path

from lan_app.conversation_metrics import (
    build_conversation_metrics,
    compute_actionability_ratio,
    count_interruptions,
    merge_speaker_turns,
    refresh_recording_metrics,
)
from lan_app.config import AppSettings
from lan_app.db import (
    create_recording,
    get_meeting_metrics,
    init_db,
    list_participant_metrics,
)


def test_merge_speaker_turns_merges_tiny_gaps_only_for_adjacent_same_speaker():
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "hello"},
        {"start": 1.6, "end": 2.1, "speaker": "S1", "text": "team"},
        {"start": 2.2, "end": 2.8, "speaker": "S2", "text": "hi"},
        {"start": 2.9, "end": 3.2, "speaker": "S1", "text": "again"},
    ]

    merged = merge_speaker_turns(turns, gap_threshold=1.0)

    assert len(merged) == 3
    assert merged[0]["speaker"] == "S1"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 2.1
    assert merged[0]["text"] == "hello team"
    assert merged[2]["speaker"] == "S1"
    assert merged[2]["text"] == "again"


def test_count_interruptions_counts_done_and_received_with_overlap_threshold():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "S1", "text": "long turn"},
        {"start": 4.6, "end": 6.0, "speaker": "S2", "text": "interruption"},
        {"start": 5.8, "end": 7.0, "speaker": "S3", "text": "tiny overlap only"},
        {"start": 6.5, "end": 8.0, "speaker": "S1", "text": "interrupts S3"},
    ]

    stats = count_interruptions(turns, overlap_threshold=0.3)

    assert stats["total"] == 2
    assert stats["done"]["S2"] == 1
    assert stats["done"]["S1"] == 1
    assert stats["received"]["S1"] == 1
    assert stats["received"]["S3"] == 1


def test_count_interruptions_ignores_simultaneous_starts():
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "S1", "text": "start"},
        {"start": 0.0, "end": 1.5, "speaker": "S2", "text": "same start"},
        {"start": 1.2, "end": 2.5, "speaker": "S3", "text": "later overlap"},
    ]

    stats = count_interruptions(turns, overlap_threshold=0.3)

    # S1/S2 same start should not create interruption by ordering alone.
    assert stats["done"]["S1"] == 0
    assert stats["done"]["S2"] == 0
    assert stats["received"]["S1"] == 1
    assert stats["received"]["S2"] == 1
    assert stats["done"]["S3"] == 2


def test_compute_actionability_ratio_uses_owner_and_deadline_presence():
    action_items = [
        {"task": "Send notes", "owner": "Alex", "deadline": "2026-02-25"},
        {"task": "Prepare demo", "owner": "", "deadline": "2026-02-26"},
        {"task": "Review metrics", "owner": "Mina", "deadline": None},
        {"task": "Close tickets", "owner": "Mina", "deadline": "2026-03-01"},
    ]

    ratio = compute_actionability_ratio(action_items)

    assert ratio == 0.5


def test_build_conversation_metrics_heuristic_questions_count_punctuation_for_unknown_language():
    payload = build_conversation_metrics(
        transcript_payload={},
        summary_payload={"questions": {}},
        speaker_turns=[
            {
                "start": 0.0,
                "end": 1.5,
                "speaker": "S1",
                "text": "Should we proceed?",
                "language": "xx",
            }
        ],
    )

    assert payload["meeting"]["questions_source"] == "heuristic"
    assert payload["meeting"]["total_questions"] == 1
    assert payload["participants"][0]["questions_count"] == 1


def test_build_conversation_metrics_interruptions_use_raw_turns_not_gap_merged_turns():
    payload = build_conversation_metrics(
        transcript_payload={},
        summary_payload={},
        speaker_turns=[
            {"start": 0.0, "end": 1.0, "speaker": "S1", "text": "first thought"},
            {"start": 1.3, "end": 2.2, "speaker": "S2", "text": "response starts"},
            {"start": 1.8, "end": 3.0, "speaker": "S1", "text": "continues after pause"},
        ],
        gap_threshold_sec=1.0,
        interruption_overlap_sec=0.3,
    )

    # S2 starts during S1's silent gap and should not count as interrupting S1.
    # S1's third turn does overlap S2 and counts as one interruption.
    assert payload["meeting"]["total_interruptions"] == 1
    by_speaker = {row["speaker"]: row for row in payload["participants"]}
    assert by_speaker["S2"]["interruptions_done"] == 0
    assert by_speaker["S2"]["interruptions_received"] == 1
    assert by_speaker["S1"]["interruptions_done"] == 1


def test_refresh_recording_metrics_persists_json_and_db(tmp_path: Path):
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    init_db(cfg)
    create_recording(
        "rec-metrics-refresh-1",
        source="drive",
        source_filename="meeting.mp3",
        settings=cfg,
    )

    derived = cfg.recordings_root / "rec-metrics-refresh-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "Do we have a final owner? We should decide now.",
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 3.0, "speaker": "S1", "text": "Do we have a final owner?"},
                {"start": 2.8, "end": 5.0, "speaker": "S2", "text": "Yes, I will own it."},
                {"start": 5.2, "end": 7.0, "speaker": "S1", "text": "Great, we decided."},
            ]
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "decisions": ["Ship this week"],
                "action_items": [
                    {"task": "Send recap", "owner": "S2", "deadline": "2026-03-01"},
                ],
                "emotional_summary": "Focused and decisive.",
                "questions": {
                    "total_count": 1,
                    "types": {
                        "open": 1,
                        "yes_no": 0,
                        "clarification": 0,
                        "status": 0,
                        "decision_seeking": 0,
                    },
                    "extracted": ["Do we have a final owner?"],
                },
            }
        ),
        encoding="utf-8",
    )
    (derived / "metrics.json").write_text(
        json.dumps({"status": "ok", "version": 1}),
        encoding="utf-8",
    )

    payload = refresh_recording_metrics("rec-metrics-refresh-1", settings=cfg)
    assert payload["meeting"]["total_questions"] == 1
    assert payload["meeting"]["actionability_ratio"] == 1.0
    assert len(payload["participants"]) == 2

    stored_metrics = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))
    assert stored_metrics["status"] == "ok"
    assert stored_metrics["meeting"]["total_interruptions"] >= 0

    meeting_row = get_meeting_metrics("rec-metrics-refresh-1", settings=cfg)
    participant_rows = list_participant_metrics("rec-metrics-refresh-1", settings=cfg)
    assert meeting_row is not None
    assert meeting_row["json"]["total_questions"] == 1
    assert len(participant_rows) == 2
