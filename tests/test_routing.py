from __future__ import annotations

import json
from pathlib import Path

from lan_app.config import AppSettings
from lan_app.constants import RECORDING_STATUS_NEEDS_REVIEW, RECORDING_STATUS_READY
from lan_app.db import (
    count_routing_training_examples,
    create_project,
    create_recording,
    create_voice_profile,
    delete_project,
    get_recording,
    increment_project_keyword_weights,
    init_db,
    list_project_keyword_weights,
    set_recording_routing_suggestion,
    set_speaker_assignment,
    upsert_calendar_match,
)
from lan_app.routing import refresh_recording_routing, train_routing_from_manual_selection


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _write_summary(cfg: AppSettings, recording_id: str, payload: dict[str, object]) -> None:
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_refresh_recording_routing_auto_selects_when_confident(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.routing_auto_select_threshold = 0.3
    init_db(cfg)
    roadmap = create_project("Roadmap", settings=cfg)
    create_project("Budget", settings=cfg)
    create_recording(
        "rec-route-auto-1",
        source="drive",
        source_filename="roadmap-sync.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-route-auto-1",
        candidates=[
            {
                "event_id": "evt-roadmap",
                "subject": "Roadmap sync",
                "organizer": "Alex",
                "attendees": ["Priya"],
                "score": 0.95,
                "rationale": "manual",
            }
        ],
        selected_event_id="evt-roadmap",
        selected_confidence=0.95,
        settings=cfg,
    )
    _write_summary(
        cfg,
        "rec-route-auto-1",
        {
            "topic": "Roadmap planning",
            "summary_bullets": ["Roadmap priorities and milestones"],
        },
    )
    increment_project_keyword_weights(
        project_id=roadmap["id"],
        keyword_deltas={
            "cal:roadmap": 3.0,
            "tag:roadmap": 3.0,
            "party:alex": 2.0,
        },
        settings=cfg,
    )

    decision = refresh_recording_routing(
        "rec-route-auto-1",
        settings=cfg,
        apply_workflow=True,
    )

    assert decision["suggested_project_id"] == roadmap["id"]
    assert decision["confidence"] >= cfg.routing_auto_select_threshold
    assert decision["status_after_routing"] == RECORDING_STATUS_READY

    recording = get_recording("rec-route-auto-1", settings=cfg)
    assert recording is not None
    assert recording["project_id"] == roadmap["id"]
    assert recording["suggested_project_id"] == roadmap["id"]


def test_refresh_recording_routing_marks_needs_review_when_low_confidence(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_project("Roadmap", settings=cfg)
    create_project("Budget", settings=cfg)
    create_recording(
        "rec-route-review-1",
        source="drive",
        source_filename="generic-meeting.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _write_summary(
        cfg,
        "rec-route-review-1",
        {"topic": "General discussion", "summary_bullets": ["Open items"]},
    )

    decision = refresh_recording_routing(
        "rec-route-review-1",
        settings=cfg,
        apply_workflow=True,
    )
    assert decision["confidence"] < cfg.routing_auto_select_threshold
    assert decision["status_after_routing"] == RECORDING_STATUS_NEEDS_REVIEW

    recording = get_recording("rec-route-review-1", settings=cfg)
    assert recording is not None
    assert recording["project_id"] is None
    assert recording["routing_confidence"] < cfg.routing_auto_select_threshold


def test_refresh_recording_routing_without_projects_forces_needs_review(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-route-no-projects-1",
        source="drive",
        source_filename="setup.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    decision = refresh_recording_routing(
        "rec-route-no-projects-1",
        settings=cfg,
        apply_workflow=True,
    )

    assert decision["suggested_project_id"] is None
    assert decision["confidence"] == 0.0
    assert decision["status_after_routing"] == RECORDING_STATUS_NEEDS_REVIEW


def test_delete_project_clears_recording_suggested_project_id(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    project = create_project("Cleanup", settings=cfg)
    create_recording(
        "rec-route-cleanup-1",
        source="drive",
        source_filename="cleanup.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    set_recording_routing_suggestion(
        "rec-route-cleanup-1",
        suggested_project_id=project["id"],
        routing_confidence=0.8,
        routing_rationale=["test"],
        settings=cfg,
    )
    before = get_recording("rec-route-cleanup-1", settings=cfg)
    assert before is not None
    assert before["suggested_project_id"] == project["id"]

    assert delete_project(project["id"], settings=cfg) is True
    after = get_recording("rec-route-cleanup-1", settings=cfg)
    assert after is not None
    assert after["suggested_project_id"] is None


def test_train_routing_from_manual_selection_persists_weights(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    project = create_project("Roadmap", settings=cfg)
    create_recording(
        "rec-route-train-1",
        source="drive",
        source_filename="roadmap-train.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-route-train-1",
        candidates=[
            {
                "event_id": "evt-1",
                "subject": "Roadmap training session",
                "organizer": "Alex",
                "attendees": ["Priya"],
                "score": 0.9,
                "rationale": "manual",
            }
        ],
        selected_event_id="evt-1",
        selected_confidence=0.9,
        settings=cfg,
    )
    _write_summary(
        cfg,
        "rec-route-train-1",
        {"topic": "Roadmap training", "summary_bullets": ["Roadmap delivery plan"]},
    )
    profile = create_voice_profile("Alex", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-route-train-1",
        diar_speaker_label="S1",
        voice_profile_id=profile["id"],
        confidence=1.0,
        settings=cfg,
    )

    out = train_routing_from_manual_selection(
        "rec-route-train-1",
        project["id"],
        settings=cfg,
    )

    assert out["project_id"] == project["id"]
    assert out["training_example_id"] is not None
    assert count_routing_training_examples(project_id=project["id"], settings=cfg) == 1
    weights = list_project_keyword_weights(project_id=project["id"], settings=cfg)
    keywords = {row["keyword"] for row in weights}
    assert "cal:roadmap" in keywords
    assert "tag:roadmap" in keywords
    assert f"voice:{profile['id']}" in keywords
