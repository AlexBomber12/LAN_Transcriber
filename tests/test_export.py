from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient

from lan_app import api, ui_routes
from lan_app.config import AppSettings
from lan_app.constants import RECORDING_STATUS_READY
from lan_app.db import (
    create_recording,
    create_voice_profile,
    init_db,
    set_speaker_assignment,
)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_ui_recording_export_zip_contains_markdown_and_manifest(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    recording_id = "rec-export-1"
    create_recording(
        recording_id,
        source="upload",
        source_filename="team-sync.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    profile = create_voice_profile("Alex Finance", settings=cfg)
    set_speaker_assignment(
        recording_id=recording_id,
        diar_speaker_label="S1",
        voice_profile_id=profile["id"],
        confidence=1.0,
        settings=cfg,
    )

    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Budget sync",
                "summary_bullets": ["Reviewed monthly spend."],
                "decisions": ["Freeze non-critical purchases."],
                "action_items": [
                    {
                        "task": "Draft revised budget",
                        "owner": "Alex",
                        "deadline": "2026-03-05",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps(
            [
                {"speaker": "S1", "text": "Let's review the budget."},
                {"speaker": "S2", "text": "We should freeze discretionary spend."},
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(api.app, follow_redirects=True)
    response = client.get(f"/ui/recordings/{recording_id}/export.zip")
    assert response.status_code == 200

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    assert "onenote.md" in names
    assert "manifest.json" in names
    assert "derived/summary.json" in names

    markdown = archive.read("onenote.md").decode("utf-8")
    assert "Budget sync" in markdown
    assert "Alex Finance (S1)" in markdown
    manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert manifest["recording_id"] == recording_id


def test_ui_recording_export_prefers_local_labels_and_ignores_unreviewed_suggestions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    recording_id = "rec-export-local-1"
    create_recording(
        recording_id,
        source="upload",
        source_filename="local.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    profile = create_voice_profile("Suggested Canonical", settings=cfg)
    set_speaker_assignment(
        recording_id=recording_id,
        diar_speaker_label="S1",
        voice_profile_id=profile["id"],
        confidence=0.88,
        candidate_matches=[{"voice_profile_id": profile["id"], "score": 0.88}],
        low_confidence=False,
        review_state="system_suggested",
        settings=cfg,
    )
    set_speaker_assignment(
        recording_id=recording_id,
        diar_speaker_label="S2",
        voice_profile_id=None,
        review_state="local_label",
        local_display_name="Meeting Guest",
        settings=cfg,
    )

    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps({"topic": "Local labels", "summary_bullets": ["Checked export labels."]}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps(
            [
                {"speaker": "S1", "text": "Suggested but not confirmed."},
                {"speaker": "S2", "text": "Only named for this meeting."},
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(api.app, follow_redirects=True)
    response = client.get(f"/ui/recordings/{recording_id}/export.zip")
    assert response.status_code == 200

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    markdown = archive.read("onenote.md").decode("utf-8")
    assert "- **S1:** Suggested but not confirmed." in markdown
    assert "- **Meeting Guest (S2):** Only named for this meeting." in markdown
    assert "Suggested Canonical (S1)" not in markdown
