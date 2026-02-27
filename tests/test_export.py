from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient

from lan_app import api, ui_routes
from lan_app.config import AppSettings
from lan_app.constants import RECORDING_STATUS_READY
from lan_app.db import create_recording, init_db


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
    manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert manifest["recording_id"] == recording_id
