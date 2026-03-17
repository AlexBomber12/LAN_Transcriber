"""Tests for the server-rendered HTML UI routes (PR-UI-SHELL-01)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import wave
import zipfile

import pytest
from fastapi.testclient import TestClient

from lan_app import api, ui_routes
from lan_app.auth import AUTH_COOKIE_NAME
from lan_app.config import AppSettings
from lan_app.ops import RecordingDeleteError
from lan_app.snippet_repair import SnippetRepairEligibility, SnippetRepairResult
from lan_app.db import (
    create_calendar_source,
    create_glossary_entry,
    count_routing_training_examples,
    create_voice_sample,
    create_job,
    create_recording,
    create_project,
    create_voice_profile,
    get_calendar_match,
    get_glossary_entry,
    get_recording,
    mark_recording_llm_chunk_started,
    mark_recording_pipeline_stage_completed,
    mark_recording_pipeline_stage_failed,
    mark_recording_pipeline_stage_started,
    list_glossary_entries,
    list_speaker_assignments,
    list_voice_samples,
    init_db,
    list_projects,
    list_recordings,
    list_calendar_sources,
    list_project_keyword_weights,
    replace_calendar_events_for_window,
    replace_participant_metrics,
    list_voice_profiles,
    set_recording_status,
    set_recording_cancel_request,
    set_recording_progress,
    set_speaker_assignment,
    update_glossary_entry,
    upsert_calendar_match,
    upsert_meeting_metrics,
)
from lan_app.jobs import RecordingJob
from lan_app.constants import (
    JOB_STATUS_FAILED,
    JOB_STATUS_FINISHED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_STARTED,
    JOB_TYPE_PRECHECK,
    JOB_TYPE_STT,
    RECORDING_STATUS_FAILED,
    RECORDING_STATUS_NEEDS_REVIEW,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_READY,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_STOPPED,
    RECORDING_STATUS_STOPPING,
)


def _stub_runtime_status() -> dict[str, object]:
    return {
        "items": [
            {
                "label": "Node status",
                "value": "Online",
                "detail": "dgx.local responded to /v1/models",
                "tone": "healthy",
                "show_dot": True,
            },
            {
                "label": "GPU runtime",
                "value": "GPU ready",
                "detail": "torch sees 1 GPU(s) · CUDA 12.6",
                "tone": "healthy",
            },
            {
                "label": "LLM:",
                "value": "gpt-oss:120b",
                "detail": "dgx.local · configured model is advertised",
                "tone": "healthy",
            },
        ],
    }


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    init_db(cfg)
    return TestClient(api.app, follow_redirects=True)


@pytest.fixture()
def seeded_client(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    init_db(cfg)
    create_recording(
        "rec-ui-1",
        source="drive",
        source_filename="meeting.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    create_job(
        "job-ui-1",
        recording_id="rec-ui-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
    )
    return TestClient(api.app, follow_redirects=True)


def _seed_speaker_artifacts(cfg: AppSettings, recording_id: str) -> None:
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"text": "hello from S1 and S2"}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.2, "speaker": "S1", "text": "hello team"},
                {"start": 1.3, "end": 2.1, "speaker": "S2", "text": "hi there"},
                {"start": 2.2, "end": 3.8, "speaker": "S1", "text": "next topic"},
            ]
        ),
        encoding="utf-8",
    )
    snippets = derived / "snippets"
    (snippets / "S1").mkdir(parents=True, exist_ok=True)
    (snippets / "S2").mkdir(parents=True, exist_ok=True)
    (snippets / "S1" / "1.wav").write_bytes(b"fake-wav-s1")
    (snippets / "S2" / "1.wav").write_bytes(b"fake-wav-s2")
    (derived / "snippets_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_kind": "turn",
                "degraded_diarization": False,
                "pad_seconds": 0.25,
                "max_clip_duration_seconds": 8.0,
                "min_clip_duration_seconds": 0.8,
                "max_snippets_per_speaker": 3,
                "speakers": {
                    "S1": [
                        {
                            "snippet_id": "S1-01",
                            "speaker": "S1",
                            "source_kind": "turn",
                            "source_start": 0.0,
                            "source_end": 1.2,
                            "clip_start": 0.0,
                            "clip_end": 1.45,
                            "duration_seconds": 1.45,
                            "overlap_seconds": 0.0,
                            "overlap_ratio": 0.0,
                            "purity_score": 0.88,
                            "ranking_position": 1,
                            "status": "accepted",
                            "recommended": True,
                            "extraction_backend": "wave",
                            "relative_path": "S1/1.wav",
                        }
                    ],
                    "S2": [
                        {
                            "snippet_id": "S2-01",
                            "speaker": "S2",
                            "source_kind": "turn",
                            "source_start": 1.3,
                            "source_end": 2.1,
                            "clip_start": 1.05,
                            "clip_end": 2.35,
                            "duration_seconds": 1.3,
                            "overlap_seconds": 0.0,
                            "overlap_ratio": 0.0,
                            "purity_score": 0.85,
                            "ranking_position": 1,
                            "status": "accepted",
                            "recommended": True,
                            "extraction_backend": "wave",
                            "relative_path": "S2/1.wav",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _stub_upload_enqueue(monkeypatch: pytest.MonkeyPatch, cfg: AppSettings) -> None:
    def _fake_enqueue(
        recording_id: str,
        *,
        job_type: str = JOB_TYPE_PRECHECK,
        settings: AppSettings | None = None,
        **_kwargs: object,
    ) -> RecordingJob:
        effective = settings or cfg
        job_id = f"job-upload-{recording_id[-6:]}"
        create_job(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
            status=JOB_STATUS_QUEUED,
            settings=effective,
        )
        set_recording_status(
            recording_id,
            RECORDING_STATUS_QUEUED,
            settings=effective,
        )
        return RecordingJob(
            job_id=job_id,
            recording_id=recording_id,
            job_type=job_type,
        )

    monkeypatch.setattr(api, "enqueue_recording_job", _fake_enqueue)


def _seed_speaker_turns_only(cfg: AppSettings, recording_id: str) -> None:
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"text": "hello from S1 and S2"}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.2, "speaker": "S1", "text": "hello team"},
                {"start": 1.3, "end": 2.1, "speaker": "S2", "text": "hi there"},
            ]
        ),
        encoding="utf-8",
    )


def _write_snippets_manifest(
    cfg: AppSettings,
    recording_id: str,
    payload: dict[str, object],
) -> None:
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "snippets_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_pcm_wav(
    path: Path, *, duration_sec: float, sample_rate: int = 16000
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(sample_rate * duration_sec), 1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _seed_snippet_repair_artifacts(
    cfg: AppSettings,
    recording_id: str,
    *,
    include_speaker_turns: bool = True,
) -> None:
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    _write_pcm_wav(derived / "audio_sanitized.wav", duration_sec=3.0)
    (derived / "precheck.json").write_text(
        json.dumps(
            {"duration_sec": 3.0, "speech_ratio": 0.8, "quarantine_reason": None}
        ),
        encoding="utf-8",
    )
    (derived / "diarization_segments.json").write_text(
        json.dumps(
            [
                {"speaker": "S1", "start": 0.0, "end": 1.2},
                {"speaker": "S2", "start": 1.3, "end": 2.2},
            ]
        ),
        encoding="utf-8",
    )
    (derived / "diarization_metadata.json").write_text(
        json.dumps({"degraded": False}),
        encoding="utf-8",
    )
    if include_speaker_turns:
        (derived / "speaker_turns.json").write_text(
            json.dumps(
                [
                    {"speaker": "S1", "start": 0.0, "end": 1.2, "text": "hello team"},
                    {"speaker": "S2", "start": 1.3, "end": 2.1, "text": "hi there"},
                    {"speaker": "S1", "start": 2.2, "end": 2.9, "text": "follow up"},
                ]
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_empty(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Control Center" in r.text
    assert 'id="control-center-workspace-header"' in r.text
    assert 'id="control-center-work-pane"' in r.text
    assert 'id="control-center-inspector-pane"' in r.text
    assert 'id="control-center-system-bar"' in r.text
    assert 'id="file-input"' in r.text
    assert 'id="control-center-recordings-panel"' in r.text
    assert "No recording selected" in r.text
    assert 'id="control-center-top-strip"' not in r.text
    assert 'href="/upload"' not in r.text
    assert 'href="/recordings"' not in r.text
    assert "LAN Transcriber" in r.text
    assert "Daily operator workspace" not in r.text


def test_dashboard_with_data(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    init_db(cfg)
    create_recording(
        "rec-dash-1", source="drive", source_filename="a.mp3", settings=cfg
    )
    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/")
    assert r.status_code == 200
    assert "Operator inbox" not in r.text
    assert "Drop audio here or browse from disk." in r.text
    assert "GPU ready" in r.text
    assert "gpt-oss:120b" in r.text
    assert "Daily workflow" not in r.text
    assert "Fallback and Admin Pages" not in r.text
    assert 'class="recordings-filter-strip filters"' not in r.text
    assert 'class="recordings-status-strip"' not in r.text
    assert "rec-dash-1" in r.text or "a.mp3" in r.text


def test_dashboard_summary_fragment_endpoints(seeded_client):
    recordings_summary = seeded_client.get(
        "/ui/control-center/dashboard/recordings-summary"
    )
    assert recordings_summary.status_code == 200
    assert "Recordings by status" in recordings_summary.text
    assert "stat-card" in recordings_summary.text
    assert "<html" not in recordings_summary.text

    jobs_summary = seeded_client.get("/ui/control-center/dashboard/jobs-summary")
    assert jobs_summary.status_code == 200
    assert "Queue by status" in jobs_summary.text
    assert "stat-card" in jobs_summary.text
    assert "<nav" not in jobs_summary.text


def test_control_center_query_state_and_direct_routes(seeded_client):
    r = seeded_client.get("/?selected=rec-ui-1&status=Ready&q=meeting&tab=speakers")
    assert r.status_code == 200
    assert 'id="control-center-workspace-header"' in r.text
    assert 'id="control-center-system-bar"' in r.text
    assert 'id="control-center-recordings-panel"' in r.text
    assert 'class="recordings-filter-strip filters"' not in r.text
    assert 'class="recordings-status-strip"' not in r.text
    assert 'data-select-href="/?selected=rec-ui-1&amp;status=Ready&amp;q=meeting&amp;tab=speakers"' in r.text
    assert 'onclick="activateControlCenterRecordingRow(this, event)"' in r.text
    assert 'data-testid="control-center-select-recording"' not in r.text
    assert "/recordings/rec-ui-1?tab=speakers" in r.text
    assert "Open Recording Page" in r.text
    assert "refresh-control-center-header" not in r.text
    assert "refresh-control-center-system-bar" in r.text
    assert "syncControlCenterShellRefreshUrlsFromPanel" in r.text
    assert "syncControlCenterShellRefreshUrlsFromHref" in r.text
    assert "refreshControlCenterShellFromPanel" in r.text
    assert "refreshControlCenterShellFromHref" in r.text
    assert "htmx:afterSwap" in r.text
    assert "window.__controlCenterShellAfterSwapBound" in r.text
    assert "target.id !== 'control-center-inspector-pane'" in r.text
    assert "htmx.trigger(document.body, 'refresh-control-center-inspector');" in r.text
    assert "params.delete('selected');" in r.text

    upload = seeded_client.get("/upload")
    assert upload.status_code == 200

    recordings = seeded_client.get("/recordings")
    assert recordings.status_code == 200

    detail = seeded_client.get("/recordings/rec-ui-1")
    assert detail.status_code == 200
    assert "meeting.mp3" in detail.text
    assert 'class="inspector-hero"' in detail.text


def test_control_center_pane_fragment_endpoints(seeded_client):
    workspace_header = seeded_client.get(
        "/ui/control-center/workspace-header?selected=rec-ui-1&status=Ready&q=meeting&tab=speakers"
    )
    assert workspace_header.status_code == 200
    assert 'id="control-center-workspace-header"' in workspace_header.text
    assert "hx-trigger" not in workspace_header.text
    assert "<h1>Control Center</h1>" in workspace_header.text
    assert "control-center-focus-card" not in workspace_header.text
    assert "meeting.mp3" not in workspace_header.text
    assert "/recordings/rec-ui-1?tab=speakers" not in workspace_header.text
    assert "<html" not in workspace_header.text

    system_bar = seeded_client.get(
        "/ui/control-center/system-bar?selected=rec-ui-1&status=Ready&q=meeting&tab=speakers"
    )
    assert system_bar.status_code == 200
    assert 'id="control-center-system-bar"' in system_bar.text
    assert (
        'hx-trigger="every 15s, refresh-control-center-system-bar from:body"'
        in system_bar.text
    )
    assert "Node status" in system_bar.text
    assert "GPU runtime" in system_bar.text
    assert "LLM:" in system_bar.text
    assert "Inbox view" not in system_bar.text
    assert "Active jobs" not in system_bar.text
    assert "Inference mode" not in system_bar.text
    assert "<html" not in system_bar.text


def test_control_center_system_bar_renders_degraded_cpu_fallback(
    seeded_client, monkeypatch
):
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: {
            "items": [
                {
                    "label": "Node status",
                    "value": "Online",
                    "detail": "dgx.local responded to /v1/models",
                    "tone": "healthy",
                    "show_dot": True,
                },
                {
                    "label": "GPU runtime",
                    "value": "GPU unavailable",
                    "detail": "visible=default · torch CUDA none",
                    "tone": "offline",
                },
                {
                    "label": "LLM:",
                    "value": "gpt-oss:120b",
                    "detail": "dgx.local · configured model is advertised",
                    "tone": "healthy",
                },
            ],
        },
    )

    system_bar = seeded_client.get("/ui/control-center/system-bar")

    assert system_bar.status_code == 200
    assert "Node status" in system_bar.text
    assert "GPU unavailable" in system_bar.text
    assert "CPU fallback" not in system_bar.text
    assert "control-center-system-item--offline" in system_bar.text

    work_pane = seeded_client.get(
        "/ui/control-center/work-pane?selected=rec-ui-1&status=Ready&q=meeting&tab=speakers"
    )
    assert work_pane.status_code == 200
    assert "Intake" not in work_pane.text
    assert "Drop audio into today" not in work_pane.text
    assert (
        "Add files, keep intake progress visible, and move straight into the operator inbox below."
        not in work_pane.text
    )
    assert "Live intake" in work_pane.text
    assert "Operator inbox" not in work_pane.text
    assert "Only in-flight uploads stay here" in work_pane.text
    assert "var removeTerminalItems = true;" in work_pane.text
    assert "Fallback and Admin Pages" not in work_pane.text
    assert "meeting.mp3" in work_pane.text
    assert 'id="control-center-recordings-panel"' in work_pane.text
    assert 'class="recordings-filter-strip filters"' not in work_pane.text
    assert 'class="recordings-status-strip"' not in work_pane.text
    assert "data-workspace-header-url" not in work_pane.text
    assert (
        'data-system-bar-url="/ui/control-center/system-bar?selected=rec-ui-1&amp;status=Ready&amp;'
        'q=meeting&amp;tab=speakers&amp;limit=25&amp;offset=0"'
    ) in work_pane.text
    assert "<html" not in work_pane.text

    inspector = seeded_client.get(
        "/ui/control-center/inspector-pane?selected=rec-ui-1&tab=speakers"
    )
    assert inspector.status_code == 200
    assert "rec-ui-1" in inspector.text
    assert "Recording Details" in inspector.text
    assert "SPEAKERS" in inspector.text
    assert "<nav" not in inspector.text

    empty_inspector = seeded_client.get("/ui/control-center/inspector-pane")
    assert empty_inspector.status_code == 200
    assert "No recording selected" in empty_inspector.text


def test_control_center_selected_recording_renders_embedded_inspector_actions(
    seeded_client,
):
    r = seeded_client.get("/?selected=rec-ui-1&status=Ready&q=meeting&tab=overview")
    assert r.status_code == 200
    assert 'id="control-center-inspector-pane"' in r.text
    assert "Recording Details" in r.text
    assert 'data-testid="recording-inspector-tab-overview"' not in r.text
    assert "Download ZIP" in r.text
    assert "Open Recording Page" in r.text
    assert "/ui/recordings/rec-ui-1/inspector?status=Ready&amp;q=meeting" in r.text


def test_control_center_processing_recording_polls_embedded_details_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    init_db(cfg)
    create_recording(
        "rec-ui-processing-1",
        source="upload",
        source_filename="processing.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )

    client = TestClient(api.app, follow_redirects=True)
    inspector = client.get("/ui/control-center/inspector-pane?selected=rec-ui-processing-1")
    assert inspector.status_code == 200
    assert "Recording Details" in inspector.text
    assert "every 2s" in inspector.text


def test_control_center_inspector_ignores_malformed_snippet_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    init_db(cfg)
    create_recording(
        "rec-ui-malformed-snippets",
        source="upload",
        source_filename="malformed.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-ui-malformed-snippets" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps({"summary_bullets": ["Speaker review pending."]}),
        encoding="utf-8",
    )
    (derived / "transcript.json").write_text(
        json.dumps({"text": "hello", "dominant_language": "en"}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    (derived / "snippets_manifest.json").write_text(
        json.dumps(
            {
                "speakers": {
                    "S1": [
                        {
                            "status": "accepted",
                            "relative_path": "S1/1.wav",
                            "clip_start": "bad",
                            "clip_end": "bad",
                            "source_start": "bad",
                            "source_end": "bad",
                            "purity_score": "bad",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    snippets = derived / "snippets" / "S1"
    snippets.mkdir(parents=True, exist_ok=True)
    (snippets / "1.wav").write_bytes(b"fake-wav")
    profile = create_voice_profile("Andrea", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-ui-malformed-snippets",
        diar_speaker_label="S1",
        voice_profile_id=int(profile["id"]),
        confidence=0.98,
        settings=cfg,
    )

    client = TestClient(api.app, follow_redirects=True)
    inspector = client.get(
        "/ui/control-center/inspector-pane?selected=rec-ui-malformed-snippets"
    )

    assert inspector.status_code == 200
    assert "Recording Details" in inspector.text
    assert "Andrea" in inspector.text
    assert "98%" in inspector.text


def test_control_center_embedded_inspector_open_page_link_preserves_shell_state(
    seeded_client,
):
    inspector = seeded_client.get(
        "/ui/recordings/rec-ui-1/inspector?status=Ready&q=meeting&tab=speakers"
    )
    assert inspector.status_code == 200
    assert "Recording Details" in inspector.text
    assert 'data-testid="recording-inspector-tab-overview"' not in inspector.text
    assert 'href="/recordings/rec-ui-1?tab=speakers"' in inspector.text

    summary_inspector = seeded_client.get(
        "/ui/recordings/rec-ui-1/inspector?status=Ready&q=meeting&tab=summary"
    )
    assert summary_inspector.status_code == 200
    assert 'href="/recordings/rec-ui-1?tab=summary"' in summary_inspector.text


def test_control_center_embedded_inspector_overview_stays_compact(seeded_client):
    overview = seeded_client.get(
        "/?selected=rec-ui-1&status=Ready&q=meeting&tab=overview"
    )
    assert overview.status_code == 200
    assert "Recording Details" in overview.text
    assert "KEY METADATA" in overview.text
    assert "SPEAKERS" in overview.text
    assert "TONE" in overview.text
    assert "SUMMARY" in overview.text
    assert ">Stage<" not in overview.text
    assert ">Blocker<" not in overview.text
    assert ">Next action<" not in overview.text
    assert "Pipeline Stages" not in overview.text
    assert "Diagnostics" not in overview.text
    assert "Save as correction" not in overview.text
    assert "Current / last stage" not in overview.text

    speakers = seeded_client.get(
        "/?selected=rec-ui-1&status=Ready&q=meeting&tab=speakers"
    )
    assert speakers.status_code == 200
    assert "Open canonical speakers page" not in speakers.text
    assert "Not available yet" in speakers.text


def test_control_center_embedded_details_card_uses_summary_artifacts(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    init_db(cfg)
    create_recording(
        "rec-ui-summary-1",
        source="upload",
        source_filename="summary.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-ui-summary-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Roadmap review",
                "summary_bullets": ["Reviewed blockers", "Aligned on next steps"],
                "decisions": ["Ship compact inspector"],
                "action_items": [{"task": "Polish export copy", "owner": "Alex"}],
                "questions": {
                    "total_count": 1,
                    "types": {"clarification": 1},
                    "extracted": ["Should export stay compact?"],
                },
                "emotional_summary": "Focused and calm.",
            }
        ),
        encoding="utf-8",
    )
    (derived / "transcript.json").write_text(
        json.dumps({"dominant_language": "en"}),
        encoding="utf-8",
    )

    client = TestClient(api.app, follow_redirects=True)
    summary = client.get("/ui/recordings/rec-ui-summary-1/inspector?tab=summary")
    assert summary.status_code == 200
    assert "Recording Details" in summary.text
    assert "Roadmap review" in summary.text
    assert "Reviewed blockers" in summary.text
    assert "Aligned on next steps" in summary.text
    assert "Focused and calm." in summary.text
    assert "Ship compact inspector" not in summary.text
    assert "Polish export copy" not in summary.text
    assert "Pipeline Stages" not in summary.text
    assert "Download ZIP" in summary.text
    assert 'href="/recordings/rec-ui-summary-1?tab=summary"' in summary.text

    export = client.get("/ui/recordings/rec-ui-summary-1/inspector?tab=export")
    assert export.status_code == 200
    assert "Recording Details" in export.text
    assert "Download ZIP" in export.text
    assert 'href="/recordings/rec-ui-summary-1?tab=export"' in export.text


def test_control_center_workflow_upload_select_speaker_decision_and_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    _stub_upload_enqueue(monkeypatch, cfg)

    c = TestClient(api.app, follow_redirects=True)
    upload = c.post(
        "/api/uploads",
        files={"file": ("2026-03-15 10_30_00.mp3", b"abc", "audio/mpeg")},
    )
    assert upload.status_code == 200
    recording_id = upload.json()["recording_id"]

    home = c.get("/")
    assert home.status_code == 200
    assert f'data-recording-id="{recording_id}"' in home.text
    assert "No recording selected" in home.text

    _seed_speaker_artifacts(cfg, recording_id)
    set_recording_status(recording_id, RECORDING_STATUS_READY, settings=cfg)

    speakers = c.get(f"/?selected={recording_id}&tab=speakers")
    assert speakers.status_code == 200
    assert "Recording Details" in speakers.text
    assert "SPEAKERS" in speakers.text
    assert "hello team" not in speakers.text
    assert "Add trusted sample" not in speakers.text

    updated_speakers = c.post(
        f"/ui/recordings/{recording_id}/speakers/local-label?return_to=control-center&return_tab=speakers",
        data={
            "diar_speaker_label": "S1",
            "local_display_name": "Guest Reviewer",
        },
    )
    assert updated_speakers.status_code == 200
    assert "Guest Reviewer" in updated_speakers.text

    assignments = list_speaker_assignments(recording_id, settings=cfg)
    assert len(assignments) == 1
    assert assignments[0]["review_state"] == "local_label"
    assert assignments[0]["local_display_name"] == "Guest Reviewer"

    overview = c.get(f"/?selected={recording_id}")
    assert overview.status_code == 200
    assert "Review blocker / Next action" in overview.text
    assert "Save as correction" not in overview.text

    correction = c.post(
        "/glossary",
        data={
            "canonical_text": "Sander",
            "aliases_text": "Sandia",
            "kind": "term",
            "source": "correction",
            "enabled": "1",
            "notes": "from workflow",
            "recording_id": recording_id,
            "return_to": "control-center",
            "return_after_save": "control-center",
            "selected": recording_id,
            "tab": "overview",
        },
    )
    assert correction.status_code == 200
    assert 'id="control-center-inspector-pane"' in correction.text

    entries = list_glossary_entries(settings=cfg)
    assert len(entries) == 1
    assert entries[0]["canonical_text"] == "Sander"
    assert entries[0]["aliases_json"] == ["Sandia"]
    assert entries[0]["metadata_json"] == {"recording_id": recording_id}


def test_control_center_recordings_panel_filters_search_and_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-cc-panel-1",
        source="upload",
        source_filename="alpha.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    create_recording(
        "rec-cc-panel-2",
        source="upload",
        source_filename="beta.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    panel = c.get(
        "/ui/control-center/recordings/panel?status=Ready&q=alpha&tab=speakers"
    )
    assert panel.status_code == 200
    assert 'id="control-center-recordings-panel"' in panel.text
    assert 'class="recordings-filter-strip filters"' not in panel.text
    assert 'class="recordings-status-strip"' not in panel.text
    assert "alpha.wav" in panel.text
    assert "beta.wav" not in panel.text
    assert ">Meeting<" in panel.text
    assert ">Recognition<" in panel.text
    assert ">Source<" not in panel.text
    assert ">Confidence<" not in panel.text
    assert "..." in panel.text
    assert 'aria-label="Open actions for rec-cc-panel-1"' in panel.text
    assert 'data-testid="control-center-select-recording"' not in panel.text
    assert 'data-select-href="/?selected=rec-cc-panel-1&amp;status=Ready&amp;q=alpha&amp;tab=speakers"' in panel.text
    assert "data-workspace-header-url" not in panel.text
    assert (
        'data-system-bar-url="/ui/control-center/system-bar?selected=&amp;status=Ready&amp;q=alpha&amp;'
        'tab=speakers&amp;limit=25&amp;offset=0"'
    ) in panel.text
    assert (
        'href="/?selected=rec-cc-panel-1&amp;status=Ready&amp;q=alpha&amp;tab=speakers"'
        in panel.text
    )
    assert panel.headers["HX-Push-Url"] == "/?status=Ready&q=alpha&tab=speakers"

    selected_panel = c.get(
        "/ui/control-center/recordings/panel?selected=rec-cc-panel-1&tab=speakers"
    )
    assert selected_panel.status_code == 200
    assert 'data-selected="true"' in selected_panel.text
    assert 'aria-current="page"' in selected_panel.text

    conservative = c.get("/ui/control-center/recordings/panel?q=upload")
    assert conservative.status_code == 200
    assert "alpha.wav" not in conservative.text
    assert "beta.wav" not in conservative.text


def test_control_center_recordings_panel_derives_meeting_titles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    create_recording(
        "rec-title-calendar",
        source="upload",
        source_filename="calendar-fallback.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived_calendar = cfg.recordings_root / "rec-title-calendar" / "derived"
    derived_calendar.mkdir(parents=True, exist_ok=True)
    (derived_calendar / "summary.json").write_text(
        json.dumps({"topic": "Ignored summary topic"}),
        encoding="utf-8",
    )
    upsert_calendar_match(
        recording_id="rec-title-calendar",
        candidates=[
            {
                "event_id": "evt-calendar",
                "subject": "Quarterly roadmap",
                "attendees": ["Alex"],
            }
        ],
        selected_event_id="evt-calendar",
        selected_confidence=0.96,
        settings=cfg,
    )

    create_recording(
        "rec-title-summary",
        source="upload",
        source_filename="summary-fallback.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived_summary = cfg.recordings_root / "rec-title-summary" / "derived"
    derived_summary.mkdir(parents=True, exist_ok=True)
    (derived_summary / "summary.json").write_text(
        json.dumps({"topic": "Weekly summary review"}),
        encoding="utf-8",
    )

    create_recording(
        "rec-title-fallback",
        source="upload",
        source_filename="fallback-title.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    panel = c.get("/ui/control-center/recordings/panel")

    assert panel.status_code == 200
    assert "rec-title-calendar" in panel.text
    assert "Quarterly roadmap" in panel.text
    assert "Ignored summary topic" not in panel.text
    assert "rec-title-summary" in panel.text
    assert "Weekly summary review" in panel.text
    assert "rec-title-fallback" in panel.text
    assert "fallback-title.wav" in panel.text


def test_control_center_selection_preserves_pagination_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    for index in range(26):
        create_recording(
            f"rec-cc-page-{index:02d}",
            source="upload",
            source_filename=f"page-{index:02d}.wav",
            status=RECORDING_STATUS_READY,
            settings=cfg,
        )

    paged_rows, total = list_recordings(
        settings=cfg,
        status=RECORDING_STATUS_READY,
        limit=25,
        offset=25,
    )
    assert total == 26
    selected_id = paged_rows[0]["id"]

    c = TestClient(api.app, follow_redirects=True)
    panel = c.get(
        "/ui/control-center/recordings/panel?status=Ready&limit=25&offset=25&tab=speakers"
    )
    assert panel.status_code == 200
    assert (
        panel.headers["HX-Push-Url"] == "/?status=Ready&tab=speakers&limit=25&offset=25"
    )
    assert (
        f'data-select-href="/?selected={selected_id}&amp;status=Ready&amp;tab=speakers&amp;'
        'limit=25&amp;offset=25"'
    ) in panel.text

    selected_page = c.get(
        f"/?selected={selected_id}&status=Ready&limit=25&offset=25&tab=speakers"
    )
    assert selected_page.status_code == 200
    assert 'data-selected="true"' in selected_page.text
    assert ">26–26 of 26<" in selected_page.text


# ---------------------------------------------------------------------------
# Recordings list
# ---------------------------------------------------------------------------


def test_recordings_empty(client):
    r = client.get("/recordings")
    assert r.status_code == 200
    assert "Recordings" in r.text
    assert "Back to Control Center" in r.text


def test_recordings_with_data(seeded_client):
    r = seeded_client.get("/recordings")
    assert r.status_code == 200
    assert "meeting.mp3" in r.text
    assert 'id="recordings-page-panel"' in r.text
    assert "Delete record?" in r.text
    assert 'id="delete-confirm-backdrop"' in r.text
    assert 'data-rlabel="meeting.mp3"' in r.text
    assert "Type DELETE to confirm deletion." not in r.text


def test_recordings_list_shows_progress_column_and_percent(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-list-progress-1",
        source="drive",
        source_filename="list-progress.mp3",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    set_recording_progress(
        "rec-ui-list-progress-1",
        stage="diarize",
        progress=0.5,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings")
    assert r.status_code == 200
    assert "Progress" in r.text
    assert "50%" in r.text


def test_recordings_fragment_endpoints_render_filters_table_and_pagination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-fragment-1",
        source="upload",
        source_filename="fragment-a.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_recording(
        "rec-ui-fragment-2",
        source="upload",
        source_filename="fragment-b.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    create_recording(
        "rec-ui-fragment-3",
        source="upload",
        source_filename="fragment-c.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    set_recording_progress(
        "rec-ui-fragment-1",
        stage="diarize",
        progress=0.5,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    filters = c.get("/ui/control-center/recordings/filters?status=Ready&limit=25")
    assert filters.status_code == 200
    assert 'class="recordings-filter-strip filters"' in filters.text
    assert 'hx-get="/ui/control-center/recordings/panel"' in filters.text
    assert '<option value="Ready" selected>' in filters.text
    assert 'placeholder="ID or filename"' in filters.text
    assert "<html" not in filters.text

    table = c.get("/ui/control-center/recordings/table?q=fragment&limit=2&offset=0")
    assert table.status_code == 200
    assert "fragment-a.wav" in table.text
    assert ">Recognition<" in table.text
    assert ">Source<" not in table.text
    assert ">Confidence<" not in table.text
    assert "52%" in table.text
    assert "Next &#187;" in table.text
    assert "Delete" in table.text
    assert 'aria-label="Open actions for rec-ui-fragment-1"' in table.text
    assert 'data-select-href="/?selected=rec-ui-fragment-1&amp;q=fragment&amp;limit=2"' in table.text
    assert "<nav" not in table.text


def test_recordings_list_shows_review_reason_text(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-review-list-1",
        source="upload",
        source_filename="needs-review.wav",
        status=RECORDING_STATUS_NEEDS_REVIEW,
        review_reason_code="routing_low_confidence",
        review_reason_text="Project routing confidence is too low.",
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings")
    assert r.status_code == 200
    assert "Project routing confidence is too low." in r.text


def test_recordings_status_filter(seeded_client):
    r = seeded_client.get("/recordings?status=Ready")
    assert r.status_code == 200
    assert "meeting.mp3" in r.text


def test_recordings_q_search_is_conservative(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-search-1",
        source="upload",
        source_filename="search-me.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    by_filename = c.get("/recordings?q=search-me")
    assert by_filename.status_code == 200
    assert "search-me.wav" in by_filename.text

    by_source = c.get("/recordings?q=upload")
    assert by_source.status_code == 200
    assert "search-me.wav" not in by_source.text


def test_recordings_invalid_status_filter_shows_all(seeded_client):
    r = seeded_client.get("/recordings?status=InvalidStatus")
    assert r.status_code == 200
    assert "meeting.mp3" in r.text


def test_recordings_pagination(seeded_client):
    r = seeded_client.get("/recordings?limit=1&offset=0")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Recording detail
# ---------------------------------------------------------------------------


def test_recording_detail_overview(seeded_client):
    r = seeded_client.get("/recordings/rec-ui-1")
    assert r.status_code == 200
    assert "rec-ui-1" in r.text
    assert "meeting.mp3" in r.text
    assert "Ready" in r.text
    assert 'data-rid="rec-ui-1"' in r.text
    assert 'data-rlabel="meeting.mp3"' in r.text
    assert 'href="/ui/recordings/rec-ui-1/export.zip"' in r.text
    assert 'hx-boost="false"' in r.text


def test_recording_shell_and_empty_inspector_fragment_endpoints(seeded_client):
    shell = seeded_client.get(
        "/ui/control-center/recordings/rec-ui-1/shell?tab=overview"
    )
    assert shell.status_code == 200
    assert 'data-rid="rec-ui-1"' in shell.text
    assert "Requeue" in shell.text
    assert "Download ZIP" in shell.text
    assert "Back to Control Center" in shell.text
    assert "<html" not in shell.text

    empty = seeded_client.get("/ui/control-center/inspector-empty")
    assert empty.status_code == 200
    assert "No recording selected" in empty.text
    assert "<nav" not in empty.text

    missing = seeded_client.get("/ui/control-center/recordings/missing/shell")
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("status", "expects_stop", "expects_disabled"),
    [
        (RECORDING_STATUS_QUEUED, True, False),
        (RECORDING_STATUS_PROCESSING, True, False),
        (RECORDING_STATUS_STOPPING, False, True),
        (RECORDING_STATUS_STOPPED, False, False),
        (RECORDING_STATUS_READY, False, False),
    ],
)
def test_recording_detail_stop_button_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expects_stop: bool,
    expects_disabled: bool,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-stop-visibility-1",
        source="upload",
        source_filename="stop.wav",
        status=status,
        settings=cfg,
    )
    if status == RECORDING_STATUS_STOPPING:
        set_recording_cancel_request(
            "rec-ui-stop-visibility-1",
            requested_by="user",
            reason_code="user_stop",
            reason_text="Stop requested by user",
            settings=cfg,
        )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-ui-stop-visibility-1")
    assert r.status_code == 200
    if expects_stop:
        assert "/ui/recordings/rec-ui-stop-visibility-1/stop" in r.text
    else:
        assert "/ui/recordings/rec-ui-stop-visibility-1/stop" not in r.text
    if expects_disabled:
        assert "Stopping..." in r.text
    else:
        assert "Stopping..." not in r.text


def test_recording_detail_transcript_tab_shows_asr_glossary_actions(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-glossary-1",
        source="upload",
        source_filename="glossary.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-ui-glossary-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "asr_glossary.json").write_text(
        json.dumps(
            {
                "entry_count": 2,
                "term_count": 3,
                "truncated": False,
                "entries": [
                    {
                        "canonical_text": "Sander",
                        "aliases": ["Sandia"],
                        "kind": "person",
                        "sources": ["correction", "speaker_bank"],
                    },
                    {
                        "canonical_text": "Quarterly Roadmap",
                        "aliases": [],
                        "kind": "project",
                        "sources": ["calendar"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-ui-glossary-1?tab=transcript")
    assert r.status_code == 200
    assert "Transcript" in r.text
    assert "Manage corrections" in r.text
    assert "Add correction from this recording" in r.text
    assert '/glossary?recording_id=rec-ui-glossary-1"' in r.text


def test_recording_detail_calendar_tab_renders_selected_candidate_and_rationale(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-calendar-1",
        source="upload",
        source_filename="roadmap.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-ui-calendar-1",
        candidates=[
            {
                "event_id": "evt-roadmap",
                "subject": "Roadmap Weekly Sync",
                "starts_at": "2026-03-01T10:00:00Z",
                "ends_at": "2026-03-01T11:00:00Z",
                "source_name": "Team Calendar",
                "organizer": "Alex Example",
                "attendee_details": [
                    {"label": "Priya Kapoor", "email": "priya@example.com"},
                    {"label": "Marco Rossi"},
                ],
                "confidence": 0.92,
                "rationale": [
                    "Capture time falls inside the event window.",
                    "Final confidence 0.92.",
                ],
            },
            {
                "event_id": "evt-other",
                "subject": "Other Meeting",
                "starts_at": "2026-03-01T12:00:00Z",
                "ends_at": "2026-03-01T13:00:00Z",
                "source_name": "Team Calendar",
                "organizer": "Jordan",
                "attendees": ["Alex Example"],
                "confidence": 0.33,
                "rationale": "Low confidence",
            },
        ],
        selected_event_id="evt-roadmap",
        selected_confidence=0.92,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    response = c.get("/recordings/rec-ui-calendar-1?tab=calendar")
    assert response.status_code == 200
    assert "Calendar Match" in response.text
    assert "Roadmap Weekly Sync" in response.text
    assert "Capture time falls inside the event window." in response.text
    assert "Priya Kapoor" in response.text
    assert "Selected" in response.text


def test_recording_detail_calendar_tab_shows_weak_and_ambiguous_warnings(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-calendar-warning-1",
        source="upload",
        source_filename="meeting.wav",
        captured_at="2026-03-01T09:05:00Z",
        captured_at_timezone="Europe/Rome",
        captured_at_inferred_from_filename=False,
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-ui-calendar-warning-1",
        candidates=[
            {
                "event_id": "evt-a",
                "subject": "Planning A",
                "starts_at": "2026-03-01T09:00:00Z",
                "ends_at": "2026-03-01T10:00:00Z",
                "confidence": 0.81,
            },
            {
                "event_id": "evt-b",
                "subject": "Planning B",
                "starts_at": "2026-03-01T09:03:00Z",
                "ends_at": "2026-03-01T10:03:00Z",
                "confidence": 0.76,
            },
        ],
        selected_event_id=None,
        selected_confidence=None,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    response = c.get("/recordings/rec-ui-calendar-warning-1?tab=calendar")
    assert response.status_code == 200
    assert "used receipt time instead of a filename capture timestamp" in response.text
    assert "Multiple nearby calendar candidates scored too closely" in response.text


def test_recording_detail_calendar_selection_and_clear_persist(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-calendar-select-1",
        source="upload",
        source_filename="planning.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-ui-calendar-select-1",
        candidates=[
            {
                "event_id": "evt-a",
                "subject": "Planning",
                "confidence": 0.81,
            }
        ],
        selected_event_id=None,
        selected_confidence=None,
        settings=cfg,
    )
    monkeypatch.setattr(ui_routes, "refresh_recording_routing", lambda *_a, **_k: {})

    c = TestClient(api.app, follow_redirects=False)
    selected = c.post(
        "/ui/recordings/rec-ui-calendar-select-1/calendar/select",
        data={"event_id": "evt-a"},
    )
    assert selected.status_code == 303
    assert (
        selected.headers["location"]
        == "/recordings/rec-ui-calendar-select-1?tab=diagnostics"
    )
    match = get_calendar_match("rec-ui-calendar-select-1", settings=cfg)
    assert match is not None
    assert match["selected_event_id"] == "evt-a"
    assert match["selected_confidence"] == 0.81

    cleared = c.post("/ui/recordings/rec-ui-calendar-select-1/calendar/select", data={})
    assert cleared.status_code == 303
    assert (
        cleared.headers["location"]
        == "/recordings/rec-ui-calendar-select-1?tab=diagnostics"
    )
    match = get_calendar_match("rec-ui-calendar-select-1", settings=cfg)
    assert match is not None
    assert match["selected_event_id"] is None


def test_recording_detail_processing_polls_progress(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-processing-1",
        source="drive",
        source_filename="processing.mp3",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-ui-processing-1")
    assert r.status_code == 200
    assert "/ui/recordings/rec-ui-processing-1/progress?tab=overview" in r.text
    assert "every 2s" in r.text


def test_recording_progress_endpoint_renders_expected_html(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-progress-1",
        source="drive",
        source_filename="progress.mp3",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    set_recording_progress(
        "rec-ui-progress-1",
        stage="diarize",
        progress=0.5,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/ui/recordings/rec-ui-progress-1/progress")
    assert r.status_code == 200
    assert "Pipeline:" in r.text
    assert "Diarization" in r.text
    assert "50%" in r.text
    assert "stage=<code>diarize</code>" in r.text


def test_recording_progress_endpoint_shows_llm_chunk_diagnostics(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-progress-llm-1",
        source="drive",
        source_filename="progress-llm.mp3",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    set_recording_progress(
        "rec-ui-progress-llm-1",
        stage="llm_chunk_2_of_4",
        progress=0.87,
        settings=cfg,
    )
    mark_recording_pipeline_stage_started(
        "rec-ui-progress-llm-1",
        stage_name="llm_extract",
        metadata={"label": "LLM Summary", "resumed": True},
        settings=cfg,
    )
    mark_recording_llm_chunk_started(
        "rec-ui-progress-llm-1",
        chunk_group="extract",
        chunk_index="2",
        chunk_total=4,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/ui/recordings/rec-ui-progress-llm-1/progress")
    assert r.status_code == 200
    assert "LLM Chunk 2 of 4" in r.text
    assert "Chunk:</strong> 2/4" in r.text
    assert "Mode:</strong> resuming" in r.text


def test_recording_progress_endpoint_redirects_when_terminal(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-terminal-1",
        source="upload",
        source_filename="terminal.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.get(
        "/ui/recordings/rec-ui-terminal-1/progress?tab=metrics",
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert r.headers["HX-Redirect"] == "/recordings/rec-ui-terminal-1?tab=summary"


def test_recording_progress_endpoint_redirects_to_control_center_when_embedded_terminal(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-terminal-embedded-1",
        source="upload",
        source_filename="terminal-embedded.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.get(
        "/ui/recordings/rec-ui-terminal-embedded-1/progress?"
        "tab=metrics&return_to=control-center&status=Ready&q=meeting&return_tab=metrics",
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert (
        r.headers["HX-Redirect"]
        == "/?selected=rec-ui-terminal-embedded-1&status=Ready&q=meeting"
    )


def test_recording_detail_shows_review_reason_and_local_timestamp(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-review-detail-1",
        source="upload",
        source_filename="review-detail.wav",
        captured_at="2026-01-10T10:00:00Z",
        status=RECORDING_STATUS_NEEDS_REVIEW,
        review_reason_code="routing_low_confidence",
        review_reason_text="Project routing confidence is too low.",
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-ui-review-detail-1")
    assert r.status_code == 200
    assert "Project routing confidence is too low." in r.text
    assert "2026-01-10 11:00:00 CET" in r.text


def test_recording_detail_shows_primary_diagnostics_section(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-diagnostics-1",
        source="upload",
        source_filename="diagnostics.wav",
        status=RECORDING_STATUS_NEEDS_REVIEW,
        review_reason_code="job_retry_limit_reached",
        review_reason_text="Processing hit the retry limit after repeated failures.",
        settings=cfg,
    )
    create_job(
        "job-ui-diagnostics-1",
        recording_id="rec-ui-diagnostics-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_FAILED,
        error="max attempts exceeded",
        settings=cfg,
    )
    mark_recording_pipeline_stage_failed(
        "rec-ui-diagnostics-1",
        stage_name="llm_extract",
        error_code="llm_chunk_timeout",
        error_text="LLM chunk 3/10 failed [llm_chunk_timeout]: timed out after 120s",
        metadata={
            "label": "LLM Summary",
            "root_cause_code": "llm_chunk_timeout",
            "root_cause_text": "LLM chunk 3/10 timed out.",
            "root_cause_detail": "timed out after 120s",
            "chunk_index": "3",
            "chunk_total": 10,
            "resumed": True,
        },
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-ui-diagnostics-1?tab=diagnostics")
    assert r.status_code == 200
    assert "Diagnostics" in r.text
    assert "LLM chunk 3/10 timed out." in r.text
    assert "<code>llm_chunk_timeout</code>" in r.text
    assert "Automatic retries hit the configured retry limit." in r.text


def test_recording_detail_persists_duration_from_sanitized_audio(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-duration-1",
        source="upload",
        source_filename="duration.wav",
        settings=cfg,
    )
    _write_pcm_wav(
        cfg.recordings_root / "rec-ui-duration-1" / "derived" / "audio_sanitized.wav",
        duration_sec=2.0,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-ui-duration-1")
    assert r.status_code == 200
    assert "00:00:02" in r.text
    recording = get_recording("rec-ui-duration-1", settings=cfg)
    assert recording is not None
    assert recording["duration_sec"] == 2.0


def test_recording_detail_log_tab(seeded_client):
    r = seeded_client.get("/recordings/rec-ui-1?tab=log")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_recording_detail_project_tab(seeded_client):
    r = seeded_client.get("/recordings/rec-ui-1?tab=project")
    assert r.status_code == 200
    assert "Project Assignment" in r.text
    assert "Save project" in r.text
    assert "Routing Rationale" in r.text


def test_recording_detail_shows_stuck_recovery_warning(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-ui-stuck-1",
        source="drive",
        source_filename="stuck.mp3",
        status=RECORDING_STATUS_NEEDS_REVIEW,
        settings=cfg,
    )
    create_job(
        "job-ui-stuck-1",
        recording_id="rec-ui-stuck-1",
        job_type=JOB_TYPE_PRECHECK,
        status=JOB_STATUS_FAILED,
        error="stuck job recovered",
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-ui-stuck-1")
    assert r.status_code == 200
    assert "recovered from a stuck job" in r.text


def test_recording_detail_project_assignment_trains_routing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    project = create_project("Roadmap", settings=cfg)
    create_recording(
        "rec-project-assign-1",
        source="drive",
        source_filename="roadmap-sync.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    profile = create_voice_profile("Alex", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-project-assign-1",
        diar_speaker_label="S1",
        voice_profile_id=profile["id"],
        confidence=1.0,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-project-assign-1",
        candidates=[
            {
                "event_id": "evt-1",
                "subject": "Roadmap weekly sync",
                "organizer": "Alex",
                "attendees": ["Priya"],
                "score": 0.95,
                "rationale": "manual-test",
            }
        ],
        selected_event_id="evt-1",
        selected_confidence=0.95,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-project-assign-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Roadmap status update",
                "summary_bullets": ["Roadmap milestones reviewed"],
                "decisions": ["Ship Q2 beta"],
                "action_items": [{"task": "Finalize roadmap", "owner": "Alex"}],
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=False)
    resp = c.post(
        "/ui/recordings/rec-project-assign-1/project",
        data={
            "project_id": str(project["id"]),
            "train_routing": "1",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/recordings/rec-project-assign-1?tab=diagnostics"

    recording = get_recording("rec-project-assign-1", settings=cfg)
    assert recording is not None
    assert recording["project_id"] == project["id"]
    assert count_routing_training_examples(project_id=project["id"], settings=cfg) == 1
    keyword_rows = list_project_keyword_weights(project_id=project["id"], settings=cfg)
    keywords = {row["keyword"] for row in keyword_rows}
    assert "cal:roadmap" in keywords
    assert "tag:roadmap" in keywords
    assert f"voice:{profile['id']}" in keywords
    assert recording["status"] in {
        RECORDING_STATUS_READY,
        RECORDING_STATUS_NEEDS_REVIEW,
    }


def test_recording_detail_speakers_tab_assignment_persists(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-1",
        source="drive",
        source_filename="speakers.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-1")
    profile = create_voice_profile("Alice Example", settings=cfg)

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-speakers-1/speakers/assign",
        data={
            "diar_speaker_label": "S1",
            "voice_profile_id": str(profile["id"]),
        },
    )
    assert r.status_code == 303

    assignments = list_speaker_assignments("rec-speakers-1", settings=cfg)
    assert len(assignments) == 1
    assert assignments[0]["diar_speaker_label"] == "S1"
    assert assignments[0]["voice_profile_id"] == profile["id"]
    assert assignments[0]["review_state"] == "confirmed_canonical"

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-1?tab=speakers"
    )
    assert page.status_code == 200
    assert ">Speakers<" in page.text
    assert "Alice Example" in page.text
    assert "Confirm match" in page.text
    assert "Mapped globally" in page.text
    assert "Add trusted sample" in page.text
    assert "Best match" in page.text
    assert "Purity 88%" in page.text
    assert "Recognition cue" in page.text


def test_recording_detail_speakers_create_and_assign(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-create-1",
        source="drive",
        source_filename="speakers-create.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-create-1")

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-speakers-create-1/speakers/create-and-assign",
        data={
            "diar_speaker_label": "S2",
            "display_name": "Bob New",
            "notes": "ops",
        },
    )
    assert r.status_code == 303

    profiles = list_voice_profiles(settings=cfg)
    assert any(row["display_name"] == "Bob New" for row in profiles)
    assignments = list_speaker_assignments("rec-speakers-create-1", settings=cfg)
    assert len(assignments) == 1
    assert assignments[0]["diar_speaker_label"] == "S2"
    assert assignments[0]["voice_profile_name"] == "Bob New"
    assert assignments[0]["review_state"] == "confirmed_canonical"


def test_recording_detail_speakers_keep_unknown_persists_intentional_review(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-unknown-1",
        source="drive",
        source_filename="speakers-unknown.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-unknown-1")
    profile = create_voice_profile("Unknown Candidate", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-speakers-unknown-1",
        diar_speaker_label="S1",
        voice_profile_id=None,
        confidence=0.61,
        candidate_matches=[{"voice_profile_id": profile["id"], "score": 0.61}],
        low_confidence=True,
        review_state="system_suggested",
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-speakers-unknown-1/speakers/keep-unknown",
        data={"diar_speaker_label": "S1"},
    )
    assert r.status_code == 303

    assignments = list_speaker_assignments("rec-speakers-unknown-1", settings=cfg)
    assert len(assignments) == 1
    assert assignments[0]["voice_profile_id"] is None
    assert assignments[0]["review_state"] == "kept_unknown"
    assert assignments[0]["low_confidence"] == 0

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-unknown-1?tab=speakers"
    )
    assert page.status_code == 200
    assert "Unknown by choice" in page.text
    assert "Below auto-match threshold 0.75" not in page.text


def test_recording_detail_speakers_local_label_shows_in_detail_and_export(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-local-1",
        source="drive",
        source_filename="speakers-local.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-local-1")
    derived = cfg.recordings_root / "rec-speakers-local-1" / "derived"
    (derived / "summary.json").write_text(
        json.dumps(
            {"topic": "Speaker review", "summary_bullets": ["Discussed labels."]}
        ),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps([{"speaker": "S1", "text": "I am only named for this meeting."}]),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-speakers-local-1/speakers/local-label",
        data={
            "diar_speaker_label": "S1",
            "local_display_name": "Design Lead",
        },
    )
    assert r.status_code == 303

    assignments = list_speaker_assignments("rec-speakers-local-1", settings=cfg)
    assert len(assignments) == 1
    assert assignments[0]["voice_profile_id"] is None
    assert assignments[0]["review_state"] == "local_label"
    assert assignments[0]["local_display_name"] == "Design Lead"
    assert list_voice_profiles(settings=cfg) == []

    speakers_page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-local-1?tab=speakers"
    )
    assert speakers_page.status_code == 200
    assert "Local label only" in speakers_page.text
    assert "Design Lead" in speakers_page.text

    export_resp = TestClient(api.app, follow_redirects=True).get(
        "/ui/recordings/rec-speakers-local-1/export.zip"
    )
    assert export_resp.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(export_resp.content))
    markdown = archive.read("onenote.md").decode("utf-8")
    assert "Design Lead (S1)" in markdown


def test_recording_detail_speakers_add_sample_links_snippet_and_audio_route(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-sample-1",
        source="drive",
        source_filename="speakers-sample.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-sample-1")
    profile = create_voice_profile("Cara Sample", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-speakers-sample-1",
        diar_speaker_label="S1",
        voice_profile_id=profile["id"],
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-speakers-sample-1/speakers/add-sample",
        data={
            "diar_speaker_label": "S1",
            "voice_profile_id": str(profile["id"]),
            "snippet_path": "S1/1.wav",
        },
    )
    assert r.status_code == 303

    samples = list_voice_samples(settings=cfg)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["voice_profile_id"] == profile["id"]
    assert sample["recording_id"] == "rec-speakers-sample-1"
    assert sample["sample_source"] == "trusted_sample"
    assert sample["snippet_path"].startswith(
        "recordings/rec-speakers-sample-1/derived/snippets/S1/"
    )

    snippet_resp = TestClient(api.app, follow_redirects=True).get(
        "/ui/recordings/rec-speakers-sample-1/snippets/S1/1.wav"
    )
    assert snippet_resp.status_code == 200
    assert snippet_resp.headers["content-type"].startswith("audio/wav")

    audio_resp = TestClient(api.app, follow_redirects=True).get(
        f"/ui/voice-samples/{sample['id']}/audio"
    )
    assert audio_resp.status_code == 200
    assert audio_resp.headers["content-type"].startswith("audio/wav")

    speakers_page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-sample-1?tab=speakers"
    )
    assert speakers_page.status_code == 200
    assert "Trusted sample saved for Cara Sample." in speakers_page.text


def test_recording_detail_speakers_add_sample_keeps_local_label_decision(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-local-sample-1",
        source="drive",
        source_filename="speakers-local-sample.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-local-sample-1")
    profile = create_voice_profile("Trusted Canonical", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-speakers-local-sample-1",
        diar_speaker_label="S1",
        voice_profile_id=None,
        review_state="local_label",
        local_display_name="Guest Speaker",
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-speakers-local-sample-1/speakers/add-sample",
        data={
            "diar_speaker_label": "S1",
            "voice_profile_id": str(profile["id"]),
            "snippet_path": "S1/1.wav",
        },
    )
    assert r.status_code == 303

    assignments = list_speaker_assignments("rec-speakers-local-sample-1", settings=cfg)
    assert len(assignments) == 1
    assert assignments[0]["review_state"] == "local_label"
    assert assignments[0]["voice_profile_id"] is None
    assert assignments[0]["local_display_name"] == "Guest Speaker"
    samples = list_voice_samples(settings=cfg)
    assert len(samples) == 1
    assert samples[0]["voice_profile_id"] == profile["id"]
    assert samples[0]["sample_source"] == "trusted_sample"

    speakers_page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-local-sample-1?tab=speakers"
    )
    assert speakers_page.status_code == 200
    assert "Local label only" in speakers_page.text
    assert "Trusted sample saved for Trusted Canonical." in speakers_page.text


def test_recording_detail_speakers_trusted_sample_state_ignores_manual_and_unscoped_samples(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-ignored-samples-1",
        source="drive",
        source_filename="speakers-ignored.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-ignored-samples-1")
    profile = create_voice_profile("Ignore Me", settings=cfg)
    create_voice_sample(
        voice_profile_id=profile["id"],
        recording_id="rec-speakers-ignored-samples-1",
        diar_speaker_label="S1",
        snippet_path="recordings/rec-speakers-ignored-samples-1/derived/snippets/S1/1.wav",
        sample_source="manual",
        settings=cfg,
    )
    create_voice_sample(
        voice_profile_id=profile["id"],
        recording_id="rec-speakers-ignored-samples-1",
        diar_speaker_label=" ",
        snippet_path="recordings/rec-speakers-ignored-samples-1/derived/snippets/S1/1.wav",
        sample_source="trusted_sample",
        settings=cfg,
    )

    speakers_page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-ignored-samples-1?tab=speakers"
    )
    assert speakers_page.status_code == 200
    assert "Trusted sample saved" not in speakers_page.text


def test_recording_detail_speakers_no_clean_snippet_shows_clear_message(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-no-clean-1",
        source="drive",
        source_filename="speakers-no-clean.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-no-clean-1")
    _write_snippets_manifest(
        cfg,
        "rec-speakers-no-clean-1",
        {
            "version": 1,
            "source_kind": "turn",
            "degraded_diarization": False,
            "pad_seconds": 0.25,
            "max_clip_duration_seconds": 8.0,
            "min_clip_duration_seconds": 0.8,
            "max_snippets_per_speaker": 3,
            "manifest_status": "no_clean_snippets",
            "speakers": {
                "S1": [
                    {
                        "snippet_id": "S1-01",
                        "speaker": "S1",
                        "source_kind": "turn",
                        "source_start": 0.0,
                        "source_end": 1.2,
                        "clip_start": 0.0,
                        "clip_end": 1.45,
                        "duration_seconds": 1.45,
                        "overlap_seconds": 0.31,
                        "overlap_ratio": 0.2138,
                        "purity_score": 0.62,
                        "ranking_position": 1,
                        "status": "rejected_overlap",
                        "recommended": False,
                        "extraction_backend": "none",
                    }
                ]
            },
        },
    )
    mark_recording_pipeline_stage_completed(
        "rec-speakers-no-clean-1",
        stage_name="snippet_export",
        metadata={"manifest_status": "no_clean_snippets"},
        settings=cfg,
    )
    profile = create_voice_profile("Blocked Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-no-clean-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Snippet export completed, but no clean snippets are available because every candidate overlaps another speaker."
        in page.text
    )
    assert (
        "No clean snippets are available because every candidate overlaps another speaker."
        in page.text
    )
    assert "rejected because it overlaps another speaker" in page.text
    assert "No snippet quality data found." not in page.text

    blocked = TestClient(api.app, follow_redirects=False).post(
        "/ui/recordings/rec-speakers-no-clean-1/speakers/add-sample",
        data={
            "diar_speaker_label": "S1",
            "voice_profile_id": str(profile["id"]),
            "snippet_path": "",
        },
    )
    assert blocked.status_code == 422
    assert "snippet_path is required" in blocked.text


def test_recording_detail_speakers_snippet_not_started_message(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-pending-1",
        source="upload",
        source_filename="pending.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-pending-1")
    set_recording_progress(
        "rec-speakers-pending-1",
        stage="speaker_turns",
        progress=0.8,
        settings=cfg,
    )
    create_voice_profile("Pending Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-pending-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Pending:</strong> The pipeline has not reached Snippet Export yet."
        in page.text
    )
    assert "Add sample will be available after Snippet Export runs." in page.text
    assert 'name="snippet_path" disabled' in page.text


def test_recording_detail_speakers_snippet_running_message(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-running-1",
        source="upload",
        source_filename="running.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-running-1")
    set_recording_progress(
        "rec-speakers-running-1",
        stage="snippet_export",
        progress=0.84,
        settings=cfg,
    )
    mark_recording_pipeline_stage_started(
        "rec-speakers-running-1",
        stage_name="snippet_export",
        settings=cfg,
    )
    create_voice_profile("Running Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-running-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Generating:</strong> Snippet export is currently generating clean clips for this recording."
        in page.text
    )
    assert (
        "Add sample will be available when clean clips finish generating." in page.text
    )


def test_recording_detail_speakers_ready_during_processing_keeps_snippets_usable(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-processing-ready-1",
        source="upload",
        source_filename="processing-ready.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-processing-ready-1")
    set_recording_progress(
        "rec-speakers-processing-ready-1",
        stage="llm_extract",
        progress=0.9,
        settings=cfg,
    )
    mark_recording_pipeline_stage_completed(
        "rec-speakers-processing-ready-1",
        stage_name="snippet_export",
        metadata={"manifest_status": "partial", "accepted_snippets": 2},
        settings=cfg,
    )
    create_voice_profile("Ready Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-processing-ready-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Ready:</strong> Clean clips are ready while processing continues in LLM Summary."
        in page.text
    )
    assert (
        "/ui/recordings/rec-speakers-processing-ready-1/snippets/S1/1.wav" in page.text
    )
    assert "Add trusted sample" in page.text
    assert 'name="snippet_path" disabled' not in page.text
    assert 'disabled>Add trusted sample' not in page.text


def test_recording_detail_speakers_add_sample_selector_keeps_all_clean_snippets(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-many-snippets-1",
        source="upload",
        source_filename="many-snippets.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-many-snippets-1")
    derived = cfg.recordings_root / "rec-speakers-many-snippets-1" / "derived"
    snippets = derived / "snippets" / "S1"
    snippets.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 5):
        (snippets / f"{idx}.wav").write_bytes(f"fake-wav-{idx}".encode("utf-8"))
    _write_snippets_manifest(
        cfg,
        "rec-speakers-many-snippets-1",
        {
            "version": 1,
            "source_kind": "turn",
            "degraded_diarization": False,
            "max_snippets_per_speaker": 4,
            "speakers": {
                "S1": [
                    {
                        "snippet_id": "S1-01",
                        "speaker": "S1",
                        "source_kind": "turn",
                        "source_start": 0.0,
                        "source_end": 1.0,
                        "clip_start": 0.0,
                        "clip_end": 1.0,
                        "purity_score": 0.95,
                        "ranking_position": 1,
                        "status": "accepted",
                        "recommended": True,
                        "relative_path": "S1/1.wav",
                    },
                    {
                        "snippet_id": "S1-02",
                        "speaker": "S1",
                        "source_kind": "turn",
                        "source_start": 1.0,
                        "source_end": 2.0,
                        "clip_start": 1.0,
                        "clip_end": 2.0,
                        "purity_score": 0.91,
                        "ranking_position": 2,
                        "status": "accepted",
                        "recommended": False,
                        "relative_path": "S1/2.wav",
                    },
                    {
                        "snippet_id": "S1-03",
                        "speaker": "S1",
                        "source_kind": "turn",
                        "source_start": 2.0,
                        "source_end": 3.0,
                        "clip_start": 2.0,
                        "clip_end": 3.0,
                        "purity_score": 0.88,
                        "ranking_position": 3,
                        "status": "accepted",
                        "recommended": False,
                        "relative_path": "S1/3.wav",
                    },
                    {
                        "snippet_id": "S1-04",
                        "speaker": "S1",
                        "source_kind": "turn",
                        "source_start": 3.0,
                        "source_end": 4.0,
                        "clip_start": 3.0,
                        "clip_end": 4.0,
                        "purity_score": 0.84,
                        "ranking_position": 4,
                        "status": "accepted",
                        "recommended": False,
                        "relative_path": "S1/4.wav",
                    },
                ]
            },
        },
    )
    create_voice_profile("Many Snippets Profile", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-many-snippets-1?tab=speakers"
    )
    assert page.status_code == 200
    assert 'option value="S1/4.wav"' in page.text


def test_recording_detail_speakers_nonfatal_snippet_failure_message(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-failed-1",
        source="upload",
        source_filename="failed.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-failed-1")
    set_recording_progress(
        "rec-speakers-failed-1",
        stage="llm_extract",
        progress=0.9,
        settings=cfg,
    )
    _write_snippets_manifest(
        cfg,
        "rec-speakers-failed-1",
        {
            "version": 1,
            "speakers": {},
            "manifest_status": "export_failed",
            "warnings": [{"code": "RuntimeError", "message": "snippet boom"}],
        },
    )
    mark_recording_pipeline_stage_completed(
        "rec-speakers-failed-1",
        stage_name="snippet_export",
        metadata={"manifest_status": "export_failed", "warning": "snippet boom"},
        settings=cfg,
    )
    create_voice_profile("Failed Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-failed-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Failed:</strong> Snippet export failed, so no clean clips are available for this speaker. The rest of processing continues. snippet boom"
        in page.text
    )
    assert "Add sample is unavailable because snippet export failed" not in page.text
    assert "snippet boom" in page.text


def test_recording_detail_speakers_legacy_missing_manifest_message(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-legacy-1",
        source="upload",
        source_filename="legacy.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-legacy-1")
    create_voice_profile("Legacy Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-legacy-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Legacy:</strong> This older recording has no snippets manifest yet."
        in page.text
    )
    assert (
        "Missing sanitized audio and no raw audio fallback is available." in page.text
    )


def test_recording_detail_speakers_regenerate_snippets_repairs_missing_manifest(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-repair-1",
        source="upload",
        source_filename="repair.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_snippet_repair_artifacts(cfg, "rec-speakers-repair-1")
    create_voice_profile("Repair Sample", settings=cfg)

    client = TestClient(api.app, follow_redirects=True)
    page = client.post(
        "/ui/recordings/rec-speakers-repair-1/speakers/regenerate-snippets"
    )

    assert page.status_code == 200
    assert "Regenerated" in page.text
    assert "/ui/recordings/rec-speakers-repair-1/snippets/S1/1.wav" in page.text
    assert (
        cfg.recordings_root
        / "rec-speakers-repair-1"
        / "derived"
        / "snippets_manifest.json"
    ).exists()


def test_recording_detail_speakers_regenerate_snippets_reports_missing_prereq(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-repair-missing-1",
        source="upload",
        source_filename="repair-missing.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_snippet_repair_artifacts(
        cfg,
        "rec-speakers-repair-missing-1",
        include_speaker_turns=False,
    )

    client = TestClient(api.app, follow_redirects=True)
    page = client.post(
        "/ui/recordings/rec-speakers-repair-missing-1/speakers/regenerate-snippets"
    )

    assert page.status_code == 200
    assert "Missing or unreadable derived/speaker_turns.json." in page.text
    assert "Regenerate snippets" not in page.text


def test_speakers_tab_repair_context_messages_cover_missing_stale_and_blank_reason(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-context-1",
        source="upload",
        source_filename="context.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-context-1")

    monkeypatch.setattr(
        ui_routes,
        "assess_snippet_repair",
        lambda *_a, **_k: SnippetRepairEligibility(
            recording_id="rec-speakers-context-1",
            available=True,
            artifact_state="missing",
        ),
    )
    context = ui_routes._speakers_tab_context(  # noqa: SLF001
        "rec-speakers-context-1",
        cfg,
    )
    assert "no snippets manifest yet" in context["snippet_repair"]["detail"]

    monkeypatch.setattr(
        ui_routes,
        "assess_snippet_repair",
        lambda *_a, **_k: SnippetRepairEligibility(
            recording_id="rec-speakers-context-1",
            available=True,
            artifact_state="stale",
        ),
    )
    context = ui_routes._speakers_tab_context(  # noqa: SLF001
        "rec-speakers-context-1",
        cfg,
    )
    assert "look incomplete" in context["snippet_repair"]["detail"]

    monkeypatch.setattr(
        ui_routes,
        "assess_snippet_repair",
        lambda *_a, **_k: SnippetRepairEligibility(
            recording_id="rec-speakers-context-1",
            available=False,
            artifact_state="missing",
            reason_text="",
        ),
    )
    context = ui_routes._speakers_tab_context(  # noqa: SLF001
        "rec-speakers-context-1",
        cfg,
    )
    assert context["snippet_repair"]["detail"] == ""


def test_snippet_repair_notice_message_and_missing_recording_route(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    assert (
        ui_routes._snippet_repair_notice_message(  # noqa: SLF001
            SnippetRepairResult(
                recording_id="rec-1",
                manifest_status="no_usable_speech",
                accepted_snippets=0,
                speaker_count=0,
                warning_count=0,
                degraded_diarization=False,
                audio_source="sanitized_audio",
                duration_sec=1.0,
                artifact_state_before="missing",
            )
        )
        == "Snippet manifest regenerated, but no speaker turns produced usable clips."
    )
    assert (
        ui_routes._snippet_repair_notice_message(  # noqa: SLF001
            SnippetRepairResult(
                recording_id="rec-2",
                manifest_status="no_clean_snippets",
                accepted_snippets=0,
                speaker_count=1,
                warning_count=1,
                degraded_diarization=False,
                audio_source="sanitized_audio",
                duration_sec=1.0,
                artifact_state_before="missing",
            )
        )
        == "Snippet manifest regenerated. No accepted clean clips were available."
    )

    response = TestClient(api.app, follow_redirects=True).post(
        "/ui/recordings/missing-recording/speakers/regenerate-snippets"
    )
    assert response.status_code == 404
    assert response.text == "Recording not found"


def test_recording_detail_speakers_stopped_before_snippet_export_is_unavailable(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-stopped-1",
        source="upload",
        source_filename="stopped.wav",
        status=RECORDING_STATUS_STOPPED,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-stopped-1")
    set_recording_progress(
        "rec-speakers-stopped-1",
        stage="speaker_turns",
        progress=0.8,
        settings=cfg,
    )
    create_voice_profile("Stopped Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-stopped-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Unavailable:</strong> This recording is no longer processing and did not reach "
        "Snippet Export, so no clean clips are available for this speaker." in page.text
    )
    assert "Legacy:</strong>" not in page.text
    assert (
        "Pending:</strong> The pipeline has not reached Snippet Export yet."
        not in page.text
    )


def test_recording_detail_speakers_needs_review_without_snippets_is_unavailable(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-needs-review-1",
        source="upload",
        source_filename="needs-review.wav",
        status=RECORDING_STATUS_NEEDS_REVIEW,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-needs-review-1")
    create_voice_profile("Needs Review Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-needs-review-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Unavailable:</strong> This recording is no longer processing and did not reach "
        "Snippet Export, so no clean clips are available for this speaker." in page.text
    )
    assert "Legacy:</strong>" not in page.text


def test_recording_detail_speakers_terminal_running_stage_is_not_generating(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-running-terminal-1",
        source="upload",
        source_filename="running-terminal.wav",
        status=RECORDING_STATUS_FAILED,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-running-terminal-1")
    mark_recording_pipeline_stage_started(
        "rec-speakers-running-terminal-1",
        stage_name="snippet_export",
        settings=cfg,
    )
    create_voice_profile("Terminal Running Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-running-terminal-1?tab=speakers"
    )
    assert page.status_code == 200
    assert "Generating:</strong>" not in page.text
    assert (
        "Unavailable:</strong> This recording is no longer processing and did not reach "
        "Snippet Export, so no clean clips are available for this speaker." in page.text
    )


def test_recording_detail_speakers_llm_alias_progress_is_not_pending(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-llm-alias-1",
        source="upload",
        source_filename="llm-alias.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    _seed_speaker_turns_only(cfg, "rec-speakers-llm-alias-1")
    set_recording_progress(
        "rec-speakers-llm-alias-1",
        stage="llm",
        progress=0.9,
        settings=cfg,
    )
    create_voice_profile("Alias Sample", settings=cfg)

    page = TestClient(api.app, follow_redirects=True).get(
        "/recordings/rec-speakers-llm-alias-1?tab=speakers"
    )
    assert page.status_code == 200
    assert (
        "Snippet export should already be available, but the snippets manifest is missing."
        in page.text
    )
    assert (
        "Pending:</strong> The pipeline has not reached Snippet Export yet."
        not in page.text
    )


def test_recording_detail_speakers_show_degraded_notice_and_low_confidence(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-speakers-review-1",
        source="drive",
        source_filename="speakers-review.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_speaker_artifacts(cfg, "rec-speakers-review-1")
    profile = create_voice_profile("Review Candidate", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-speakers-review-1",
        diar_speaker_label="S1",
        voice_profile_id=None,
        confidence=0.61,
        candidate_matches=[{"voice_profile_id": profile["id"], "score": 0.61}],
        low_confidence=True,
        review_state="system_suggested",
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-speakers-review-1" / "derived"
    (derived / "diarization_status.json").write_text(
        json.dumps(
            {
                "degraded": True,
                "mode": "fallback",
                "reason": "pyannote unavailable",
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-speakers-review-1?tab=speakers")
    assert r.status_code == 200
    assert "degraded fallback mode" in r.text
    assert "Needs review" in r.text
    assert "Below auto-match threshold 0.75" in r.text
    assert "Review Candidate (0.61)" in r.text


def test_recording_detail_metrics_tab_uses_summary_payload(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-metrics-tab-1",
        source="drive",
        source_filename="metrics.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-metrics-tab-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Weekly sync",
                "summary_bullets": ["Reviewed blockers"],
                "decisions": ["Ship on Friday"],
                "action_items": [
                    {
                        "task": "Send notes",
                        "owner": "Alex",
                        "deadline": "2026-02-23",
                        "confidence": 0.9,
                    }
                ],
                "emotional_summary": "Focused.",
                "questions": {
                    "total_count": 1,
                    "types": {
                        "open": 0,
                        "yes_no": 0,
                        "clarification": 0,
                        "status": 1,
                        "decision_seeking": 0,
                    },
                    "extracted": ["Is QA complete?"],
                },
            }
        ),
        encoding="utf-8",
    )
    upsert_meeting_metrics(
        recording_id="rec-metrics-tab-1",
        payload={
            "total_interruptions": 2,
            "total_questions": 1,
            "decisions_count": 1,
            "action_items_count": 1,
            "actionability_ratio": 1.0,
            "emotional_summary": "Focused.",
            "total_speech_time_seconds": 42.5,
        },
        settings=cfg,
    )
    replace_participant_metrics(
        recording_id="rec-metrics-tab-1",
        rows=[
            {
                "diar_speaker_label": "S1",
                "voice_profile_id": None,
                "payload": {
                    "speaker": "S1",
                    "airtime_seconds": 24.0,
                    "airtime_share": 0.56,
                    "turns": 6,
                    "interruptions_done": 1,
                    "interruptions_received": 0,
                    "questions_count": 1,
                    "role_hint": "Leader",
                },
            }
        ],
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-metrics-tab-1?tab=metrics")
    assert r.status_code == 200
    assert "Meeting Metrics" in r.text
    assert "Participants" in r.text
    assert "42.5s" in r.text
    assert "Leader" in r.text
    assert "Decisions" in r.text
    assert "Ship on Friday" in r.text
    assert "Send notes" in r.text
    assert "Is QA complete?" in r.text


def test_recording_detail_metrics_tab_backfills_missing_db_side_from_artifact(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-metrics-backfill-1",
        source="drive",
        source_filename="metrics-backfill.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    derived = cfg.recordings_root / "rec-metrics-backfill-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    (derived / "metrics.json").write_text(
        json.dumps(
            {
                "meeting": {
                    "total_interruptions": 3,
                    "total_questions": 5,
                    "decisions_count": 2,
                    "action_items_count": 4,
                    "actionability_ratio": 0.75,
                    "emotional_summary": "Constructive.",
                    "total_speech_time_seconds": 60.0,
                },
                "participants": [
                    {
                        "speaker": "S2",
                        "airtime_seconds": 21.0,
                        "airtime_share": 0.35,
                        "turns": 4,
                        "interruptions_done": 2,
                        "interruptions_received": 1,
                        "questions_count": 2,
                        "role_hint": "Facilitator",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Simulate partial DB persistence: meeting row exists but is incomplete, participant rows missing.
    upsert_meeting_metrics(
        recording_id="rec-metrics-backfill-1",
        payload={"total_interruptions": 3},
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-metrics-backfill-1?tab=metrics")
    assert r.status_code == 200
    assert "Constructive." in r.text
    assert "75.0%" in r.text
    assert "Facilitator" in r.text  # participant row backfilled from artifact


def test_recording_detail_summary_tab_shows_topic_and_emotional_summary(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-overview-summary-1",
        source="drive",
        source_filename="overview.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-overview-summary-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Quarterly roadmap",
                "summary_bullets": ["Roadmap reviewed"],
                "summary": "- Roadmap reviewed",
                "emotional_summary": "Calm and constructive.",
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-overview-summary-1?tab=summary")
    assert r.status_code == 200
    assert "Quarterly roadmap" in r.text
    assert "Roadmap reviewed" in r.text
    assert "Calm and constructive." in r.text


def test_recording_detail_language_tab_renders_spans(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-tab-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-lang-tab-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hello hola",
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "es",
                "language_distribution": {"en": 40.0, "es": 60.0},
                "language_spans": [
                    {"start": 0.0, "end": 1.0, "lang": "en"},
                    {"start": 1.0, "end": 2.0, "lang": "es"},
                ],
            }
        ),
        encoding="utf-8",
    )
    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-lang-tab-1?tab=language")
    assert r.status_code == 200
    assert "Language Distribution" in r.text
    assert "Language Spans" in r.text
    assert "Re-summarize (LLM only)" in r.text


def test_recording_detail_language_tab_keeps_auto_target_selected_when_unset(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-auto-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-lang-auto-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hola equipo",
                "language": {"detected": "es", "confidence": 0.95},
                "dominant_language": "es",
                "target_summary_language": "es",
                "language_distribution": {"es": 100.0},
                "language_spans": [{"start": 0.0, "end": 2.0, "lang": "es"}],
            }
        ),
        encoding="utf-8",
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/recordings/rec-lang-auto-1?tab=language")
    assert r.status_code == 200
    target_select = r.text.split('id="target_summary_language"', 1)[1].split(
        "</select>", 1
    )[0]
    assert 'value="" selected>Auto (dominant language)</option>' in target_select
    assert 'value="es" selected' not in target_select
    assert "Spanish (es)" in r.text


def test_recording_detail_not_found(client):
    r = client.get("/recordings/nonexistent-id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def test_projects_empty(client):
    r = client.get("/projects")
    assert r.status_code == 200
    assert "Projects" in r.text


def test_projects_create_and_list(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)

    r = c.post("/projects", data={"name": "AlphaProject"})
    assert r.status_code == 200
    assert "AlphaProject" in r.text

    projects = list_projects(settings=cfg)
    assert any(p["name"] == "AlphaProject" for p in projects)


def test_projects_create_blank_name_ignored(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)
    c.post("/projects", data={"name": "   "})
    assert list_projects(settings=cfg) == []


def test_projects_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    proj = create_project("ToDelete", settings=cfg)
    c = TestClient(api.app, follow_redirects=True)
    r = c.post(f"/projects/{proj['id']}/delete")
    assert r.status_code == 200
    assert list_projects(settings=cfg) == []


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


def test_glossary_page_create_edit_and_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=False)

    prefilled = TestClient(api.app, follow_redirects=True).get(
        "/glossary?recording_id=rec-ui-glossary-create-1&source=correction&kind=project&notes=Seen+on+call"
    )
    assert prefilled.status_code == 200
    assert "Corrections / ASR Memory" in prefilled.text
    assert "Prefilled from recording rec-ui-glossary-create-1." in prefilled.text
    assert "Add correction" in prefilled.text
    assert "Advanced fields" in prefilled.text
    assert 'value="Seen on call"' in prefilled.text
    assert 'value="rec-ui-glossary-create-1"' in prefilled.text

    created = c.post(
        "/glossary",
        data={
            "canonical_text": "Sander",
            "aliases_text": "Sandia\nSandoor",
            "kind": "person",
            "source": "correction",
            "enabled": "1",
            "notes": "common transcription issue",
            "recording_id": "rec-ui-glossary-create-1",
        },
    )
    assert created.status_code == 303
    created_entry = list_glossary_entries(settings=cfg)[0]
    assert created_entry["aliases_json"] == ["Sandia", "Sandoor"]
    assert created_entry["metadata_json"] == {
        "recording_id": "rec-ui-glossary-create-1"
    }
    seeded_entry = update_glossary_entry(
        int(created_entry["id"]),
        settings=cfg,
        metadata={
            "recording_id": "rec-ui-glossary-create-1",
            "import_source": "csv",
            "audit": {"by": "importer"},
        },
    )
    assert seeded_entry["metadata_json"]["import_source"] == "csv"

    edit_page = TestClient(api.app, follow_redirects=True).get(
        f"/glossary?edit_id={created_entry['id']}"
    )
    assert edit_page.status_code == 200
    assert "Edit correction" in edit_page.text

    updated = c.post(
        f"/glossary/{created_entry['id']}",
        data={
            "canonical_text": "Sander Van Doorn",
            "aliases_text": "Sandia",
            "kind": "person",
            "source": "manual",
            "notes": "updated",
            "recording_id": "rec-ui-glossary-create-2",
        },
    )
    assert updated.status_code == 303
    updated_entry = get_glossary_entry(int(created_entry["id"]), settings=cfg)
    assert updated_entry is not None
    assert updated_entry["canonical_text"] == "Sander Van Doorn"
    assert updated_entry["enabled"] == 0
    assert updated_entry["source"] == "manual"
    assert updated_entry["metadata_json"] == {
        "recording_id": "rec-ui-glossary-create-2",
        "import_source": "csv",
        "audit": {"by": "importer"},
    }

    listing = TestClient(api.app, follow_redirects=True).get("/glossary")
    assert listing.status_code == 200
    assert "Corrections / ASR Memory" in listing.text
    assert "Sander Van Doorn" in listing.text
    assert "Always-on memory" in listing.text
    assert "Saved but paused" in listing.text
    assert "Linked to recording rec-ui-glossary-create-2" in listing.text

    deleted = c.post(f"/glossary/{created_entry['id']}/delete")
    assert deleted.status_code == 303
    assert list_glossary_entries(settings=cfg) == []


def test_glossary_return_context_links_and_redirects(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    entry = create_glossary_entry(
        "Sander",
        aliases=["Sandia"],
        term_kind="person",
        source="manual",
        settings=cfg,
    )
    return_query = (
        "return_to=control-center&selected=rec-ui-1&status=Ready&q=meeting&tab=speakers"
    )

    page = TestClient(api.app, follow_redirects=True).get(f"/glossary?{return_query}")
    assert page.status_code == 200
    assert "Return to Control Center" in page.text
    assert 'href="/recordings/rec-ui-1?tab=speakers"' in page.text
    assert (
        f'href="/glossary?edit_id={entry["id"]}&return_to=control-center&amp;'
        "selected=rec-ui-1&amp;status=Ready&amp;q=meeting&amp;tab=speakers#glossary-"
        f'{entry["id"]}"'
    ) in page.text

    c = TestClient(api.app, follow_redirects=False)
    created = c.post(
        "/glossary",
        data={
            "canonical_text": "Alex",
            "aliases_text": "Alek",
            "kind": "person",
            "source": "manual",
            "return_to": "control-center",
            "selected": "rec-ui-1",
            "status": "Ready",
            "q": "meeting",
            "tab": "speakers",
        },
    )
    assert created.status_code == 303
    assert created.headers["location"] == f"/glossary?{return_query}#glossary-2"

    updated = c.post(
        f"/glossary/{entry['id']}",
        data={
            "canonical_text": "Sander Updated",
            "aliases_text": "Sandia",
            "kind": "person",
            "source": "manual",
            "return_to": "control-center",
            "selected": "rec-ui-1",
            "status": "Ready",
            "q": "meeting",
            "tab": "speakers",
        },
    )
    assert updated.status_code == 303
    assert (
        updated.headers["location"]
        == f"/glossary?{return_query}#glossary-{entry['id']}"
    )

    deleted = c.post(
        f"/glossary/{entry['id']}/delete",
        data={
            "return_to": "control-center",
            "selected": "rec-ui-1",
            "status": "Ready",
            "q": "meeting",
            "tab": "speakers",
        },
    )
    assert deleted.status_code == 303
    assert deleted.headers["location"] == f"/glossary?{return_query}"


def test_glossary_create_redirect_without_entry_anchor_when_backend_returns_no_id(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    monkeypatch.setattr(ui_routes, "create_glossary_entry", lambda **_kwargs: {})

    c = TestClient(api.app, follow_redirects=False)
    response = c.post(
        "/glossary",
        data={
            "canonical_text": "Sander",
            "aliases_text": "Sandia",
            "kind": "person",
            "source": "manual",
            "return_to": "control-center",
            "status": "Ready",
        },
    )
    assert response.status_code == 303
    assert (
        response.headers["location"]
        == "/glossary?return_to=control-center&status=Ready"
    )


def test_glossary_summary_fragment_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_glossary_entry(
        "Sander",
        aliases=["Sandia"],
        term_kind="person",
        source="manual",
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/ui/control-center/glossary-summary")
    assert r.status_code == 200
    assert "Corrections Snapshot" in r.text
    assert "Sander" in r.text
    assert "Sandia" in r.text
    assert "Always-on memory" in r.text
    assert "Manage corrections" in r.text
    assert "<html" not in r.text


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


def test_voices_empty(client):
    r = client.get("/voices")
    assert r.status_code == 200
    assert "Canonical Speakers" in r.text


def test_voices_create_and_list(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)

    r = c.post("/voices", data={"display_name": "Alice Smith", "notes": "sales team"})
    assert r.status_code == 200
    assert "Alice Smith" in r.text

    profiles = list_voice_profiles(settings=cfg)
    assert any(v["display_name"] == "Alice Smith" for v in profiles)


def test_voices_create_blank_name_ignored(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    c = TestClient(api.app, follow_redirects=True)
    c.post("/voices", data={"display_name": "  ", "notes": ""})
    assert list_voice_profiles(settings=cfg) == []


def test_voices_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    vp = create_voice_profile("Bob", settings=cfg)
    c = TestClient(api.app, follow_redirects=True)
    r = c.post(f"/voices/{vp['id']}/delete")
    assert r.status_code == 200
    assert list_voice_profiles(settings=cfg) == []


def test_voices_return_context_links_and_redirects(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    source = create_voice_profile("Source Voice", settings=cfg)
    target = create_voice_profile("Target Voice", settings=cfg)
    return_query = (
        "return_to=control-center&selected=rec-ui-1&status=Ready&q=meeting&tab=speakers"
    )

    page = TestClient(api.app, follow_redirects=True).get(f"/voices?{return_query}")
    assert page.status_code == 200
    assert "Return to Control Center" in page.text
    assert 'href="/recordings/rec-ui-1?tab=speakers"' in page.text

    c = TestClient(api.app, follow_redirects=False)
    created = c.post(
        "/voices",
        data={
            "display_name": "Alex Speaker",
            "notes": "host",
            "return_to": "control-center",
            "selected": "rec-ui-1",
            "status": "Ready",
            "q": "meeting",
            "tab": "speakers",
        },
    )
    assert created.status_code == 303
    assert created.headers["location"] == f"/voices?{return_query}#voice-3"
    created_profile_id = max(
        int(profile["id"]) for profile in list_voice_profiles(settings=cfg)
    )

    merged = c.post(
        f"/voices/{source['id']}/merge",
        data={
            "target_profile_id": str(target["id"]),
            "return_to": "control-center",
            "selected": "rec-ui-1",
            "status": "Ready",
            "q": "meeting",
            "tab": "speakers",
        },
    )
    assert merged.status_code == 303
    assert merged.headers["location"] == f"/voices?{return_query}#voice-{target['id']}"

    deleted = c.post(
        f"/voices/{created_profile_id}/delete",
        data={
            "return_to": "control-center",
            "selected": "rec-ui-1",
            "status": "Ready",
            "q": "meeting",
            "tab": "speakers",
        },
    )
    assert deleted.status_code == 303
    assert deleted.headers["location"] == f"/voices?{return_query}"


def test_voices_page_renders_duplicate_candidates_and_sample_inspection(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-voices-1",
        source="upload",
        source_filename="voices.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    left = create_voice_profile("Alice Canonical", settings=cfg)
    right = create_voice_profile("Alicia Duplicate", settings=cfg)
    sample = create_voice_sample(
        voice_profile_id=left["id"],
        recording_id="rec-voices-1",
        diar_speaker_label="S1",
        snippet_path="recordings/rec-voices-1/derived/snippets/S1/1.wav",
        candidate_matches=[
            {"voice_profile_id": left["id"], "score": 0.97},
            {"voice_profile_id": right["id"], "score": 0.83},
        ],
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    r = c.get("/voices")
    assert r.status_code == 200
    assert "Potential duplicates" in r.text
    assert "Alicia Duplicate (1 samples, 0.83)" in r.text
    assert "Inspect samples" in r.text
    assert f"/ui/voice-samples/{sample['id']}/audio" in r.text
    assert "Merge into" in r.text


def test_voices_merge_route_calls_backend_and_redirects_to_target_anchor(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    source = create_voice_profile("Source Voice", settings=cfg)
    target = create_voice_profile("Target Voice", settings=cfg)
    seen: dict[str, int | AppSettings] = {}

    def _merge(
        source_profile_id: int, target_profile_id: int, *, settings: AppSettings
    ):
        seen["source"] = source_profile_id
        seen["target"] = target_profile_id
        seen["settings"] = settings
        return {"target": target_profile_id}

    monkeypatch.setattr(ui_routes, "merge_canonical_speakers", _merge)

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        f"/voices/{source['id']}/merge",
        data={"target_profile_id": str(target["id"])},
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/voices#voice-{target['id']}"
    assert seen == {"source": source["id"], "target": target["id"], "settings": cfg}


# ---------------------------------------------------------------------------
# Queue / Jobs
# ---------------------------------------------------------------------------


def test_queue_empty(client):
    r = client.get("/queue")
    assert r.status_code == 200
    assert "Queue" in r.text


def test_queue_with_data(seeded_client):
    r = seeded_client.get("/queue")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_queue_status_filter(seeded_client):
    r = seeded_client.get("/queue?status=queued")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_queue_recording_filter(seeded_client):
    r = seeded_client.get("/queue?recording_id=rec-ui-1")
    assert r.status_code == 200
    assert "precheck" in r.text


def test_queue_invalid_status_shows_all(seeded_client):
    r = seeded_client.get("/queue?status=bogus")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def test_upload_page(client):
    r = client.get("/upload")
    assert r.status_code == 200
    assert "Standalone Upload" in r.text
    assert "Back to Control Center" in r.text
    assert 'id="file-input"' in r.text
    assert "Drop audio here or browse from disk." in r.text
    assert "Recent uploads stay here while you work" in r.text
    assert (
        "Completed uploads stay listed here so standalone uploads keep their in-page progress and Open recording link."
        in r.text
    )
    assert "Completed uploads stay visible here until you leave the page." in r.text
    assert (
        "No uploads yet. New files appear here with upload and processing progress."
        in r.text
    )
    assert "var removeTerminalItems = false;" in r.text


def test_upload_panel_fragment_endpoint(client):
    r = client.get("/ui/control-center/upload/panel")
    assert r.status_code == 200
    assert 'id="file-input"' in r.text
    assert 'id="upload-rows"' in r.text
    assert "Only in-flight uploads stay here" in r.text
    assert "Finished recordings move into the main worklist below" in r.text
    assert (
        "No active uploads. New files appear here until they enter the main inbox."
        in r.text
    )
    assert "<html" not in r.text


def test_ui_root_redirects_to_login_when_auth_enabled(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.api_bearer_token = "secret-ui-token"
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    c = TestClient(api.app, follow_redirects=False)
    r = c.get("/ui")
    assert r.status_code == 303
    assert r.headers["location"].startswith("/ui/login")


def test_ui_login_sets_cookie_and_allows_protected_post(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.api_bearer_token = "secret-ui-token"
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-auth-ui-1", source="drive", source_filename="auth.mp3", settings=cfg
    )

    monkeypatch.setattr(
        ui_routes,
        "purge_pending_recording_jobs",
        lambda *_args, **_kwargs: 0,
    )

    c = TestClient(api.app, follow_redirects=False)
    blocked = c.post(
        "/ui/recordings/rec-auth-ui-1/delete",
    )
    assert blocked.status_code == 401

    login = c.post(
        "/ui/login",
        data={"token": "secret-ui-token", "next": "/ui"},
    )
    assert login.status_code == 303
    assert AUTH_COOKIE_NAME in login.headers.get("set-cookie", "")

    allowed = c.post(
        "/ui/recordings/rec-auth-ui-1/delete",
    )
    assert allowed.status_code == 200


def test_ui_login_sets_secure_cookie_for_https_host(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.api_bearer_token = "secret-ui-token"
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    c = TestClient(api.app, follow_redirects=False, base_url="https://staging.example")
    login = c.post(
        "/ui/login",
        data={"token": "secret-ui-token", "next": "/ui"},
    )

    assert login.status_code == 303
    set_cookie = login.headers.get("set-cookie", "")
    assert AUTH_COOKIE_NAME in set_cookie
    assert "Secure" in set_cookie


def test_ui_login_does_not_set_secure_cookie_for_http_lan_host(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.api_bearer_token = "secret-ui-token"
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)

    c = TestClient(api.app, follow_redirects=False, base_url="http://192.168.1.10")
    login = c.post(
        "/ui/login",
        data={"token": "secret-ui-token", "next": "/ui"},
    )

    assert login.status_code == 303
    set_cookie = login.headers.get("set-cookie", "")
    assert AUTH_COOKIE_NAME in set_cookie
    assert "Secure" not in set_cookie


# ---------------------------------------------------------------------------
# Inline action endpoints
# ---------------------------------------------------------------------------


def test_ui_action_quarantine(seeded_client):
    r = seeded_client.post("/ui/recordings/rec-ui-1/quarantine")
    assert r.status_code == 200


def test_ui_action_quarantine_control_center_stays_on_root(seeded_client):
    r = seeded_client.post(
        "/ui/recordings/rec-ui-1/quarantine?return_to=control-center"
    )
    assert r.status_code == 200
    assert "HX-Redirect" not in r.headers


def test_ui_action_quarantine_not_found(client):
    r = client.post("/ui/recordings/no-such-rec/quarantine")
    assert r.status_code == 404


def test_ui_action_stop_queued_recording_immediately_stops_and_finishes_queued_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-stop-queued-1",
        source="upload",
        source_filename="queued.wav",
        status=RECORDING_STATUS_QUEUED,
        settings=cfg,
    )
    create_job(
        "job-stop-queued-1",
        recording_id="rec-stop-queued-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_QUEUED,
    )
    set_recording_progress(
        "rec-stop-queued-1",
        stage="asr",
        progress=0.4,
        settings=cfg,
    )
    monkeypatch.setattr(
        ui_routes,
        "purge_pending_recording_jobs",
        lambda *_args, **_kwargs: 1,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-stop-queued-1/stop", data={"tab": "overview"})
    assert r.status_code == 303
    assert r.headers["location"] == "/recordings/rec-stop-queued-1"

    recording = get_recording("rec-stop-queued-1", settings=cfg) or {}
    job = ui_routes.get_job("job-stop-queued-1", settings=cfg) or {}
    assert recording["status"] == RECORDING_STATUS_STOPPED
    assert recording["cancel_requested_by"] == "user"
    assert recording["cancel_reason_code"] == "user_stop"
    assert recording["cancel_reason_text"] == "Cancelled by user"
    assert recording["pipeline_stage"] is None
    assert recording["pipeline_progress"] is None
    assert job["status"] == JOB_STATUS_FINISHED
    assert job["error"] == "cancelled_by_user"


def test_ui_action_stop_processing_recording_sets_stopping_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-stop-processing-1",
        source="upload",
        source_filename="processing.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    create_job(
        "job-stop-processing-1",
        recording_id="rec-stop-processing-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_STARTED,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-stop-processing-1/stop",
        data={"tab": "log"},
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/recordings/rec-stop-processing-1?tab=diagnostics"

    recording = get_recording("rec-stop-processing-1", settings=cfg) or {}
    job = ui_routes.get_job("job-stop-processing-1", settings=cfg) or {}
    assert recording["status"] == RECORDING_STATUS_STOPPING
    assert recording["cancel_requested_at"]
    assert recording["cancel_requested_by"] == "user"
    assert recording["cancel_reason_code"] == "user_stop"
    assert recording["cancel_reason_text"] == "Stop requested by user"
    assert job["status"] == JOB_STATUS_STARTED


def test_ui_action_stop_duplicate_request_does_not_corrupt_existing_stop_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-stop-dup-1",
        source="upload",
        source_filename="dup.wav",
        status=RECORDING_STATUS_STOPPING,
        settings=cfg,
    )
    create_job(
        "job-stop-dup-1",
        recording_id="rec-stop-dup-1",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_STARTED,
    )
    set_recording_cancel_request(
        "rec-stop-dup-1",
        requested_by="user",
        reason_code="user_stop",
        reason_text="Stop requested by user",
        settings=cfg,
    )
    before = get_recording("rec-stop-dup-1", settings=cfg) or {}

    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-stop-dup-1/stop", data={"tab": "overview"})
    assert r.status_code == 303

    after = get_recording("rec-stop-dup-1", settings=cfg) or {}
    assert after["status"] == RECORDING_STATUS_STOPPING
    assert after["cancel_requested_at"] == before["cancel_requested_at"]
    assert after["cancel_reason_text"] == "Stop requested by user"


def test_ui_action_delete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-del-1", source="drive", source_filename="x.mp3", settings=cfg)

    def _fake_purge(recording_id: str, *, settings=None) -> int:
        return 0

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", _fake_purge)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-del-1/delete")
    assert r.status_code in (200, 307, 302)


def test_ui_action_delete_not_found(client):
    r = client.post("/ui/recordings/no-such-rec/delete")
    assert r.status_code == 404


def test_ui_action_delete_does_not_require_confirmation(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-del-confirm-1", source="drive", source_filename="x.mp3", settings=cfg
    )
    monkeypatch.setattr(
        ui_routes,
        "purge_pending_recording_jobs",
        lambda *_args, **_kwargs: 0,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-del-confirm-1/delete")
    assert r.status_code in (200, 307, 302)
    assert get_recording("rec-del-confirm-1", settings=cfg) is None


def test_ui_action_delete_control_center_stays_on_root(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-del-control-center-1",
        source="drive",
        source_filename="x.mp3",
        settings=cfg,
    )
    monkeypatch.setattr(
        ui_routes,
        "purge_pending_recording_jobs",
        lambda *_args, **_kwargs: 0,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-del-control-center-1/delete?return_to=control-center"
    )
    assert r.status_code == 200
    assert "HX-Redirect" not in r.headers
    assert get_recording("rec-del-control-center-1", settings=cfg) is None


def test_ui_action_requeue(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-rq-1", source="drive", source_filename="y.mp3", settings=cfg)

    observed: dict[str, object] = {}

    def _fake_enqueue(
        recording_id: str,
        *,
        settings=None,
        job_type=JOB_TYPE_PRECHECK,
        reset_pipeline_state: bool = False,
    ):
        observed["recording_id"] = recording_id
        observed["job_type"] = job_type
        observed["reset_pipeline_state"] = reset_pipeline_state
        return None

    monkeypatch.setattr(ui_routes, "enqueue_recording_job", _fake_enqueue)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-rq-1/requeue")
    assert r.status_code in (200, 307, 302)
    assert observed == {
        "recording_id": "rec-rq-1",
        "job_type": JOB_TYPE_PRECHECK,
        "reset_pipeline_state": True,
    }


def test_ui_action_requeue_control_center_stays_on_root(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-rq-control-center-1", source="drive", source_filename="y.mp3", settings=cfg
    )

    monkeypatch.setattr(
        ui_routes,
        "enqueue_recording_job",
        lambda *_args, **_kwargs: None,
    )
    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-rq-control-center-1/requeue?return_to=control-center"
    )
    assert r.status_code == 200
    assert "HX-Redirect" not in r.headers


def test_ui_action_requeue_not_found(client):
    r = client.post("/ui/recordings/no-such-rec/requeue")
    assert r.status_code == 404


def test_ui_action_requeue_failure_returns_503(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-rqf-1", source="drive", source_filename="z.mp3", settings=cfg)

    def _fail_enqueue(
        recording_id: str,
        *,
        settings=None,
        job_type=JOB_TYPE_PRECHECK,
        reset_pipeline_state: bool = False,
    ):
        raise RuntimeError("redis down")

    monkeypatch.setattr(ui_routes, "enqueue_recording_job", _fail_enqueue)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-rqf-1/requeue")
    assert r.status_code == 503
    assert "redis down" in r.text


def test_ui_action_retry_failed_step(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-rtry-1", source="drive", source_filename="retry-step.mp3", settings=cfg
    )
    create_job(
        "job-rtry-1",
        recording_id="rec-rtry-1",
        job_type=JOB_TYPE_STT,
        settings=cfg,
        status=JOB_STATUS_FAILED,
    )

    observed: dict[str, str] = {}

    def _fake_enqueue(
        recording_id: str,
        *,
        settings=None,
        job_type=JOB_TYPE_PRECHECK,
        reset_pipeline_state: bool = False,
    ):
        observed["recording_id"] = recording_id
        observed["job_type"] = job_type
        observed["reset_pipeline_state"] = reset_pipeline_state
        return None

    monkeypatch.setattr(ui_routes, "enqueue_recording_job", _fake_enqueue)
    c = TestClient(api.app, follow_redirects=False)

    r = c.post("/ui/recordings/rec-rtry-1/jobs/job-rtry-1/retry")
    assert r.status_code == 303
    assert r.headers["location"] == "/recordings/rec-rtry-1?tab=diagnostics"
    assert observed == {
        "recording_id": "rec-rtry-1",
        "job_type": JOB_TYPE_PRECHECK,
        "reset_pipeline_state": False,
    }


def test_ui_action_retry_failed_step_rejects_non_failed_job(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-rtry-2", source="drive", source_filename="retry-step.mp3", settings=cfg
    )
    create_job(
        "job-rtry-2",
        recording_id="rec-rtry-2",
        job_type=JOB_TYPE_PRECHECK,
        settings=cfg,
        status=JOB_STATUS_QUEUED,
    )

    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-rtry-2/jobs/job-rtry-2/retry")
    assert r.status_code == 422
    assert "failed jobs" in r.text


def test_ui_language_resummarize_uses_target_language_override(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-rsum-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    derived = cfg.recordings_root / "rec-lang-rsum-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hello team and hola equipo",
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
                "language_distribution": {"en": 55.0, "es": 45.0},
                "language_spans": [
                    {"start": 0.0, "end": 2.0, "lang": "en"},
                    {"start": 2.0, "end": 4.0, "lang": "es"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text(
        json.dumps({"friendly": 0, "model": "test-llm-model", "summary": "- old"}),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    async def _fake_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ):
        captured["system_prompt"] = system_prompt
        captured["model"] = model or ""
        return {
            "content": json.dumps(
                {
                    "topic": "Resumen semanal",
                    "summary_bullets": ["Bloqueadores revisados."],
                    "decisions": ["Publicar el viernes."],
                    "action_items": [
                        {
                            "task": "Enviar notas",
                            "owner": "Alex",
                            "deadline": "2026-02-23",
                            "confidence": 0.85,
                        }
                    ],
                    "emotional_summary": "Enfoque positivo.",
                    "questions": {
                        "total_count": 1,
                        "types": {
                            "open": 0,
                            "yes_no": 0,
                            "clarification": 1,
                            "status": 0,
                            "decision_seeking": 0,
                        },
                        "extracted": ["Quien valida QA?"],
                    },
                }
            )
        }

    monkeypatch.setattr(ui_routes.LLMClient, "generate", _fake_generate)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-lang-rsum-1/language/resummarize",
        data={
            "target_summary_language": "es",
            "transcript_language_override": "en",
        },
    )
    assert r.status_code == 303

    recording = get_recording("rec-lang-rsum-1", settings=cfg)
    assert recording is not None
    assert recording["target_summary_language"] == "es"
    assert recording["language_override"] == "en"

    summary_payload = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["summary_bullets"] == ["Bloqueadores revisados."]
    assert summary_payload["topic"] == "Resumen semanal"
    assert summary_payload["target_summary_language"] == "es"
    assert captured["model"] == cfg.llm_model
    assert "in Spanish." in captured["system_prompt"]


def test_ui_language_resummarize_without_speaker_turns_uses_full_transcript(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-rsum-legacy-1",
        source="drive",
        source_filename="legacy.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )

    derived = cfg.recordings_root / "rec-lang-rsum-legacy-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    long_text = " ".join(f"token{i}" for i in range(260))
    normalized_long_text = " ".join(long_text.split())
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": long_text,
                "language": {"detected": "en", "confidence": 0.9},
                "dominant_language": "en",
            }
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text(
        json.dumps({"friendly": 0, "model": "test-llm-model", "summary": "- old"}),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    async def _fake_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ):
        captured["user_prompt"] = user_prompt
        return {
            "content": json.dumps(
                {
                    "topic": "Legacy",
                    "summary_bullets": ["ok"],
                    "decisions": [],
                    "action_items": [],
                    "emotional_summary": "Neutral.",
                    "questions": {"total_count": 0, "types": {}, "extracted": []},
                }
            )
        }

    monkeypatch.setattr(ui_routes.LLMClient, "generate", _fake_generate)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-lang-rsum-legacy-1/language/resummarize",
        data={
            "target_summary_language": "en",
            "transcript_language_override": "",
        },
    )
    assert r.status_code == 303

    prompt_payload = json.loads(captured["user_prompt"])
    speaker_turns = prompt_payload["speaker_turns"]
    assert len(speaker_turns) > 1
    reconstructed = " ".join(str(turn["text"]) for turn in speaker_turns)
    assert reconstructed == normalized_long_text


def test_ui_language_retranscribe_enqueues_precheck_and_saves_overrides(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-lang-rtr-1",
        source="drive",
        source_filename="lang.mp3",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    called: dict[str, object] = {}

    def _fake_enqueue(
        recording_id: str,
        *,
        settings=None,
        job_type=JOB_TYPE_PRECHECK,
        reset_pipeline_state: bool = False,
    ):
        called["recording_id"] = recording_id
        called["job_type"] = job_type
        called["reset_pipeline_state"] = reset_pipeline_state
        return None

    monkeypatch.setattr(ui_routes, "enqueue_recording_job", _fake_enqueue)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post(
        "/ui/recordings/rec-lang-rtr-1/language/retranscribe",
        data={
            "target_summary_language": "es",
            "transcript_language_override": "en",
        },
    )
    assert r.status_code == 303
    assert called["recording_id"] == "rec-lang-rtr-1"
    assert called["job_type"] == JOB_TYPE_PRECHECK
    assert called["reset_pipeline_state"] is True

    recording = get_recording("rec-lang-rtr-1", settings=cfg)
    assert recording is not None
    assert recording["target_summary_language"] == "es"
    assert recording["language_override"] == "en"


def test_ui_action_delete_purge_failure_returns_503(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording("rec-pf-1", source="drive", source_filename="pf.mp3", settings=cfg)

    def _fail_purge(recording_id: str, *, settings=None) -> int:
        raise RuntimeError("redis down")

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", _fail_purge)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-pf-1/delete")
    assert r.status_code == 503
    assert "redis down" in r.text


def test_ui_action_delete_cleanup_failure_returns_500(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-del-fail-1",
        source="drive",
        source_filename="fail.mp3",
        settings=cfg,
    )

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        ui_routes,
        "delete_recording_with_artifacts",
        lambda *_a, **_k: (_ for _ in ()).throw(RecordingDeleteError("disk busy")),
    )
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-del-fail-1/delete")
    assert r.status_code == 500
    assert "disk busy" in r.text


def test_ui_action_delete_returns_404_when_cleanup_reports_missing(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-del-missing-1",
        source="drive",
        source_filename="missing.mp3",
        settings=cfg,
    )

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        ui_routes, "delete_recording_with_artifacts", lambda *_a, **_k: False
    )
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-del-missing-1/delete")
    assert r.status_code == 404
    assert "Not found" in r.text


def test_ui_action_delete_removes_disk_artifacts(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-disk-1", source="drive", source_filename="w.mp3", settings=cfg
    )
    rec_dir = cfg.recordings_root / "rec-disk-1"
    rec_dir.mkdir(parents=True)
    (rec_dir / "audio.mp3").write_text("fake")

    def _fake_purge(recording_id: str, *, settings=None) -> int:
        return 0

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", _fake_purge)
    c = TestClient(api.app, follow_redirects=False)
    c.post("/ui/recordings/rec-disk-1/delete")
    assert not rec_dir.exists()


def test_ui_action_delete_cascades_voice_samples_for_recording(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_recording(
        "rec-del-sample-1",
        source="drive",
        source_filename="del-sample.mp3",
        settings=cfg,
    )
    profile = create_voice_profile("Voice Owner", settings=cfg)
    create_voice_sample(
        voice_profile_id=profile["id"],
        recording_id="rec-del-sample-1",
        diar_speaker_label="S1",
        snippet_path="recordings/rec-del-sample-1/derived/snippets/S1/1.wav",
        settings=cfg,
    )
    assert len(list_voice_samples(settings=cfg)) == 1

    def _fake_purge(recording_id: str, *, settings=None) -> int:
        return 0

    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", _fake_purge)
    c = TestClient(api.app, follow_redirects=False)
    r = c.post("/ui/recordings/rec-del-sample-1/delete")
    assert r.status_code in (200, 307, 302)
    assert list_voice_samples(settings=cfg) == []


def test_projects_duplicate_name_handled(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    create_project("DupeTest", settings=cfg)
    c = TestClient(api.app, follow_redirects=True)
    r = c.post("/projects", data={"name": "DupeTest"})
    assert r.status_code == 200
    projects = list_projects(settings=cfg)
    assert len(projects) == 1


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------


def test_static_htmx_served(client):
    r = client.get("/static/htmx.min.js")
    assert r.status_code == 200
    assert "htmx" in r.text


def test_dashboard_references_local_htmx(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/static/htmx.min.js" in r.text
    assert "unpkg.com" not in r.text


# ---------------------------------------------------------------------------
# DB helpers: projects and voice_profiles
# ---------------------------------------------------------------------------


def test_db_list_projects_empty(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    assert list_projects(settings=cfg) == []


def test_db_create_and_delete_project(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    p = create_project("Gamma", settings=cfg)
    assert p["name"] == "Gamma"
    assert p["auto_publish"] == 0
    projects = list_projects(settings=cfg)
    assert len(projects) == 1

    from lan_app.db import delete_project

    assert delete_project(p["id"], settings=cfg) is True
    assert list_projects(settings=cfg) == []


def test_db_list_voice_profiles_empty(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    assert list_voice_profiles(settings=cfg) == []


def test_db_create_and_delete_voice_profile(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    vp = create_voice_profile("Carol", notes="engineer", settings=cfg)
    assert vp["display_name"] == "Carol"
    assert vp["notes"] == "engineer"
    profiles = list_voice_profiles(settings=cfg)
    assert len(profiles) == 1

    from lan_app.db import delete_voice_profile

    assert delete_voice_profile(vp["id"], settings=cfg) is True
    assert list_voice_profiles(settings=cfg) == []


def test_db_set_and_clear_speaker_assignment(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-assignment-1",
        source="drive",
        source_filename="assignment.mp3",
        settings=cfg,
    )
    profile = create_voice_profile("Dana", settings=cfg)

    set_speaker_assignment(
        recording_id="rec-assignment-1",
        diar_speaker_label="S1",
        voice_profile_id=profile["id"],
        settings=cfg,
    )
    rows = list_speaker_assignments("rec-assignment-1", settings=cfg)
    assert len(rows) == 1
    assert rows[0]["voice_profile_name"] == "Dana"

    set_speaker_assignment(
        recording_id="rec-assignment-1",
        diar_speaker_label="S1",
        voice_profile_id=None,
        settings=cfg,
    )
    assert list_speaker_assignments("rec-assignment-1", settings=cfg) == []


def test_db_create_and_list_voice_samples(tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-sample-1",
        source="drive",
        source_filename="sample.mp3",
        settings=cfg,
    )
    profile = create_voice_profile("Evan", settings=cfg)

    sample = create_voice_sample(
        voice_profile_id=profile["id"],
        recording_id="rec-sample-1",
        diar_speaker_label="S1",
        snippet_path="recordings/rec-sample-1/derived/snippets/S1/1.wav",
        settings=cfg,
    )
    assert sample["voice_profile_name"] == "Evan"

    rows = list_voice_samples(settings=cfg)
    assert len(rows) == 1
    assert rows[0]["recording_id"] == "rec-sample-1"

    filtered = list_voice_samples(voice_profile_id=profile["id"], settings=cfg)
    assert len(filtered) == 1
    assert filtered[0]["diar_speaker_label"] == "S1"


def test_calendars_page_renders_seeded_sources_and_events(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    source = create_calendar_source(
        name="Local Team Calendar",
        kind="file",
        file_ics="BEGIN:VCALENDAR\nEND:VCALENDAR",
        settings=cfg,
    )
    replace_calendar_events_for_window(
        source_id=int(source["id"]),
        window_start="2026-02-01T00:00:00Z",
        window_end="2026-02-10T00:00:00Z",
        events=[
            {
                "uid": "evt-seeded-1",
                "starts_at": "2026-02-03T10:00:00Z",
                "ends_at": "2026-02-03T11:00:00Z",
                "all_day": False,
                "summary": "Seeded calendar event",
                "location": "Room 5",
                "description": "fixture",
                "organizer": "Alex",
                "attendees": [{"label": "Priya Kapoor"}],
                "updated_at": "2026-02-01T00:00:00Z",
            },
            {
                "uid": "evt-seeded-end-date-1",
                "starts_at": "2026-02-10T15:00:00Z",
                "ends_at": "2026-02-10T16:00:00Z",
                "all_day": False,
                "summary": "End date event",
                "location": "Room 9",
                "description": "fixture",
                "organizer": "Alex",
                "updated_at": "2026-02-01T00:00:00Z",
            },
        ],
        settings=cfg,
    )

    c = TestClient(api.app, follow_redirects=True)
    response = c.get("/calendars?from=2026-02-01&to=2026-02-10")
    assert response.status_code == 200
    assert "Calendars" in response.text
    assert "Local Team Calendar" in response.text
    assert ">2<" in response.text
    assert "Seeded calendar event" in response.text
    assert "Priya Kapoor" in response.text
    assert "2026-02-03 11:00:00 CET" in response.text
    assert "End date event" in response.text


def test_calendars_create_and_sync_file_source(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    init_db(cfg)
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    starts_at = now + timedelta(days=1)
    ends_at = starts_at + timedelta(hours=1)
    file_payload = "\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//LAN Transcriber//UI Calendar Test//EN",
            "BEGIN:VEVENT",
            "UID:ui-event-1",
            f"DTSTART:{starts_at.strftime('%Y%m%dT%H%M%SZ')}",
            f"DTEND:{ends_at.strftime('%Y%m%dT%H%M%SZ')}",
            "SUMMARY:UI Sync Event",
            "LOCATION:Room 2",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
    )

    c = TestClient(api.app, follow_redirects=False)
    created = c.post(
        "/calendars/sources",
        data={
            "name": "UI File Calendar",
            "kind": "file",
            "file": file_payload,
        },
    )
    assert created.status_code == 303
    assert created.headers["location"].startswith("/calendars?source_id=")

    sources = list_calendar_sources(settings=cfg)
    assert len(sources) == 1
    source_id = int(sources[0]["id"])

    synced = c.post(f"/calendars/sources/{source_id}/sync")
    assert synced.status_code == 303
    assert synced.headers["location"] == f"/calendars?source_id={source_id}"

    sources_after = list_calendar_sources(settings=cfg)
    assert sources_after[0]["last_synced_at"]

    page = TestClient(api.app, follow_redirects=True).get("/calendars")
    assert page.status_code == 200
    assert "UI Sync Event" in page.text
