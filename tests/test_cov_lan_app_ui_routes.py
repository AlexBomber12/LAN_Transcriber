from __future__ import annotations

import asyncio
from datetime import timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from lan_app import api, ui_routes
from lan_app.calendar.service import CalendarSyncError
from lan_app.config import AppSettings
from lan_app.constants import (
    JOB_STATUS_FAILED,
    RECORDING_STATUS_FAILED,
    RECORDING_STATUS_NEEDS_REVIEW,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_PUBLISHED,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_QUARANTINE,
    RECORDING_STATUS_READY,
    RECORDING_STATUS_STOPPED,
    RECORDING_STATUS_STOPPING,
)
from lan_app.db import (
    create_calendar_source,
    create_glossary_entry,
    create_job,
    create_recording,
    create_voice_profile,
    init_db,
    set_speaker_assignment,
    upsert_calendar_match,
)
from lan_app.jobs import DuplicateRecordingJobError


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
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AppSettings, TestClient]:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(api, "_settings", cfg)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    init_db(cfg)
    return cfg, TestClient(api.app, follow_redirects=False)


def _seed_recording(cfg: AppSettings, recording_id: str = "rec-cov-ui-routes-1") -> str:
    create_recording(
        recording_id,
        source="upload",
        source_filename=f"{recording_id}.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    return recording_id


def test_language_and_json_helper_edge_paths() -> None:
    class _NoStrip(str):
        def strip(self) -> "_NoStrip":  # type: ignore[override]
            return self

    class _BulletOnly:
        def __str__(self) -> _NoStrip:
            return _NoStrip("- ")

    assert ui_routes._language_display_name("unknown") == "Unknown"  # noqa: SLF001

    with pytest.raises(ValueError, match="target must be a language code"):
        ui_routes._parse_language_form_value("??", field_name="target")  # noqa: SLF001

    warning = ui_routes._recording_recovery_warning(  # noqa: SLF001
        [{"error": "stuck job recovered", "finished_at": "2026-01-02T03:04:05Z"}]
    )
    assert warning and "2026-01-02 04:04:05 CET" in warning

    warning_no_ts = ui_routes._recording_recovery_warning(  # noqa: SLF001
        [{"error": "stuck job recovered"}]
    )
    assert warning_no_ts and "recovered from a stuck job" in warning_no_ts

    assert ui_routes._normalise_text_items(  # noqa: SLF001
        ["", "  ", _BulletOnly(), "- first", "second", "third"],
        max_items=2,
    ) == [
        "first",
        "second",
    ]


def test_control_center_helper_contexts_cover_fragment_builders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        ui_routes,
        "collect_control_center_runtime_status",
        lambda _settings: _stub_runtime_status(),
    )
    monkeypatch.setattr(ui_routes, "_status_counts", lambda settings: {"Ready": 1})
    monkeypatch.setattr(ui_routes, "_job_counts", lambda settings: {"queued": 2})
    monkeypatch.setattr(
        ui_routes,
        "list_recordings",
        lambda **_kwargs: ([{"id": "rec-helper-1"}], 3),
    )
    monkeypatch.setattr(
        ui_routes,
        "list_glossary_entries",
        lambda settings: [
            {
                "id": 1,
                "canonical_text": "Sander",
                "aliases_json": ["Sandia"],
                "kind": "person",
                "source": "manual",
            }
        ],
    )
    monkeypatch.setattr(
        ui_routes,
        "list_voice_profiles",
        lambda settings: [
            {
                "id": 7,
                "display_name": "Alex Helper",
                "notes": "host",
                "updated_at": "2026-01-02T03:04:05Z",
            }
        ],
    )
    monkeypatch.setattr(
        ui_routes,
        "list_voice_samples",
        lambda settings: [
            {"id": 4, "voice_profile_id": 7},
            {"id": 5, "voice_profile_id": None},
        ],
    )
    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda recording_id, settings: {
            "id": recording_id,
            "source": "upload",
            "source_filename": "helper.wav",
            "status": "Ready",
            "captured_at": "2026-01-02T03:04:05Z",
            "created_at": "2026-01-02T03:04:05Z",
            "updated_at": "2026-01-02T03:04:05Z",
            "duration_sec": 12.0,
        },
    )
    dashboard = ui_routes._dashboard_status_context(cfg)  # noqa: SLF001
    assert dashboard["recordings_summary_strip"]["title"] == "Recordings by status"
    assert dashboard["jobs_summary_strip"]["counts"] == {"queued": 2}
    assert dashboard["recent"] == [{"id": "rec-helper-1"}]

    state = ui_routes._control_center_state_context(  # noqa: SLF001
        selected=" rec-helper-1 ",
        status="Ready",
        q=" helper ",
        tab="speakers",
        limit=100,
        offset=25,
    )
    assert state["selected"] == "rec-helper-1"
    assert state["status"] == "Ready"
    assert state["q"] == "helper"
    assert state["tab"] == "speakers"
    assert state["limit"] == 100
    assert state["offset"] == 25
    assert state["selected_detail_href"] == "/recordings/rec-helper-1?tab=speakers"
    assert state["system_bar_url"] == (
        "/ui/control-center/system-bar?"
        "selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=100&offset=25"
    )
    assert "status=Ready" in state["work_pane_url"]
    assert "limit=100" in state["work_pane_url"]
    assert "offset=25" in state["clear_selection_href"]

    summary_state = ui_routes._control_center_state_context(  # noqa: SLF001
        selected="rec-helper-1",
        status="Ready",
        q="helper",
        tab="summary",
    )
    assert summary_state["tab"] == "summary"
    assert summary_state["selected_detail_href"] == "/recordings/rec-helper-1?tab=summary"
    assert summary_state["tab_label"] == "Summary"

    fallback_state = ui_routes._control_center_state_context(  # noqa: SLF001
        selected=None,
        status="unknown",
        q=None,
        tab="mystery",
    )
    assert fallback_state["status"] == ""
    assert fallback_state["tab"] == "overview"
    assert fallback_state["clear_selection_href"] == "/"

    assert (
        ui_routes._control_center_shell_href(  # noqa: SLF001
            selected="rec helper",
            status_filter="Ready",
            search_query="demo",
            tab="log",
            limit=100,
            offset=25,
        )
        == "/?selected=rec+helper&status=Ready&q=demo&limit=100&offset=25"
    )
    assert (
        ui_routes._control_center_shell_href(  # noqa: SLF001
            selected="rec helper",
            offset=25,
        )
        == "/?selected=rec+helper&limit=25&offset=25"
    )
    assert ui_routes._control_center_shell_href(  # noqa: SLF001
        selected="rec-helper-1",
        calendar_error="calendar drift",
        speakers_notice="snippets fixed",
        speakers_error="speaker mismatch",
    ) == (
        "/?selected=rec-helper-1&calendar_error=calendar+drift&"
        "speakers_notice=snippets+fixed&speakers_error=speaker+mismatch"
    )
    assert ui_routes._control_center_recordings_panel_url(  # noqa: SLF001
        selected="rec-helper-1",
        status_filter="Ready",
        search_query="demo",
        tab="log",
        limit=25,
        offset=2,
    ) == (
        "/ui/control-center/recordings/panel?"
        "selected=rec-helper-1&status=Ready&q=demo&tab=log&limit=25&offset=2"
    )
    assert (
        ui_routes._control_center_inspector_path(  # noqa: SLF001
            "rec helper",
            status_filter="Ready",
            search_query="demo",
            tab="mystery",
            limit=25,
            offset=2,
        )
        == "/ui/recordings/rec%20helper/inspector?status=Ready&q=demo&limit=25&offset=2"
    )
    assert (
        ui_routes._control_center_return_query(  # noqa: SLF001
            status_filter="Ready",
            search_query="demo",
            return_tab="mystery",
            limit=25,
            offset=2,
        )
        == "?return_to=control-center&return_tab=overview&status=Ready&q=demo&limit=25&offset=2"
    )
    assert ui_routes._workflow_return_query_pairs(  # noqa: SLF001
        return_to="control-center",
        selected="rec-helper-1",
        status="Ready",
        q="helper",
        tab="speakers",
        limit=100,
        offset=25,
    ) == [
        ("return_to", "control-center"),
        ("selected", "rec-helper-1"),
        ("status", "Ready"),
        ("q", "helper"),
        ("tab", "speakers"),
        ("limit", "100"),
        ("offset", "25"),
    ]
    assert ui_routes._workflow_page_href(  # noqa: SLF001
        "/glossary",
        return_to="control-center",
        selected="rec-helper-1",
        status="Ready",
        q="helper",
        tab="speakers",
        limit=100,
        offset=25,
        extra_params=[("recording_id", "rec-helper-1")],
    ) == (
        "/glossary?return_to=control-center&selected=rec-helper-1&status=Ready&"
        "q=helper&tab=speakers&limit=100&offset=25&recording_id=rec-helper-1"
    )
    assert (
        ui_routes._workflow_page_href(  # noqa: SLF001
            "/glossary",
            return_to="control-center",
            selected="",
            status="Ready",
            q="",
            tab="overview",
            extra_params=[("recording_id", "")],
        )
        == "/glossary?return_to=control-center&status=Ready"
    )
    workflow_return = ui_routes._workflow_return_context(  # noqa: SLF001
        return_to="control-center",
        selected="rec-helper-1",
        status="Ready",
        q="helper",
        tab="speakers",
        limit=100,
        offset=25,
        default_href="/glossary",
    )
    assert workflow_return["active"] is True
    assert (
        workflow_return["href"]
        == "/?selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=100&offset=25"
    )
    assert (
        workflow_return["selected_detail_href"]
        == "/recordings/rec-helper-1?tab=speakers"
    )
    queue_return = ui_routes._workflow_return_context(  # noqa: SLF001
        return_to="control-center",
        selected="",
        status="Ready",
        q="",
        tab="overview",
        default_href="/glossary",
    )
    assert queue_return["active"] is True
    assert "same queue state" in queue_return["message"]
    inactive_return = ui_routes._workflow_return_context(  # noqa: SLF001
        return_to="",
        selected="rec-helper-1",
        status="Ready",
        q="helper",
        tab="speakers",
        limit=100,
        offset=25,
        default_href="/glossary",
    )
    assert inactive_return["active"] is False
    assert inactive_return["href"] == "/glossary"
    assert (
        ui_routes._recording_inspector_return_path(  # noqa: SLF001
            "rec-helper-1",
            return_tab="mystery",
        )
        == "/recordings/rec-helper-1"
    )
    assert (
        ui_routes._recording_inspector_return_path(  # noqa: SLF001
            "rec-helper-1",
            return_tab="overview",
            calendar_error="calendar drift",
        )
        == "/recordings/rec-helper-1?calendar_error=calendar+drift"
    )
    assert (
        ui_routes._recording_inspector_return_path(  # noqa: SLF001
            "rec-helper-1",
            return_to="control-center",
            return_tab="speakers",
            status="Ready",
            q="helper",
            limit=25,
            offset=2,
        )
        == "/?selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=25&offset=2"
    )
    assert (
        ui_routes._recording_inspector_return_path(  # noqa: SLF001
            "rec-helper-1",
            return_to="control-center",
            return_tab="summary",
            status="Ready",
        )
        == "/?selected=rec-helper-1&status=Ready&tab=summary"
    )
    assert (
        ui_routes._recording_detail_path("rec-helper-1", tab="summary")
        == "/recordings/rec-helper-1?tab=summary"
    )

    monkeypatch.setattr(
        ui_routes,
        "_recordings_list_items_context",
        lambda items, *, settings: [
            {
                "id": item["id"],
                "status": "Ready",
                "source_filename": "helper.wav",
            }
            for item in items
        ],
    )
    panel_context = ui_routes._recordings_panel_context(  # noqa: SLF001
        cfg,
        mode="control_center",
        selected="rec-helper-1",
        status="Ready",
        q="helper",
        limit=25,
        offset=0,
        tab="speakers",
    )
    assert panel_context["total"] == 3
    assert panel_context["recordings_filters"]["status_filter"] == "Ready"
    assert (
        panel_context["recordings_filters"]["hidden_fields"][0]["value"]
        == "rec-helper-1"
    )
    assert panel_context["recordings_table"]["rows"][0]["detail_href"] == (
        "/recordings/rec-helper-1?tab=speakers"
    )
    assert panel_context["recordings_table"]["rows"][0]["select_href"] == (
        "/?selected=rec-helper-1&status=Ready&q=helper&tab=speakers"
    )
    assert panel_context["recordings_table"]["rows"][0]["selected"] is True
    assert panel_context["recordings_table"]["rows"][0]["meeting_title"] == "helper.wav"
    assert panel_context["recordings_table"]["rows"][0]["status_dot_tone"] == "ready"
    assert panel_context["recordings_table"]["rows"][0]["progress_percent"] == 100
    assert panel_context["recordings_table"]["mode"] == "control_center"
    assert panel_context["status_cards"][0]["status"] == "All"
    assert panel_context["status_cards"][0]["active"] is False
    assert any(
        card["status"] == "Ready" and card["active"]
        for card in panel_context["status_cards"]
    )

    work_pane = ui_routes._control_center_work_pane_context(  # noqa: SLF001
        cfg,
        state=state,
    )
    assert (
        work_pane["recordings_panel"]["panel_id"] == "control-center-recordings-panel"
    )
    assert work_pane["recordings_panel"]["title"] == ""
    assert "daily loop" in work_pane["preview_message"]
    assert work_pane["recordings_panel"]["recordings_filters"]["limit"] == 100
    assert (
        work_pane["workflow_links"]["selected_detail_href"]
        == "/recordings/rec-helper-1?tab=speakers"
    )
    assert work_pane["workflow_links"]["corrections_href"] == (
        "/glossary?return_to=control-center&selected=rec-helper-1&status=Ready&"
        "q=helper&tab=speakers&limit=100&offset=25&recording_id=rec-helper-1"
    )
    assert (
        work_pane["glossary_summary"]["manage_href"]
        == work_pane["workflow_links"]["corrections_href"]
    )
    assert work_pane["voice_summary"]["profile_count"] == 1
    assert work_pane["voice_summary"]["sample_count"] == 2
    assert work_pane["voice_summary"]["profiles"][0]["display_name"] == "Alex Helper"
    assert work_pane["voice_summary"]["profiles"][0]["sample_count"] == 1

    system_bar = ui_routes._control_center_system_bar_context(  # noqa: SLF001
        cfg,
    )
    assert len(system_bar["items"]) == 3
    assert system_bar["items"][0]["label"] == "Node status"
    assert system_bar["items"][0]["show_dot"] is True
    assert system_bar["items"][1]["value"] == "GPU ready"
    assert system_bar["items"][2]["label"] == "LLM:"

    assert ui_routes._control_center_visible_total(  # noqa: SLF001
        cfg,
        state=state,
    ) == 3

    filters = ui_routes._recordings_filters_context(  # noqa: SLF001
        mode="control_center",
        selected="rec-helper-1",
        status_filter="Ready",
        search_query="helper",
        tab="speakers",
        limit=25,
    )
    assert filters["limit_options"] == [25, 50, 100, 200]
    assert filters["hx_target"] == "#control-center-recordings-panel"
    assert work_pane["recordings_panel"]["system_bar_url"] == (
        "/ui/control-center/system-bar?"
        "selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=100&offset=0"
    )

    table = ui_routes._recordings_table_context(  # noqa: SLF001
        mode="control_center",
        selected="rec-helper-1",
        items=[{"id": "rec-helper-1"}],
        total=3,
        limit=2,
        offset=0,
        status_filter="Ready",
        search_query="helper",
        tab="speakers",
    )
    assert table["has_prev"] is False
    assert table["has_next"] is True
    assert (
        table["next_href"]
        == "/?selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=2&offset=2"
    )
    assert table["next_hx_get"].endswith("limit=2&offset=2")
    assert table["rows"][0]["select_href"] == (
        "/?selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=2"
    )
    assert table["rows"][0]["selected"] is True
    assert table["mode"] == "control_center"

    paged_table = ui_routes._recordings_table_context(  # noqa: SLF001
        mode="control_center",
        selected="rec-helper-1",
        items=[{"id": "rec-helper-2"}],
        total=5,
        limit=2,
        offset=2,
        status_filter="Ready",
        search_query="helper",
        tab="speakers",
    )
    assert (
        paged_table["prev_href"]
        == "/?selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=2"
    )
    assert paged_table["next_href"] == (
        "/?selected=rec-helper-1&status=Ready&q=helper&tab=speakers&limit=2&offset=4"
    )

    upload_shell = ui_routes._upload_shell_context(mode="control_center")  # noqa: SLF001
    assert upload_shell["file_input_id"] == "file-input"
    assert upload_shell["empty_row_id"] == "upload-empty"
    assert upload_shell["show_compact_queue"] is True
    assert upload_shell["show_empty_state_text"] is False
    assert upload_shell["remove_terminal_items"] is True
    assert upload_shell["section_title"] == "UPLOAD"
    assert upload_shell["active_counter_id"] == "upload-active-count"
    standalone_upload_shell = ui_routes._upload_shell_context(mode="standalone")  # noqa: SLF001
    assert standalone_upload_shell["show_compact_queue"] is False
    assert standalone_upload_shell["show_empty_state_text"] is True
    assert standalone_upload_shell["remove_terminal_items"] is False
    assert (
        standalone_upload_shell["queue_title"]
        == "Recent uploads stay here while you work"
    )

    page_notice = ui_routes._page_notice_context("  Recovered from a stuck job.  ")  # noqa: SLF001
    assert page_notice == {"message": "Recovered from a stuck job."}

    selected_shell = ui_routes._selected_recording_summary_shell_context(  # noqa: SLF001
        {
            "id": "rec-helper-1",
            "source_filename": "meeting.wav",
            "captured_at_display": "—",
            "stop_eligible": False,
            "stop_in_progress": False,
        },
        current_tab="overview",
        recovery_warning=None,
    )
    assert selected_shell["notices"] == []
    assert selected_shell["action_bar"]["current_tab"] == "overview"


def test_control_center_system_bar_route_avoids_work_pane_builder(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg, c = client

    def _unexpected_work_pane(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("_control_center_work_pane_context should not run here")

    def _fake_system_bar_context(
        settings: AppSettings,
        *,
        runtime_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert settings is ui_routes._settings  # noqa: SLF001
        assert runtime_status == {"items": []}
        return {
            "items": [
                {
                    "label": "Node status",
                    "value": "Online",
                    "detail": "synthetic node detail",
                    "tone": "healthy",
                    "show_dot": True,
                }
            ],
        }

    monkeypatch.setattr(
        ui_routes, "_control_center_work_pane_context", _unexpected_work_pane
    )
    monkeypatch.setattr(
        ui_routes,
        "_control_center_recordings_panel_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("_control_center_recordings_panel_context should not run here")
        ),
    )
    monkeypatch.setattr(
        ui_routes,
        "list_recordings",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("list_recordings should not run here")
        ),
    )

    async def _fake_run_in_threadpool(func, settings):
        assert func is ui_routes.collect_control_center_runtime_status
        assert settings is ui_routes._settings  # noqa: SLF001
        return {"items": []}

    monkeypatch.setattr(ui_routes, "run_in_threadpool", _fake_run_in_threadpool)
    monkeypatch.setattr(
        ui_routes, "_control_center_system_bar_context", _fake_system_bar_context
    )

    response = c.get(
        "/ui/control-center/system-bar"
        "?selected=rec-route-1&status=Ready&q=meeting&tab=speakers&limit=100&offset=25"
    )

    assert response.status_code == 200
    assert 'id="control-center-system-bar"' in response.text
    assert "Node status" in response.text
    assert "Online" in response.text


def test_recordings_panel_context_clamps_offset_to_last_available_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    calls: list[int] = []

    def _list_recordings(**kwargs):
        offset = int(kwargs["offset"])
        calls.append(offset)
        if offset == 25:
            return [], 25
        return [{"id": "rec-helper-1"}], 25

    monkeypatch.setattr(ui_routes, "list_recordings", _list_recordings)
    monkeypatch.setattr(ui_routes, "_status_counts", lambda settings: {"Ready": 25})
    monkeypatch.setattr(
        ui_routes,
        "_recordings_list_items_context",
        lambda items, *, settings: [
            {
                "id": item["id"],
                "status": "Ready",
                "source_filename": "helper.wav",
            }
            for item in items
        ],
    )

    panel_context = ui_routes._recordings_panel_context(  # noqa: SLF001
        cfg,
        mode="control_center",
        selected="rec-helper-1",
        status="Ready",
        q="helper",
        limit=25,
        offset=25,
        tab="speakers",
    )

    assert calls[:2] == [25, 0]
    assert panel_context["recordings_table"]["offset"] == 0
    assert panel_context["recordings_table"]["rows"][0]["id"] == "rec-helper-1"
    assert panel_context["refresh_url"].endswith("limit=25&offset=0")

    selected_shell_warning = ui_routes._selected_recording_summary_shell_context(  # noqa: SLF001
        {
            "id": "rec-helper-2",
            "source_filename": "meeting.wav",
            "captured_at_display": "2026-01-10 11:00:00 CET",
            "stop_eligible": True,
            "stop_in_progress": True,
        },
        current_tab="log",
        recovery_warning="Recovered from a stuck job.",
    )
    assert selected_shell_warning["notices"] == [
        {"message": "Recovered from a stuck job."}
    ]
    assert selected_shell_warning["action_bar"]["back_href"] == "/"
    assert (
        selected_shell_warning["action_bar"]["back_label"] == "Back to Control Center"
    )
    selected_shell_notice = ui_routes._selected_recording_summary_shell_context(  # noqa: SLF001
        {
            "id": "rec-helper-2",
            "source_filename": "meeting.wav",
            "captured_at_display": "2026-01-10 11:00:00 CET",
            "stop_eligible": False,
            "stop_in_progress": False,
        },
        current_tab="overview",
        recovery_warning=None,
        workflow_notice="Saved correction for Sander.",
    )
    assert selected_shell_notice["notices"] == [
        {"message": "Saved correction for Sander."}
    ]

    empty_shell = ui_routes._empty_inspector_shell_context()  # noqa: SLF001
    assert empty_shell["title"] == "No recording selected"
    assert empty_shell["message"] == ""

    control_center_empty = ui_routes._control_center_empty_inspector_context()  # noqa: SLF001
    assert control_center_empty["title"] == "No recording selected"
    assert control_center_empty["message"] == ""

    monkeypatch.setattr(
        ui_routes,
        "list_glossary_entries",
        lambda settings: [
            {
                "id": 1,
                "canonical_text": "Sander",
                "kind": "person",
                "aliases_json": ["Sandia"],
                "source": "speaker_bank",
            },
            {
                "id": 2,
                "canonical_text": "Roadmap",
                "kind": "project",
                "aliases_json": "invalid",
                "source": None,
            },
        ],
    )
    glossary_summary = ui_routes._compact_glossary_summary_context(cfg, limit=2)  # noqa: SLF001
    assert glossary_summary["entry_count"] == 2
    assert glossary_summary["entries"][0]["aliases"] == ["Sandia"]
    assert glossary_summary["entries"][1]["aliases"] == []
    assert glossary_summary["entries"][0]["kind"] == "Person or speaker name"
    assert glossary_summary["entries"][1]["kind"] == "Project"
    assert glossary_summary["entries"][1]["source_label"] == "Always-on memory"


def test_embedded_recording_details_context_prefers_confirmed_title_and_calendar_match(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    recording = create_recording(
        "rec-details-confirmed",
        source="upload",
        source_filename="meeting.wav",
        captured_at="2026-03-15T10:30:00Z",
        duration_sec=65.0,
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-details-confirmed" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Ignored summary title",
                "summary_bullets": ["Reviewed blockers", "Aligned on next steps"],
                "emotional_summary": "Focused and calm. Everyone aligned.",
            }
        ),
        encoding="utf-8",
    )
    (derived / "transcript.json").write_text(
        json.dumps({"dominant_language": "en"}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.2, "text": "hello"}]),
        encoding="utf-8",
    )
    profile = create_voice_profile("Andrea", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-details-confirmed",
        diar_speaker_label="S1",
        voice_profile_id=int(profile["id"]),
        confidence=0.98,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-details-confirmed",
        candidates=[{"event_id": "evt-1", "subject": "Quarterly roadmap"}],
        selected_event_id="evt-1",
        selected_confidence=0.96,
        settings=cfg,
    )

    prepared = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {**recording, "display_title": "Confirmed daily title"},
        settings=cfg,
    )
    details = ui_routes._embedded_recording_details_context(  # noqa: SLF001
        "rec-details-confirmed",
        recording=prepared,
        diagnostics={"primary_reason_text": ""},
        current_tab="speakers",
        stage_rows=[],
        settings=cfg,
    )

    assert details["primary_title"] == "Confirmed daily title"
    assert details["primary_title_source"] == "confirmed"
    assert details["summary_lines"] == ["Reviewed blockers", "Aligned on next steps"]
    assert details["tone"] == "Focused and calm."
    assert details["speaker_rows"] == [
        {
            "primary_label": "Andrea",
            "secondary_label": "S1",
            "confidence_display": "98%",
        }
    ]
    assert details["metadata_rows"][-1] == {
        "label": "Matched meeting",
        "value": "Quarterly roadmap",
    }
    assert details["metadata_rows"][4]["value"] == "Ready for export."
    assert details["open_recording_page_href"] == "/recordings/rec-details-confirmed?tab=speakers"


def test_embedded_recording_details_context_summary_and_job_id_fallbacks(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)

    summary_recording = create_recording(
        "rec-details-summary",
        source="upload",
        source_filename="summary.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    summary_derived = cfg.recordings_root / "rec-details-summary" / "derived"
    summary_derived.mkdir(parents=True, exist_ok=True)
    (summary_derived / "summary.json").write_text(
        json.dumps(
            {
                "topic": "Summary fallback title",
                "summary": "First sentence. Second sentence. Third sentence.",
            }
        ),
        encoding="utf-8",
    )
    summary_prepared = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        summary_recording,
        settings=cfg,
    )
    summary_details = ui_routes._embedded_recording_details_context(  # noqa: SLF001
        "rec-details-summary",
        recording=summary_prepared,
        diagnostics={"primary_reason_text": ""},
        current_tab="overview",
        stage_rows=[],
        settings=cfg,
    )

    assert summary_details["primary_title"] == "Summary fallback title"
    assert summary_details["primary_title_source"] == "summary_topic"
    assert summary_details["summary_lines"] == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    assert summary_details["tone"] == "Not available yet"
    assert summary_details["speaker_rows"] == []
    assert all(
        row["label"] != "Matched meeting" for row in summary_details["metadata_rows"]
    )

    job_recording = create_recording(
        "rec-details-job",
        source="upload",
        source_filename="fallback.wav",
        status=RECORDING_STATUS_PROCESSING,
        settings=cfg,
    )
    job_prepared = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        job_recording,
        settings=cfg,
    )
    job_details = ui_routes._embedded_recording_details_context(  # noqa: SLF001
        "rec-details-job",
        recording=job_prepared,
        diagnostics={"primary_reason_text": "Check speakers"},
        current_tab="export",
        stage_rows=[],
        settings=cfg,
    )

    assert job_details["primary_title"] == "rec-details-job"
    assert job_details["primary_title_source"] == "job_id"
    assert job_details["filename"] == "fallback.wav"
    assert job_details["tone"] == "Not available yet"
    assert job_details["summary_lines"] == []
    assert job_details["metadata_rows"][4]["value"] == "Check speakers"
    assert job_details["open_recording_page_href"] == "/recordings/rec-details-job?tab=export"


def test_embedded_recording_details_context_handles_summary_fallback_and_bad_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    derived = cfg.recordings_root / "rec-details-edge" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"text": "edge transcript"}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps([{"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ui_routes,
        "_summary_context",
        lambda *_a, **_k: {
            "summary_text": "-",
            "summary_bullets": [],
            "emotional_summary": "",
        },
    )
    monkeypatch.setattr(
        ui_routes,
        "list_speaker_assignments",
        lambda *_a, **_k: [
            {
                "diar_speaker_label": "S1",
                "voice_profile_id": 1,
                "voice_profile_name": "Andrea",
                "local_display_name": "",
                "confidence": "bad",
            }
        ],
    )
    monkeypatch.setattr(
        ui_routes,
        "_control_center_meeting_title_context",
        lambda *_a, **_k: {
            "meeting_title": "edge.wav",
            "meeting_title_source": "filename",
        },
    )
    monkeypatch.setattr(
        ui_routes,
        "_recording_dominant_language_display",
        lambda *_a, **_k: "Unknown",
    )

    details = ui_routes._embedded_recording_details_context(  # noqa: SLF001
        "rec-details-edge",
        recording={
            "id": "rec-details-edge",
            "status": RECORDING_STATUS_READY,
            "captured_at_display": "—",
            "duration_display": "—",
            "source_filename": "edge.wav",
            "review_reason_text_display": "",
            "status_reason_text_display": "",
        },
        diagnostics={"primary_reason_text": ""},
        current_tab="overview",
        stage_rows=[],
        settings=cfg,
    )

    assert details["primary_title"] == "rec-details-edge"
    assert details["summary_lines"] == ["-"]
    assert details["speaker_rows"] == [
        {
            "primary_label": "Andrea",
            "secondary_label": "S1",
            "confidence_display": "",
        }
    ]


def test_embedded_speaker_summary_rows_supports_turns_and_assignment_only_speakers(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-speaker-summary",
        source="upload",
        source_filename="speaker-summary.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-speaker-summary" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"text": "hello world"}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps(
            [
                {"speaker": "S1", "start": 0.0, "end": 1.0, "text": "hello"},
                {"speaker": "S2", "start": 1.0, "end": 2.0, "text": "world"},
            ]
        ),
        encoding="utf-8",
    )
    profile = create_voice_profile("Andrea", settings=cfg)
    set_speaker_assignment(
        recording_id="rec-speaker-summary",
        diar_speaker_label="S1",
        voice_profile_id=int(profile["id"]),
        confidence=0.98,
        settings=cfg,
    )
    set_speaker_assignment(
        recording_id="rec-speaker-summary",
        diar_speaker_label="S3",
        voice_profile_id=None,
        local_display_name="Guest",
        settings=cfg,
    )

    rows = ui_routes._embedded_speaker_summary_rows(  # noqa: SLF001
        "rec-speaker-summary",
        settings=cfg,
    )

    assert rows == [
        {
            "primary_label": "Andrea",
            "secondary_label": "S1",
            "confidence_display": "98%",
        },
        {
            "primary_label": "S2",
            "secondary_label": "Unknown",
            "confidence_display": "",
        },
        {
            "primary_label": "Guest",
            "secondary_label": "S3",
            "confidence_display": "100%",
        },
    ]


def test_embedded_speaker_summary_rows_fall_back_to_transcript_chunks(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-speaker-summary-fallback",
        source="upload",
        source_filename="speaker-summary-fallback.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-speaker-summary-fallback" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"text": "fallback transcript only"}),
        encoding="utf-8",
    )

    rows = ui_routes._embedded_speaker_summary_rows(  # noqa: SLF001
        "rec-speaker-summary-fallback",
        settings=cfg,
    )

    assert rows == [
        {
            "primary_label": "S1",
            "secondary_label": "Unknown",
            "confidence_display": "",
        }
    ]


def test_embedded_speaker_summary_rows_normalizes_blank_turn_labels(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-speaker-summary-blank",
        source="upload",
        source_filename="speaker-summary-blank.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-speaker-summary-blank" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"text": "blank speaker label"}),
        encoding="utf-8",
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps([{"speaker": "", "start": 0.0, "end": 1.0, "text": "hello"}]),
        encoding="utf-8",
    )

    rows = ui_routes._embedded_speaker_summary_rows(  # noqa: SLF001
        "rec-speaker-summary-blank",
        settings=cfg,
    )

    assert rows == [
        {
            "primary_label": "S1",
            "secondary_label": "Unknown",
            "confidence_display": "",
        }
    ]


def test_compact_inspector_helpers_cover_next_action_branches(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    control_center_state = {
        "status": "Ready",
        "q": "helper",
        "limit": 25,
        "offset": 0,
    }
    derived = cfg.recordings_root / "rec-compact-1" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"dominant_language": "en"}),
        encoding="utf-8",
    )

    ready_overview = ui_routes._compact_inspector_overview_context(  # noqa: SLF001
        "rec-compact-1",
        recording={
            "status": RECORDING_STATUS_READY,
            "captured_at_display": "2026-01-02 03:04:05 CET",
            "pipeline_updated_at_display": "2026-01-02 03:05:00 CET",
            "source": "upload",
            "language_auto": "it",
            "status_reason_text_display": "",
        },
        diagnostics={
            "current_stage_label": "Done",
            "current_stage_code": "done",
            "primary_reason_text": "",
        },
        control_center_state=control_center_state,
        settings=cfg,
    )
    assert ready_overview["next_action"]["href"].endswith("tab=export")
    assert ready_overview["metadata"][2]["value"] == "English (en)"

    processing_overview = ui_routes._compact_inspector_overview_context(  # noqa: SLF001
        "rec-compact-1",
        recording={
            "status": RECORDING_STATUS_PROCESSING,
            "captured_at_display": "2026-01-02 03:04:05 CET",
            "pipeline_updated_at_display": "—",
            "pipeline_stage": "asr",
            "source": "upload",
            "language_auto": "it",
            "status_reason_text_display": "",
        },
        diagnostics={
            "current_stage_label": "Waiting",
            "current_stage_code": "asr",
            "primary_reason_text": "",
        },
        control_center_state=control_center_state,
        settings=cfg,
    )
    assert processing_overview["stage_label"] == "ASR / VAD"
    assert (
        processing_overview["stage_detail"] == "The worker is updating this stage live."
    )

    blocked_overview = ui_routes._compact_inspector_overview_context(  # noqa: SLF001
        "rec-compact-1",
        recording={
            "status": RECORDING_STATUS_NEEDS_REVIEW,
            "captured_at_display": "2026-01-02 03:04:05 CET",
            "pipeline_updated_at_display": "2026-01-02 03:05:00 CET",
            "source": "upload",
            "language_auto": "it",
            "status_reason_text_display": "Speaker review required.",
        },
        diagnostics={
            "current_stage_label": "Speaker Turns",
            "current_stage_code": "speaker_turns",
            "primary_reason_text": "Speaker review required.",
        },
        control_center_state=control_center_state,
        settings=cfg,
    )
    assert blocked_overview["blocker_text"] == "Speaker review required."

    review_speakers = ui_routes._compact_inspector_next_action_context(  # noqa: SLF001
        "rec-compact-1",
        recording={
            "status": RECORDING_STATUS_NEEDS_REVIEW,
            "status_reason_text_display": "",
        },
        diagnostics={
            "current_stage_code": "speaker_turns",
            "primary_reason_text": "Speaker labels need review",
        },
        control_center_state=control_center_state,
    )
    assert review_speakers["href"].endswith("tab=speakers")

    review_summary = ui_routes._compact_inspector_next_action_context(  # noqa: SLF001
        "rec-compact-1",
        recording={
            "status": RECORDING_STATUS_NEEDS_REVIEW,
            "status_reason_text_display": "",
        },
        diagnostics={
            "current_stage_code": "llm_merge",
            "primary_reason_text": "Summary output needs review",
        },
        control_center_state=control_center_state,
    )
    assert review_summary["href"].endswith("tab=summary")

    stopped = ui_routes._compact_inspector_next_action_context(  # noqa: SLF001
        "rec-compact-1",
        recording={
            "status": RECORDING_STATUS_STOPPED,
            "status_reason_text_display": "",
        },
        diagnostics={"current_stage_code": "waiting", "primary_reason_text": ""},
        control_center_state=control_center_state,
    )
    assert stopped["title"] == "Requeue when ready"

    quarantine = ui_routes._compact_inspector_next_action_context(  # noqa: SLF001
        "rec-compact-1",
        recording={
            "status": RECORDING_STATUS_QUARANTINE,
            "status_reason_text_display": "",
        },
        diagnostics={"current_stage_code": "done", "primary_reason_text": ""},
        control_center_state=control_center_state,
    )
    assert quarantine["external"] is True
    assert quarantine["href"] == "/recordings/rec-compact-1"

    fallback = ui_routes._compact_inspector_next_action_context(  # noqa: SLF001
        "rec-compact-1",
        recording={"status": "Mystery", "status_reason_text_display": ""},
        diagnostics={"current_stage_code": "", "primary_reason_text": ""},
        control_center_state=control_center_state,
    )
    assert fallback["button_label"] == "Open Full Page"


def test_full_page_helpers_cover_overview_and_next_action_branches(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)

    review_speakers = ui_routes._full_page_overview_next_action_context(  # noqa: SLF001
        "rec-full-1",
        recording={"status": RECORDING_STATUS_NEEDS_REVIEW},
        diagnostics={
            "current_stage_code": "diarization",
            "primary_reason_text": "Speaker labels need review",
        },
    )
    assert review_speakers["href"].endswith("tab=speakers")

    failed = ui_routes._full_page_overview_next_action_context(  # noqa: SLF001
        "rec-full-1",
        recording={"status": RECORDING_STATUS_FAILED},
        diagnostics={"current_stage_code": "llm_extract", "primary_reason_text": ""},
    )
    assert failed["title"] == "Triage before retrying"
    assert failed["href"].endswith("tab=diagnostics")

    fallback = ui_routes._full_page_overview_next_action_context(  # noqa: SLF001
        "rec-full-1",
        recording={"status": "Mystery"},
        diagnostics={"current_stage_code": "", "primary_reason_text": ""},
    )
    assert fallback["href"].endswith("tab=transcript")

    detailed_overview = ui_routes._full_page_overview_context(  # noqa: SLF001
        "rec-full-1",
        recording={
            "status": RECORDING_STATUS_READY,
            "pipeline_updated_at_display": "2026-01-02 03:05:00 CET",
            "captured_at_display": "2026-01-02 03:04:05 CET",
            "duration_display": "00:00:12",
            "source": "upload",
            "project_name": "Roadmap",
            "suggested_project_name": "Roadmap",
            "routing_confidence": 0.91,
            "review_reason_text_display": "Needs a manual check",
            "quarantine_reason": "Sensitive content",
            "cancel_requested_at_display": "2026-01-02 03:06:00 CET",
            "cancel_requested_by_display": "Alex",
        },
        diagnostics={
            "current_stage_label": "LLM Summary",
            "current_stage_status_label": "Completed",
            "current_stage_code": "llm_extract",
            "primary_reason_text": "Primary blocker",
        },
        settings=cfg,
    )
    assert "Last pipeline update 2026-01-02 03:05:00 CET." in detailed_overview[
        "state_detail"
    ]
    assert "Current stage status: Completed." in detailed_overview["state_detail"]
    assert detailed_overview["metadata"][-1]["value"] == "Roadmap · 0.91"
    assert detailed_overview["focus_items"] == [
        {"label": "Primary reason", "value": "Primary blocker"},
        {"label": "Review reason", "value": "Needs a manual check"},
        {"label": "Quarantine reason", "value": "Sensitive content"},
        {"label": "Routing signal", "value": "Roadmap · 0.91"},
        {
            "label": "Stop requested",
            "value": "2026-01-02 03:06:00 CET by Alex",
        },
    ]

    fallback_overview = ui_routes._full_page_overview_context(  # noqa: SLF001
        "rec-full-1",
        recording={
            "status": "Mystery",
            "pipeline_stage": "sanitize_audio",
            "captured_at_display": "—",
            "duration_display": "—",
            "source": "upload",
            "routing_confidence": 0.42,
        },
        diagnostics={
            "current_stage_label": "Waiting",
            "current_stage_status_label": "Queued",
            "current_stage_code": "",
            "primary_reason_text": "",
        },
        settings=cfg,
    )
    assert fallback_overview["state_detail"] == (
        "Current / last pipeline stage: Sanitize Audio."
    )
    assert fallback_overview["metadata"][-1]["value"] == "0.42"

    branch_overview = ui_routes._full_page_overview_context(  # noqa: SLF001
        "rec-full-1",
        recording={
            "status": RECORDING_STATUS_READY,
            "captured_at_display": "—",
            "duration_display": "—",
            "source": "upload",
            "suggested_project_id": 7,
            "cancel_requested_at_display": "2026-01-02 03:07:00 CET",
            "cancel_requested_by_display": "",
        },
        diagnostics={
            "current_stage_label": "Done",
            "current_stage_status_label": "Ready",
            "current_stage_code": "done",
            "primary_reason_text": "",
        },
        settings=cfg,
    )
    assert branch_overview["metadata"][-1]["value"] == "#7"
    assert branch_overview["focus_items"] == [
        {
            "label": "Routing signal",
            "value": "#7",
        },
        {
            "label": "Stop requested",
            "value": "2026-01-02 03:07:00 CET",
        },
    ]


def test_display_helpers_cover_timezone_duration_and_prepare_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)

    assert ui_routes._pipeline_stage_label("precheck") == "Sanitize & Precheck"  # noqa: SLF001
    assert ui_routes._pipeline_stage_label("snippet_export") == "Snippet Export"  # noqa: SLF001
    assert ui_routes._pipeline_stage_label("llm_chunk_2_of_5") == "LLM Chunk 2 of 5"  # noqa: SLF001
    assert ui_routes._pipeline_stage_label("llm_chunk_bad") == "Llm Chunk Bad"  # noqa: SLF001
    assert ui_routes._pipeline_stage_label("custom_stage") == "Custom Stage"  # noqa: SLF001
    assert ui_routes._recording_source_display(None) == "Unknown"  # noqa: SLF001
    assert ui_routes._recording_source_display("manual_upload") == "Manual Upload"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(None) == "—"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(0) == "—"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(2.0) == "00:00:02"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(2.345) == "00:00:02"  # noqa: SLF001
    assert ui_routes._format_duration_seconds(4699.26) == "01:18:19"  # noqa: SLF001
    assert ui_routes._format_local_timestamp("") == "—"  # noqa: SLF001
    assert ui_routes._format_local_timestamp("bad-timestamp") == "bad-timestamp"  # noqa: SLF001
    assert "CET" in ui_routes._format_local_timestamp("2026-01-10T10:00:00Z")  # noqa: SLF001

    observed_update: dict[str, object] = {}
    monkeypatch.setattr(
        ui_routes,
        "_recording_audio_candidates",
        lambda *_a, **_k: [Path("/tmp/fake.wav")],
    )
    monkeypatch.setattr(ui_routes, "_probe_duration_seconds", lambda *_a, **_k: 3.5)
    monkeypatch.setattr(
        ui_routes,
        "set_recording_duration",
        lambda recording_id,
        duration_sec,
        *,
        settings=None,
        touch_updated_at=True: observed_update.update(
            {
                "recording_id": recording_id,
                "duration_sec": duration_sec,
                "touch_updated_at": touch_updated_at,
            }
        )
        or True,
    )

    prepared = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {
            "id": "rec-helper-1",
            "duration_sec": None,
            "captured_at": "2026-01-10T10:00:00Z",
            "created_at": "",
            "updated_at": "bad",
            "pipeline_updated_at": "2026-01-10T10:05:00Z",
            "review_reason_text": "  Needs a closer look.  ",
        },
        settings=cfg,
    )
    assert observed_update == {
        "recording_id": "rec-helper-1",
        "duration_sec": 3.5,
        "touch_updated_at": False,
    }
    assert prepared["duration_display"] == "00:00:03"
    assert prepared["captured_at_display"].endswith("CET")
    assert prepared["created_at_display"] == "—"
    assert prepared["updated_at_display"] == "bad"
    assert prepared["pipeline_updated_at_display"].endswith("CET")
    assert prepared["review_reason_text_display"] == "Needs a closer look."
    assert prepared["status_reason_text_display"] == "Needs a closer look."
    assert prepared["stop_eligible"] is False
    assert prepared["stop_in_progress"] is False

    observed_update.clear()
    prepared_existing_duration = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {
            "id": "rec-helper-2",
            "duration_sec": 1.0,
            "captured_at": None,
            "created_at": None,
            "updated_at": None,
            "pipeline_updated_at": None,
            "review_reason_text": None,
        },
        settings=cfg,
    )
    assert observed_update == {}
    assert prepared_existing_duration["duration_display"] == "00:00:01"

    prepared_stopping = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {
            "id": "rec-helper-3",
            "status": RECORDING_STATUS_STOPPING,
            "duration_sec": 1.0,
            "captured_at": None,
            "created_at": None,
            "updated_at": None,
            "pipeline_updated_at": None,
            "cancel_requested_at": "2026-01-10T10:06:00Z",
            "cancel_requested_by": "user",
            "cancel_reason_text": "Stop requested by user",
        },
        settings=cfg,
    )
    assert prepared_stopping["status_reason_text_display"] == "Stop requested by user"
    assert prepared_stopping["cancel_requested_at_display"].endswith("CET")
    assert prepared_stopping["stop_eligible"] is True
    assert prepared_stopping["stop_in_progress"] is True

    prepared_stopped = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {
            "id": "rec-helper-4",
            "status": RECORDING_STATUS_STOPPED,
            "duration_sec": 1.0,
            "captured_at": None,
            "created_at": None,
            "updated_at": None,
            "pipeline_updated_at": None,
            "cancel_requested_at": None,
            "cancel_requested_by": None,
            "cancel_reason_text": "Cancelled by user",
        },
        settings=cfg,
    )
    assert prepared_stopped["status_reason_text_display"] == "Cancelled by user"
    assert prepared_stopped["stop_eligible"] is False


@pytest.mark.parametrize(
    ("recording", "expected"),
    [
        (
            {"status_reason_text_display": "Needs an operator review."},
            "Needs an operator review.",
        ),
        (
            {"status": RECORDING_STATUS_QUEUED},
            "Waiting for the worker to pick this up.",
        ),
        (
            {"status": RECORDING_STATUS_PROCESSING, "pipeline_stage": "diarize"},
            "Running Diarization.",
        ),
        (
            {"status": RECORDING_STATUS_STOPPING},
            "Stop requested. Waiting for a safe checkpoint.",
        ),
        ({"status": RECORDING_STATUS_STOPPED}, "Stopped by an operator."),
        (
            {"status": RECORDING_STATUS_NEEDS_REVIEW, "routing_confidence": 0.42},
            "Routing confidence 0.42. Review before publish.",
        ),
        (
            {"status": RECORDING_STATUS_NEEDS_REVIEW},
            "Manual review is still required.",
        ),
        (
            {"status": RECORDING_STATUS_NEEDS_REVIEW, "routing_confidence": "bad"},
            "Manual review is still required.",
        ),
        ({"status": RECORDING_STATUS_READY}, "Ready for export."),
        ({"status": RECORDING_STATUS_PUBLISHED}, "Published output is available."),
        (
            {"status": RECORDING_STATUS_QUARANTINE},
            "Quarantined. Inspect before requeue.",
        ),
        (
            {"status": RECORDING_STATUS_FAILED},
            "Open the recording to inspect the failed stage.",
        ),
        ({"status": "Unknown"}, ""),
    ],
)
def test_recording_worklist_hint_covers_statuses(
    recording: dict[str, Any],
    expected: str,
) -> None:
    assert ui_routes._recording_worklist_hint(recording) == expected  # noqa: SLF001


def test_recordings_list_items_context_adds_source_and_worklist_fields(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)

    items = ui_routes._recordings_list_items_context(  # noqa: SLF001
        [
            {
                "id": "rec-worklist-1",
                "status": RECORDING_STATUS_NEEDS_REVIEW,
                "source": "manual_upload",
                "source_filename": "worklist.wav",
                "duration_sec": 1.0,
                "pipeline_progress": 0.5,
                "pipeline_stage": "diarize",
                "captured_at": None,
                "created_at": None,
                "updated_at": None,
                "pipeline_updated_at": None,
                "review_reason_text": None,
            }
        ],
        settings=cfg,
    )

    assert items[0]["progress_percent"] == 50
    assert items[0]["progress_stage_label"] == "Diarization"
    assert items[0]["source_display"] == "Manual Upload"
    assert items[0]["worklist_hint"] == "Manual review is still required."


def test_control_center_worklist_title_and_progress_helpers(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)

    create_recording(
        "rec-title-candidate",
        source="upload",
        source_filename="candidate.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived_candidate = cfg.recordings_root / "rec-title-candidate" / "derived"
    derived_candidate.mkdir(parents=True, exist_ok=True)
    (derived_candidate / "summary.json").write_text(
        json.dumps({"topic": "Ignored summary"}),
        encoding="utf-8",
    )
    upsert_calendar_match(
        recording_id="rec-title-candidate",
        candidates=[
            {"event_id": "evt-blank", "subject": "   "},
            {"event_id": "evt-1", "subject": "Best candidate title"},
        ],
        selected_event_id=None,
        selected_confidence=None,
        settings=cfg,
    )

    create_recording(
        "rec-title-summary-only",
        source="upload",
        source_filename="summary-only.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived_summary = cfg.recordings_root / "rec-title-summary-only" / "derived"
    derived_summary.mkdir(parents=True, exist_ok=True)
    (derived_summary / "summary.json").write_text(
        json.dumps({"topic": "Summary fallback title"}),
        encoding="utf-8",
    )

    create_recording(
        "rec-title-selected-missing",
        source="upload",
        source_filename="selected-missing.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived_selected_missing = (
        cfg.recordings_root / "rec-title-selected-missing" / "derived"
    )
    derived_selected_missing.mkdir(parents=True, exist_ok=True)
    (derived_selected_missing / "summary.json").write_text(
        json.dumps({"topic": "Selected match missing fallback"}),
        encoding="utf-8",
    )
    upsert_calendar_match(
        recording_id="rec-title-selected-missing",
        candidates=[
            {"event_id": "evt-other", "subject": "   "},
        ],
        selected_event_id="evt-selected",
        selected_confidence=0.82,
        settings=cfg,
    )

    create_recording(
        "rec-title-selected-blank",
        source="upload",
        source_filename="selected-blank.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    derived_selected_blank = cfg.recordings_root / "rec-title-selected-blank" / "derived"
    derived_selected_blank.mkdir(parents=True, exist_ok=True)
    (derived_selected_blank / "summary.json").write_text(
        json.dumps({"topic": "Selected blank fallback"}),
        encoding="utf-8",
    )
    upsert_calendar_match(
        recording_id="rec-title-selected-blank",
        candidates=[
            {"event_id": "evt-other", "subject": "   "},
            {"event_id": "evt-selected", "subject": "   ", "summary": "   "},
        ],
        selected_event_id="evt-selected",
        selected_confidence=0.77,
        settings=cfg,
    )

    candidate_title = ui_routes._control_center_meeting_title_context(  # noqa: SLF001
        {"id": "rec-title-candidate", "source_filename": "candidate.wav"},
        settings=cfg,
    )
    assert candidate_title == {
        "meeting_title": "Best candidate title",
        "meeting_title_source": "calendar_candidate",
    }

    summary_title = ui_routes._control_center_meeting_title_context(  # noqa: SLF001
        {"id": "rec-title-summary-only", "source_filename": "summary-only.wav"},
        settings=cfg,
    )
    assert summary_title == {
        "meeting_title": "Summary fallback title",
        "meeting_title_source": "summary_topic",
    }

    selected_missing_title = ui_routes._control_center_meeting_title_context(  # noqa: SLF001
        {"id": "rec-title-selected-missing", "source_filename": "selected-missing.wav"},
        settings=cfg,
    )
    assert selected_missing_title == {
        "meeting_title": "Selected match missing fallback",
        "meeting_title_source": "summary_topic",
    }

    selected_blank_title = ui_routes._control_center_meeting_title_context(  # noqa: SLF001
        {"id": "rec-title-selected-blank", "source_filename": "selected-blank.wav"},
        settings=cfg,
    )
    assert selected_blank_title == {
        "meeting_title": "Selected blank fallback",
        "meeting_title_source": "summary_topic",
    }

    fallback_title = ui_routes._control_center_meeting_title_context(  # noqa: SLF001
        {"id": "rec-title-fallback", "source_filename": "fallback.wav"},
        settings=cfg,
    )
    assert fallback_title == {
        "meeting_title": "fallback.wav",
        "meeting_title_source": "filename",
    }

    missing_id_title = ui_routes._control_center_meeting_title_context(  # noqa: SLF001
        {"id": "", "source_filename": ""},
        settings=cfg,
    )
    assert missing_id_title == {
        "meeting_title": "Recording",
        "meeting_title_source": "filename",
    }

    assert ui_routes._control_center_status_dot_tone(  # noqa: SLF001
        RECORDING_STATUS_NEEDS_REVIEW
    ) == "review"
    assert ui_routes._control_center_status_dot_tone(  # noqa: SLF001
        RECORDING_STATUS_QUARANTINE
    ) == "quarantine"
    assert ui_routes._control_center_status_dot_tone(  # noqa: SLF001
        RECORDING_STATUS_FAILED
    ) == "failed"
    assert ui_routes._control_center_status_dot_tone(  # noqa: SLF001
        RECORDING_STATUS_STOPPED
    ) == "stopped"
    assert ui_routes._control_center_status_dot_tone("mystery") == "unknown"  # noqa: SLF001

    queued_progress = ui_routes._control_center_progress_context(  # noqa: SLF001
        {"status": RECORDING_STATUS_QUEUED, "pipeline_progress": None}
    )
    assert queued_progress == {
        "progress_text": "5%",
        "progress_note": "Uploaded",
        "progress_percent": 5,
        "show_progress_bar": True,
    }

    processing_progress = ui_routes._control_center_progress_context(  # noqa: SLF001
        {
            "status": RECORDING_STATUS_PROCESSING,
            "pipeline_progress": 0.5,
            "pipeline_stage": "diarize",
        }
    )
    assert processing_progress == {
        "progress_text": "52%",
        "progress_note": "Diarization",
        "progress_percent": 52,
        "show_progress_bar": True,
    }

    stopping_progress = ui_routes._control_center_progress_context(  # noqa: SLF001
        {"status": RECORDING_STATUS_STOPPING, "pipeline_progress": None}
    )
    assert stopping_progress == {
        "progress_text": "5%",
        "progress_note": "Stop requested",
        "progress_percent": 5,
        "show_progress_bar": True,
    }

    staged_processing_progress = ui_routes._control_center_progress_context(  # noqa: SLF001
        {"status": RECORDING_STATUS_PROCESSING, "pipeline_progress": None, "pipeline_stage": "sanitize"}
    )
    assert staged_processing_progress == {
        "progress_text": "5%",
        "progress_note": "Sanitize",
        "progress_percent": 5,
        "show_progress_bar": True,
    }

    failed_progress = ui_routes._control_center_progress_context(  # noqa: SLF001
        {"status": RECORDING_STATUS_FAILED, "pipeline_progress": None}
    )
    assert failed_progress == {
        "progress_text": "100%",
        "progress_note": "Failed",
        "progress_percent": 100,
        "show_progress_bar": True,
    }

    unknown_progress = ui_routes._control_center_progress_context(  # noqa: SLF001
        {"status": "mystery", "pipeline_progress": None}
    )
    assert unknown_progress == {
        "progress_text": "—",
        "progress_note": "",
        "progress_percent": 0,
        "show_progress_bar": False,
    }


def test_stage_rows_and_diagnostics_context_cover_new_observability_helpers() -> None:
    rows = ui_routes._pipeline_stage_rows_for_display(  # noqa: SLF001
        "rec-observability-1",
        rows=[
            {
                "stage_name": "llm_extract",
                "status": "failed",
                "attempt": 2,
                "updated_at": "2026-03-12T12:00:04Z",
                "started_at": "2026-03-12T12:00:00Z",
                "finished_at": "2026-03-12T12:00:04Z",
                "duration_ms": 4000,
                "error_text": "LLM chunk 3/10 failed [llm_chunk_timeout]: timed out after 120s",
                "metadata_json": {
                    "root_cause_code": "llm_chunk_timeout",
                    "root_cause_text": "LLM chunk 3/10 timed out.",
                    "cancel_chunk_total": "bad",
                },
            }
        ],
    )
    assert rows[0]["root_cause_code"] == "llm_chunk_timeout"
    assert rows[0]["chunk_total"] is None
    assert ui_routes._format_elapsed_seconds(12.2) == "00:00:12"  # noqa: SLF001

    diagnostics_payload = ui_routes._recording_diagnostics_context(  # noqa: SLF001
        recording={"status": "Processing"},
        stage_rows=[],
        chunk_rows=[],
        jobs=[],
    )
    assert diagnostics_payload["chunk_text"] == "—"

    diagnostics_payload = ui_routes._recording_diagnostics_context(  # noqa: SLF001
        recording={
            "status": "Processing",
            "pipeline_stage": "llm_chunk_3",
        },
        stage_rows=[],
        chunk_rows=[
            {
                "chunk_index": "3",
                "chunk_total": None,
                "status": "running",
                "attempt": 1,
            }
        ],
        jobs=[],
    )
    assert diagnostics_payload["chunk_text"] == "3"


def test_prepare_recording_for_display_ignores_duration_backfill_write_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        ui_routes,
        "_recording_audio_candidates",
        lambda *_a, **_k: [Path("/tmp/fake.wav")],
    )
    monkeypatch.setattr(ui_routes, "_probe_duration_seconds", lambda *_a, **_k: 4.25)
    monkeypatch.setattr(
        ui_routes,
        "set_recording_duration",
        lambda *_a, **_k: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )

    with caplog.at_level("WARNING"):
        prepared = ui_routes._prepare_recording_for_display(  # noqa: SLF001
            {
                "id": "rec-helper-err-1",
                "duration_sec": None,
                "captured_at": None,
                "created_at": None,
                "updated_at": None,
                "pipeline_updated_at": None,
                "review_reason_text": None,
            },
            settings=cfg,
        )

    assert prepared["duration_sec"] == 4.25
    assert prepared["duration_display"] == "00:00:04"
    assert (
        "Failed to backfill display duration for recording rec-helper-err-1"
        in caplog.text
    )

    monkeypatch.setattr(
        ui_routes,
        "ZoneInfo",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ui_routes.ZoneInfoNotFoundError("missing tzdata")
        ),
        raising=False,
    )
    assert ui_routes._display_timezone() is timezone.utc  # noqa: SLF001


def test_ui_action_stop_helper_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ui_routes, "_settings", cfg)
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})

    monkeypatch.setattr(ui_routes, "get_recording", lambda *_a, **_k: None)
    missing = asyncio.run(
        ui_routes.ui_action_stop("rec-missing", request=request, tab="overview")
    )
    assert missing.status_code == 404
    assert missing.body.decode("utf-8") == "Not found"

    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda *_a, **_k: {"id": "rec-ready", "status": RECORDING_STATUS_READY},
    )
    not_eligible = asyncio.run(
        ui_routes.ui_action_stop("rec-ready", request=request, tab="calendar")
    )
    assert not_eligible.status_code == 303
    assert not_eligible.headers["location"] == "/recordings/rec-ready?tab=diagnostics"

    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda *_a, **_k: {"id": "rec-queued", "status": RECORDING_STATUS_QUEUED},
    )
    monkeypatch.setattr(ui_routes, "list_jobs", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(
        ui_routes,
        "purge_pending_recording_jobs",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("queue unavailable")),
    )
    queue_error = asyncio.run(
        ui_routes.ui_action_stop("rec-queued", request=request, tab="overview")
    )
    assert queue_error.status_code == 503
    assert (
        queue_error.body.decode("utf-8")
        == "Stop failed (queue unavailable): queue unavailable"
    )

    calls: list[str] = []
    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda *_a, **_k: {
            "id": "rec-stopping",
            "status": RECORDING_STATUS_STOPPING,
            "cancel_requested_at": "2026-01-10T10:06:00Z",
        },
    )
    monkeypatch.setattr(
        ui_routes, "has_started_job_for_recording", lambda *_a, **_k: False
    )
    monkeypatch.setattr(
        ui_routes,
        "acknowledge_recording_cancel_request",
        lambda *_a, **_k: calls.append("ack") or True,
    )
    monkeypatch.setattr(
        ui_routes,
        "set_recording_status_if_current_in",
        lambda *_a, **_k: calls.append("status") or True,
    )
    monkeypatch.setattr(
        ui_routes,
        "clear_recording_progress",
        lambda *_a, **_k: calls.append("clear") or True,
    )
    acknowledged = asyncio.run(
        ui_routes.ui_action_stop("rec-stopping", request=request, tab="overview")
    )
    assert acknowledged.status_code == 303
    assert acknowledged.headers["location"] == "/recordings/rec-stopping"
    assert calls == ["status", "ack", "clear"]

    race_calls: list[str] = []
    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda *_a, **_k: {"id": "rec-race", "status": RECORDING_STATUS_QUEUED},
    )
    monkeypatch.setattr(ui_routes, "list_jobs", lambda **_kwargs: ([], 0))
    monkeypatch.setattr(ui_routes, "purge_pending_recording_jobs", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        ui_routes, "has_started_job_for_recording", lambda *_a, **_k: False
    )
    monkeypatch.setattr(
        ui_routes,
        "set_recording_status_if_current_in",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        ui_routes,
        "set_recording_cancel_request",
        lambda *_a, **_k: race_calls.append("set_cancel") or True,
    )
    monkeypatch.setattr(
        ui_routes,
        "acknowledge_recording_cancel_request",
        lambda *_a, **_k: race_calls.append("ack") or True,
    )
    monkeypatch.setattr(
        ui_routes,
        "clear_recording_progress",
        lambda *_a, **_k: race_calls.append("clear") or True,
    )
    raced = asyncio.run(
        ui_routes.ui_action_stop("rec-race", request=request, tab="overview")
    )
    assert raced.status_code == 303
    assert raced.headers["location"] == "/recordings/rec-race"
    assert race_calls == []

    htmx_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [(b"hx-request", b"true")],
        }
    )
    htmx_response = ui_routes._ui_recording_post_response(  # noqa: SLF001
        htmx_request,
        return_to="",
        redirect_to="/recordings/rec-helper-1?tab=log",
    )
    assert htmx_response.headers["HX-Redirect"] == "/recordings/rec-helper-1?tab=log"


def test_control_center_inspector_routes_handle_missing_selection(client):
    _cfg_obj, c = client

    page = c.get("/?selected=missing")
    assert page.status_code == 200
    assert "Selected recording not found" in page.text

    pane = c.get("/ui/control-center/inspector-pane?selected=missing")
    assert pane.status_code == 200
    assert "Selected recording not found" in pane.text

    focused = c.get("/ui/recordings/missing/inspector")
    assert focused.status_code == 200
    assert "Selected recording not found" in focused.text


def test_load_json_and_chunk_helpers_cover_error_paths(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert ui_routes._load_json_dict(broken) == {}  # noqa: SLF001
    assert ui_routes._load_json_list(broken) == []  # noqa: SLF001

    as_list = tmp_path / "list.json"
    as_list.write_text("[1,2]", encoding="utf-8")
    assert ui_routes._load_json_dict(as_list) == {}  # noqa: SLF001
    as_dict = tmp_path / "dict.json"
    as_dict.write_text('{"k": 1}', encoding="utf-8")
    assert ui_routes._load_json_list(as_dict) == []  # noqa: SLF001

    assert ui_routes._chunk_text_for_turns("   ") == []  # noqa: SLF001
    # Long single word forces split path and leaves no trailing buffered chunk.
    chunks = ui_routes._chunk_text_for_turns("abcdefghij", chunk_size=4)  # noqa: SLF001
    assert chunks == ["abcd", "efgh", "ij"]


def test_summary_context_and_metrics_merge_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-summary-helpers"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(
        json.dumps(
            {
                "summary": "- bullet-1\n\nbullet-2",
                "action_items": [
                    "bad-row",
                    {},
                    {"task": ""},
                    {
                        "task": "task-1",
                        "owner": " ",
                        "deadline": " ",
                        "confidence": "bad",
                    },
                ],
                "questions": {
                    "types": {"open": "NaN"},
                    "total_count": "NaN",
                    "extracted": "Q1\n\nQ2",
                },
            }
        ),
        encoding="utf-8",
    )
    summary = ui_routes._summary_context(recording_id, cfg)  # noqa: SLF001
    assert summary["summary_bullets"][:2] == ["bullet-1", "bullet-2"]
    assert summary["action_items"][0]["confidence"] == 0.5
    assert summary["questions"]["total_count"] == 2

    (derived / "summary.json").write_text(
        json.dumps(
            {"questions": {"types": "bad", "total_count": "x", "extracted": []}}
        ),
        encoding="utf-8",
    )
    summary_no_types = ui_routes._summary_context(recording_id, cfg)  # noqa: SLF001
    assert summary_no_types["questions"]["types"]["open"] == 0

    (derived / "metrics.json").write_text(
        json.dumps(
            {
                "meeting": {
                    "total_interruptions": "7",
                    "total_questions": "bad",
                    "actionability_ratio": "bad",
                },
                "participants": [
                    {"speaker": "S1", "airtime_seconds": "10.2", "turns": "3"},
                    {"speaker": "", "airtime_seconds": 12},
                    {"speaker": "S2", "airtime_seconds": "6.4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ui_routes, "get_meeting_metrics", lambda *_a, **_k: {"json": "bad"}
    )
    monkeypatch.setattr(
        ui_routes,
        "list_participant_metrics",
        lambda *_a, **_k: [
            {"json": "bad", "diar_speaker_label": "Sbad"},
            {"json": {"speaker": ""}, "diar_speaker_label": ""},
            {
                "json": {"speaker": "S1", "questions_count": "x"},
                "diar_speaker_label": "S1",
            },
        ],
    )
    metrics = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert metrics["meeting"]["total_interruptions"] == 7
    assert metrics["meeting"]["total_questions"] == 0
    assert [row["speaker"] for row in metrics["participants"]] == ["S1", "S2"]

    # Cover additional artifact branches: no meeting dict and participants payload not list.
    (derived / "metrics.json").write_text(
        json.dumps({"meeting": "bad", "participants": "bad"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui_routes, "list_participant_metrics", lambda *_a, **_k: [])
    metrics_no_participants = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert metrics_no_participants["participants"] == []

    # Cover participant skip path in final rendering loop.
    (derived / "metrics.json").write_text(
        json.dumps(
            {"participants": [{"speaker": ""}, {"speaker": "S9", "airtime_seconds": 1}]}
        ),
        encoding="utf-8",
    )
    metrics_skip_blank = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert [row["speaker"] for row in metrics_skip_blank["participants"]] == ["S9"]

    # Force the merge path where an existing participant row is skipped due to empty speaker.
    (derived / "metrics.json").write_text(
        json.dumps({"participants": [{"speaker": "S1", "airtime_seconds": 2.0}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ui_routes,
        "list_participant_metrics",
        lambda *_a, **_k: [
            {
                "json": {"speaker": "S1", "airtime_seconds": 1.0},
                "diar_speaker_label": "S1",
            }
        ],
    )
    real_str = str
    seen: dict[str, int] = {"count": 0}

    def _stateful_str(value: object) -> str:
        seen["count"] += 1
        # First conversion keeps S1 during participant ingestion; second triggers line-346 skip.
        if seen["count"] == 2:
            return ""
        return real_str(value)

    monkeypatch.setattr(ui_routes, "str", _stateful_str, raising=False)
    merged = ui_routes._metrics_tab_context(recording_id, cfg)  # noqa: SLF001
    assert merged["participants"]


def test_asr_glossary_context_and_route_error_paths(
    client: tuple[AppSettings, TestClient],
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-glossary-helper-1")
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    assert ui_routes._asr_glossary_context(recording_id, cfg)["available"] is False  # noqa: SLF001

    (derived / "asr_glossary.json").write_text(
        json.dumps(
            {
                "entry_count": "bad",
                "term_count": "bad",
                "truncated": 1,
                "entries": [
                    "skip",
                    {"canonical_text": "", "aliases": ["skip"]},
                    {
                        "canonical_text": "Sander",
                        "aliases": [" Sandia ", ""],
                        "kind": "person",
                        "sources": ["correction", "speaker_bank"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    glossary_ctx = ui_routes._asr_glossary_context(recording_id, cfg)  # noqa: SLF001
    assert glossary_ctx["available"] is True
    assert glossary_ctx["entries"][0]["aliases"] == ["Sandia"]
    assert glossary_ctx["entries"][0]["sources_label"] == "correction, speaker bank"
    assert glossary_ctx["entry_count"] == 1
    assert glossary_ctx["term_count"] == 2
    assert glossary_ctx["manage_href"] == "/glossary"
    assert glossary_ctx["quick_add_href"] == f"/glossary?recording_id={recording_id}"

    (derived / "asr_glossary.json").write_text(
        json.dumps({"entries": "bad"}),
        encoding="utf-8",
    )
    glossary_ctx_no_entries = ui_routes._asr_glossary_context(recording_id, cfg)  # noqa: SLF001
    assert glossary_ctx_no_entries["available"] is True
    assert glossary_ctx_no_entries["entries"] == []
    assert glossary_ctx_no_entries["quick_add_href"].endswith(recording_id)

    assert (
        c.post(
            "/glossary",
            data={
                "canonical_text": "   ",
                "aliases_text": "",
                "kind": "term",
                "source": "manual",
                "enabled": "1",
            },
        ).status_code
        == 422
    )
    created = create_glossary_entry("Known", settings=cfg)
    assert (
        c.post(
            f"/glossary/{created['id']}",
            data={
                "canonical_text": "Known",
                "aliases_text": "",
                "kind": "person",
                "source": "unsupported",
                "enabled": "1",
            },
        ).status_code
        == 422
    )
    assert (
        c.post(
            "/glossary/999",
            data={
                "canonical_text": "Missing",
                "aliases_text": "",
                "kind": "term",
                "source": "manual",
            },
        ).status_code
        == 404
    )

    payload = ui_routes._glossary_form_payload(  # noqa: SLF001
        canonical_text="Known",
        aliases_text="Alias",
        kind="term",
        source="manual",
        enabled="1",
        notes="note",
        recording_id="   ",
        existing_metadata={"recording_id": "rec-1", "import_source": "csv"},
    )
    assert payload["metadata"] == {"import_source": "csv"}
    assert ui_routes._glossary_kind_label(None) == "General term"  # noqa: SLF001
    assert ui_routes._glossary_kind_label("term") == "General term"  # noqa: SLF001
    assert ui_routes._glossary_kind_label("custom_value") == "Custom Value"  # noqa: SLF001
    assert ui_routes._glossary_source_label(None) == "Always-on memory"  # noqa: SLF001
    assert ui_routes._glossary_source_label("custom_value") == "Custom Value"  # noqa: SLF001
    assert ui_routes._glossary_quick_entry_href() == "/glossary"  # noqa: SLF001
    assert ui_routes._glossary_quick_entry_href(  # noqa: SLF001
        recording_id=" rec-quick-1 ",
        canonical_text=" Sander ",
        aliases_text="Sandia",
        kind="person",
        source="manual",
        notes=" Seen on call ",
    ) == (
        "/glossary?recording_id=rec-quick-1&canonical_text=Sander&"
        "aliases_text=Sandia&kind=person&source=manual&notes=Seen+on+call"
    )
    defaults = ui_routes._glossary_form_defaults(  # noqa: SLF001
        canonical_text="  Sander  ",
        aliases_text=" Sandia ",
        kind="unsupported",
        source="unsupported",
        notes=" Seen on call ",
        recording_id=" rec-quick-2 ",
    )
    assert defaults == {
        "canonical_text": "Sander",
        "aliases_text": "Sandia",
        "kind": "term",
        "source": "correction",
        "enabled": True,
        "notes": "Seen on call",
        "recording_id": "rec-quick-2",
        "advanced_open": True,
        "prefill_notice": (
            "Prefilled from recording rec-quick-2. "
            "The recording link is already attached under Advanced."
        ),
    }
    blank_defaults = ui_routes._glossary_form_defaults()  # noqa: SLF001
    assert blank_defaults["advanced_open"] is False
    assert blank_defaults["prefill_notice"] is None


def test_fallback_turns_and_path_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    turns = ui_routes._fallback_speaker_turns_from_transcript(  # noqa: SLF001
        {
            "segments": [
                "skip",
                {"text": "   "},
                {"text": "hello", "start": "bad", "end": "bad"},
            ]
        }
    )
    assert turns == [
        {"start": 0.0, "end": 0.0, "speaker": "S1", "text": "hello", "language": None}
    ]

    fallback = ui_routes._fallback_speaker_turns_from_transcript(
        {"text": "hello world"}
    )  # noqa: SLF001
    assert fallback and fallback[0]["speaker"] == "S1"

    fallback_with_empty_segments = ui_routes._fallback_speaker_turns_from_transcript(  # noqa: SLF001
        {"segments": ["skip", {"text": "  "}], "text": "from transcript"}
    )
    assert (
        fallback_with_empty_segments
        and fallback_with_empty_segments[0]["text"] == "from transcript"
    )

    def _boom_resolve(_self: Path) -> Path:
        raise OSError("resolve-failed")

    monkeypatch.setattr(Path, "resolve", _boom_resolve)
    assert ui_routes._safe_path(tmp_path / "x", root=tmp_path) is None  # noqa: SLF001


def test_audio_snippet_helpers_and_speakers_context_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-speakers-helpers"
    derived = cfg.recordings_root / recording_id / "derived"
    snippets_dir = derived / "snippets" / "S1"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    (snippets_dir / "clip.txt").write_text("x", encoding="utf-8")
    (snippets_dir / "clip.wav").write_bytes(b"wav")
    (snippets_dir / "subdir").mkdir()

    assert ui_routes._safe_path(tmp_path / "x", root=tmp_path / "other") is None  # noqa: SLF001
    assert ui_routes._safe_audio_path(tmp_path / "clip.mp3", root=tmp_path) is None  # noqa: SLF001
    assert ui_routes._speaker_snippet_files(recording_id, "S1", settings=cfg) == [  # noqa: SLF001
        snippets_dir / "clip.wav"
    ]
    assert ui_routes._as_data_relative_path(Path("/etc/passwd"), settings=cfg) is None  # noqa: SLF001

    (derived / "transcript.json").write_text(
        json.dumps({"text": "fallback only"}), encoding="utf-8"
    )
    (derived / "speaker_turns.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        ui_routes,
        "list_speaker_assignments",
        lambda *_a, **_k: [{"diar_speaker_label": "S1", "voice_profile_id": "x"}],
    )
    monkeypatch.setattr(ui_routes, "list_voice_profiles", lambda *_a, **_k: [])
    fallback_ctx = ui_routes._speakers_tab_context(recording_id, cfg)  # noqa: SLF001
    assert fallback_ctx["speaker_rows"]

    (derived / "speaker_turns.json").write_text(
        json.dumps([{"speaker": "S1", "start": "bad", "end": "bad", "text": "sample"}]),
        encoding="utf-8",
    )
    parsed_ctx = ui_routes._speakers_tab_context(recording_id, cfg)  # noqa: SLF001
    assert parsed_ctx["speaker_rows"][0]["duration_sec"] == 0.0
    assert parsed_ctx["speaker_rows"][0]["voice_profile_id"] is None


def test_snippet_manifest_helpers_cover_context_and_validation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-snippet-manifest-helpers"
    derived = cfg.recordings_root / recording_id / "derived"
    snippets_root = derived / "snippets"
    (snippets_root / "S1").mkdir(parents=True, exist_ok=True)
    (snippets_root / "S1" / "1.wav").write_bytes(b"wav")
    (snippets_root / "S1" / "nested").mkdir(parents=True, exist_ok=True)
    (snippets_root / "S1" / "nested" / "1.wav").write_bytes(b"wav")
    (derived / "snippets_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {
                    "S1": [
                        "skip",
                        {
                            "snippet_id": "skip-speaker",
                            "speaker": "S9",
                            "ranking_position": 0,
                        },
                        {
                            "snippet_id": "S1-02",
                            "speaker": "S1",
                            "clip_start": 4.0,
                            "clip_end": 5.0,
                            "source_kind": "turn",
                            "source_start": 4.0,
                            "source_end": 4.7,
                            "purity_score": 0.61,
                            "ranking_position": 2,
                            "status": "rejected_overlap",
                        },
                        {
                            "snippet_id": "S1-01",
                            "speaker": "S1",
                            "clip_start": 0.0,
                            "clip_end": 1.0,
                            "source_kind": "turn",
                            "source_start": 0.0,
                            "source_end": 0.8,
                            "purity_score": 0.88,
                            "ranking_position": 1,
                            "status": "accepted",
                            "recommended": True,
                            "relative_path": "S1/1.wav",
                        },
                        {
                            "snippet_id": "S1-03",
                            "speaker": "S1",
                            "clip_start": 6.0,
                            "clip_end": 6.5,
                            "source_kind": "turn",
                            "source_start": 6.0,
                            "source_end": 6.2,
                            "purity_score": 0.52,
                            "ranking_position": 3,
                            "status": "rejected_failed_extract",
                        },
                        {
                            "snippet_id": "S1-04",
                            "speaker": "S1",
                            "clip_start": 7.0,
                            "clip_end": 8.0,
                            "source_kind": "turn",
                            "source_start": 7.0,
                            "source_end": 7.8,
                            "purity_score": 0.74,
                            "ranking_position": 4,
                            "status": "accepted",
                            "recommended": False,
                            "relative_path": "S1/missing.wav",
                        },
                        {
                            "snippet_id": "S1-05",
                            "speaker": "S1",
                            "clip_start": 8.0,
                            "clip_end": 9.0,
                            "source_kind": "turn",
                            "source_start": 8.0,
                            "source_end": 8.8,
                            "purity_score": 0.73,
                            "ranking_position": 5,
                            "status": "accepted",
                            "recommended": False,
                            "relative_path": "S1/nested/1.wav",
                        },
                        {
                            "snippet_id": "S1-06",
                            "speaker": "S1",
                            "clip_start": 9.0,
                            "clip_end": 10.0,
                            "source_kind": "turn",
                            "source_start": 9.0,
                            "source_end": 9.8,
                            "purity_score": 0.72,
                            "ranking_position": 6,
                            "status": "accepted",
                            "recommended": False,
                            "relative_path": "../evil.wav",
                        },
                    ],
                    "S2": [
                        {
                            "snippet_id": "S2-01",
                            "speaker": "S2",
                            "clip_start": 1.5,
                            "clip_end": 2.0,
                            "source_kind": "turn",
                            "source_start": 1.5,
                            "source_end": 1.9,
                            "purity_score": 0.77,
                            "ranking_position": 1,
                            "status": "accepted",
                            "recommended": True,
                            "relative_path": "S1/1.wav",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    rows = ui_routes._speaker_snippet_manifest_entries(  # noqa: SLF001
        recording_id,
        "S1",
        settings=cfg,
    )
    assert ui_routes._speaker_snippet_files(recording_id, "missing", settings=cfg) == []  # noqa: SLF001
    assert [row["snippet_id"] for row in rows] == [
        "S1-01",
        "S1-02",
        "S1-03",
        "S1-04",
        "S1-05",
        "S1-06",
    ]
    assert ui_routes._snippet_audio_url(recording_id, "bad") is None  # noqa: SLF001
    assert ui_routes._snippet_audio_url(recording_id, "S1/1.wav") == (  # noqa: SLF001
        f"/ui/recordings/{recording_id}/snippets/S1/1.wav"
    )
    assert ui_routes._snippet_choice_label(rows[0]).startswith(
        "Recommended: 0.00s-1.00s"
    )  # noqa: SLF001
    assert ui_routes._snippet_warning_messages(rows) == [  # noqa: SLF001
        "1 snippet candidate was rejected because it overlaps another speaker.",
        "1 snippet candidate could not be extracted from the sanitized WAV.",
    ]
    assert ui_routes._no_clean_snippet_message([]) == (  # noqa: SLF001
        "No snippet quality data is available for this speaker yet."
    )

    context = ui_routes._speaker_snippet_context(  # noqa: SLF001
        recording_id,
        "S1",
        settings=cfg,
    )
    assert context["clean_snippets"][0]["relative_path"] == "S1/1.wav"
    assert context["clean_snippets"][0]["recommended"] is True
    assert context["no_clean_snippet_message"] is None
    blocked_context = ui_routes._speaker_snippet_context(  # noqa: SLF001
        recording_id,
        "S2",
        settings=cfg,
    )
    assert blocked_context["clean_snippets"] == []

    selected = ui_routes._selected_clean_snippet(  # noqa: SLF001
        recording_id,
        "S1",
        "S1/1.wav",
        settings=cfg,
    )
    assert selected == (snippets_root / "S1" / "1.wav").resolve()

    with pytest.raises(
        ValueError, match="Selected snippet is not a clean snippet for this speaker"
    ):
        ui_routes._selected_clean_snippet(  # noqa: SLF001
            recording_id,
            "S1",
            "S1/2.wav",
            settings=cfg,
        )

    with pytest.raises(
        ValueError, match="Selected snippet does not belong to this speaker"
    ):
        ui_routes._selected_clean_snippet(  # noqa: SLF001
            recording_id,
            "S2",
            "S1/1.wav",
            settings=cfg,
        )

    with pytest.raises(ValueError, match="Selected snippet file does not exist"):
        ui_routes._selected_clean_snippet(  # noqa: SLF001
            recording_id,
            "S1",
            "S1/missing.wav",
            settings=cfg,
        )

    with pytest.raises(ValueError, match="Selected snippet path is invalid"):
        ui_routes._selected_clean_snippet(  # noqa: SLF001
            recording_id,
            "S1",
            "../evil.wav",
            settings=cfg,
        )


def test_snippet_message_helpers_and_display_backfill_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-ui-display-backfill"
    raw_dir = cfg.recordings_root / recording_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "audio.mp3").write_bytes(b"raw")
    (raw_dir / "audio.wav").write_bytes(b"raw")

    calls: list[Path] = []

    def _fake_wave_duration(path: Path) -> float | None:
        calls.append(path)
        return None if path.suffix == ".mp3" else 12.0

    monkeypatch.setattr(ui_routes, "_audio_duration_from_wave", _fake_wave_duration)
    monkeypatch.setattr(ui_routes, "_audio_duration_from_ffprobe", lambda _path: None)
    monkeypatch.setattr(ui_routes, "set_recording_duration", lambda *_a, **_k: None)

    item = ui_routes._prepare_recording_for_display(  # noqa: SLF001
        {"id": recording_id, "duration_sec": None},
        settings=cfg,
    )
    assert item["duration_display"] == "00:00:12"
    assert calls == [raw_dir / "audio.mp3", raw_dir / "audio.wav"]
    assert ui_routes._snippet_warning_messages(
        [  # noqa: SLF001
            {"status": "rejected_degraded"},
            {"status": "rejected_short"},
            {"status": "rejected_short"},
        ]
    ) == [
        "Diarization ran in degraded mode, so snippet samples from this speaker were blocked.",
        "2 snippet candidates were too short to trust as a voice sample.",
    ]
    assert ui_routes._no_clean_snippet_message([{"status": "rejected_degraded"}]) == (  # noqa: SLF001
        "No clean snippets are available because diarization ran in degraded mode."
    )
    assert ui_routes._no_clean_snippet_message(
        [{"status": "rejected_failed_extract"}]
    ) == (  # noqa: SLF001
        "No clean snippets are available because extraction failed for the clean candidates."
    )
    assert ui_routes._no_clean_snippet_message([{"status": "rejected_short"}]) == (  # noqa: SLF001
        "No clean snippets are available because every candidate was too short."
    )
    assert ui_routes._no_clean_snippet_message([{"status": "rejected_rank_limit"}]) == (  # noqa: SLF001
        "No accepted clean snippets are available for this speaker."
    )


def test_snippet_ui_state_helper_covers_manifest_and_stage_edges() -> None:
    assert ui_routes._pipeline_stage_order("not-a-stage") is None  # noqa: SLF001
    assert ui_routes._pipeline_stage_order("llm") == ui_routes._pipeline_stage_order(
        "llm_extract"
    )  # noqa: SLF001
    assert ui_routes._stage_row_metadata(None) == {}  # noqa: SLF001
    assert ui_routes._snippet_manifest_warning_messages(  # noqa: SLF001
        {"warnings": ["skip", {"message": ""}, {"message": "boom"}]}
    ) == ["boom"]
    assert (
        ui_routes._snippet_ready_message(  # noqa: SLF001
            {"status": RECORDING_STATUS_PROCESSING, "pipeline_stage": "snippet_export"}
        )
        == "Clean clips are ready while processing continues."
    )
    assert ui_routes._snippet_completed_without_clean_message(None) == (  # noqa: SLF001
        "Snippet export completed, but no accepted clean snippets are available for this speaker."
    )
    assert (
        ui_routes._snippet_completed_without_clean_message(  # noqa: SLF001
            "No snippet quality data is available for this speaker yet."
        )
        == "Snippet export completed, but no accepted clean snippets are available for this speaker."
    )
    assert ui_routes._snippet_completed_without_clean_message("state unavailable") == (  # noqa: SLF001
        "Snippet export completed, but state unavailable"
    )

    running_state = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={
            "status": RECORDING_STATUS_PROCESSING,
            "pipeline_stage": "snippet_export",
        },
        stage_rows=[],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert running_state["code"] == "running"

    failed_no_warning = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY},
        stage_rows=[
            {
                "stage_name": "snippet_export",
                "status": "failed",
                "metadata_json": {},
                "error_text": "",
            }
        ],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert failed_no_warning["detail"] == (
        "Snippet export failed, so no clean clips are available for this speaker."
    )

    accepted_missing = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY, "pipeline_stage": "llm_extract"},
        stage_rows=[
            {
                "stage_name": "snippet_export",
                "status": "completed",
                "metadata_json": {"manifest_status": "ok"},
            }
        ],
        manifest_exists=True,
        manifest={"manifest_status": "ok"},
        entries=[{"status": "accepted"}],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert accepted_missing["code"] == "unavailable"
    assert "audio files are missing from disk" in accepted_missing["detail"]

    unreadable = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY},
        stage_rows=[],
        manifest_exists=True,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert unreadable["code"] == "unavailable"
    assert "could not be read" in unreadable["detail"]

    unavailable = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY},
        stage_rows=[
            {
                "stage_name": "snippet_export",
                "status": "completed",
                "metadata_json": {"manifest_status": "no_usable_speech"},
            }
        ],
        manifest_exists=True,
        manifest={
            "manifest_status": "no_usable_speech",
            "warnings": [{"message": "No speaker turns were available."}],
        },
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert unavailable["code"] == "unavailable"
    assert unavailable["detail"] == "No speaker turns were available."

    cancelled = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY},
        stage_rows=[
            {
                "stage_name": "snippet_export",
                "status": "cancelled",
                "metadata_json": {},
            }
        ],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert cancelled["code"] == "unavailable"
    assert "cancelled" in cancelled["detail"]

    legacy = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY},
        stage_rows=[],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert legacy["code"] == "legacy_missing_manifest"

    stopped_before_snippets = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={
            "status": RECORDING_STATUS_STOPPED,
            "pipeline_stage": "speaker_turns",
        },
        stage_rows=[],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert stopped_before_snippets["code"] == "unavailable"
    assert (
        stopped_before_snippets["detail"]
        == "This recording is no longer processing and did not reach Snippet Export, so no clean clips are available for this speaker."
    )

    failed_before_snippets = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={
            "status": RECORDING_STATUS_FAILED,
            "pipeline_stage": "speaker_turns",
        },
        stage_rows=[],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert failed_before_snippets["code"] == "unavailable"

    stale_running_terminal = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={
            "status": RECORDING_STATUS_FAILED,
            "pipeline_stage": "snippet_export",
        },
        stage_rows=[
            {
                "stage_name": "snippet_export",
                "status": "running",
                "metadata_json": {},
            }
        ],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert stale_running_terminal["code"] == "unavailable"

    needs_review_without_stage = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={
            "status": RECORDING_STATUS_NEEDS_REVIEW,
            "pipeline_stage": "speaker_turns",
        },
        stage_rows=[],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert needs_review_without_stage["code"] == "unavailable"

    llm_alias = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_PROCESSING, "pipeline_stage": "llm"},
        stage_rows=[],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert llm_alias["code"] == "unavailable"

    missing_after_stage = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={
            "status": RECORDING_STATUS_PROCESSING,
            "pipeline_stage": "llm_extract",
        },
        stage_rows=[
            {
                "stage_name": "snippet_export",
                "status": "completed",
                "metadata_json": {"manifest_status": "ok"},
            }
        ],
        manifest_exists=False,
        manifest={},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert missing_after_stage["code"] == "unavailable"

    no_clean_ready = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY},
        stage_rows=[
            {
                "stage_name": "snippet_export",
                "status": "completed",
                "metadata_json": {"manifest_status": "no_clean_snippets"},
            }
        ],
        manifest_exists=True,
        manifest={"manifest_status": "no_clean_snippets"},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message="No clean snippets are available because every candidate overlaps another speaker.",
    )
    assert no_clean_ready["code"] == "ready_no_clean_snippets"
    assert "every candidate overlaps another speaker" in no_clean_ready["detail"]

    unknown_manifest = ui_routes._resolve_speaker_snippet_ui_state(  # noqa: SLF001
        recording={"status": RECORDING_STATUS_READY},
        stage_rows=[],
        manifest_exists=True,
        manifest={"manifest_status": "mystery"},
        entries=[],
        clean_snippets=[],
        no_clean_snippet_message=None,
    )
    assert unknown_manifest["detail"] == (
        "Snippet export finished, but its manifest state is unavailable."
    )


def test_speaker_helper_paths_cover_duplicates_labels_and_notices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-speaker-helper-extra"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "diarization_status.json").write_text(
        json.dumps({"degraded": True, "mode": "fallback"}),
        encoding="utf-8",
    )
    (derived / "diarization_metadata.json").write_text(
        json.dumps({"reason": "no pyannote"}),
        encoding="utf-8",
    )

    candidate_rows = ui_routes._candidate_match_rows(  # noqa: SLF001
        [
            "skip",
            {"voice_profile_id": "bad", "score": 0.4},
            {"voice_profile_id": "1", "score": "bad"},
            {"voice_profile_id": 2, "score": 0.6, "display_name": "Bea"},
        ],
        voice_profiles_by_id={1: {"display_name": "Alex"}},
    )
    assert candidate_rows == [
        {"voice_profile_id": 2, "display_name": "Bea", "score": 0.6},
        {"voice_profile_id": 1, "display_name": "Alex", "score": 0.0},
    ]

    monkeypatch.setattr(
        ui_routes,
        "list_speaker_assignments",
        lambda *_a, **_k: [
            {
                "diar_speaker_label": "S1",
                "voice_profile_name": "Alex",
                "review_state": "confirmed_canonical",
            },
            {
                "diar_speaker_label": "S2",
                "local_display_name": "Meeting Guest",
                "review_state": "local_label",
            },
            {
                "diar_speaker_label": "S3",
                "voice_profile_name": "Suggested",
                "review_state": "system_suggested",
            },
            {"diar_speaker_label": " ", "voice_profile_name": "skip"},
        ],
    )
    assert ui_routes._recording_speaker_name_map(recording_id, settings=cfg) == {  # noqa: SLF001
        "S1": "Alex",
        "S2": "Meeting Guest",
    }
    assert (
        ui_routes._speaker_display_label("S1", speaker_name_map={"S1": "Alex"})
        == "Alex (S1)"
    )  # noqa: SLF001
    assert (
        ui_routes._speaker_display_label("S2", speaker_name_map={"S1": "Alex"}) == "S2"
    )  # noqa: SLF001
    assert (
        ui_routes._speaker_review_state({"local_display_name": "Meeting Guest"})
        == "local_label"
    )  # noqa: SLF001
    assert (
        ui_routes._speaker_review_state(  # noqa: SLF001
            {"voice_profile_id": 1, "voice_profile_name": "Alex"}
        )
        == "confirmed_canonical"
    )
    assert (
        ui_routes._speaker_review_state(  # noqa: SLF001
            {"candidate_matches_json": [{"voice_profile_id": 1, "score": 0.4}]}
        )
        == "system_suggested"
    )
    assert (
        ui_routes._speaker_review_state({"review_state": "bad"}) == "system_suggested"
    )  # noqa: SLF001
    assert (
        ui_routes._speaker_assignment_display_name(  # noqa: SLF001
            {"review_state": "system_suggested", "voice_profile_name": "Suggested"}
        )
        == ""
    )
    status_ctx = ui_routes._speaker_assignment_status_context(  # noqa: SLF001
        "S1",
        {"review_state": "local_label", "local_display_name": "Meeting Guest"},
    )
    assert status_ctx["badge_label"] == "Local label only"
    suggested_ctx = ui_routes._speaker_assignment_status_context(  # noqa: SLF001
        "S3",
        {"voice_profile_name": "Suggested"},
    )
    assert suggested_ctx["mapping_title"] == "Suggested global match: Suggested"
    assert ui_routes._trusted_sample_state([]) is None  # noqa: SLF001
    assert ui_routes._trusted_sample_state(  # noqa: SLF001
        [{"voice_profile_id": 1, "voice_profile_name": "Alex"}]
    ) == {
        "badge_label": "1 saved",
        "detail": "Trusted sample saved for Alex.",
    }
    assert ui_routes._trusted_sample_state(  # noqa: SLF001
        [{"voice_profile_id": 3, "voice_profile_name": ""}]
    ) == {
        "badge_label": "1 saved",
        "detail": "Trusted sample saved for Canonical #3.",
    }
    assert ui_routes._trusted_sample_state(  # noqa: SLF001
        [
            {"voice_profile_id": 1, "voice_profile_name": "Alex"},
            {"voice_profile_id": 2, "voice_profile_name": "Bea"},
        ]
    ) == {
        "badge_label": "2 saved",
        "detail": "Trusted samples saved across 2 canonical speakers.",
    }
    assert ui_routes._trusted_sample_state(  # noqa: SLF001
        [{"voice_profile_id": None, "voice_profile_name": ""}]
    ) == {
        "badge_label": "1 saved",
        "detail": "Trusted sample saved from this recording for future matching.",
    }

    duplicates = ui_routes._voice_duplicate_candidates(  # noqa: SLF001
        voice_samples=[
            {
                "voice_profile_id": 1,
                "candidate_matches_json": [
                    {"voice_profile_id": 1, "score": 0.95},
                    {"voice_profile_id": 2, "score": 0.81},
                    {"voice_profile_id": 2, "score": 0.8},
                ],
            },
            {
                "voice_profile_id": None,
                "candidate_matches_json": [{"voice_profile_id": 2, "score": 0.5}],
            },
        ],
        voice_profiles_by_id={1: {"display_name": "Alex"}, 2: {"display_name": "Bea"}},
    )
    assert duplicates == {
        1: [
            {
                "voice_profile_id": 2,
                "display_name": "Bea",
                "best_score": 0.81,
                "match_count": 1,
            }
        ]
    }

    assert ui_routes._speaker_review_notices(  # noqa: SLF001
        recording_id,
        low_confidence_count=2,
        settings=cfg,
    ) == [
        "Diarization ran in degraded fallback mode (fallback): no pyannote.",
        "2 speaker matches are low confidence and need manual review.",
    ]

    (derived / "diarization_status.json").write_text(
        json.dumps({"degraded": True, "mode": "pyannote"}),
        encoding="utf-8",
    )
    (derived / "diarization_metadata.json").write_text("{}", encoding="utf-8")
    assert ui_routes._speaker_review_notices(  # noqa: SLF001
        recording_id,
        low_confidence_count=0,
        settings=cfg,
    ) == [
        "Diarization ran in degraded fallback mode; speaker results may need manual review."
    ]


def test_project_language_and_resummarize_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-proj-lang-helpers"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    assert ui_routes._as_int("bad") is None  # noqa: SLF001
    assert ui_routes._as_int(None) is None  # noqa: SLF001

    monkeypatch.setattr(
        ui_routes,
        "refresh_recording_routing",
        lambda *_a, **_k: {
            "suggested_project_id": None,
            "suggested_project_name": "",
            "confidence": "bad",
            "rationale": "bad",
            "threshold": 0.9,
        },
    )
    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda *_a, **_k: {
            "project_id": "5",
            "suggested_project_id": "9",
            "suggested_project_name": "Fallback Name",
            "routing_confidence": "bad",
            "routing_rationale_json": [" reason "],
        },
    )
    monkeypatch.setattr(
        ui_routes, "list_projects", lambda *_a, **_k: [{"id": 5, "name": "Roadmap"}]
    )
    monkeypatch.setattr(
        ui_routes,
        "count_routing_training_examples",
        lambda *_a, **_k: 3,
    )
    project_ctx = ui_routes._project_tab_context(recording_id, {"project_id": 5}, cfg)  # noqa: SLF001
    assert project_ctx["selected_project_name"] == "Roadmap"
    assert project_ctx["suggested_project_name"] == "Fallback Name"
    assert project_ctx["rationale"] == ["reason"]
    assert project_ctx["confidence"] == 0.0

    monkeypatch.setattr(
        ui_routes,
        "refresh_recording_routing",
        lambda *_a, **_k: {
            "suggested_project_id": 7,
            "suggested_project_name": "Decision Name",
            "confidence": 0.8,
            "rationale": ["decision"],
            "threshold": 0.9,
        },
    )
    monkeypatch.setattr(
        ui_routes,
        "list_projects",
        lambda *_a, **_k: [{"id": 1, "name": "A"}, {"id": 5, "name": "Roadmap"}],
    )
    decision_ctx = ui_routes._project_tab_context(recording_id, {"project_id": 5}, cfg)  # noqa: SLF001
    assert decision_ctx["suggested_project_id"] == 7
    assert decision_ctx["suggested_project_name"] == "Decision Name"

    options = ui_routes._language_options(  # noqa: SLF001
        distribution_codes=["unknown", "en"],
        target_summary_language="fr",
        transcript_language_override="de",
    )
    assert any(row["code"] == "fr" for row in options)
    assert any(row["code"] == "de" for row in options)
    duplicate_options = ui_routes._language_options(  # noqa: SLF001
        distribution_codes=["en", "en"],
        target_summary_language=None,
        transcript_language_override=None,
    )
    assert duplicate_options[0]["code"] == "en"

    transcript_payload = {
        "language_distribution": {"en": "bad", "fr": 22.5},
        "language_spans": ["skip", {"start": "bad", "end": "bad", "lang": "en"}],
    }
    (derived / "transcript.json").write_text(
        json.dumps(transcript_payload), encoding="utf-8"
    )
    (derived / "summary.json").write_text("{}", encoding="utf-8")
    lang_ctx = ui_routes._language_tab_context(recording_id, {}, cfg)  # noqa: SLF001
    assert lang_ctx["distribution"][0]["code"] == "fr"
    assert lang_ctx["spans"] == []

    with pytest.raises(ValueError, match="No transcript.json"):
        ui_routes._resummarize_recording(
            recording_id="missing", settings=cfg, target_summary_language="en"
        )  # noqa: SLF001

    (cfg.recordings_root / "empty" / "derived").mkdir(parents=True, exist_ok=True)
    (cfg.recordings_root / "empty" / "derived" / "transcript.json").write_text(
        json.dumps({"text": ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Transcript text is empty"):
        ui_routes._resummarize_recording(
            recording_id="empty", settings=cfg, target_summary_language="en"
        )  # noqa: SLF001


def test_transcript_tab_context_covers_fallback_turns_and_copy_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)

    speaker_recording = cfg.recordings_root / "rec-transcript-fallback" / "derived"
    speaker_recording.mkdir(parents=True, exist_ok=True)
    (speaker_recording / "transcript.json").write_text(
        json.dumps({"text": "raw transcript", "language": {"detected": "es"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ui_routes,
        "_fallback_speaker_turns_from_transcript",
        lambda *_a, **_k: [
            {"speaker": "S9", "text": "   "},
            {"speaker": "", "start": "bad", "end": "", "text": "Hola", "language": "??"},
            {
                "speaker": "S2",
                "start": 1.2,
                "end": "bad",
                "text": "Hello",
                "language": "en",
            },
        ],
    )

    transcript = ui_routes._transcript_tab_context(  # noqa: SLF001
        "rec-transcript-fallback",
        cfg,
    )
    assert transcript["available"] is True
    assert transcript["source_label"] == "Speaker-attributed turns"
    assert transcript["dominant_language_label"] == "Spanish (es)"
    assert transcript["turn_count"] == 2
    assert transcript["turns"][0] == {
        "speaker": "S1",
        "timestamp": "",
        "time_range": "— - —",
        "text": "Hola",
        "language_label": "",
    }
    assert transcript["turns"][1]["time_range"] == "00:00:01 - 00:00:01"
    assert transcript["turns"][1]["timestamp"] == "00:01"
    assert transcript["turns"][1]["language_label"] == "English (en)"
    assert "[— - —] S1: Hola" in transcript["copy_text"]
    assert "[00:00:01 - 00:00:01] S2: Hello" in transcript["copy_text"]

    raw_only_recording = cfg.recordings_root / "rec-transcript-raw" / "derived"
    raw_only_recording.mkdir(parents=True, exist_ok=True)
    (raw_only_recording / "transcript.json").write_text(
        json.dumps({"text": "raw only", "dominant_language": "en"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ui_routes,
        "_fallback_speaker_turns_from_transcript",
        lambda *_a, **_k: [],
    )

    raw_only = ui_routes._transcript_tab_context(  # noqa: SLF001
        "rec-transcript-raw",
        cfg,
    )
    assert raw_only["available"] is True
    assert raw_only["source_label"] == "Transcript text"
    assert raw_only["turn_count"] == 0
    assert raw_only["turns"] == []
    assert raw_only["copy_text"] == "raw only"
    assert raw_only["dominant_language_label"] == "English (en)"

    file_backed_recording = cfg.recordings_root / "rec-transcript-file" / "derived"
    file_backed_recording.mkdir(parents=True, exist_ok=True)
    (file_backed_recording / "transcript.json").write_text(
        json.dumps({"text": "ignored raw", "dominant_language": "en"}),
        encoding="utf-8",
    )
    (file_backed_recording / "speaker_turns.json").write_text(
        json.dumps(
            [
                {
                    "speaker": "S3",
                    "start": 2.0,
                    "end": 3.0,
                    "text": "From file",
                    "language": "en",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ui_routes,
        "_fallback_speaker_turns_from_transcript",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("fallback not expected")),
    )

    file_backed = ui_routes._transcript_tab_context(  # noqa: SLF001
        "rec-transcript-file",
        cfg,
    )
    assert file_backed["turn_count"] == 1
    assert file_backed["turns"][0]["speaker"] == "S3"
    assert file_backed["copy_text"] == "[00:00:02 - 00:00:03] S3: From file"


def test_resummarize_recording_default_turn_and_attendees_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-resummarize-edge"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hello world",
                "calendar_attendees": [" Alex ", " ", "Priya"],
            }
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text(
        json.dumps({"friendly": "bad"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        ui_routes, "_fallback_speaker_turns_from_transcript", lambda *_a, **_k: []
    )
    monkeypatch.setattr(
        ui_routes,
        "PipelineSettings",
        lambda **kwargs: type("S", (), {"llm_model": "test-model", **kwargs})(),
    )
    prompts_seen: dict[str, Any] = {}

    def _prompts(
        turns: list[dict[str, Any]], *_a: Any, **kwargs: Any
    ) -> tuple[str, str]:
        prompts_seen["turns"] = turns
        prompts_seen["attendees"] = kwargs.get("calendar_attendees")
        return "sys", "usr"

    class _FakeLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"content": '{"summary":"ok"}'}

    writes: dict[Path, dict[str, Any]] = {}

    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        writes[path] = dict(payload)

    monkeypatch.setattr(ui_routes, "build_structured_summary_prompts", _prompts)
    monkeypatch.setattr(ui_routes, "LLMClient", lambda: _FakeLLM())
    monkeypatch.setattr(
        ui_routes,
        "build_summary_payload",
        lambda **_k: {"summary": "ok", "friendly": 0},
    )
    monkeypatch.setattr(ui_routes, "atomic_write_json", _write_json)
    monkeypatch.setattr(ui_routes, "refresh_recording_metrics", lambda *_a, **_k: None)

    ui_routes._resummarize_recording(
        recording_id, settings=cfg, target_summary_language=None
    )  # noqa: SLF001
    assert prompts_seen["turns"][0]["speaker"] == "S1"
    assert prompts_seen["attendees"] == ["Alex", "Priya"]
    assert any(path.name == "summary.json" for path in writes)


def test_resummarize_recording_prefers_selected_calendar_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-resummarize-selected-cal",
        source="upload",
        source_filename="selected.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    upsert_calendar_match(
        recording_id="rec-resummarize-selected-cal",
        candidates=[
            {
                "event_id": "evt-selected",
                "subject": "Selected Calendar Event",
                "attendee_details": [{"label": "Priya Kapoor"}],
            }
        ],
        selected_event_id="evt-selected",
        selected_confidence=0.9,
        settings=cfg,
    )
    derived = cfg.recordings_root / "rec-resummarize-selected-cal" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps(
            {
                "text": "hello world",
                "calendar_title": "Stale title",
                "calendar_attendees": ["Stale attendee"],
            }
        ),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text(json.dumps({"friendly": 0}), encoding="utf-8")

    monkeypatch.setattr(
        ui_routes, "_fallback_speaker_turns_from_transcript", lambda *_a, **_k: []
    )
    monkeypatch.setattr(
        ui_routes,
        "PipelineSettings",
        lambda **kwargs: type("S", (), {"llm_model": "test-model", **kwargs})(),
    )
    prompts_seen: dict[str, Any] = {}

    def _prompts(
        turns: list[dict[str, Any]], *_a: Any, **kwargs: Any
    ) -> tuple[str, str]:
        prompts_seen["turns"] = turns
        prompts_seen["title"] = kwargs.get("calendar_title")
        prompts_seen["attendees"] = kwargs.get("calendar_attendees")
        return "sys", "usr"

    class _FakeLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"content": '{"summary":"ok"}'}

    monkeypatch.setattr(ui_routes, "build_structured_summary_prompts", _prompts)
    monkeypatch.setattr(ui_routes, "LLMClient", lambda: _FakeLLM())
    monkeypatch.setattr(
        ui_routes,
        "build_summary_payload",
        lambda **_k: {"summary": "ok", "friendly": 0},
    )
    monkeypatch.setattr(ui_routes, "atomic_write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(ui_routes, "refresh_recording_metrics", lambda *_a, **_k: None)

    ui_routes._resummarize_recording(  # noqa: SLF001
        "rec-resummarize-selected-cal",
        settings=cfg,
        target_summary_language=None,
    )
    assert prompts_seen["turns"][0]["speaker"] == "S1"
    assert prompts_seen["title"] == "Selected Calendar Event"
    assert prompts_seen["attendees"] == ["Priya Kapoor"]


def test_resummarize_recording_uses_existing_speaker_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    recording_id = "rec-resummarize-existing-turns"
    derived = cfg.recordings_root / recording_id / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "transcript.json").write_text(
        json.dumps({"text": "hello"}), encoding="utf-8"
    )
    (derived / "speaker_turns.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "speaker": "S1", "text": "hello"}]),
        encoding="utf-8",
    )
    (derived / "summary.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        ui_routes,
        "PipelineSettings",
        lambda **kwargs: type("S", (), {"llm_model": "test-model", **kwargs})(),
    )
    monkeypatch.setattr(
        ui_routes, "build_structured_summary_prompts", lambda *_a, **_k: ("sys", "usr")
    )

    class _FakeLLM:
        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"content": '{"summary":"ok"}'}

    monkeypatch.setattr(ui_routes, "LLMClient", lambda: _FakeLLM())
    monkeypatch.setattr(
        ui_routes,
        "build_summary_payload",
        lambda **_k: {"summary": "ok", "friendly": 1},
    )
    monkeypatch.setattr(ui_routes, "atomic_write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(ui_routes, "refresh_recording_metrics", lambda *_a, **_k: None)

    ui_routes._resummarize_recording(
        recording_id, settings=cfg, target_summary_language="en"
    )  # noqa: SLF001


def test_datetime_and_calendar_parse_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="from is required"):
        ui_routes._parse_iso_datetime("", field_name="from")  # noqa: SLF001
    with pytest.raises(ValueError, match="from must be ISO-8601 datetime"):
        ui_routes._parse_iso_datetime("bad", field_name="from")  # noqa: SLF001
    parsed = ui_routes._parse_iso_datetime("2026-02-01T10:00:00", field_name="from")  # noqa: SLF001
    assert parsed.tzinfo is timezone.utc
    parsed_tz = ui_routes._parse_iso_datetime("2026-02-01T10:00:00Z", field_name="from")  # noqa: SLF001
    assert parsed_tz.tzinfo is timezone.utc

    with pytest.raises(ValueError, match="to is required"):
        ui_routes._parse_ymd_date("", field_name="to")  # noqa: SLF001
    with pytest.raises(ValueError, match="to must be YYYY-MM-DD"):
        ui_routes._parse_ymd_date("2026/02/01", field_name="to")  # noqa: SLF001

    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ui_routes, "list_calendar_sources", lambda *_a, **_k: [])
    monkeypatch.setattr(ui_routes, "list_calendar_events", lambda *_a, **_k: [])
    with pytest.raises(ValueError, match="to must be after from"):
        ui_routes._calendar_page_data(  # noqa: SLF001
            date_from="2026-02-10",
            date_to="2026-02-01",
            source_id=None,
            settings=cfg,
        )


def test_calendar_ui_helper_context_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    assert ui_routes._calendar_attendee_labels("bad") == []  # noqa: SLF001
    assert ui_routes._calendar_attendee_labels(  # noqa: SLF001
        [{"label": ""}, {"label": "Alex"}, {"name": "Alex"}, "Priya", "Priya"]
    ) == ["Alex", "Priya"]
    assert ui_routes._calendar_rationale_rows(123) == []  # noqa: SLF001
    assert ui_routes._calendar_rationale_rows(" one ") == ["one"]  # noqa: SLF001
    candidate = ui_routes._calendar_candidate_context(  # noqa: SLF001
        {
            "event_id": "evt-1",
            "subject": "Calendar event",
            "starts_at": "2026-03-01T10:00:00Z",
            "ends_at": "2026-03-01T11:00:00Z",
            "attendee_details": [{"label": "Alex"}],
            "source_kind": "file",
            "rationale": "manual",
        },
        selected_event_id="evt-1",
    )
    assert candidate["selected"] is True
    assert candidate["attendees_label"] == "Alex"
    assert candidate["source_label"] == "file"

    monkeypatch.setattr(
        ui_routes,
        "get_calendar_match",
        lambda *_a, **_k: {"selected_event_id": "evt-1", "selected_confidence": 0.7},
    )
    monkeypatch.setattr(
        ui_routes,
        "get_recording",
        lambda *_a, **_k: {
            "source": "upload",
            "captured_at": "2026-03-01T09:05:00Z",
            "captured_at_timezone": "Europe/Rome",
            "captured_at_inferred_from_filename": 0,
        },
    )
    monkeypatch.setattr(
        ui_routes,
        "selected_calendar_candidate",
        lambda *_a, **_k: {"event_id": "evt-1", "subject": "Selected"},
    )
    monkeypatch.setattr(
        ui_routes,
        "calendar_match_candidates",
        lambda *_a, **_k: [{"event_id": "evt-1", "subject": "Selected"}],
    )
    context = ui_routes._calendar_tab_context("rec-helper", cfg)  # noqa: SLF001
    assert context["selected"]["subject_display"] == "Selected"
    assert context["candidates"][0]["selected"] is True
    assert len(context["warnings"]) == 1


def test_calendar_detail_error_and_invalid_confidence_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    _seed_recording(cfg, "rec-calendar-ui-edge")
    monkeypatch.setattr(
        ui_routes,
        "get_calendar_match",
        lambda *_a, **_k: {"selected_event_id": "", "selected_confidence": None},
    )
    monkeypatch.setattr(ui_routes, "selected_calendar_candidate", lambda *_a, **_k: {})
    monkeypatch.setattr(ui_routes, "calendar_match_candidates", lambda *_a, **_k: [])
    detail = c.get(
        "/recordings/rec-calendar-ui-edge?tab=calendar&calendar_error=Need+review"
    )
    assert detail.status_code == 200
    assert "Need review" in detail.text

    monkeypatch.setattr(
        ui_routes,
        "calendar_match_candidates",
        lambda *_a, **_k: [
            {"event_id": "evt-other", "score": 0.1},
            {"event_id": "evt-edge", "score": "bad"},
        ],
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        ui_routes,
        "set_calendar_match_selection",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    monkeypatch.setattr(ui_routes, "refresh_recording_routing", lambda *_a, **_k: {})
    selected = c.post(
        "/ui/recordings/rec-calendar-ui-edge/calendar/select",
        data={"event_id": "evt-edge"},
    )
    assert selected.status_code == 303
    assert captured["selected_confidence"] is None


def test_auth_route_edge_paths(
    client: tuple[AppSettings, TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    _cfg, c = client
    monkeypatch.setattr(ui_routes, "auth_enabled", lambda *_a, **_k: False)
    assert c.get("/ui").headers["location"] == "/"
    assert c.get("/ui/login").headers["location"] == "/"
    assert c.post("/ui/login", data={"token": "x"}).headers["location"] == "/"
    assert c.get("/ui/logout").headers["location"] == "/"

    monkeypatch.setattr(ui_routes, "auth_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(ui_routes, "request_is_authenticated", lambda *_a, **_k: True)
    assert c.get("/ui/login?next=/queue").headers["location"] == "/queue"

    monkeypatch.setattr(
        ui_routes, "expected_bearer_token", lambda *_a, **_k: "expected"
    )
    bad_login = c.post("/ui/login", data={"token": "bad", "next": "/ui"})
    assert bad_login.status_code == 401
    assert "Invalid token." in bad_login.text
    monkeypatch.setattr(ui_routes, "request_is_authenticated", lambda *_a, **_k: False)
    login_page = c.get("/ui/login?next=/ui")
    assert login_page.status_code == 200
    assert "Invalid token." not in login_page.text

    assert c.get("/ui/logout").headers["location"] == "/ui/login"


def test_recording_progress_export_and_snippet_not_found_paths(
    client: tuple[AppSettings, TestClient],
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-snippet-missing-1")
    assert c.get("/ui/recordings/missing/progress").status_code == 404
    assert c.get("/ui/recordings/missing/export.zip").status_code == 404
    assert c.get("/ui/recordings/missing/snippets/S1/a.wav").status_code == 404
    assert (
        c.get(f"/ui/recordings/{recording_id}/snippets/S1/missing.wav").status_code
        == 404
    )


def test_recording_export_zip_route_returns_zip_bytes(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-export-zip-ok")
    seen: dict[str, object] = {}

    def _fake_build_export_zip_bytes(
        recording_id_arg: str,
        *,
        settings: AppSettings,
        include_snippets: bool,
    ) -> bytes:
        seen["recording_id"] = recording_id_arg
        seen["settings"] = settings
        seen["include_snippets"] = include_snippets
        return b"zip-bytes"

    monkeypatch.setattr(
        ui_routes, "build_export_zip_bytes", _fake_build_export_zip_bytes
    )
    response = c.get(f"/ui/recordings/{recording_id}/export.zip?include_snippets=1")
    assert response.status_code == 200
    assert response.content == b"zip-bytes"
    assert response.headers["content-type"].startswith("application/zip")
    assert seen == {
        "recording_id": recording_id,
        "settings": cfg,
        "include_snippets": True,
    }


def test_assign_speaker_validation_and_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg)
    base = f"/ui/recordings/{recording_id}/speakers/assign"

    assert (
        c.post(
            "/ui/recordings/missing/speakers/assign",
            data={"diar_speaker_label": "S1", "voice_profile_id": ""},
        ).status_code
        == 404
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": " ", "voice_profile_id": ""}
        ).status_code
        == 422
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": "S1", "voice_profile_id": ""}
        ).status_code
        == 422
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": "S1", "voice_profile_id": "bad"}
        ).status_code
        == 422
    )

    monkeypatch.setattr(
        ui_routes,
        "set_speaker_assignment",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.IntegrityError("missing")),
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"}
        ).status_code
        == 404
    )

    monkeypatch.setattr(
        ui_routes,
        "set_speaker_assignment",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad assignment")),
    )
    failed = c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"})
    assert failed.status_code == 422
    assert "bad assignment" in failed.text

    monkeypatch.setattr(ui_routes, "set_speaker_assignment", lambda *_a, **_k: None)
    ok = c.post(base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"})
    assert ok.status_code == 303


def test_keep_unknown_and_local_label_validation_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-speaker-decisions-1")
    keep_unknown_base = f"/ui/recordings/{recording_id}/speakers/keep-unknown"
    local_label_base = f"/ui/recordings/{recording_id}/speakers/local-label"

    assert (
        c.post(
            "/ui/recordings/missing/speakers/keep-unknown",
            data={"diar_speaker_label": "S1"},
        ).status_code
        == 404
    )
    assert (
        c.post(keep_unknown_base, data={"diar_speaker_label": " "}).status_code == 422
    )

    monkeypatch.setattr(
        ui_routes,
        "set_speaker_assignment",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("unknown failed")),
    )
    failed_keep_unknown = c.post(keep_unknown_base, data={"diar_speaker_label": "S1"})
    assert failed_keep_unknown.status_code == 422
    assert "unknown failed" in failed_keep_unknown.text

    monkeypatch.setattr(ui_routes, "set_speaker_assignment", lambda *_a, **_k: None)
    assert (
        c.post(keep_unknown_base, data={"diar_speaker_label": "S1"}).status_code == 303
    )

    assert (
        c.post(
            "/ui/recordings/missing/speakers/local-label",
            data={"diar_speaker_label": "S1", "local_display_name": "Guest"},
        ).status_code
        == 404
    )
    assert (
        c.post(
            local_label_base,
            data={"diar_speaker_label": " ", "local_display_name": "Guest"},
        ).status_code
        == 422
    )
    assert (
        c.post(
            local_label_base,
            data={"diar_speaker_label": "S1", "local_display_name": " "},
        ).status_code
        == 422
    )

    monkeypatch.setattr(
        ui_routes,
        "set_speaker_assignment",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("local failed")),
    )
    failed_local_label = c.post(
        local_label_base,
        data={"diar_speaker_label": "S1", "local_display_name": "Guest"},
    )
    assert failed_local_label.status_code == 422
    assert "local failed" in failed_local_label.text

    monkeypatch.setattr(ui_routes, "set_speaker_assignment", lambda *_a, **_k: None)
    assert (
        c.post(
            local_label_base,
            data={"diar_speaker_label": "S1", "local_display_name": "Guest"},
        ).status_code
        == 303
    )


def test_create_and_assign_speaker_validation_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-create-assign-1")
    base = f"/ui/recordings/{recording_id}/speakers/create-and-assign"

    assert (
        c.post(
            "/ui/recordings/missing/speakers/create-and-assign",
            data={"diar_speaker_label": "S1", "display_name": "Alex"},
        ).status_code
        == 404
    )
    assert (
        c.post(base, data={"diar_speaker_label": " ", "display_name": "A"}).status_code
        == 422
    )
    assert (
        c.post(base, data={"diar_speaker_label": "S1", "display_name": " "}).status_code
        == 422
    )

    monkeypatch.setattr(
        ui_routes, "create_voice_profile", lambda *_a, **_k: {"id": "bad"}
    )
    resp = c.post(base, data={"diar_speaker_label": "S1", "display_name": "Alex"})
    assert resp.status_code == 503


def test_merge_voice_validation_and_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    source = create_voice_profile("Source", settings=cfg)
    target = create_voice_profile("Target", settings=cfg)
    base = f"/voices/{source['id']}/merge"

    assert c.post(base, data={"target_profile_id": ""}).status_code == 422
    assert c.post(base, data={"target_profile_id": "bad"}).status_code == 422

    monkeypatch.setattr(
        ui_routes,
        "merge_canonical_speakers",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ValueError("target_profile_id was not found")
        ),
    )
    assert (
        c.post(base, data={"target_profile_id": str(target["id"])}).status_code == 404
    )

    monkeypatch.setattr(
        ui_routes,
        "merge_canonical_speakers",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ValueError("source_profile_id and target_profile_id must differ")
        ),
    )
    assert (
        c.post(base, data={"target_profile_id": str(target["id"])}).status_code == 422
    )

    monkeypatch.setattr(ui_routes, "merge_canonical_speakers", lambda *_a, **_k: {})
    ok = c.post(base, data={"target_profile_id": str(target["id"])})
    assert ok.status_code == 303


def test_add_speaker_sample_validation_and_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-add-sample-1")
    base = f"/ui/recordings/{recording_id}/speakers/add-sample"

    assert (
        c.post(
            "/ui/recordings/missing/speakers/add-sample",
            data={"diar_speaker_label": "S1", "voice_profile_id": "1"},
        ).status_code
        == 404
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": " ", "voice_profile_id": "1"}
        ).status_code
        == 422
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": "S1", "voice_profile_id": ""}
        ).status_code
        == 422
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": "S1", "voice_profile_id": "bad"}
        ).status_code
        == 422
    )
    assert (
        c.post(
            base, data={"diar_speaker_label": "S1", "voice_profile_id": "1"}
        ).status_code
        == 422
    )

    rel_bad = tmp_path / "outside.wav"
    rel_bad.write_bytes(b"wav")
    monkeypatch.setattr(
        ui_routes,
        "_selected_clean_snippet",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ValueError("Selected snippet path is invalid")
        ),
    )
    invalid = c.post(
        base,
        data={
            "diar_speaker_label": "S1",
            "voice_profile_id": "1",
            "snippet_path": "../bad.wav",
        },
    )
    assert invalid.status_code == 422
    assert "Selected snippet path is invalid" in invalid.text

    monkeypatch.setattr(ui_routes, "_selected_clean_snippet", lambda *_a, **_k: rel_bad)
    monkeypatch.setattr(ui_routes, "_as_data_relative_path", lambda *_a, **_k: None)
    assert (
        c.post(
            base,
            data={
                "diar_speaker_label": "S1",
                "voice_profile_id": "1",
                "snippet_path": "S1/1.wav",
            },
        ).status_code
        == 422
    )

    monkeypatch.setattr(
        ui_routes, "_as_data_relative_path", lambda *_a, **_k: "recordings/x.wav"
    )
    monkeypatch.setattr(
        ui_routes,
        "create_voice_sample",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.IntegrityError("missing")),
    )
    assert (
        c.post(
            base,
            data={
                "diar_speaker_label": "S1",
                "voice_profile_id": "1",
                "snippet_path": "S1/1.wav",
            },
        ).status_code
        == 404
    )

    monkeypatch.setattr(
        ui_routes,
        "create_voice_sample",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad snippet")),
    )
    bad_value = c.post(
        base,
        data={
            "diar_speaker_label": "S1",
            "voice_profile_id": "1",
            "snippet_path": "S1/1.wav",
        },
    )
    assert bad_value.status_code == 422
    assert "bad snippet" in bad_value.text


def test_set_recording_project_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-project-errors-1")
    base = f"/ui/recordings/{recording_id}/project"

    assert c.post("/ui/recordings/missing/project", data={}).status_code == 404
    assert c.post(base, data={"project_id": "bad"}).status_code == 422

    monkeypatch.setattr(
        ui_routes,
        "set_recording_project",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.IntegrityError("missing")),
    )
    assert c.post(base, data={"project_id": "1"}).status_code == 404

    monkeypatch.setattr(ui_routes, "set_recording_project", lambda *_a, **_k: False)
    assert c.post(base, data={"project_id": "1"}).status_code == 404

    monkeypatch.setattr(ui_routes, "set_recording_project", lambda *_a, **_k: True)
    monkeypatch.setattr(
        ui_routes,
        "train_routing_from_manual_selection",
        lambda *_a, **_k: (_ for _ in ()).throw(KeyError("missing")),
    )
    failed_train = c.post(base, data={"project_id": "1", "train_routing": "yes"})
    assert failed_train.status_code == 404

    monkeypatch.setattr(ui_routes, "set_recording_project", lambda *_a, **_k: True)
    monkeypatch.setattr(
        ui_routes, "refresh_recording_routing", lambda *_a, **_k: {"ok": True}
    )
    no_project = c.post(base, data={"project_id": "", "train_routing": "yes"})
    assert no_project.status_code == 303


def test_voices_page_and_audio_route_edge_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg, c = client
    monkeypatch.setattr(
        ui_routes, "list_voice_profiles", lambda *_a, **_k: [{"id": "bad"}]
    )
    monkeypatch.setattr(
        ui_routes,
        "list_voice_samples",
        lambda *_a, **_k: [
            {"id": 1, "voice_profile_id": "bad"},
            {"id": 2, "voice_profile_id": "1"},
        ],
    )
    assert c.get("/voices").status_code == 200

    monkeypatch.setattr(ui_routes, "get_voice_sample", lambda *_a, **_k: None)
    assert c.get("/ui/voice-samples/1/audio").status_code == 404

    monkeypatch.setattr(
        ui_routes, "get_voice_sample", lambda *_a, **_k: {"snippet_path": "  "}
    )
    assert c.get("/ui/voice-samples/2/audio").status_code == 404

    missing_rel = {"snippet_path": "recordings/missing.wav"}
    monkeypatch.setattr(ui_routes, "get_voice_sample", lambda *_a, **_k: missing_rel)
    assert c.get("/ui/voice-samples/3/audio").status_code == 404

    good_rel = cfg.data_root / "recordings" / "sample.wav"
    good_rel.parent.mkdir(parents=True, exist_ok=True)
    good_rel.write_bytes(b"wav")
    monkeypatch.setattr(
        ui_routes,
        "get_voice_sample",
        lambda *_a, **_k: {"snippet_path": "recordings/sample.wav"},
    )
    ok = c.get("/ui/voice-samples/4/audio")
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("audio/wav")

    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"wav")
    monkeypatch.setattr(
        ui_routes,
        "get_voice_sample",
        lambda *_a, **_k: {"snippet_path": str(outside)},
    )
    assert c.get("/ui/voice-samples/5/audio").status_code == 404


def test_calendar_route_edge_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    bad_range = c.get("/calendars?from=2026-02-10&to=2026-02-01")
    assert bad_range.status_code == 200
    assert "to must be after from" in bad_range.text

    created = c.post(
        "/calendars/sources",
        data={"name": "ICS URL", "kind": "url", "url": "https://example.com/team.ics"},
    )
    assert created.status_code == 303
    assert created.headers["location"].startswith("/calendars?source_id=")

    bad_create = c.post(
        "/calendars/sources",
        data={"name": "Bad URL", "kind": "url", "url": "not-a-url"},
    )
    assert bad_create.status_code == 303
    assert "error=" in bad_create.headers["location"]

    assert c.post("/calendars/sources/999/sync").status_code == 404

    source = create_calendar_source(
        name="sync-source",
        kind="file",
        file_ics="BEGIN:VCALENDAR\nEND:VCALENDAR",
        settings=cfg,
    )
    source_id = int(source["id"])

    async def _raise_sync(*_args: Any, **_kwargs: Any) -> Any:
        raise CalendarSyncError("sync boom")

    monkeypatch.setattr(ui_routes, "run_in_threadpool", _raise_sync)
    failed_sync = c.post(f"/calendars/sources/{source_id}/sync")
    assert failed_sync.status_code == 303
    assert "error=" in failed_sync.headers["location"]

    _seed_recording(cfg, "rec-calendar-select-edge")
    missing_recording = c.post("/ui/recordings/missing/calendar/select", data={})
    assert missing_recording.status_code == 404

    bad_select = c.post(
        "/ui/recordings/rec-calendar-select-edge/calendar/select",
        data={"event_id": "evt-missing"},
    )
    assert bad_select.status_code == 422


def test_queue_action_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-queue-errors-1")

    monkeypatch.setattr(
        ui_routes,
        "enqueue_recording_job",
        lambda *_a, **_k: (_ for _ in ()).throw(
            DuplicateRecordingJobError(recording_id=recording_id, job_id="job-dup")
        ),
    )
    requeue = c.post(f"/ui/recordings/{recording_id}/requeue")
    assert requeue.status_code == 409
    assert "already active" in requeue.text

    assert c.post("/ui/recordings/missing/jobs/job-1/retry").status_code == 404
    assert (
        c.post(f"/ui/recordings/{recording_id}/jobs/missing/retry").status_code == 404
    )

    create_job(
        "job-failed-ui-cov",
        recording_id=recording_id,
        job_type="precheck",
        status=JOB_STATUS_FAILED,
        settings=cfg,
    )

    retry_dup = c.post(f"/ui/recordings/{recording_id}/jobs/job-failed-ui-cov/retry")
    assert retry_dup.status_code == 409

    monkeypatch.setattr(
        ui_routes,
        "enqueue_recording_job",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("queue-down")),
    )
    retry_failed = c.post(f"/ui/recordings/{recording_id}/jobs/job-failed-ui-cov/retry")
    assert retry_failed.status_code == 503


def test_language_action_error_paths(
    client: tuple[AppSettings, TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, c = client
    recording_id = _seed_recording(cfg, "rec-lang-errors-1")

    assert c.post("/ui/recordings/missing/language/settings").status_code == 404

    monkeypatch.setattr(
        ui_routes,
        "_save_language_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad language")),
    )
    bad_settings = c.post(
        f"/ui/recordings/{recording_id}/language/settings",
        data={"target_summary_language": "xx"},
    )
    assert bad_settings.status_code == 422

    monkeypatch.setattr(
        ui_routes, "_save_language_settings", lambda *_a, **_k: ("en", None)
    )
    ok_settings = c.post(f"/ui/recordings/{recording_id}/language/settings")
    assert ok_settings.status_code == 303

    assert c.post("/ui/recordings/missing/language/resummarize").status_code == 404
    bad_resummarize = c.post(f"/ui/recordings/{recording_id}/language/resummarize")
    assert bad_resummarize.status_code == 422

    monkeypatch.setattr(
        ui_routes,
        "_save_language_settings",
        lambda *_a, **_k: (None, None),
    )

    async def _raise_resummarize(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("summarize down")

    monkeypatch.setattr(ui_routes, "run_in_threadpool", _raise_resummarize)
    re_summary_error = c.post(f"/ui/recordings/{recording_id}/language/resummarize")
    assert re_summary_error.status_code == 503

    assert c.post("/ui/recordings/missing/language/retranscribe").status_code == 404
    monkeypatch.setattr(
        ui_routes,
        "_save_language_settings",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad override")),
    )
    retr_bad = c.post(f"/ui/recordings/{recording_id}/language/retranscribe")
    assert retr_bad.status_code == 422

    monkeypatch.setattr(
        ui_routes, "_save_language_settings", lambda *_a, **_k: (None, None)
    )
    monkeypatch.setattr(
        ui_routes,
        "enqueue_recording_job",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("enqueue down")),
    )
    retr_err = c.post(f"/ui/recordings/{recording_id}/language/retranscribe")
    assert retr_err.status_code == 503
