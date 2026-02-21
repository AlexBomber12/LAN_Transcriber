"""OneNote project browsing and publish helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote, unquote

from .config import AppSettings
from .constants import RECORDING_STATUS_NEEDS_REVIEW, RECORDING_STATUS_READY
from .db import (
    get_calendar_match,
    get_meeting_metrics,
    get_project,
    get_recording,
    list_participant_metrics,
    set_recording_publish_result,
)
from .ms_graph import MicrosoftGraphClient

_ALLOWED_PUBLISH_STATUSES = {
    RECORDING_STATUS_READY,
    RECORDING_STATUS_NEEDS_REVIEW,
}
_PAGE_ID_RE = re.compile(r"/pages/([^/?#]+)", re.IGNORECASE)


class PublishPreconditionError(ValueError):
    """Raised when publish preconditions are not met."""


def list_onenote_notebooks(
    *,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    cfg = settings or AppSettings()
    client = MicrosoftGraphClient(cfg)
    response = client.graph_get("/me/onenote/notebooks?$top=200")
    items = response.get("value")
    if not isinstance(items, list):
        return []
    notebooks = [_normalise_onenote_item(item) for item in items]
    out = [item for item in notebooks if item is not None]
    out.sort(key=lambda row: (row["display_name"].lower(), row["id"]))
    return out


def list_onenote_sections(
    notebook_id: str,
    *,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    clean_notebook_id = str(notebook_id).strip()
    if not clean_notebook_id:
        raise ValueError("notebook_id is required")
    cfg = settings or AppSettings()
    client = MicrosoftGraphClient(cfg)
    encoded_notebook_id = quote(clean_notebook_id, safe="")
    response = client.graph_get(
        f"/me/onenote/notebooks/{encoded_notebook_id}/sections?$top=200"
    )
    items = response.get("value")
    if not isinstance(items, list):
        return []
    sections = [_normalise_onenote_item(item) for item in items]
    out = [item for item in sections if item is not None]
    out.sort(key=lambda row: (row["display_name"].lower(), row["id"]))
    return out


def publish_recording_to_onenote(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    recording = get_recording(recording_id, settings=cfg)
    if recording is None:
        raise KeyError(recording_id)

    status = str(recording.get("status") or "").strip()
    if status not in _ALLOWED_PUBLISH_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_PUBLISH_STATUSES))
        raise PublishPreconditionError(
            f"Recording status must be one of: {allowed} (current: {status or 'unknown'})"
        )

    project_id_raw = recording.get("project_id")
    if project_id_raw is None:
        raise PublishPreconditionError("Recording has no project assigned.")
    try:
        project_id = int(project_id_raw)
    except (TypeError, ValueError) as exc:
        raise PublishPreconditionError("Recording project_id is invalid.") from exc
    project = get_project(project_id, settings=cfg)
    if project is None:
        raise PublishPreconditionError("Recording project mapping does not exist.")

    section_id = str(project.get("onenote_section_id") or "").strip()
    if not section_id:
        raise PublishPreconditionError("Project OneNote section_id is not configured.")
    notebook_id = str(project.get("onenote_notebook_id") or "").strip() or None

    summary = _load_summary_context(recording_id, cfg)
    metrics = _load_metrics_context(recording_id, cfg)
    calendar = _load_calendar_context(recording_id, cfg)
    links = _build_link_context(recording, cfg)

    participants = _participants_for_title(metrics, calendar)
    title = _build_publish_title(recording, summary, participants)
    html_payload = _build_onenote_html(
        title=title,
        summary=summary,
        metrics=metrics,
        calendar=calendar,
        links=links,
    )

    client = MicrosoftGraphClient(cfg)
    encoded_section_id = quote(section_id, safe="")
    response = client.graph_post_html(
        f"/me/onenote/sections/{encoded_section_id}/pages",
        html_payload,
    )
    page_id = _extract_page_id(response)
    if not page_id:
        raise RuntimeError("Graph publish succeeded but page ID was not returned.")
    page_url = _extract_page_url(response)

    saved = set_recording_publish_result(
        recording_id,
        onenote_page_id=page_id,
        onenote_page_url=page_url,
        settings=cfg,
    )
    if not saved:
        raise KeyError(recording_id)

    return {
        "recording_id": recording_id,
        "project_id": project_id,
        "onenote_notebook_id": notebook_id,
        "onenote_section_id": section_id,
        "onenote_page_id": page_id,
        "onenote_page_url": page_url,
        "title": title,
    }


def _normalise_onenote_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    item_id = str(item.get("id") or "").strip()
    if not item_id:
        return None
    name = str(item.get("displayName") or "").strip() or item_id
    return {
        "id": item_id,
        "display_name": name,
        "web_url": _extract_page_url(item),
    }


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _normalise_text_items(value: Any, *, max_items: int) -> list[str]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, str):
        rows = value.splitlines()
    else:
        return []

    out: list[str] = []
    for row in rows:
        if len(out) >= max_items:
            break
        text = str(row).strip()
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        if text:
            out.append(text)
    return out


def _load_summary_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    summary_path = settings.recordings_root / recording_id / "derived" / "summary.json"
    payload = _load_json_dict(summary_path)
    summary_bullets = _normalise_text_items(payload.get("summary_bullets"), max_items=20)
    if not summary_bullets:
        summary_bullets = _normalise_text_items(payload.get("summary"), max_items=20)
    decisions = _normalise_text_items(payload.get("decisions"), max_items=30)

    action_items: list[dict[str, str | None]] = []
    raw_action_items = payload.get("action_items")
    if isinstance(raw_action_items, list):
        for row in raw_action_items[:40]:
            if not isinstance(row, dict):
                continue
            task = str(row.get("task") or "").strip()
            if not task:
                continue
            owner = str(row.get("owner") or "").strip() or None
            deadline = str(row.get("deadline") or "").strip() or None
            action_items.append(
                {
                    "task": task,
                    "owner": owner,
                    "deadline": deadline,
                }
            )

    topic = str(payload.get("topic") or "").strip()
    return {
        "topic": topic or "Untitled",
        "summary_bullets": summary_bullets,
        "decisions": decisions,
        "action_items": action_items,
    }


def _load_metrics_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    meeting_payload: dict[str, Any] = {}
    participants_payload: list[dict[str, Any]] = []

    meeting_row = get_meeting_metrics(recording_id, settings=settings) or {}
    meeting_json = meeting_row.get("json")
    if isinstance(meeting_json, dict):
        meeting_payload = dict(meeting_json)

    for row in list_participant_metrics(recording_id, settings=settings):
        payload = row.get("json")
        participant = payload if isinstance(payload, dict) else {}
        if not participant:
            continue
        speaker = str(
            participant.get("speaker") or row.get("diar_speaker_label") or ""
        ).strip()
        if not speaker:
            continue
        participants_payload.append(
            {
                "speaker": speaker,
                "airtime_seconds": _to_float(participant.get("airtime_seconds")),
                "airtime_share": _to_float(participant.get("airtime_share")),
                "turns": _to_int(participant.get("turns")),
                "interruptions_done": _to_int(participant.get("interruptions_done")),
                "interruptions_received": _to_int(
                    participant.get("interruptions_received")
                ),
                "questions_count": _to_int(participant.get("questions_count")),
                "role_hint": str(participant.get("role_hint") or "").strip() or None,
            }
        )

    if not meeting_payload or not participants_payload:
        metrics_path = settings.recordings_root / recording_id / "derived" / "metrics.json"
        fallback = _load_json_dict(metrics_path)
        meeting_raw = fallback.get("meeting")
        if isinstance(meeting_raw, dict):
            for key, value in meeting_raw.items():
                meeting_payload.setdefault(key, value)
        participants_raw = fallback.get("participants")
        if isinstance(participants_raw, list):
            for row in participants_raw:
                if not isinstance(row, dict):
                    continue
                speaker = str(row.get("speaker") or "").strip()
                if not speaker:
                    continue
                participants_payload.append(
                    {
                        "speaker": speaker,
                        "airtime_seconds": _to_float(row.get("airtime_seconds")),
                        "airtime_share": _to_float(row.get("airtime_share")),
                        "turns": _to_int(row.get("turns")),
                        "interruptions_done": _to_int(row.get("interruptions_done")),
                        "interruptions_received": _to_int(
                            row.get("interruptions_received")
                        ),
                        "questions_count": _to_int(row.get("questions_count")),
                        "role_hint": str(row.get("role_hint") or "").strip() or None,
                    }
                )

    participants_payload.sort(
        key=lambda row: (-float(row.get("airtime_seconds") or 0.0), row["speaker"])
    )

    meeting = {
        "total_interruptions": _to_int(meeting_payload.get("total_interruptions")),
        "total_questions": _to_int(meeting_payload.get("total_questions")),
        "decisions_count": _to_int(meeting_payload.get("decisions_count")),
        "action_items_count": _to_int(meeting_payload.get("action_items_count")),
        "actionability_ratio": _to_float(meeting_payload.get("actionability_ratio")),
        "total_speech_time_seconds": _to_float(
            meeting_payload.get("total_speech_time_seconds")
        ),
        "emotional_summary": str(meeting_payload.get("emotional_summary") or "").strip()
        or None,
    }
    return {"meeting": meeting, "participants": participants_payload}


def _load_calendar_context(recording_id: str, settings: AppSettings) -> dict[str, Any]:
    row = get_calendar_match(recording_id, settings=settings) or {}
    selected_event_id = str(row.get("selected_event_id") or "").strip()
    selected_confidence = row.get("selected_confidence")
    if not selected_event_id:
        return {
            "selected_event_id": None,
            "selected_confidence": selected_confidence,
            "selected_event": None,
        }
    candidates = row.get("candidates_json")
    if not isinstance(candidates, list):
        candidates = []
    selected_event: dict[str, Any] | None = None
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if str(item.get("event_id") or "").strip() != selected_event_id:
            continue
        selected_event = {
            "subject": str(item.get("subject") or "").strip() or "(no subject)",
            "start": str(item.get("start") or "").strip() or None,
            "end": str(item.get("end") or "").strip() or None,
            "organizer": str(item.get("organizer") or "").strip() or None,
            "attendees": [
                str(attendee).strip()
                for attendee in (item.get("attendees") or [])
                if str(attendee).strip()
            ],
            "location": str(item.get("location") or "").strip() or None,
        }
        break
    return {
        "selected_event_id": selected_event_id,
        "selected_confidence": selected_confidence,
        "selected_event": selected_event,
    }


def _build_link_context(recording: dict[str, Any], settings: AppSettings) -> dict[str, str | None]:
    recording_id = str(recording.get("id") or "").strip()
    recording_dir = settings.recordings_root / recording_id
    links: dict[str, str | None] = {
        "drive_artifact_folder_url": None,
        "drive_source_file_url": None,
        "local_artifacts_path": None,
        "local_artifacts_url": None,
        "raw_audio_path": None,
        "raw_audio_url": None,
    }

    drive_folder_url = str(recording.get("drive_artifact_folder_url") or "").strip()
    if drive_folder_url:
        links["drive_artifact_folder_url"] = drive_folder_url
    drive_folder_id = str(recording.get("drive_artifact_folder_id") or "").strip()
    if drive_folder_id and links["drive_artifact_folder_url"] is None:
        links["drive_artifact_folder_url"] = (
            f"https://drive.google.com/drive/folders/{drive_folder_id}"
        )

    drive_file_id = str(recording.get("drive_file_id") or "").strip()
    if drive_file_id:
        links["drive_source_file_url"] = f"https://drive.google.com/file/d/{drive_file_id}/view"

    if recording_dir.exists():
        resolved_dir = recording_dir.resolve()
        links["local_artifacts_path"] = str(resolved_dir)
        links["local_artifacts_url"] = resolved_dir.as_uri()

    raw_dir = recording_dir / "raw"
    for candidate in sorted(raw_dir.glob("audio.*")):
        if not candidate.is_file():
            continue
        resolved_file = candidate.resolve()
        links["raw_audio_path"] = str(resolved_file)
        links["raw_audio_url"] = resolved_file.as_uri()
        break
    return links


def _participants_for_title(
    metrics: dict[str, Any],
    calendar: dict[str, Any],
) -> list[str]:
    names: list[str] = []
    for row in metrics.get("participants") or []:
        if not isinstance(row, dict):
            continue
        speaker = str(row.get("speaker") or "").strip()
        if speaker and speaker not in names:
            names.append(speaker)
    if not names:
        selected_event = calendar.get("selected_event")
        if isinstance(selected_event, dict):
            for attendee in selected_event.get("attendees") or []:
                text = str(attendee).strip()
                if text and text not in names:
                    names.append(text)
    if not names:
        return ["Participants unavailable"]
    return names[:4]


def _build_publish_title(
    recording: dict[str, Any],
    summary: dict[str, Any],
    participants: list[str],
) -> str:
    captured = _parse_iso_datetime(str(recording.get("captured_at") or ""))
    if captured is None:
        captured = datetime.now(tz=timezone.utc)
    timestamp = captured.strftime("%Y-%m-%d %H:%M")
    topic = str(summary.get("topic") or "").strip() or "Untitled"
    people = ", ".join(participants) if participants else "Participants unavailable"
    duration = _format_duration(recording.get("duration_sec"))
    return f"{timestamp} | {topic} | {people} | {duration}"


def _build_onenote_html(
    *,
    title: str,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    calendar: dict[str, Any],
    links: dict[str, str | None],
) -> str:
    rows: list[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{escape(title)}</title>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        "<h2>Summary</h2>",
    ]

    summary_bullets = summary.get("summary_bullets") or []
    if summary_bullets:
        rows.append("<ul>")
        for bullet in summary_bullets:
            rows.append(f"<li>{escape(str(bullet))}</li>")
        rows.append("</ul>")
    else:
        rows.append("<p>No summary available.</p>")

    rows.append("<h2>Decisions</h2>")
    decisions = summary.get("decisions") or []
    if decisions:
        rows.append("<ul>")
        for decision in decisions:
            rows.append(f"<li>{escape(str(decision))}</li>")
        rows.append("</ul>")
    else:
        rows.append("<p>No decisions captured.</p>")

    rows.append("<h2>Action items</h2>")
    action_items = summary.get("action_items") or []
    if action_items:
        rows.extend(
            [
                '<table border="1" cellpadding="6" cellspacing="0">',
                "<tr><th>Task</th><th>Owner</th><th>Deadline</th></tr>",
            ]
        )
        for row in action_items:
            if not isinstance(row, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{escape(str(row.get('task') or ''))}</td>"
                f"<td>{escape(str(row.get('owner') or '—'))}</td>"
                f"<td>{escape(str(row.get('deadline') or '—'))}</td>"
                "</tr>"
            )
        rows.append("</table>")
    else:
        rows.append("<p>No action items captured.</p>")

    rows.append("<h2>Metrics</h2>")
    meeting = metrics.get("meeting") if isinstance(metrics.get("meeting"), dict) else {}
    rows.extend(
        [
            "<ul>",
            f"<li>Total interruptions: {escape(str(meeting.get('total_interruptions', 0)))}</li>",
            f"<li>Total questions: {escape(str(meeting.get('total_questions', 0)))}</li>",
            f"<li>Decisions count: {escape(str(meeting.get('decisions_count', 0)))}</li>",
            f"<li>Action items count: {escape(str(meeting.get('action_items_count', 0)))}</li>",
            (
                "<li>Actionability ratio: "
                f"{escape(_format_decimal(meeting.get('actionability_ratio')))}</li>"
            ),
            (
                "<li>Total speech time (sec): "
                f"{escape(_format_decimal(meeting.get('total_speech_time_seconds')))}</li>"
            ),
            "</ul>",
        ]
    )

    participants = metrics.get("participants") or []
    if participants:
        rows.extend(
            [
                '<table border="1" cellpadding="6" cellspacing="0">',
                (
                    "<tr><th>Participant</th><th>Airtime (s)</th><th>Airtime share</th>"
                    "<th>Turns</th><th>Questions</th></tr>"
                ),
            ]
        )
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{escape(str(participant.get('speaker') or '—'))}</td>"
                f"<td>{escape(_format_decimal(participant.get('airtime_seconds')))}</td>"
                f"<td>{escape(_format_decimal(participant.get('airtime_share')))}</td>"
                f"<td>{escape(str(participant.get('turns') or 0))}</td>"
                f"<td>{escape(str(participant.get('questions_count') or 0))}</td>"
                "</tr>"
            )
        rows.append("</table>")
    else:
        rows.append("<p>No participant metrics available.</p>")

    rows.append("<h2>Calendar context</h2>")
    selected_event = (
        calendar.get("selected_event") if isinstance(calendar.get("selected_event"), dict) else None
    )
    if selected_event:
        rows.extend(
            [
                "<ul>",
                f"<li>Subject: {escape(str(selected_event.get('subject') or '—'))}</li>",
                f"<li>Start: {escape(str(selected_event.get('start') or '—'))}</li>",
                f"<li>End: {escape(str(selected_event.get('end') or '—'))}</li>",
                f"<li>Organizer: {escape(str(selected_event.get('organizer') or '—'))}</li>",
                (
                    "<li>Attendees: "
                    f"{escape(', '.join(selected_event.get('attendees') or []) or '—')}</li>"
                ),
                f"<li>Location: {escape(str(selected_event.get('location') or '—'))}</li>",
                (
                    "<li>Match score: "
                    f"{escape(_format_decimal(calendar.get('selected_confidence')))}</li>"
                ),
                "</ul>",
            ]
        )
    else:
        rows.append("<p>No calendar event selected.</p>")

    rows.append("<h2>Links</h2>")
    rows.append("<ul>")
    if links.get("drive_artifact_folder_url"):
        url = str(links["drive_artifact_folder_url"])
        rows.append(
            f'<li>Drive artifact folder: <a href="{escape(url)}">{escape(url)}</a></li>'
        )
    if links.get("drive_source_file_url"):
        url = str(links["drive_source_file_url"])
        rows.append(f'<li>Drive source file: <a href="{escape(url)}">{escape(url)}</a></li>')
    if links.get("local_artifacts_path"):
        path = str(links["local_artifacts_path"])
        rows.append(f"<li>Local artifact folder: {escape(path)}</li>")
    if links.get("raw_audio_path"):
        path = str(links["raw_audio_path"])
        if links.get("raw_audio_url"):
            url = str(links["raw_audio_url"])
            rows.append(f'<li>Raw audio path: <a href="{escape(url)}">{escape(path)}</a></li>')
        else:
            rows.append(f"<li>Raw audio path: {escape(path)}</li>")
    rows.append("</ul>")

    rows.extend(["</body>", "</html>"])
    return "\n".join(rows)


def _parse_iso_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_duration(value: Any) -> str:
    try:
        duration_sec = int(float(value))
    except (TypeError, ValueError):
        duration_sec = 0
    if duration_sec <= 0:
        return "n/a"
    minutes, seconds = divmod(duration_sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _extract_page_id(payload: dict[str, Any]) -> str | None:
    candidate = str(payload.get("id") or "").strip()
    if candidate:
        return candidate
    for key in ("location", "content_location", "self"):
        url = str(payload.get(key) or "").strip()
        if not url:
            continue
        match = _PAGE_ID_RE.search(url)
        if match:
            return unquote(match.group(1))
    return None


def _extract_page_url(payload: dict[str, Any]) -> str | None:
    links = payload.get("links")
    if isinstance(links, dict):
        web = links.get("oneNoteWebUrl")
        if isinstance(web, dict):
            href = str(web.get("href") or "").strip()
            if href:
                return href
    one_note_web_url = payload.get("oneNoteWebUrl")
    if isinstance(one_note_web_url, dict):
        href = str(one_note_web_url.get("href") or "").strip()
        if href:
            return href
    for key in ("contentUrl", "webUrl", "location", "content_location"):
        candidate = str(payload.get(key) or "").strip()
        if candidate:
            return candidate
    return None


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_decimal(value: Any) -> str:
    return f"{_to_float(value):.3f}"


__all__ = [
    "PublishPreconditionError",
    "list_onenote_notebooks",
    "list_onenote_sections",
    "publish_recording_to_onenote",
]
