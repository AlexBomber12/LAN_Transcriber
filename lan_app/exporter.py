"""Export helpers for OneNote-ready markdown and ZIP bundles."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile
from typing import Any

from .config import AppSettings
from .db import get_recording

_OPTIONAL_DERIVED_FILES = (
    "summary.json",
    "transcript.json",
    "speaker_turns.json",
    "metrics.json",
    "segments.json",
    "lang_spans.json",
)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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


def _load_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return payload


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split()).strip()


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
        text = _normalize_text(row)
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        if text:
            out.append(text)
    return out


def _metadata_lines(recording: dict[str, Any], *, language: str | None) -> list[str]:
    recording_id = str(recording.get("id") or "").strip() or "unknown"
    filename = str(recording.get("source_filename") or "").strip() or "unknown"
    duration_raw = recording.get("duration_sec")
    try:
        duration = f"{int(duration_raw)} sec" if duration_raw is not None else "unknown"
    except (TypeError, ValueError):
        duration = str(duration_raw).strip() or "unknown"

    lines = [
        "## Metadata",
        f"- Recording ID: `{recording_id}`",
        f"- Filename: `{filename}`",
        f"- Duration: {duration}",
    ]
    if language:
        lines.append(f"- Language: {language}")
    return lines


def _summary_section(summary_payload: dict[str, Any]) -> list[str]:
    summary_bullets = _normalise_text_items(summary_payload.get("summary_bullets"), max_items=30)
    if not summary_bullets:
        summary_bullets = _normalise_text_items(summary_payload.get("summary"), max_items=30)
    if not summary_bullets:
        return ["## Summary", "Summary is not available yet."]
    return ["## Summary", *[f"- {bullet}" for bullet in summary_bullets]]


def _decisions_section(summary_payload: dict[str, Any]) -> list[str]:
    decisions = _normalise_text_items(summary_payload.get("decisions"), max_items=40)
    if not decisions:
        return []
    return ["## Decisions", *[f"- {decision}" for decision in decisions]]


def _action_items_section(summary_payload: dict[str, Any]) -> list[str]:
    raw_items = summary_payload.get("action_items")
    if not isinstance(raw_items, list):
        return []

    lines = ["## Action Items"]
    wrote_any = False
    for row in raw_items[:50]:
        if isinstance(row, dict):
            task = _normalize_text(row.get("task") or "")
            owner = _normalize_text(row.get("owner") or "")
            deadline = _normalize_text(row.get("deadline") or "")
            if not task:
                continue
            details: list[str] = []
            if owner:
                details.append(f"owner: {owner}")
            if deadline:
                details.append(f"deadline: {deadline}")
            suffix = f" ({'; '.join(details)})" if details else ""
            lines.append(f"- [ ] {task}{suffix}")
            wrote_any = True
            continue
        task_text = _normalize_text(row)
        if task_text:
            lines.append(f"- [ ] {task_text}")
            wrote_any = True
    return lines if wrote_any else []


def _questions_section(summary_payload: dict[str, Any]) -> list[str]:
    questions_payload = summary_payload.get("questions")
    extracted: list[str] = []
    if isinstance(questions_payload, dict):
        extracted = _normalise_text_items(questions_payload.get("extracted"), max_items=40)
    else:
        extracted = _normalise_text_items(questions_payload, max_items=40)
    if not extracted:
        return []
    return ["## Questions", *[f"- {question}" for question in extracted]]


def _emotion_section(summary_payload: dict[str, Any]) -> list[str]:
    emotional_summary = _normalize_text(summary_payload.get("emotional_summary") or "")
    if not emotional_summary:
        return []
    return ["## Emotional Summary", emotional_summary]


def _metrics_section(metrics_payload: dict[str, Any]) -> list[str]:
    meeting_payload = metrics_payload.get("meeting")
    if not isinstance(meeting_payload, dict):
        return []

    rows: list[tuple[str, Any]] = [
        ("Interruptions", meeting_payload.get("total_interruptions")),
        ("Questions", meeting_payload.get("total_questions")),
        ("Decisions", meeting_payload.get("decisions_count")),
        ("Action items", meeting_payload.get("action_items_count")),
    ]
    present = [(label, value) for label, value in rows if value is not None]
    if not present:
        return []

    return ["## Metrics", *[f"- {label}: {value}" for label, value in present]]


def _transcript_section(
    transcript_payload: dict[str, Any],
    speaker_turns_payload: list[dict[str, Any]],
) -> list[str]:
    if speaker_turns_payload:
        lines = ["## Transcript"]
        wrote_any = False
        for turn in speaker_turns_payload:
            speaker = _normalize_text(turn.get("speaker") or "S1") or "S1"
            text = _normalize_text(turn.get("text") or "")
            if not text:
                continue
            lines.append(f"- **{speaker}:** {text}")
            wrote_any = True
        if wrote_any:
            return lines

    transcript_text = str(transcript_payload.get("text") or "").strip()
    if transcript_text:
        return ["## Transcript", transcript_text]
    return ["## Transcript", "Transcript is not available yet."]


def build_onenote_markdown(recording: dict[str, Any], *, settings: AppSettings) -> str:
    """Build an export markdown note from recording metadata + derived artifacts."""

    recording_id = str(recording.get("id") or "").strip()
    derived = settings.recordings_root / recording_id / "derived"

    summary_payload = _load_json_dict(derived / "summary.json")
    transcript_payload = _load_json_dict(derived / "transcript.json")
    speaker_turns_raw = _load_json_list(derived / "speaker_turns.json")
    speaker_turns_payload = [row for row in speaker_turns_raw if isinstance(row, dict)]
    metrics_payload = _load_json_dict(derived / "metrics.json")

    topic = _normalize_text(summary_payload.get("topic") or "")
    fallback_title = _normalize_text(recording.get("source_filename") or "")
    title = topic or fallback_title or (recording_id or "Recording")
    captured_at = _normalize_text(recording.get("captured_at") or "")
    title_line = f"# {title} ({captured_at})" if captured_at else f"# {title}"

    language = _normalize_text(recording.get("language_auto") or "")
    if not language:
        language_payload = transcript_payload.get("language")
        if isinstance(language_payload, dict):
            language = _normalize_text(language_payload.get("detected") or "")
    if not language:
        language = _normalize_text(transcript_payload.get("dominant_language") or "")
    language = language or None

    sections: list[str] = [title_line, "\n".join(_metadata_lines(recording, language=language))]

    for block in (
        _summary_section(summary_payload),
        _decisions_section(summary_payload),
        _action_items_section(summary_payload),
        _questions_section(summary_payload),
        _emotion_section(summary_payload),
        _metrics_section(metrics_payload),
        _transcript_section(transcript_payload, speaker_turns_payload),
    ):
        if block:
            sections.append("\n".join(block))

    return "\n\n".join(sections).strip() + "\n"


def _try_read_bytes(path: Path) -> bytes | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def build_export_zip_bytes(
    recording_id: str,
    *,
    settings: AppSettings,
    include_snippets: bool = False,
) -> bytes:
    """Build a ZIP bundle with markdown export + available derived artifacts."""

    recording = get_recording(recording_id, settings=settings)
    if recording is None:
        raise KeyError(recording_id)

    recording_root = settings.recordings_root / recording_id
    derived_root = recording_root / "derived"
    markdown = build_onenote_markdown(recording, settings=settings)

    included_files: list[str] = ["onenote.md"]
    payload_buffer = io.BytesIO()
    with zipfile.ZipFile(payload_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("onenote.md", markdown)

        for filename in _OPTIONAL_DERIVED_FILES:
            source = derived_root / filename
            data = _try_read_bytes(source)
            if data is None:
                continue
            arcname = f"derived/{filename}"
            archive.writestr(arcname, data)
            included_files.append(arcname)

        if include_snippets:
            snippets_root = derived_root / "snippets"
            if snippets_root.exists() and snippets_root.is_dir():
                for path in sorted(snippets_root.rglob("*.wav")):
                    data = _try_read_bytes(path)
                    if data is None:
                        continue
                    try:
                        arcname = path.relative_to(recording_root).as_posix()
                    except ValueError:
                        continue
                    archive.writestr(arcname, data)
                    included_files.append(arcname)

        manifest = {
            "recording_id": recording_id,
            "created_at": _utc_now_iso(),
            "filenames_included": sorted([*included_files, "manifest.json"]),
        }
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        )

    return payload_buffer.getvalue()


__all__ = ["build_onenote_markdown", "build_export_zip_bytes"]
