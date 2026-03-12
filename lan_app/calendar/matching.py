from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any
import unicodedata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lan_app.config import AppSettings
from lan_app.db import (
    get_calendar_match,
    get_recording,
    list_calendar_events,
    upsert_calendar_match,
)

_AUTO_SELECT_MIN_MARGIN = 0.10
_AUTO_SELECT_MIN_SCORE = 0.74
_CONSERVATIVE_AUTO_SELECT_MIN_MARGIN = 0.15
_CONSERVATIVE_AUTO_SELECT_MIN_SCORE = 0.82
_PRESTART_GRACE_MINUTES = 15.0
_PROXIMITY_WINDOW_MINUTES = 180.0
_SEARCH_WINDOW_AFTER = timedelta(hours=12)
_SEARCH_WINDOW_BEFORE = timedelta(hours=8)
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_STOPWORDS = {
    "and",
    "audio",
    "call",
    "meeting",
    "recording",
    "sync",
    "the",
    "weekly",
    "with",
}


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_duration_seconds(value: object) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return duration


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_token_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _tokenize(value: object) -> set[str]:
    text = _normalize_token_text(value)
    if not text:
        return set()
    return {token for token in _TOKEN_RE.findall(text) if token not in _STOPWORDS}


def _recording_tokens(recording: dict[str, Any]) -> set[str]:
    filename = Path(str(recording.get("source_filename") or "").strip()).stem
    return _tokenize(filename)


def _candidate_rows(raw: object) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def calendar_match_candidates(
    recording_id: str,
    *,
    settings: AppSettings,
) -> list[dict[str, Any]]:
    row = get_calendar_match(recording_id, settings=settings) or {}
    return _candidate_rows(row.get("candidates_json"))


def _candidate_attendee_details(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in value:
        if isinstance(row, dict):
            name = str(row.get("name") or "").strip()
            email = str(row.get("email") or "").strip()
            label = str(row.get("label") or name or email).strip()
        else:
            name = ""
            email = ""
            label = str(row or "").strip()
        if not label:
            continue
        dedupe_key = (email.lower(), label.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        payload = {"label": label}
        if name:
            payload["name"] = name
        if email:
            payload["email"] = email
        out.append(payload)
    return out


def _candidate_attendees(candidate: dict[str, Any]) -> list[str]:
    attendees = candidate.get("attendees")
    if isinstance(attendees, list):
        cleaned = [str(value).strip() for value in attendees if str(value).strip()]
        if cleaned:
            return cleaned
    return [row["label"] for row in _candidate_attendee_details(candidate.get("attendee_details"))]


def selected_calendar_candidate(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    row = get_calendar_match(recording_id, settings=settings) or {}
    selected_event_id = str(row.get("selected_event_id") or "").strip()
    if not selected_event_id:
        return {}
    for candidate in _candidate_rows(row.get("candidates_json")):
        if str(candidate.get("event_id") or "").strip() == selected_event_id:
            normalized = dict(candidate)
            normalized["attendees"] = _candidate_attendees(normalized)
            return normalized
    return {}


def calendar_summary_context(
    recording_id: str,
    *,
    settings: AppSettings,
) -> tuple[str | None, list[str]]:
    selected = selected_calendar_candidate(recording_id, settings=settings)
    title = str(selected.get("subject") or selected.get("summary") or "").strip() or None
    return title, _candidate_attendees(selected)


def _event_identifier(event: dict[str, Any]) -> str:
    source_id = str(event.get("source_id") or "").strip()
    uid = str(event.get("uid") or "").strip()
    starts_at = str(event.get("starts_at") or "").strip()
    return f"{source_id}:{uid}:{starts_at}"


def _event_attendee_details(event: dict[str, Any]) -> list[dict[str, str]]:
    return _candidate_attendee_details(event.get("attendees_json"))


def _event_attendees(event: dict[str, Any]) -> list[str]:
    return [row["label"] for row in _event_attendee_details(event)]


def _capture_time_assessment(recording: dict[str, Any]) -> dict[str, Any]:
    captured_at = _parse_datetime(recording.get("captured_at"))
    recording_source = str(recording.get("source") or "").strip().lower()
    source_text = str(recording.get("captured_at_source") or "").strip()
    timezone_name = str(recording.get("captured_at_timezone") or "").strip()
    inferred_from_filename = _coerce_bool(recording.get("captured_at_inferred_from_filename"))

    base = {
        "status": "stored_utc",
        "penalty": 0.0,
        "warning": None,
        "rationale": None,
        "auto_select_mode": "normal",
    }
    if captured_at is None:
        return {
            **base,
            "status": "missing",
            "auto_select_mode": "blocked",
            "warning": "This recording has no usable capture timestamp, so calendar matching needs manual review.",
            "rationale": "Capture time is missing, so automatic calendar selection is disabled.",
        }

    if inferred_from_filename:
        if not source_text or not timezone_name:
            return {
                **base,
                "status": "suspicious",
                "penalty": 0.18,
                "warning": (
                    "The capture time provenance looks incomplete, so automatic calendar matching is intentionally conservative."
                ),
                "rationale": "Filename-derived capture time is missing provenance details and should be reviewed.",
                "auto_select_mode": "blocked",
            }
        try:
            source_local = datetime.fromisoformat(source_text)
            source_zone = ZoneInfo(timezone_name)
        except (ValueError, ZoneInfoNotFoundError):
            return {
                **base,
                "status": "suspicious",
                "penalty": 0.18,
                "warning": (
                    "The capture time provenance looks inconsistent, so automatic calendar matching is intentionally conservative."
                ),
                "rationale": "Filename-derived capture time could not be validated against its stored timezone.",
                "auto_select_mode": "blocked",
            }
        if source_local.tzinfo is None:
            source_local = source_local.replace(tzinfo=source_zone)
        expected_utc = source_local.astimezone(timezone.utc)
        if abs((expected_utc - captured_at).total_seconds()) > 90:
            return {
                **base,
                "status": "suspicious",
                "penalty": 0.18,
                "warning": (
                    "The capture time provenance does not line up with the stored UTC timestamp, so automatic calendar matching is intentionally conservative."
                ),
                "rationale": "Stored UTC capture time does not match the filename-derived local timestamp.",
                "auto_select_mode": "blocked",
            }
        return {
            **base,
            "status": "corrected_local",
            "rationale": (
                f"Capture time uses the corrected filename timestamp in {timezone_name}."
            ),
        }

    if recording_source == "upload":
        if source_text:
            return {
                **base,
                "status": "suspicious",
                "penalty": 0.18,
                "warning": (
                    "The upload capture timestamp metadata is inconsistent, so automatic calendar matching is intentionally conservative."
                ),
                "rationale": "Upload capture provenance is inconsistent and should be reviewed.",
                "auto_select_mode": "blocked",
            }
        if not timezone_name:
            return {
                **base,
                "status": "suspicious",
                "penalty": 0.18,
                "warning": (
                    "The upload capture timestamp is missing timezone provenance, so automatic calendar matching is intentionally conservative."
                ),
                "rationale": "Upload capture time is missing timezone provenance and should be reviewed.",
                "auto_select_mode": "blocked",
            }
        return {
            **base,
            "status": "weak_upload_time",
            "penalty": 0.08,
            "warning": (
                "This upload used receipt time instead of a filename capture timestamp, so calendar auto-selection is conservative."
            ),
            "rationale": (
                "Capture time came from upload receipt time instead of filename metadata."
            ),
            "auto_select_mode": "conservative",
        }

    return base


def _presence_score(
    *,
    captured_at: datetime,
    starts_at: datetime,
    ends_at: datetime,
) -> tuple[float, bool]:
    if starts_at <= captured_at <= ends_at:
        return 1.0, True
    minutes_before_start = (starts_at - captured_at).total_seconds() / 60.0
    if 0.0 < minutes_before_start <= _PRESTART_GRACE_MINUTES:
        return max(0.45, 0.9 - (minutes_before_start / 30.0)), False
    return 0.0, False


def _overlap_signals(
    *,
    captured_at: datetime,
    starts_at: datetime,
    ends_at: datetime,
    duration_sec: float | None,
) -> dict[str, float]:
    event_duration_seconds = max((ends_at - starts_at).total_seconds(), 60.0)
    if duration_sec is None:
        return {
            "overlap_seconds": 0.0,
            "recording_overlap_ratio": 0.0,
            "event_overlap_ratio": 0.0,
            "overlap_score": 0.0,
            "duration_similarity": 0.0,
            "duration_score": 0.0,
        }

    recording_end = captured_at + timedelta(seconds=duration_sec)
    overlap_seconds = max(
        0.0,
        (min(recording_end, ends_at) - max(captured_at, starts_at)).total_seconds(),
    )
    recording_overlap_ratio = min(1.0, overlap_seconds / max(duration_sec, 1.0))
    event_overlap_ratio = min(1.0, overlap_seconds / max(event_duration_seconds, 1.0))
    overlap_score = (0.45 * recording_overlap_ratio) + (0.55 * event_overlap_ratio)
    duration_similarity = max(
        0.0,
        1.0
        - (
            abs(event_duration_seconds - duration_sec)
            / max(event_duration_seconds, duration_sec, 1.0)
        ),
    )
    if overlap_seconds <= 0:
        duration_score = 0.0
    else:
        duration_score = duration_similarity * (0.35 + (0.65 * event_overlap_ratio))
    return {
        "overlap_seconds": overlap_seconds,
        "recording_overlap_ratio": recording_overlap_ratio,
        "event_overlap_ratio": event_overlap_ratio,
        "overlap_score": overlap_score,
        "duration_similarity": duration_similarity,
        "duration_score": duration_score,
    }


def _event_token_buckets(event: dict[str, Any]) -> dict[str, set[str]]:
    attendee_values: list[object] = [
        event.get("organizer"),
        event.get("organizer_name"),
    ]
    attendee_values.extend(_event_attendees(event))
    attendees: set[str] = set()
    for value in attendee_values:
        attendees.update(_tokenize(value))
    return {
        "subject": _tokenize(event.get("summary")),
        "location": _tokenize(event.get("location")),
        "attendees": attendees,
    }


def _text_match_details(
    *,
    event: dict[str, Any],
    recording_tokens: set[str],
) -> dict[str, Any]:
    buckets = _event_token_buckets(event)
    shared = {
        name: sorted(recording_tokens & tokens)
        for name, tokens in buckets.items()
    }
    if not recording_tokens:
        return {
            "score": 0.0,
            "shared": shared,
            "subject_score": 0.0,
            "attendee_score": 0.0,
            "location_score": 0.0,
        }
    subject_score = min(1.0, len(shared["subject"]) / max(min(len(recording_tokens), 3), 1))
    attendee_score = min(1.0, len(shared["attendees"]) / 2.0)
    location_score = min(1.0, len(shared["location"]) / 2.0)
    score = min(
        1.0,
        (0.65 * subject_score) + (0.25 * attendee_score) + (0.10 * location_score),
    )
    return {
        "score": score,
        "shared": shared,
        "subject_score": subject_score,
        "attendee_score": attendee_score,
        "location_score": location_score,
    }


def _candidate_score_value(candidate: dict[str, Any]) -> float:
    raw = candidate.get("score", candidate.get("confidence"))
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _score_event(
    *,
    event: dict[str, Any],
    captured_at: datetime,
    duration_sec: float | None,
    recording_tokens: set[str],
    capture_time_assessment: dict[str, Any],
) -> dict[str, Any] | None:
    starts_at = _parse_datetime(event.get("starts_at"))
    ends_at = _parse_datetime(event.get("ends_at"))
    if starts_at is None or ends_at is None or ends_at <= starts_at:
        return None

    signed_start_delta_minutes = (captured_at - starts_at).total_seconds() / 60.0
    start_delta_minutes = abs(signed_start_delta_minutes)
    proximity_score = max(0.0, 1.0 - (start_delta_minutes / _PROXIMITY_WINDOW_MINUTES))
    presence_score, capture_inside = _presence_score(
        captured_at=captured_at,
        starts_at=starts_at,
        ends_at=ends_at,
    )
    overlap = _overlap_signals(
        captured_at=captured_at,
        starts_at=starts_at,
        ends_at=ends_at,
        duration_sec=duration_sec,
    )
    text_match = _text_match_details(event=event, recording_tokens=recording_tokens)
    penalty = float(capture_time_assessment.get("penalty") or 0.0)

    if duration_sec is None:
        raw_score = (
            (0.56 * proximity_score)
            + (0.24 * presence_score)
            + (0.20 * float(text_match["score"]))
        )
    else:
        raw_score = (
            (0.24 * proximity_score)
            + (0.18 * presence_score)
            + (0.30 * float(overlap["overlap_score"]))
            + (0.18 * float(overlap["duration_score"]))
            + (0.10 * float(text_match["score"]))
        )
    confidence = round(max(0.0, min(raw_score - penalty, 1.0)), 4)

    if signed_start_delta_minutes == 0:
        rationale = ["Capture starts exactly at the event start."]
    elif signed_start_delta_minutes < 0:
        rationale = [
            f"Capture starts {int(round(abs(signed_start_delta_minutes)))}m before the event."
        ]
    else:
        rationale = [
            f"Capture starts {int(round(signed_start_delta_minutes))}m after the event begins."
        ]
    if capture_inside:
        rationale.append("Capture time falls inside the event window.")
    elif presence_score > 0:
        rationale.append("Capture starts shortly before the event begins.")
    if duration_sec is not None:
        rationale.append(
            "Overlap covers "
            f"{int(round(float(overlap['recording_overlap_ratio']) * 100))}% of the recording and "
            f"{int(round(float(overlap['event_overlap_ratio']) * 100))}% of the event."
        )
        rationale.append(
            f"Duration signal is {int(round(float(overlap['duration_score']) * 100))}% after overlap adjustment."
        )
    for label, shared_tokens in (
        ("subject", text_match["shared"]["subject"]),
        ("attendee", text_match["shared"]["attendees"]),
        ("location", text_match["shared"]["location"]),
    ):
        if shared_tokens:
            rationale.append(f"Shared {label} tokens: {', '.join(shared_tokens)}.")
    capture_time_rationale = str(capture_time_assessment.get("rationale") or "").strip()
    if capture_time_rationale:
        rationale.append(capture_time_rationale)
    rationale.append(f"Final confidence {confidence:.2f}.")

    return {
        "event_id": _event_identifier(event),
        "uid": str(event.get("uid") or "").strip(),
        "source_id": event.get("source_id"),
        "source_name": str(event.get("source_name") or "").strip() or None,
        "source_kind": str(event.get("source_kind") or "").strip() or None,
        "subject": str(event.get("summary") or "").strip() or None,
        "summary": str(event.get("summary") or "").strip() or None,
        "description": str(event.get("description") or "").strip() or None,
        "location": str(event.get("location") or "").strip() or None,
        "starts_at": _utc_iso(starts_at),
        "ends_at": _utc_iso(ends_at),
        "all_day": bool(event.get("all_day")),
        "organizer": str(event.get("organizer") or "").strip() or None,
        "organizer_name": str(event.get("organizer_name") or "").strip() or None,
        "organizer_email": str(event.get("organizer_email") or "").strip() or None,
        "attendees": _event_attendees(event),
        "attendee_details": _event_attendee_details(event),
        "score": confidence,
        "confidence": confidence,
        "rationale": rationale,
        "capture_time_status": str(capture_time_assessment.get("status") or "stored_utc"),
        "score_details": {
            "proximity": round(proximity_score, 4),
            "presence": round(presence_score, 4),
            "overlap": round(float(overlap["overlap_score"]), 4),
            "duration": round(float(overlap["duration_score"]), 4),
            "text": round(float(text_match["score"]), 4),
            "penalty": round(penalty, 4),
            "raw_score": round(raw_score, 4),
            "signed_start_delta_minutes": round(signed_start_delta_minutes, 2),
            "recording_overlap_ratio": round(float(overlap["recording_overlap_ratio"]), 4),
            "event_overlap_ratio": round(float(overlap["event_overlap_ratio"]), 4),
            "duration_similarity": round(float(overlap["duration_similarity"]), 4),
            "shared_tokens": {
                "subject": list(text_match["shared"]["subject"]),
                "attendees": list(text_match["shared"]["attendees"]),
                "location": list(text_match["shared"]["location"]),
            },
        },
    }


def _auto_selection_decision(
    candidates: list[dict[str, Any]],
    *,
    capture_time_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assessment = capture_time_assessment or {
        "auto_select_mode": "normal",
        "penalty": 0.0,
    }
    min_score = _AUTO_SELECT_MIN_SCORE
    min_margin = _AUTO_SELECT_MIN_MARGIN
    if str(assessment.get("auto_select_mode") or "") == "conservative":
        min_score = max(min_score, _CONSERVATIVE_AUTO_SELECT_MIN_SCORE)
        min_margin = max(min_margin, _CONSERVATIVE_AUTO_SELECT_MIN_MARGIN)
    if not candidates:
        return {
            "selected_event_id": None,
            "selected_confidence": None,
            "reason_code": "no_candidates",
            "reason": "No calendar candidates are currently stored for this recording.",
            "min_score": min_score,
            "min_margin": min_margin,
        }
    if str(assessment.get("auto_select_mode") or "") == "blocked":
        return {
            "selected_event_id": None,
            "selected_confidence": None,
            "reason_code": "blocked_suspicious_time",
            "reason": "Automatic selection is disabled because the capture timestamp should be reviewed first.",
            "min_score": min_score,
            "min_margin": min_margin,
        }
    top_score = _candidate_score_value(candidates[0])
    if top_score < min_score:
        return {
            "selected_event_id": None,
            "selected_confidence": None,
            "reason_code": "below_threshold",
            "reason": (
                "Top candidate stayed below the auto-select threshold "
                f"({top_score:.2f} < {min_score:.2f})."
            ),
            "min_score": min_score,
            "min_margin": min_margin,
        }
    if len(candidates) > 1:
        next_score = _candidate_score_value(candidates[1])
        margin = top_score - next_score
        if margin < min_margin:
            return {
                "selected_event_id": None,
                "selected_confidence": None,
                "reason_code": "margin_too_small",
                "reason": (
                    "Top candidates are too close together for automatic selection "
                    f"({margin:.2f} < {min_margin:.2f})."
                ),
                "min_score": min_score,
                "min_margin": min_margin,
            }
        return {
            "selected_event_id": str(candidates[0].get("event_id") or "").strip() or None,
            "selected_confidence": top_score,
            "reason_code": "selected",
            "reason": f"Beat the next candidate by {margin:.2f} confidence.",
            "min_score": min_score,
            "min_margin": min_margin,
        }
    return {
        "selected_event_id": str(candidates[0].get("event_id") or "").strip() or None,
        "selected_confidence": top_score,
        "reason_code": "selected",
        "reason": "Only candidate and above the auto-select threshold.",
        "min_score": min_score,
        "min_margin": min_margin,
    }


def _auto_selection(
    candidates: list[dict[str, Any]],
    *,
    capture_time_assessment: dict[str, Any] | None = None,
) -> tuple[str | None, float | None]:
    decision = _auto_selection_decision(
        candidates,
        capture_time_assessment=capture_time_assessment,
    )
    return (
        decision["selected_event_id"],
        decision["selected_confidence"],
    )


def _annotate_candidates(
    candidates: list[dict[str, Any]],
    *,
    decision: dict[str, Any],
) -> None:
    if not candidates:
        return
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index
    top = candidates[0]
    top["auto_select_reason"] = decision["reason_code"]
    top["auto_select_note"] = decision["reason"]
    top["auto_select_min_score"] = decision["min_score"]
    top["auto_select_min_margin"] = decision["min_margin"]
    if len(candidates) > 1:
        margin = _candidate_score_value(candidates[0]) - _candidate_score_value(candidates[1])
        top["margin_to_next"] = round(margin, 4)
    note = str(decision.get("reason") or "").strip()
    rationale = top.get("rationale")
    if note and isinstance(rationale, list) and note not in rationale:
        rationale.append(note)


def calendar_match_warnings(
    recording: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    *,
    selected_event_id: str | None,
) -> list[str]:
    assessment = _capture_time_assessment(recording or {})
    warnings: list[str] = []
    warning_text = str(assessment.get("warning") or "").strip()
    if warning_text:
        warnings.append(warning_text)
    if selected_event_id:
        return warnings
    decision = _auto_selection_decision(
        candidates,
        capture_time_assessment=assessment,
    )
    if decision["reason_code"] == "below_threshold":
        warnings.append(
            "No calendar candidate is strong enough for auto-selection yet. Review the candidates on this tab."
        )
    if decision["reason_code"] == "margin_too_small":
        warnings.append(
            "Multiple nearby calendar candidates scored too closely, so the recording was left for manual selection."
        )
    elif len(candidates) > 1:
        margin = _candidate_score_value(candidates[0]) - _candidate_score_value(candidates[1])
        if margin < float(decision.get("min_margin") or _AUTO_SELECT_MIN_MARGIN):
            warnings.append(
                "Multiple nearby calendar candidates scored too closely, so the recording was left for manual selection."
            )
    deduped: list[str] = []
    seen: set[str] = set()
    for row in warnings:
        if row in seen:
            continue
        seen.add(row)
        deduped.append(row)
    return deduped


def match_recording_to_calendar(
    recording: dict[str, Any],
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    recording_id = str(recording.get("id") or "").strip()
    captured_at = _parse_datetime(recording.get("captured_at"))
    if captured_at is None:
        return {
            "recording_id": recording_id,
            "candidates": [],
            "selected_event_id": None,
            "selected_confidence": None,
            "warnings": calendar_match_warnings(
                recording,
                [],
                selected_event_id=None,
            ),
        }

    capture_time_assessment = _capture_time_assessment(recording)
    duration_sec = _safe_duration_seconds(recording.get("duration_sec"))
    window_end = captured_at + max(
        _SEARCH_WINDOW_AFTER,
        timedelta(seconds=duration_sec or 0.0) + timedelta(hours=2),
    )
    events = list_calendar_events(
        starts_from=_utc_iso(captured_at - _SEARCH_WINDOW_BEFORE),
        ends_to=_utc_iso(window_end),
        settings=settings,
    )
    recording_tokens = _recording_tokens(recording)
    candidates = [
        candidate
        for event in events
        if (
            candidate := _score_event(
                event=event,
                captured_at=captured_at,
                duration_sec=duration_sec,
                recording_tokens=recording_tokens,
                capture_time_assessment=capture_time_assessment,
            )
        )
        is not None
    ]
    candidates.sort(
        key=lambda candidate: (
            -_candidate_score_value(candidate),
            str(candidate.get("starts_at") or ""),
            str(candidate.get("event_id") or ""),
        )
    )
    decision = _auto_selection_decision(
        candidates,
        capture_time_assessment=capture_time_assessment,
    )
    _annotate_candidates(candidates, decision=decision)
    selected_event_id = decision["selected_event_id"]
    return {
        "recording_id": recording_id,
        "candidates": candidates,
        "selected_event_id": selected_event_id,
        "selected_confidence": decision["selected_confidence"],
        "warnings": calendar_match_warnings(
            recording,
            candidates,
            selected_event_id=selected_event_id,
        ),
    }


def refresh_recording_calendar_match(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    recording = get_recording(recording_id, settings=settings)
    if recording is None:
        raise KeyError(recording_id)
    existing = get_calendar_match(recording_id, settings=settings) or {}
    match = match_recording_to_calendar(recording, settings=settings)
    if not match["candidates"] and _candidate_rows(existing.get("candidates_json")):
        return existing
    return upsert_calendar_match(
        recording_id=recording_id,
        candidates=match["candidates"],
        selected_event_id=match["selected_event_id"],
        selected_confidence=match["selected_confidence"],
        settings=settings,
    )


__all__ = [
    "calendar_match_candidates",
    "calendar_match_warnings",
    "calendar_summary_context",
    "match_recording_to_calendar",
    "refresh_recording_calendar_match",
    "selected_calendar_candidate",
]
