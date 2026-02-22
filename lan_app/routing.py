"""Project routing: suggest project with confidence and learn from corrections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .config import AppSettings
from .constants import RECORDING_STATUS_NEEDS_REVIEW, RECORDING_STATUS_READY
from .db import (
    count_routing_training_examples,
    create_routing_training_example,
    get_calendar_match,
    get_project,
    get_recording,
    increment_project_keyword_weights,
    list_project_keyword_weights,
    list_projects,
    list_speaker_assignments,
    set_recording_project,
    set_recording_routing_suggestion,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
_SIGNAL_WEIGHTS = {
    "calendar_subject": 0.40,
    "participants": 0.15,
    "llm_tags": 0.30,
    "voice_profiles": 0.15,
}
_KEY_SCALE = 3.0


@dataclass(frozen=True)
class RoutingSignals:
    calendar_subject_tokens: list[str]
    participant_tokens: list[str]
    llm_tags: list[str]
    voice_profile_ids: list[int]


def refresh_recording_routing(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
    apply_workflow: bool = False,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    recording = get_recording(recording_id, settings=cfg)
    if recording is None:
        raise KeyError(recording_id)
    current_project_id = recording.get("project_id")
    current_assignment_source = str(
        recording.get("project_assignment_source") or ""
    ).strip().lower()

    projects = list_projects(settings=cfg)
    signals = _build_routing_signals(recording_id, settings=cfg)
    threshold = float(cfg.routing_auto_select_threshold)

    if not projects:
        rationale = [
            "No projects exist yet, so routing cannot suggest a project.",
            f"Confidence 0.00 is below threshold {threshold:.2f}; manual review required.",
        ]
        set_recording_routing_suggestion(
            recording_id,
            suggested_project_id=None,
            routing_confidence=0.0,
            routing_rationale=rationale,
            settings=cfg,
        )
        return {
            "recording_id": recording_id,
            "suggested_project_id": None,
            "suggested_project_name": None,
            "confidence": 0.0,
            "threshold": threshold,
            "rationale": rationale,
            "status_after_routing": (
                RECORDING_STATUS_NEEDS_REVIEW if apply_workflow else recording.get("status")
            ),
            "auto_selected": False,
            "training_examples_total": 0,
        }

    weight_rows = list_project_keyword_weights(settings=cfg)
    weights_by_project: dict[int, dict[str, float]] = {}
    for row in weight_rows:
        try:
            project_id = int(row.get("project_id"))
            keyword = str(row.get("keyword") or "").strip()
            weight = float(row.get("weight"))
        except (TypeError, ValueError):
            continue
        if not keyword:
            continue
        weights_by_project.setdefault(project_id, {})[keyword] = weight

    scored: list[dict[str, Any]] = []
    for project in projects:
        try:
            project_id = int(project["id"])
        except (TypeError, ValueError, KeyError):
            continue
        score_payload = _project_score(
            project=project,
            signals=signals,
            learned_weights=weights_by_project.get(project_id, {}),
        )
        scored.append(
            {
                "project_id": project_id,
                "project_name": str(project.get("name") or f"Project {project_id}"),
                **score_payload,
            }
        )
    scored.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)

    top = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    top_score = float(top.get("score") or 0.0) if top else 0.0
    runner_score = float(runner_up.get("score") or 0.0) if runner_up else 0.0
    margin = max(0.0, top_score - runner_score)
    confidence = min(1.0, max(0.0, (0.75 * top_score) + (0.25 * margin)))
    if top is None or top_score <= 0.0:
        suggested_project_id = None
        suggested_project_name = None
        confidence = 0.0
    else:
        suggested_project_id = int(top["project_id"])
        suggested_project_name = str(top.get("project_name") or "")

    rationale = _build_rationale(
        top=top if suggested_project_id is not None else None,
        runner_up=runner_up if suggested_project_id is not None else None,
        confidence=confidence,
        threshold=threshold,
        signals=signals,
    )

    set_recording_routing_suggestion(
        recording_id,
        suggested_project_id=suggested_project_id,
        routing_confidence=confidence,
        routing_rationale=rationale,
        settings=cfg,
    )

    auto_selected = False
    status_after_routing: str = str(recording.get("status") or RECORDING_STATUS_READY)
    if apply_workflow:
        if suggested_project_id is not None and confidence >= threshold:
            set_recording_project(
                recording_id,
                suggested_project_id,
                settings=cfg,
                assignment_source="auto",
            )
            auto_selected = True
            status_after_routing = RECORDING_STATUS_READY
        else:
            if current_project_id is not None and current_assignment_source == "auto":
                set_recording_project(
                    recording_id,
                    None,
                    settings=cfg,
                )
            status_after_routing = RECORDING_STATUS_NEEDS_REVIEW

    return {
        "recording_id": recording_id,
        "suggested_project_id": suggested_project_id,
        "suggested_project_name": suggested_project_name,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "rationale": rationale,
        "status_after_routing": status_after_routing,
        "auto_selected": auto_selected,
        "training_examples_total": count_routing_training_examples(settings=cfg),
    }


def train_routing_from_manual_selection(
    recording_id: str,
    project_id: int,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    recording = get_recording(recording_id, settings=cfg)
    if recording is None:
        raise KeyError(recording_id)
    project = get_project(int(project_id), settings=cfg)
    if project is None:
        raise KeyError(project_id)

    signals = _build_routing_signals(recording_id, settings=cfg)
    example = create_routing_training_example(
        recording_id=recording_id,
        project_id=int(project_id),
        calendar_subject_tokens=signals.calendar_subject_tokens,
        tags=signals.llm_tags,
        voice_profile_ids=signals.voice_profile_ids,
        settings=cfg,
    )

    deltas: dict[str, float] = {}
    for token in signals.calendar_subject_tokens:
        deltas[f"cal:{token}"] = deltas.get(f"cal:{token}", 0.0) + 1.0
    for token in signals.participant_tokens:
        deltas[f"party:{token}"] = deltas.get(f"party:{token}", 0.0) + 0.6
    for token in signals.llm_tags:
        deltas[f"tag:{token}"] = deltas.get(f"tag:{token}", 0.0) + 0.8
    for voice_profile_id in signals.voice_profile_ids:
        key = f"voice:{voice_profile_id}"
        deltas[key] = deltas.get(key, 0.0) + 1.2

    updated_keywords = increment_project_keyword_weights(
        project_id=int(project_id),
        keyword_deltas=deltas,
        settings=cfg,
    )

    return {
        "recording_id": recording_id,
        "project_id": int(project_id),
        "training_example_id": example.get("id"),
        "updated_keyword_count": updated_keywords,
    }


def _project_score(
    *,
    project: dict[str, Any],
    signals: RoutingSignals,
    learned_weights: dict[str, float],
) -> dict[str, Any]:
    project_name_tokens = _tokenize(str(project.get("name") or ""))

    calendar_component = _component_score(
        tokens=signals.calendar_subject_tokens,
        keyword_prefix="cal",
        learned_weights=learned_weights,
        fallback_tokens=project_name_tokens,
        fallback_score=0.4,
    )
    participant_component = _component_score(
        tokens=signals.participant_tokens,
        keyword_prefix="party",
        learned_weights=learned_weights,
        fallback_tokens=set(),
        fallback_score=0.0,
    )
    llm_component = _component_score(
        tokens=signals.llm_tags,
        keyword_prefix="tag",
        learned_weights=learned_weights,
        fallback_tokens=project_name_tokens,
        fallback_score=0.35,
    )
    voice_component = _component_score(
        tokens=[str(value) for value in signals.voice_profile_ids],
        keyword_prefix="voice",
        learned_weights=learned_weights,
        fallback_tokens=set(),
        fallback_score=0.0,
    )

    component_rows = {
        "calendar_subject": calendar_component,
        "participants": participant_component,
        "llm_tags": llm_component,
        "voice_profiles": voice_component,
    }
    available_weight = 0.0
    score_total = 0.0
    for key, payload in component_rows.items():
        if payload["token_count"] <= 0:
            continue
        weight = _SIGNAL_WEIGHTS[key]
        available_weight += weight
        score_total += weight * payload["score"]

    if available_weight <= 0.0:
        score = 0.0
    else:
        score = min(1.0, max(0.0, score_total / available_weight))

    return {
        "score": round(score, 4),
        "components": component_rows,
    }


def _component_score(
    *,
    tokens: list[str],
    keyword_prefix: str,
    learned_weights: dict[str, float],
    fallback_tokens: set[str],
    fallback_score: float,
) -> dict[str, Any]:
    unique_tokens = sorted({token for token in tokens if token})
    if not unique_tokens:
        return {
            "score": 0.0,
            "token_count": 0,
            "matched_count": 0,
        }

    signal_scores: list[float] = []
    matched_count = 0
    for token in unique_tokens:
        lookup_key = f"{keyword_prefix}:{token}"
        learned = float(learned_weights.get(lookup_key, 0.0))
        if learned > 0.0:
            contribution = min(learned / _KEY_SCALE, 1.0)
            matched_count += 1
        elif token in fallback_tokens:
            contribution = fallback_score
            if contribution > 0.0:
                matched_count += 1
        else:
            contribution = 0.0
        signal_scores.append(contribution)

    score = sum(signal_scores) / float(len(signal_scores))
    return {
        "score": round(score, 4),
        "token_count": len(unique_tokens),
        "matched_count": matched_count,
    }


def _build_rationale(
    *,
    top: dict[str, Any] | None,
    runner_up: dict[str, Any] | None,
    confidence: float,
    threshold: float,
    signals: RoutingSignals,
) -> list[str]:
    if top is None:
        return [
            "No project could be scored from available routing signals.",
            f"Confidence {confidence:.2f} is below threshold {threshold:.2f}; manual review required.",
        ]

    components = top.get("components", {})
    calendar_component = components.get("calendar_subject", {})
    participant_component = components.get("participants", {})
    llm_component = components.get("llm_tags", {})
    voice_component = components.get("voice_profiles", {})

    rationale = [
        (
            f"Top project: {top.get('project_name')} "
            f"(score={float(top.get('score') or 0.0):.2f})."
        ),
        (
            "Calendar subject match "
            f"{float(calendar_component.get('score') or 0.0):.2f} "
            f"({int(calendar_component.get('matched_count') or 0)}/"
            f"{int(calendar_component.get('token_count') or 0)} tokens)."
        ),
        (
            "Organizer/attendee match "
            f"{float(participant_component.get('score') or 0.0):.2f} "
            f"({int(participant_component.get('matched_count') or 0)}/"
            f"{int(participant_component.get('token_count') or 0)} tokens)."
        ),
        (
            "LLM keyword match "
            f"{float(llm_component.get('score') or 0.0):.2f} "
            f"({int(llm_component.get('matched_count') or 0)}/"
            f"{int(llm_component.get('token_count') or 0)} tags)."
        ),
        (
            "Voice profile match "
            f"{float(voice_component.get('score') or 0.0):.2f} "
            f"({int(voice_component.get('matched_count') or 0)}/"
            f"{int(voice_component.get('token_count') or 0)} profiles)."
        ),
    ]
    if runner_up is not None:
        rationale.append(
            (
                f"Runner-up: {runner_up.get('project_name')} "
                f"(score={float(runner_up.get('score') or 0.0):.2f})."
            )
        )
    rationale.append(
        (
            f"Final confidence {confidence:.2f} "
            f"({'meets' if confidence >= threshold else 'below'} threshold {threshold:.2f})."
        )
    )
    if not signals.calendar_subject_tokens:
        rationale.append("No calendar subject tokens were available.")
    if not signals.llm_tags:
        rationale.append("No LLM tags/keywords were available yet.")
    if not signals.voice_profile_ids:
        rationale.append("No assigned voice profiles were available yet.")
    return rationale


def _build_routing_signals(
    recording_id: str,
    *,
    settings: AppSettings,
) -> RoutingSignals:
    selected_calendar = _selected_calendar_candidate(recording_id, settings=settings)
    subject_tokens = _tokenize(str(selected_calendar.get("subject") or ""))

    participant_rows = [str(selected_calendar.get("organizer") or "").strip()]
    attendee_rows = selected_calendar.get("attendees")
    if isinstance(attendee_rows, list):
        participant_rows.extend(str(row).strip() for row in attendee_rows)
    participant_tokens: set[str] = set()
    for row in participant_rows:
        participant_tokens.update(_tokenize(row))

    llm_tags = _llm_keywords(recording_id, settings=settings)

    voice_profile_ids: set[int] = set()
    for assignment in list_speaker_assignments(recording_id, settings=settings):
        try:
            profile_id_raw = assignment.get("voice_profile_id")
            if profile_id_raw is None:
                continue
            voice_profile_ids.add(int(profile_id_raw))
        except (TypeError, ValueError):
            continue

    return RoutingSignals(
        calendar_subject_tokens=sorted(subject_tokens),
        participant_tokens=sorted(participant_tokens),
        llm_tags=sorted(llm_tags),
        voice_profile_ids=sorted(voice_profile_ids),
    )


def _selected_calendar_candidate(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    row = get_calendar_match(recording_id, settings=settings) or {}
    selected_event_id = str(row.get("selected_event_id") or "").strip()
    if not selected_event_id:
        return {}

    candidates = row.get("candidates_json")
    if not isinstance(candidates, list):
        return {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("event_id") or "").strip() == selected_event_id:
            return candidate
    return {}


def _llm_keywords(recording_id: str, *, settings: AppSettings) -> set[str]:
    summary_path = settings.recordings_root / recording_id / "derived" / "summary.json"
    payload = _load_json_dict(summary_path)
    snippets: list[str] = []
    snippets.append(str(payload.get("topic") or ""))
    snippets.extend(_text_list(payload.get("summary_bullets")))
    snippets.extend(_text_list(payload.get("decisions")))
    snippets.extend(_action_item_keywords(payload.get("action_items")))

    tags: set[str] = set()
    for snippet in snippets:
        tags.update(_tokenize(snippet))
    return tags


def _action_item_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        task = str(row.get("task") or "").strip()
        owner = str(row.get("owner") or "").strip()
        deadline = str(row.get("deadline") or "").strip()
        if task:
            out.append(task)
        if owner:
            out.append(owner)
        if deadline:
            out.append(deadline)
    return out


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for row in value:
        text = str(row).strip()
        if text:
            out.append(text)
    return out


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


def _tokenize(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in _TOKEN_RE.findall(value):
        token = raw_token.lower()
        if len(token) < 2:
            continue
        if token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


__all__ = [
    "refresh_recording_routing",
    "train_routing_from_manual_selection",
]
