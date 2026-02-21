from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from lan_transcriber.artifacts import atomic_write_json

from .config import AppSettings
from .db import replace_participant_metrics, upsert_meeting_metrics

DEFAULT_GAP_THRESHOLD_SEC = 1.0
DEFAULT_INTERRUPTION_OVERLAP_SEC = 0.3

_QUESTION_TYPE_KEYS = (
    "open",
    "yes_no",
    "clarification",
    "status",
    "decision_seeking",
)

_DECISION_CUES = (
    "decide",
    "decision",
    "agree",
    "approved",
    "approve",
    "commit",
    "resolved",
    "finalize",
)

_TASK_CUES = (
    "action item",
    "follow up",
    "todo",
    "to-do",
    "next step",
    "owner",
    "deadline",
    "i will",
    "we will",
    "we need",
)

def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out < 0:
        return default
    return out


def _normalise_language_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    token = raw.replace("_", "-").split("-", 1)[0]
    if len(token) == 2 and token.isalpha():
        return token
    return None


def _normalise_turns(turns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in turns:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        start = _safe_float(row.get("start"), default=0.0)
        end = _safe_float(row.get("end"), default=start)
        if end < start:
            end = start
        speaker = str(row.get("speaker") or "S1").strip() or "S1"
        payload: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
            "text": text,
        }
        language = _normalise_language_code(row.get("language"))
        if language:
            payload["language"] = language
        out.append(payload)
    out.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return out


def _fallback_speaker_turns_from_transcript(transcript_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = transcript_payload.get("segments")
    if isinstance(raw_segments, list):
        turns: list[dict[str, Any]] = []
        for row in raw_segments:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            start = _safe_float(row.get("start"), default=0.0)
            end = _safe_float(row.get("end"), default=start)
            if end < start:
                end = start
            payload: dict[str, Any] = {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": "S1",
                "text": text,
            }
            language = _normalise_language_code(row.get("language"))
            if language:
                payload["language"] = language
            turns.append(payload)
        if turns:
            return turns

    text = str(transcript_payload.get("text") or "").strip()
    if not text:
        return []
    return [{"start": 0.0, "end": 0.0, "speaker": "S1", "text": text}]


def merge_speaker_turns(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD_SEC,
) -> list[dict[str, Any]]:
    """Merge adjacent turns from the same speaker with tiny gaps."""
    turns = _normalise_turns(speaker_turns)
    if not turns:
        return []

    safe_gap = max(gap_threshold, 0.0)
    merged: list[dict[str, Any]] = []
    current = dict(turns[0])

    for row in turns[1:]:
        same_speaker = str(row["speaker"]) == str(current["speaker"])
        if same_speaker and _safe_float(row.get("start")) <= _safe_float(current.get("end")) + safe_gap:
            current["end"] = round(
                max(_safe_float(current.get("end")), _safe_float(row.get("end"))),
                3,
            )
            current["text"] = f"{current['text']} {row['text']}".strip()
            if "language" not in current and "language" in row:
                current["language"] = row["language"]
            continue
        merged.append(current)
        current = dict(row)

    merged.append(current)
    return merged


def count_interruptions(
    speaker_turns: Sequence[dict[str, Any]],
    *,
    overlap_threshold: float = DEFAULT_INTERRUPTION_OVERLAP_SEC,
) -> dict[str, Any]:
    """Count interruption events and per-speaker done/received counters."""
    turns = _normalise_turns(speaker_turns)
    safe_overlap = max(overlap_threshold, 0.0)

    done: dict[str, int] = {}
    received: dict[str, int] = {}
    total = 0

    for idx, turn in enumerate(turns):
        interrupter = str(turn["speaker"])
        done.setdefault(interrupter, 0)
        received.setdefault(interrupter, 0)

        turn_start = _safe_float(turn.get("start"))
        turn_end = _safe_float(turn.get("end"), default=turn_start)
        if turn_end <= turn_start:
            continue

        seen_receivers: set[str] = set()
        for previous in turns[:idx]:
            receiver = str(previous["speaker"])
            if receiver == interrupter or receiver in seen_receivers:
                continue

            previous_end = _safe_float(previous.get("end"), default=_safe_float(previous.get("start")))
            if turn_start >= previous_end:
                continue

            overlap = min(previous_end, turn_end) - turn_start
            if overlap < safe_overlap:
                continue

            done[interrupter] = done.get(interrupter, 0) + 1
            received[receiver] = received.get(receiver, 0) + 1
            total += 1
            seen_receivers.add(receiver)

    return {
        "total": total,
        "done": done,
        "received": received,
    }


def _normalise_text_items(value: Any, *, max_items: int) -> list[str]:
    rows: list[Any]
    if isinstance(value, list):
        rows = value
    elif isinstance(value, str):
        rows = [line.strip() for line in value.splitlines() if line.strip()]
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


def _normalise_action_items(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        task = str(row.get("task") or "").strip()
        if not task:
            continue
        owner = str(row.get("owner") or "").strip() or None
        deadline = str(row.get("deadline") or "").strip() or None
        out.append(
            {
                "task": task,
                "owner": owner,
                "deadline": deadline,
            }
        )
    return out


def _normalise_question_types(value: Any) -> dict[str, int]:
    out = {key: 0 for key in _QUESTION_TYPE_KEYS}
    if not isinstance(value, dict):
        return out
    for key in _QUESTION_TYPE_KEYS:
        try:
            out[key] = max(0, int(value.get(key, 0)))
        except (TypeError, ValueError):
            out[key] = 0
    return out


def _count_question_punctuation(text: str, _language: str | None) -> int:
    return text.count("?") + text.count("？")


def _tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\b[\w-]{2,}\b", text, flags=re.UNICODE)]


def _tokenize_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"\b[\w-]{5,}\b", text, flags=re.UNICODE)
    }


def _allocate_by_weight(total: int, weights: dict[str, int]) -> dict[str, int]:
    keys = sorted(weights)
    if total <= 0 or not keys:
        return {key: 0 for key in keys}

    positive_weights = {key: max(int(weights.get(key, 0)), 0) for key in keys}
    weight_sum = sum(positive_weights.values())
    if weight_sum == 0:
        out = {key: 0 for key in keys}
        for idx in range(total):
            out[keys[idx % len(keys)]] += 1
        return out

    raw: dict[str, float] = {
        key: total * (positive_weights[key] / weight_sum)
        for key in keys
    }
    out = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(out.values())
    ranked = sorted(
        keys,
        key=lambda key: (raw[key] - out[key], positive_weights[key], key),
        reverse=True,
    )
    for key in ranked[:remainder]:
        out[key] += 1
    return out


def _question_counts_by_speaker(
    merged_turns: Sequence[dict[str, Any]],
    summary_payload: dict[str, Any],
) -> tuple[dict[str, int], int, str, dict[str, int]]:
    speakers = sorted({str(row.get("speaker") or "S1") for row in merged_turns})
    heuristic = {speaker: 0 for speaker in speakers}
    for row in merged_turns:
        speaker = str(row.get("speaker") or "S1")
        text = str(row.get("text") or "")
        language = _normalise_language_code(row.get("language"))
        heuristic[speaker] = heuristic.get(speaker, 0) + _count_question_punctuation(
            text,
            language,
        )

    questions_payload = summary_payload.get("questions")
    questions_obj = questions_payload if isinstance(questions_payload, dict) else {}
    llm_extracted = _normalise_text_items(questions_obj.get("extracted"), max_items=100)
    question_types = _normalise_question_types(questions_obj.get("types"))
    try:
        llm_total = max(0, int(questions_obj.get("total_count", 0)))
    except (TypeError, ValueError):
        llm_total = 0

    if llm_total == 0 and llm_extracted:
        llm_total = len(llm_extracted)

    llm_counts = {speaker: 0 for speaker in speakers}
    normalized_turns: list[tuple[str, str, set[str]]] = []
    for row in merged_turns:
        speaker = str(row.get("speaker") or "S1")
        text = " ".join(str(row.get("text") or "").lower().split())
        normalized_turns.append((speaker, text, set(_tokenize_words(text))))

    for question in llm_extracted:
        q_text = " ".join(question.lower().split())
        q_text = q_text.strip("?？ ")
        if not q_text:
            continue
        q_tokens = set(_tokenize_words(q_text))
        best_speaker: str | None = None
        best_score = 0.0

        for speaker, turn_text, turn_tokens in normalized_turns:
            if q_text in turn_text:
                score = 2.0
            elif q_tokens:
                overlap = len(q_tokens & turn_tokens)
                score = overlap / float(max(len(q_tokens), 1))
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_speaker = speaker

        if best_speaker is not None and best_score >= 0.5:
            llm_counts[best_speaker] = llm_counts.get(best_speaker, 0) + 1

    llm_assigned = sum(llm_counts.values())
    llm_available = llm_total > 0 or bool(llm_extracted)

    if llm_available:
        target_total = llm_total or llm_assigned
        counts = dict(llm_counts)
        if target_total > llm_assigned:
            weight_source = {
                speaker: max(1, heuristic.get(speaker, 0))
                for speaker in speakers
            }
            extra = _allocate_by_weight(target_total - llm_assigned, weight_source)
            for speaker in speakers:
                counts[speaker] = counts.get(speaker, 0) + extra.get(speaker, 0)
        return counts, target_total, "llm", question_types

    heuristic_total = sum(heuristic.values())
    if heuristic_total > 0 and sum(question_types.values()) == 0:
        question_types["open"] = heuristic_total
    return heuristic, heuristic_total, "heuristic", question_types


def _speaker_signal_counts(merged_turns: Sequence[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in merged_turns:
        speaker = str(row.get("speaker") or "S1")
        text = str(row.get("text") or "").lower()
        signals = 0
        if any(cue in text for cue in _DECISION_CUES):
            signals += 1
        if any(cue in text for cue in _TASK_CUES):
            signals += 1
        out[speaker] = out.get(speaker, 0) + signals
    return out


def _domain_terms_from_summary(summary_payload: dict[str, Any]) -> set[str]:
    decisions = _normalise_text_items(summary_payload.get("decisions"), max_items=50)
    action_items = _normalise_action_items(summary_payload.get("action_items"))
    questions_payload = summary_payload.get("questions")
    questions_obj = questions_payload if isinstance(questions_payload, dict) else {}
    extracted_questions = _normalise_text_items(questions_obj.get("extracted"), max_items=50)

    chunks: list[str] = [
        str(summary_payload.get("topic") or ""),
        *decisions,
        *[str(row.get("task") or "") for row in action_items],
        *extracted_questions,
    ]

    terms: set[str] = set()
    for chunk in chunks:
        terms.update(_tokenize_terms(chunk))
    return terms


def _terminology_density_by_speaker(
    merged_turns: Sequence[dict[str, Any]],
    *,
    domain_terms: set[str],
) -> dict[str, float]:
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}

    for row in merged_turns:
        speaker = str(row.get("speaker") or "S1")
        tokens = _tokenize_terms(str(row.get("text") or ""))
        totals[speaker] = totals.get(speaker, 0) + len(tokens)
        hits[speaker] = hits.get(speaker, 0) + len(tokens & domain_terms)

    density: dict[str, float] = {}
    for speaker in totals:
        token_total = totals[speaker]
        if token_total <= 0:
            density[speaker] = 0.0
            continue
        density[speaker] = round(hits.get(speaker, 0) / float(token_total), 4)
    return density


def compute_actionability_ratio(action_items: Sequence[dict[str, Any]]) -> float:
    normalized = _normalise_action_items(list(action_items))
    if not normalized:
        return 0.0
    actionable = sum(
        1
        for row in normalized
        if str(row.get("owner") or "").strip() and str(row.get("deadline") or "").strip()
    )
    return round(actionable / float(len(normalized)), 4)


def _infer_role_hint(
    *,
    airtime_share: float,
    turns: int,
    questions_count: int,
    decision_task_signals: int,
    avg_turn_duration_sec: float,
    terminology_density: float,
    avg_turns: float,
    avg_turn_duration: float,
    meeting_questions: int,
) -> str:
    passive_turn_limit = max(1, int(round(avg_turns * 0.6)))
    facilitator_turn_limit = max(3, int(round(avg_turns * 1.15)))
    facilitator_question_limit = max(1, int(round(meeting_questions * 0.3)))

    if airtime_share <= 0.15 and turns <= passive_turn_limit:
        return "Passive"
    if airtime_share >= 0.35 and decision_task_signals >= 1:
        return "Leader"
    if turns >= facilitator_turn_limit and questions_count >= facilitator_question_limit:
        return "Facilitator"
    if avg_turn_duration_sec >= max(4.0, avg_turn_duration * 1.2) and terminology_density >= 0.08:
        return "Expert"

    scores = {
        "Leader": airtime_share * 1.8 + min(decision_task_signals, 4) * 0.6,
        "Facilitator": (turns / max(avg_turns, 1.0)) * 0.9 + questions_count * 0.45,
        "Expert": (avg_turn_duration_sec / max(avg_turn_duration, 1.0)) * 0.8
        + terminology_density * 4.0,
        "Passive": (1.0 - airtime_share) + max(0.0, 1.0 - (turns / max(avg_turns, 1.0))),
    }
    ordered_roles = ("Leader", "Facilitator", "Expert", "Passive")
    return max(ordered_roles, key=lambda role: (scores[role], -ordered_roles.index(role)))


def build_conversation_metrics(
    *,
    transcript_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    speaker_turns: Sequence[dict[str, Any]],
    gap_threshold_sec: float = DEFAULT_GAP_THRESHOLD_SEC,
    interruption_overlap_sec: float = DEFAULT_INTERRUPTION_OVERLAP_SEC,
) -> dict[str, Any]:
    turns_source = list(speaker_turns)
    if not turns_source:
        turns_source = _fallback_speaker_turns_from_transcript(transcript_payload)

    merged_turns = merge_speaker_turns(turns_source, gap_threshold=gap_threshold_sec)
    interruption_stats = count_interruptions(
        merged_turns,
        overlap_threshold=interruption_overlap_sec,
    )

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in merged_turns:
        speaker = str(row.get("speaker") or "S1")
        by_speaker.setdefault(speaker, []).append(row)

    speakers = sorted(by_speaker)
    airtime_seconds: dict[str, float] = {}
    for speaker in speakers:
        total = sum(
            max(0.0, _safe_float(row.get("end")) - _safe_float(row.get("start")))
            for row in by_speaker[speaker]
        )
        airtime_seconds[speaker] = round(total, 3)

    total_speech_time = round(sum(airtime_seconds.values()), 3)
    question_counts, total_questions, questions_source, question_types = _question_counts_by_speaker(
        merged_turns,
        summary_payload,
    )
    signal_counts = _speaker_signal_counts(merged_turns)
    terminology_density = _terminology_density_by_speaker(
        merged_turns,
        domain_terms=_domain_terms_from_summary(summary_payload),
    )

    turns_per_speaker = {speaker: len(by_speaker[speaker]) for speaker in speakers}
    avg_turns = sum(turns_per_speaker.values()) / float(max(len(speakers), 1))

    avg_turn_duration_by_speaker: dict[str, float] = {}
    for speaker in speakers:
        turns = max(turns_per_speaker.get(speaker, 0), 1)
        avg_turn_duration_by_speaker[speaker] = round(
            airtime_seconds.get(speaker, 0.0) / float(turns),
            3,
        )
    avg_turn_duration = sum(avg_turn_duration_by_speaker.values()) / float(max(len(speakers), 1))

    participants: list[dict[str, Any]] = []
    for speaker in speakers:
        speaker_airtime = airtime_seconds.get(speaker, 0.0)
        share = 0.0
        if total_speech_time > 0:
            share = speaker_airtime / total_speech_time

        participant = {
            "speaker": speaker,
            "airtime_seconds": round(speaker_airtime, 3),
            "airtime_share": round(share, 4),
            "turns": int(turns_per_speaker.get(speaker, 0)),
            "interruptions_done": int(interruption_stats["done"].get(speaker, 0)),
            "interruptions_received": int(interruption_stats["received"].get(speaker, 0)),
            "questions_count": int(question_counts.get(speaker, 0)),
        }
        participant["role_hint"] = _infer_role_hint(
            airtime_share=participant["airtime_share"],
            turns=participant["turns"],
            questions_count=participant["questions_count"],
            decision_task_signals=signal_counts.get(speaker, 0),
            avg_turn_duration_sec=avg_turn_duration_by_speaker.get(speaker, 0.0),
            terminology_density=terminology_density.get(speaker, 0.0),
            avg_turns=avg_turns,
            avg_turn_duration=avg_turn_duration,
            meeting_questions=total_questions,
        )
        participants.append(participant)

    participants.sort(key=lambda row: (-row["airtime_seconds"], row["speaker"]))

    decisions = _normalise_text_items(summary_payload.get("decisions"), max_items=100)
    action_items = _normalise_action_items(summary_payload.get("action_items"))
    emotional_summary = str(summary_payload.get("emotional_summary") or "").strip() or "—"

    meeting = {
        "participants_count": len(participants),
        "total_speech_time_seconds": total_speech_time,
        "total_interruptions": int(interruption_stats["total"]),
        "total_questions": int(total_questions),
        "question_types": question_types,
        "decisions_count": len(decisions),
        "action_items_count": len(action_items),
        "actionability_ratio": compute_actionability_ratio(action_items),
        "emotional_summary": emotional_summary,
        "questions_source": questions_source,
    }

    return {
        "version": 1,
        "config": {
            "gap_threshold_sec": float(gap_threshold_sec),
            "interruption_overlap_sec": float(interruption_overlap_sec),
        },
        "meeting": meeting,
        "participants": participants,
    }


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def refresh_recording_metrics(
    recording_id: str,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    """Compute and persist meeting+participant metrics for a recording."""
    derived = settings.recordings_root / recording_id / "derived"
    transcript_path = derived / "transcript.json"
    summary_path = derived / "summary.json"
    speaker_turns_path = derived / "speaker_turns.json"
    metrics_path = derived / "metrics.json"

    transcript_payload = _load_json_dict(transcript_path)
    summary_payload = _load_json_dict(summary_path)
    speaker_turns = _load_json_list(speaker_turns_path)

    computed = build_conversation_metrics(
        transcript_payload=transcript_payload,
        summary_payload=summary_payload,
        speaker_turns=speaker_turns,
    )

    existing_metrics = _load_json_dict(metrics_path)
    merged_metrics = dict(existing_metrics)
    merged_metrics["conversation_version"] = computed["version"]
    merged_metrics["conversation_config"] = computed["config"]
    merged_metrics["meeting"] = computed["meeting"]
    merged_metrics["participants"] = computed["participants"]
    atomic_write_json(metrics_path, merged_metrics)

    upsert_meeting_metrics(
        recording_id=recording_id,
        payload=computed["meeting"],
        settings=settings,
    )
    replace_participant_metrics(
        recording_id=recording_id,
        rows=[
            {
                "voice_profile_id": None,
                "diar_speaker_label": str(row.get("speaker") or "S1"),
                "payload": row,
            }
            for row in computed["participants"]
        ],
        settings=settings,
    )
    return computed


__all__ = [
    "DEFAULT_GAP_THRESHOLD_SEC",
    "DEFAULT_INTERRUPTION_OVERLAP_SEC",
    "merge_speaker_turns",
    "count_interruptions",
    "compute_actionability_ratio",
    "build_conversation_metrics",
    "refresh_recording_metrics",
]
