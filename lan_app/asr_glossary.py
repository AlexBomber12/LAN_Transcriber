from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from .config import AppSettings
from .db import (
    get_recording,
    list_glossary_entries,
    list_project_keyword_weights,
    list_voice_profiles,
)

_MAX_GLOSSARY_ENTRIES = 24
_MAX_GLOSSARY_TERMS = 48
_MAX_GLOSSARY_TERM_CHARS = 480
_MAX_HOTWORDS_CHARS = 640
_MAX_PROMPT_CHARS = 720
_SOURCE_PRIORITY = {
    "manual": 0,
    "correction": 1,
    "project": 2,
    "calendar": 3,
    "speaker_bank": 4,
    "system": 5,
}


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _term_key(value: object) -> str:
    return _clean_text(value).lower()


def _clean_terms(values: Sequence[object] | None) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        text = _clean_text(value)
        key = _term_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _calendar_attendee_name(value: object) -> str:
    text = _clean_text(value)
    if text.lower().startswith("mailto:"):
        text = _clean_text(text.split(":", 1)[1])
    if "<" in text:
        prefix = _clean_text(text.split("<", 1)[0])
        if prefix:
            return prefix
    return text


def _source_rank(source: object) -> int:
    return _SOURCE_PRIORITY.get(_term_key(source), 99)


def _maybe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _manual_source_entries(settings: AppSettings) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list_glossary_entries(enabled=True, settings=settings):
        canonical = _clean_text(row.get("canonical_text"))
        if not canonical:
            continue
        metadata = row.get("metadata_json")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        detail: dict[str, Any] = {
            "source": _term_key(row.get("source")) or "manual",
            "entry_id": _maybe_int(row.get("id")),
        }
        recording_id = _clean_text(metadata_dict.get("recording_id"))
        notes = _clean_text(row.get("notes"))
        if recording_id:
            detail["recording_id"] = recording_id
        if notes:
            detail["notes"] = notes
        out.append(
            {
                "canonical_text": canonical,
                "aliases": row.get("aliases_json")
                if isinstance(row.get("aliases_json"), list)
                else [],
                "kind": _term_key(row.get("kind")) or "term",
                "source": detail["source"],
                "detail": detail,
            }
        )
    return out


def _speaker_source_entries(settings: AppSettings) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list_voice_profiles(settings=settings):
        canonical = _clean_text(row.get("display_name"))
        if not canonical:
            continue
        out.append(
            {
                "canonical_text": canonical,
                "aliases": [],
                "kind": "person",
                "source": "speaker_bank",
                "detail": {
                    "source": "speaker_bank",
                    "voice_profile_id": _maybe_int(row.get("id")),
                },
            }
        )
    return out


def _calendar_source_entries(
    recording_id: str,
    *,
    calendar_title: str | None,
    calendar_attendees: Sequence[str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    title = _clean_text(calendar_title)
    if title:
        out.append(
            {
                "canonical_text": title,
                "aliases": [],
                "kind": "project",
                "source": "calendar",
                "detail": {
                    "source": "calendar",
                    "recording_id": recording_id,
                    "field": "title",
                },
            }
        )
    for attendee in calendar_attendees or ():
        name = _calendar_attendee_name(attendee)
        if not name:
            continue
        out.append(
            {
                "canonical_text": name,
                "aliases": [],
                "kind": "person",
                "source": "calendar",
                "detail": {
                    "source": "calendar",
                    "recording_id": recording_id,
                    "field": "attendee",
                },
            }
        )
    return out


def _project_source_entries(
    recording_id: str,
    *,
    project_id: int | None,
    project_name: str | None,
    settings: AppSettings,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    clean_project_name = _clean_text(project_name)
    if clean_project_name:
        out.append(
            {
                "canonical_text": clean_project_name,
                "aliases": [],
                "kind": "project",
                "source": "project",
                "detail": {
                    "source": "project",
                    "project_id": project_id,
                    "recording_id": recording_id,
                    "field": "project_name",
                },
            }
        )
    if project_id is None:
        return out

    keyword_rows: list[tuple[float, str]] = []
    for row in list_project_keyword_weights(project_id=project_id, settings=settings):
        keyword = _clean_text(row.get("keyword"))
        if not keyword or _term_key(keyword) == _term_key(clean_project_name):
            continue
        try:
            weight = float(row.get("weight"))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        keyword_rows.append((weight, keyword))
    keyword_rows.sort(key=lambda item: (-item[0], item[1].lower(), item[1]))

    for _weight, keyword in keyword_rows[:5]:
        out.append(
            {
                "canonical_text": keyword,
                "aliases": [],
                "kind": "term",
                "source": "project",
                "detail": {
                    "source": "project",
                    "project_id": project_id,
                    "recording_id": recording_id,
                    "field": "keyword",
                },
            }
        )
    return out


def _merge_source_entries(source_entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in source_entries:
        canonical = _clean_text(row.get("canonical_text"))
        if not canonical:
            continue
        key = _term_key(canonical)
        aliases = _clean_terms(
            row.get("aliases") if isinstance(row.get("aliases"), Sequence) else []
        )
        source = _term_key(row.get("source")) or "system"
        kind = _term_key(row.get("kind")) or "term"
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        bucket = merged.setdefault(
            key,
            {
                "canonical_text": canonical,
                "kind": kind,
                "sources": [],
                "source_details": [],
                "_aliases": {},
                "_sort_key": (_source_rank(source), key, canonical),
            },
        )
        candidate_sort_key = (_source_rank(source), _term_key(canonical), canonical)
        if candidate_sort_key < bucket["_sort_key"]:
            bucket["canonical_text"] = canonical
            bucket["kind"] = kind
            bucket["_sort_key"] = candidate_sort_key
        if source not in bucket["sources"]:
            bucket["sources"].append(source)

        serialised_detail = {"source": source}
        for field in (
            "entry_id",
            "voice_profile_id",
            "project_id",
            "recording_id",
            "field",
            "notes",
        ):
            value = detail.get(field)
            if value in (None, "", [], {}):
                continue
            serialised_detail[field] = value
        if serialised_detail not in bucket["source_details"]:
            bucket["source_details"].append(serialised_detail)

        for alias in aliases:
            alias_key = _term_key(alias)
            if alias_key == key:
                continue
            bucket["_aliases"].setdefault(alias_key, alias)

    candidates: list[dict[str, Any]] = []
    for bucket in merged.values():
        sources = sorted(bucket["sources"], key=lambda item: (_source_rank(item), item))
        source_details = sorted(
            bucket["source_details"],
            key=lambda item: (
                _source_rank(item.get("source")),
                str(item.get("source") or ""),
                str(item.get("field") or ""),
                str(
                    item.get("entry_id")
                    or item.get("voice_profile_id")
                    or item.get("project_id")
                    or item.get("recording_id")
                    or ""
                ),
            ),
        )
        aliases = [
            bucket["_aliases"][alias_key]
            for alias_key in sorted(bucket["_aliases"])
        ]
        candidates.append(
            {
                "canonical_text": bucket["canonical_text"],
                "aliases": aliases,
                "kind": bucket["kind"],
                "sources": sources,
                "source_details": source_details,
                "_sort_key": bucket["_sort_key"],
            }
        )
    candidates.sort(key=lambda item: item["_sort_key"])
    for item in candidates:
        item.pop("_sort_key", None)
    return candidates


def _fit_entries(
    candidates: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], int, bool]:
    selected_entries: list[dict[str, Any]] = []
    selected_terms: list[str] = []
    seen_terms: set[str] = set()
    term_chars = 0
    truncated = False

    for candidate in candidates:
        if len(selected_entries) >= _MAX_GLOSSARY_ENTRIES:
            truncated = True
            break

        canonical = _clean_text(candidate.get("canonical_text"))
        canonical_key = _term_key(canonical)
        if not canonical or canonical_key in seen_terms:
            continue
        if len(selected_terms) >= _MAX_GLOSSARY_TERMS:
            truncated = True
            break
        if term_chars + len(canonical) > _MAX_GLOSSARY_TERM_CHARS:
            truncated = True
            continue

        selected_aliases: list[str] = []
        seen_terms.add(canonical_key)
        selected_terms.append(canonical)
        term_chars += len(canonical)

        for alias in candidate.get("aliases", []):
            alias_text = _clean_text(alias)
            alias_key = _term_key(alias_text)
            if not alias_text or alias_key in seen_terms:
                continue
            if len(selected_terms) >= _MAX_GLOSSARY_TERMS:
                truncated = True
                break
            if term_chars + len(alias_text) > _MAX_GLOSSARY_TERM_CHARS:
                truncated = True
                continue
            seen_terms.add(alias_key)
            selected_terms.append(alias_text)
            selected_aliases.append(alias_text)
            term_chars += len(alias_text)

        selected_entries.append(
            {
                "canonical_text": canonical,
                "aliases": selected_aliases,
                "kind": _term_key(candidate.get("kind")) or "term",
                "sources": list(candidate.get("sources") or []),
                "source_details": list(candidate.get("source_details") or []),
                "terms": [canonical, *selected_aliases],
                "term_count": 1 + len(selected_aliases),
            }
        )

    return selected_entries, selected_terms, term_chars, truncated


def _limited_join(
    terms: Sequence[str],
    *,
    separator: str,
    prefix: str,
    max_chars: int,
) -> tuple[str | None, list[str]]:
    if max_chars <= len(prefix):
        return None, []
    parts: list[str] = []
    used_terms: list[str] = []
    total_chars = len(prefix)
    for term in terms:
        extra_chars = len(term) + (len(separator) if parts else 0)
        if total_chars + extra_chars > max_chars:
            break
        parts.append(term)
        used_terms.append(term)
        total_chars += extra_chars
    if not parts:
        return None, []
    return f"{prefix}{separator.join(parts)}", used_terms


def build_recording_asr_glossary(
    recording_id: str,
    *,
    calendar_title: str | None = None,
    calendar_attendees: Sequence[str] | None = None,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    recording = get_recording(recording_id, settings=cfg) or {}
    project_name = _clean_text(recording.get("project_name"))
    project_id = _maybe_int(recording.get("project_id"))

    candidates = _merge_source_entries(
        [
            *_manual_source_entries(cfg),
            *_speaker_source_entries(cfg),
            *_calendar_source_entries(
                recording_id,
                calendar_title=calendar_title,
                calendar_attendees=calendar_attendees,
            ),
            *_project_source_entries(
                recording_id,
                project_id=project_id,
                project_name=project_name,
                settings=cfg,
            ),
        ]
    )
    selected_entries, selected_terms, selected_term_chars, truncated = _fit_entries(
        candidates
    )
    hotwords, hotword_terms = _limited_join(
        selected_terms,
        separator=", ",
        prefix="",
        max_chars=_MAX_HOTWORDS_CHARS,
    )
    initial_prompt, prompt_terms = _limited_join(
        selected_terms,
        separator="; ",
        prefix="Glossary: ",
        max_chars=_MAX_PROMPT_CHARS,
    )
    if len(hotword_terms) < len(selected_terms) or len(prompt_terms) < len(selected_terms):
        truncated = True

    source_counts: dict[str, int] = {}
    for entry in selected_entries:
        for source in entry.get("sources", []):
            source_key = _term_key(source)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

    return {
        "version": 1,
        "recording_id": _clean_text(recording_id) or recording_id,
        "generated_at": _utc_now(),
        "entries": selected_entries,
        "terms": selected_terms,
        "entry_count": len(selected_entries),
        "term_count": len(selected_terms),
        "term_chars": selected_term_chars,
        "truncated": bool(truncated),
        "source_counts": dict(sorted(source_counts.items())),
        "budgets": {
            "max_entries": _MAX_GLOSSARY_ENTRIES,
            "max_terms": _MAX_GLOSSARY_TERMS,
            "max_term_chars": _MAX_GLOSSARY_TERM_CHARS,
            "max_hotwords_chars": _MAX_HOTWORDS_CHARS,
            "max_prompt_chars": _MAX_PROMPT_CHARS,
        },
        "hotwords": hotwords,
        "hotword_terms": hotword_terms,
        "initial_prompt": initial_prompt,
        "prompt_terms": prompt_terms,
    }


__all__ = ["build_recording_asr_glossary"]
