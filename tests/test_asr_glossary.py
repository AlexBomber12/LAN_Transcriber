from __future__ import annotations

from pathlib import Path

import pytest

from lan_app import asr_glossary
from lan_app.config import AppSettings
from lan_app.db import (
    create_glossary_entry,
    create_project,
    create_recording,
    create_voice_profile,
    increment_project_keyword_weights,
    init_db,
    set_recording_project,
)


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_build_recording_asr_glossary_merges_manual_speaker_calendar_and_project_sources(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-glossary-build-1",
        source="upload",
        source_filename="meeting.wav",
        settings=cfg,
    )
    project = create_project("Apollo Launch", settings=cfg)
    set_recording_project("rec-glossary-build-1", int(project["id"]), settings=cfg)
    increment_project_keyword_weights(
        project_id=int(project["id"]),
        keyword_deltas={"roadmap": 2.0, "launchpad": 1.5},
        settings=cfg,
    )
    create_glossary_entry(
        "Sander",
        aliases=["Sandia", " sandia "],
        term_kind="person",
        source="correction",
        settings=cfg,
    )
    create_glossary_entry(
        "Quarterly roadmap",
        aliases=["Q roadmap"],
        term_kind="project",
        source="manual",
        settings=cfg,
    )
    create_voice_profile("Priya Kapoor", settings=cfg)
    create_voice_profile("Alex Example", settings=cfg)

    payload = asr_glossary.build_recording_asr_glossary(
        "rec-glossary-build-1",
        calendar_title="Quarterly Roadmap",
        calendar_attendees=["Priya Kapoor", "Alex Example <alex@example.com>"],
        settings=cfg,
    )

    entry_order = [
        (row["sources"][0], row["canonical_text"])
        for row in payload["entries"]
    ]
    assert entry_order == sorted(
        entry_order,
        key=lambda item: (
            asr_glossary._source_rank(item[0]),  # noqa: SLF001
            item[1].lower(),
            item[1],
        ),
    )
    sander_entry = next(
        row for row in payload["entries"] if row["canonical_text"] == "Sander"
    )
    assert sander_entry["aliases"] == ["Sandia"]
    assert sander_entry["sources"] == ["correction"]
    assert "Priya Kapoor" in payload["terms"]
    assert "Alex Example" in payload["terms"]
    assert "Apollo Launch" in payload["terms"]
    assert "roadmap" in payload["terms"]
    assert payload["hotwords"]
    assert payload["initial_prompt"]
    assert payload["source_counts"]["correction"] >= 1
    assert payload["source_counts"]["speaker_bank"] >= 1
    assert payload["source_counts"]["calendar"] >= 1
    assert payload["source_counts"]["project"] >= 1


def test_build_recording_asr_glossary_deduplicates_aliases_and_enforces_budgets(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-glossary-build-2",
        source="upload",
        source_filename="meeting.wav",
        settings=cfg,
    )

    for index in range(80):
        create_glossary_entry(
            f"Term {index:02d}",
            aliases=[f"Alias {index:02d}", f"alias {index:02d}"],
            term_kind="term",
            source="manual",
            settings=cfg,
        )

    payload = asr_glossary.build_recording_asr_glossary(
        "rec-glossary-build-2",
        settings=cfg,
    )

    assert payload["truncated"] is True
    assert payload["entry_count"] <= payload["budgets"]["max_entries"]
    assert payload["term_count"] <= payload["budgets"]["max_terms"]
    assert payload["term_chars"] <= payload["budgets"]["max_term_chars"]
    assert len(payload["terms"]) == len({term.lower() for term in payload["terms"]})
    assert all(
        len(entry["terms"]) == entry["term_count"] for entry in payload["entries"]
    )


def test_asr_glossary_private_helpers_cover_edge_paths(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    assert asr_glossary._clean_terms(["  Alex  ", "alex", "", None]) == ["Alex"]  # noqa: SLF001
    assert asr_glossary._calendar_attendee_name("mailto:alex@example.com") == "alex@example.com"  # noqa: SLF001
    assert asr_glossary._calendar_attendee_name("Alex <alex@example.com>") == "Alex"  # noqa: SLF001
    assert asr_glossary._calendar_attendee_name("<alex@example.com>") == "<alex@example.com>"  # noqa: SLF001
    assert asr_glossary._source_rank("unknown") == 99  # noqa: SLF001
    assert asr_glossary._maybe_int("bad") is None  # noqa: SLF001
    assert asr_glossary._limited_join(  # noqa: SLF001
        ["Term A", "Term B"],
        separator=", ",
        prefix="",
        max_chars=0,
    ) == (None, [])
    assert asr_glossary._project_source_entries(  # noqa: SLF001
        "rec-glossary-build-3",
        project_id=None,
        project_name="",
        settings=cfg,
    ) == []
    assert asr_glossary._merge_source_entries(  # noqa: SLF001
        [
            {"canonical_text": " ", "aliases": [], "kind": "term", "source": "manual"},
            {
                "canonical_text": "Sander",
                "aliases": ["Sandia"],
                "kind": "person",
                "source": "manual",
                "detail": {"source": "manual", "entry_id": 1},
            },
            {
                "canonical_text": "sander",
                "aliases": ["Sandor"],
                "kind": "person",
                "source": "correction",
                "detail": {"source": "correction", "entry_id": 2},
            },
        ]
    )[0]["aliases"] == ["Sandia", "Sandor"]


def test_asr_glossary_source_helpers_cover_branch_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)

    monkeypatch.setattr(
        asr_glossary,
        "list_glossary_entries",
        lambda **_kwargs: [
            {"canonical_text": " ", "aliases_json": [], "kind": "term", "source": "manual"},
            {
                "id": "x",
                "canonical_text": "Sander",
                "aliases_json": "bad",
                "kind": "person",
                "source": "manual",
                "metadata_json": {"recording_id": "rec-1"},
                "notes": "remember this",
            },
        ],
    )
    manual_entries = asr_glossary._manual_source_entries(cfg)  # noqa: SLF001
    assert manual_entries == [
        {
            "canonical_text": "Sander",
            "aliases": [],
            "kind": "person",
            "source": "manual",
            "detail": {
                "source": "manual",
                "entry_id": None,
                "recording_id": "rec-1",
                "notes": "remember this",
            },
        }
    ]

    monkeypatch.setattr(
        asr_glossary,
        "list_voice_profiles",
        lambda **_kwargs: [
            {"display_name": " ", "id": 1},
            {"display_name": "Alex Example", "id": "bad"},
        ],
    )
    assert asr_glossary._speaker_source_entries(cfg) == [  # noqa: SLF001
        {
            "canonical_text": "Alex Example",
            "aliases": [],
            "kind": "person",
            "source": "speaker_bank",
            "detail": {"source": "speaker_bank", "voice_profile_id": None},
        }
    ]

    calendar_entries = asr_glossary._calendar_source_entries(  # noqa: SLF001
        "rec-2",
        calendar_title=None,
        calendar_attendees=[" ", "<alex@example.com>"],
    )
    assert calendar_entries == [
        {
            "canonical_text": "<alex@example.com>",
            "aliases": [],
            "kind": "person",
            "source": "calendar",
            "detail": {
                "source": "calendar",
                "recording_id": "rec-2",
                "field": "attendee",
            },
        }
    ]

    monkeypatch.setattr(
        asr_glossary,
        "list_project_keyword_weights",
        lambda **_kwargs: [
            {"keyword": "Roadmap", "weight": 2.0},
            {"keyword": "broken", "weight": "bad"},
            {"keyword": "ignored", "weight": 0.0},
            {"keyword": "apollo", "weight": 1.5},
        ],
    )
    assert asr_glossary._project_source_entries(  # noqa: SLF001
        "rec-3",
        project_id=7,
        project_name="Roadmap",
        settings=cfg,
    ) == [
        {
            "canonical_text": "Roadmap",
            "aliases": [],
            "kind": "project",
            "source": "project",
            "detail": {
                "source": "project",
                "project_id": 7,
                "recording_id": "rec-3",
                "field": "project_name",
            },
        },
        {
            "canonical_text": "apollo",
            "aliases": [],
            "kind": "term",
            "source": "project",
            "detail": {
                "source": "project",
                "project_id": 7,
                "recording_id": "rec-3",
                "field": "keyword",
            },
        },
    ]


def test_asr_glossary_merge_fit_join_and_truncation_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    merged = asr_glossary._merge_source_entries(  # noqa: SLF001
        [
            {
                "canonical_text": "Sander",
                "aliases": ["Sander", "Sandia"],
                "kind": "person",
                "source": "manual",
                "detail": {"source": "manual", "entry_id": 1},
            },
            {
                "canonical_text": "Sander",
                "aliases": ["Sandia"],
                "kind": "person",
                "source": "manual",
                "detail": {"source": "manual", "entry_id": 1},
            },
        ]
    )
    assert merged == [
        {
            "canonical_text": "Sander",
            "aliases": ["Sandia"],
            "kind": "person",
            "sources": ["manual"],
            "source_details": [{"source": "manual", "entry_id": 1}],
        }
    ]

    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_ENTRIES", 1)
    selected_entries, selected_terms, _term_chars, truncated = asr_glossary._fit_entries(  # noqa: SLF001
        [
            {
                "canonical_text": "Term A",
                "aliases": [],
                "kind": "term",
                "sources": ["manual"],
                "source_details": [],
            },
            {
                "canonical_text": "Term B",
                "aliases": [],
                "kind": "term",
                "sources": ["manual"],
                "source_details": [],
            },
        ]
    )
    assert truncated is True
    assert selected_terms == ["Term A"]
    assert selected_entries[0]["canonical_text"] == "Term A"

    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_ENTRIES", 24)
    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_TERMS", 0)
    assert asr_glossary._fit_entries(  # noqa: SLF001
        [
            {
                "canonical_text": "Term A",
                "aliases": [],
                "kind": "term",
                "sources": ["manual"],
                "source_details": [],
            }
        ]
    ) == ([], [], 0, True)

    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_TERMS", 48)
    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_TERM_CHARS", 2)
    selected_entries, selected_terms, _term_chars, truncated = asr_glossary._fit_entries(  # noqa: SLF001
        [
            {
                "canonical_text": "Long",
                "aliases": [],
                "kind": "term",
                "sources": ["manual"],
                "source_details": [],
            }
        ]
    )
    assert selected_entries == []
    assert selected_terms == []
    assert truncated is True

    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_TERM_CHARS", 20)
    selected_entries, selected_terms, _term_chars, truncated = asr_glossary._fit_entries(  # noqa: SLF001
        [
            {
                "canonical_text": "Term A",
                "aliases": ["", "Term A", "Alias Too Long For Budget", "B"],
                "kind": "term",
                "sources": ["manual"],
                "source_details": [],
            },
            {
                "canonical_text": "Term A",
                "aliases": [],
                "kind": "term",
                "sources": ["manual"],
                "source_details": [],
            },
        ]
    )
    assert selected_entries[0]["aliases"] == ["B"]
    assert selected_terms == ["Term A", "B"]
    assert truncated is True

    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_TERM_CHARS", 480)
    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_TERMS", 1)
    selected_entries, selected_terms, _term_chars, truncated = asr_glossary._fit_entries(  # noqa: SLF001
        [
            {
                "canonical_text": "Term A",
                "aliases": ["Alias"],
                "kind": "term",
                "sources": ["manual"],
                "source_details": [],
            }
        ]
    )
    assert selected_entries[0]["aliases"] == []
    assert selected_terms == ["Term A"]
    assert truncated is True

    joined, used_terms = asr_glossary._limited_join(  # noqa: SLF001
        ["Term A", "Term B"],
        separator=", ",
        prefix="",
        max_chars=8,
    )
    assert joined == "Term A"
    assert used_terms == ["Term A"]

    monkeypatch.setattr(asr_glossary, "_MAX_GLOSSARY_TERMS", 48)
    monkeypatch.setattr(asr_glossary, "_MAX_HOTWORDS_CHARS", 6)
    monkeypatch.setattr(asr_glossary, "_MAX_PROMPT_CHARS", 12)
    create_recording(
        "rec-glossary-build-4",
        source="upload",
        source_filename="meeting.wav",
        settings=cfg,
    )
    create_glossary_entry(
        "Sander",
        aliases=["Sandia"],
        settings=cfg,
    )
    payload = asr_glossary.build_recording_asr_glossary(
        "rec-glossary-build-4",
        settings=cfg,
    )
    assert payload["truncated"] is True
    assert payload["hotword_terms"] == ["Sander"]
    assert payload["prompt_terms"] == []
