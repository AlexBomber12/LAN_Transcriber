from __future__ import annotations

from pathlib import Path

import pytest

from lan_app.config import AppSettings
from lan_app.db import (
    create_recording,
    create_voice_profile,
    init_db,
    list_speaker_assignments,
    list_voice_samples,
)
from lan_app import speaker_bank


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def test_resolve_one_to_one_assignments_avoids_double_match():
    decisions = speaker_bank.resolve_one_to_one_assignments(
        [
            {
                "diar_speaker_label": "S1",
                "candidate_matches": [
                    {"voice_profile_id": 1, "score": 0.96, "display_name": "Alice"},
                    {"voice_profile_id": 2, "score": 0.61},
                ],
            },
            "skip-me",
            {
                "speaker": "S2",
                "candidate_matches": [
                    {"voice_profile_id": 1, "score": 0.94},
                    {"voice_profile_id": 2, "score": 0.88},
                ],
            },
        ],
        min_confidence=0.75,
    )

    assert decisions == [
        {
            "diar_speaker_label": "S1",
            "voice_profile_id": 1,
            "confidence": 0.96,
            "low_confidence": False,
            "candidate_matches": [
                {"voice_profile_id": 1, "score": 0.96, "display_name": "Alice"},
                {"voice_profile_id": 2, "score": 0.61},
            ],
        },
        {
            "diar_speaker_label": "S2",
            "voice_profile_id": 2,
            "confidence": 0.88,
            "low_confidence": False,
            "candidate_matches": [
                {"voice_profile_id": 1, "score": 0.94},
                {"voice_profile_id": 2, "score": 0.88},
            ],
        },
    ]


def test_speaker_bank_helper_paths():
    assert speaker_bank._clamp_score("bad-score") == 0.0  # noqa: SLF001
    assert speaker_bank._normalise_candidate_matches(  # noqa: SLF001
        ["skip", {"voice_profile_id": "4", "score": "bad"}]
    ) == [{"voice_profile_id": 4, "score": 0.0}]
    assert speaker_bank._assignment_tiebreak((1, None, 2)) == (  # noqa: SLF001
        1,
        1_000_000_000,
        2,
    )
    assert speaker_bank._is_better_result(  # noqa: SLF001
        1.0,
        2,
        (1, None),
        1.0,
        1,
        (1, 2),
    )
    assert speaker_bank._is_better_result(  # noqa: SLF001
        1.0,
        1,
        (1, None),
        1.0,
        1,
        (2, None),
    )


def test_resolve_one_to_one_assignments_keeps_low_confidence_reviewable():
    assert speaker_bank.resolve_one_to_one_assignments([]) == []

    decisions = speaker_bank.resolve_one_to_one_assignments(
        [
            {
                "diar_speaker_label": "S1",
                "candidate_matches": [
                    {"voice_profile_id": 7, "score": 0.61},
                    {"voice_profile_id": "bad"},
                    {"voice_profile_id": 7, "score": 0.6},
                ],
            }
        ],
        min_confidence=0.75,
    )

    assert decisions == [
        {
            "diar_speaker_label": "S1",
            "voice_profile_id": None,
            "confidence": 0.61,
            "low_confidence": True,
            "candidate_matches": [{"voice_profile_id": 7, "score": 0.61}],
        }
    ]

    with pytest.raises(ValueError, match="diar_speaker_label is required"):
        speaker_bank.resolve_one_to_one_assignments(
            [{"diar_speaker_label": " ", "candidate_matches": []}]
        )


def test_assign_speakers_to_recording_persists_matches_and_skips_empty_rows(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-speaker-bank-1",
        source="upload",
        source_filename="speaker-bank.wav",
        settings=cfg,
    )
    left = create_voice_profile("Left Voice", settings=cfg)
    right = create_voice_profile("Right Voice", settings=cfg)

    persisted = speaker_bank.assign_speakers_to_recording(
        "rec-speaker-bank-1",
        [
            {
                "diar_speaker_label": "S1",
                "candidate_matches": [
                    {"voice_profile_id": left["id"], "score": 0.93},
                    {"voice_profile_id": right["id"], "score": 0.52},
                ],
            },
            {
                "diar_speaker_label": "S2",
                "candidate_matches": [
                    {"voice_profile_id": left["id"], "score": 0.9},
                    {"voice_profile_id": right["id"], "score": 0.87},
                ],
            },
            {"diar_speaker_label": "S3", "candidate_matches": []},
        ],
        min_confidence=0.75,
        settings=cfg,
    )

    assert persisted[0]["voice_profile_id"] == left["id"]
    assert persisted[1]["voice_profile_id"] == right["id"]
    assert persisted[2]["recording_id"] == "rec-speaker-bank-1"
    assert persisted[2]["voice_profile_id"] is None
    assert persisted[2]["voice_profile_name"] == ""

    stored_rows = list_speaker_assignments("rec-speaker-bank-1", settings=cfg)
    assert len(stored_rows) == 2
    assert [row["voice_profile_id"] for row in stored_rows] == [left["id"], right["id"]]
    assert {row["review_state"] for row in stored_rows} == {"system_suggested"}


def test_register_voice_sample_attaches_or_defers_based_on_confidence(tmp_path: Path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-speaker-bank-sample-1",
        source="upload",
        source_filename="speaker-sample.wav",
        settings=cfg,
    )
    profile = create_voice_profile("Attached Voice", settings=cfg)

    attached = speaker_bank.register_voice_sample(
        snippet_path="recordings/rec-speaker-bank-sample-1/derived/snippets/S1/1.wav",
        candidate_matches=[
            {"voice_profile_id": profile["id"], "score": 0.92},
            {"voice_profile_id": profile["id"], "score": 0.91},
        ],
        attach_threshold=0.85,
        recording_id="rec-speaker-bank-sample-1",
        diar_speaker_label="S1",
        sample_start_sec=1.0,
        sample_end_sec=2.0,
        embedding=[0.2, 0.3],
        settings=cfg,
    )
    assert attached["voice_profile_id"] == profile["id"]
    assert attached["needs_review"] == 0
    assert attached["sample_source"] == "speaker-bank"

    review_only = speaker_bank.register_voice_sample(
        snippet_path="recordings/rec-speaker-bank-sample-1/derived/snippets/S2/1.wav",
        candidate_matches=[{"voice_profile_id": profile["id"], "score": 0.51}],
        attach_threshold=0.85,
        recording_id="rec-speaker-bank-sample-1",
        diar_speaker_label="S2",
        settings=cfg,
    )
    assert review_only["voice_profile_id"] is None
    assert review_only["needs_review"] == 1
    assert review_only["candidate_matches_json"] == [
        {"voice_profile_id": profile["id"], "score": 0.51}
    ]

    samples = list_voice_samples(settings=cfg)
    assert len(samples) == 2


def test_merge_canonical_speakers_wrapper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        speaker_bank,
        "merge_voice_profiles",
        lambda source_profile_id, target_profile_id, settings=None: {
            "source": source_profile_id,
            "target": target_profile_id,
            "settings": settings,
        },
    )

    merged = speaker_bank.merge_canonical_speakers(1, 2, settings="cfg")
    assert merged == {"source": 1, "target": 2, "settings": "cfg"}
