"""Tests for the speaker_merge post-diarization pass."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import pytest

from lan_transcriber.pipeline_steps import orchestrator as pipeline_orchestrator
from lan_transcriber.pipeline_steps.speaker_merge import (
    DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD,
    DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
    compute_pairwise_similarity,
    extract_speaker_embeddings,
    merge_similar_speakers,
    speakers_overlap,
)


_AUDIO_PATH = Path("/tmp/fake-audio.wav")


def _segment(speaker: str, start: float, end: float) -> dict[str, object]:
    return {"speaker": speaker, "start": start, "end": end}


def _make_embedding_model(
    embeddings_by_speaker: dict[str, Sequence[float]],
    segments_by_speaker: dict[str, Sequence[tuple[float, float]]],
):
    """Return an embedding model that maps (start, end) back to a speaker.

    Tests carefully set up diarization segments so that each (start, end)
    pair belongs to exactly one speaker in ``segments_by_speaker``. The fake
    model looks up the speaker via the midpoint of the requested window and
    returns the deterministic vector associated with that speaker. When a
    long segment was cropped to its middle window the midpoint still lies
    inside the original segment, so we fall back to the segment that
    contains the midpoint when no exact match exists.
    """

    midpoint_index: dict[float, str] = {}
    ranges: list[tuple[float, float, str]] = []
    for speaker, segments in segments_by_speaker.items():
        for start, end in segments:
            midpoint_index[round((start + end) / 2.0, 6)] = speaker
            ranges.append((start, end, speaker))

    def _model(audio_path: Path, start: float, end: float) -> list[float]:
        assert audio_path == _AUDIO_PATH
        midpoint = round((start + end) / 2.0, 6)
        speaker = midpoint_index.get(midpoint)
        if speaker is None:
            for seg_start, seg_end, seg_speaker in ranges:
                if seg_start <= midpoint <= seg_end:
                    speaker = seg_speaker
                    break
        assert speaker is not None, (start, end)
        return list(embeddings_by_speaker[speaker])

    return _model


def test_compute_pairwise_similarity_returns_sorted_descending() -> None:
    embeddings = {
        "A": [1.0, 0.0],
        "B": [1.0, 0.0],
        "C": [0.0, 1.0],
    }
    pairs = compute_pairwise_similarity(embeddings)
    assert pairs[0] == ("A", "B", pytest.approx(1.0))
    assert pairs[-1][2] == pytest.approx(0.0)
    assert all(a < b for a, b, _ in pairs)


def test_compute_pairwise_similarity_skips_zero_norm_vectors() -> None:
    embeddings = {
        "A": [0.0, 0.0],
        "B": [1.0, 0.0],
        "C": [0.0, 1.0],
    }
    pairs = compute_pairwise_similarity(embeddings)
    speakers_in_pairs = {(left, right) for left, right, _ in pairs}
    # A cannot appear because its norm is 0.
    assert ("B", "C") in speakers_in_pairs
    assert all("A" not in pair for pair in speakers_in_pairs)


def test_compute_pairwise_similarity_skips_shape_mismatch() -> None:
    embeddings = {
        "A": [1.0, 0.0],
        "B": [1.0, 0.0, 0.5],
    }
    assert compute_pairwise_similarity(embeddings) == []


def test_speakers_overlap_detects_overlap() -> None:
    segments = [
        _segment("A", 0.0, 2.0),
        _segment("B", 1.0, 3.0),
    ]
    assert speakers_overlap("A", "B", segments) is True


def test_speakers_overlap_tolerates_jitter() -> None:
    segments = [
        _segment("A", 0.0, 2.05),
        _segment("B", 2.0, 4.0),
    ]
    # With a 0.1s tolerance the 0.05s overlap is absorbed as jitter.
    assert speakers_overlap("A", "B", segments, tolerance_sec=0.1) is False


def test_speakers_overlap_returns_false_for_same_speaker() -> None:
    segments = [_segment("A", 0.0, 2.0)]
    assert speakers_overlap("A", "A", segments) is False


def test_speakers_overlap_ignores_zero_length_segments() -> None:
    segments = [
        _segment("A", 1.0, 1.0),
        _segment("B", 1.0, 1.0),
        _segment("B", 5.0, 6.0),
    ]
    assert speakers_overlap("A", "B", segments) is False


def test_speakers_overlap_skips_zero_length_b_segment() -> None:
    segments = [
        _segment("A", 0.0, 10.0),
        _segment("B", 5.0, 5.0),  # zero-length, must be skipped
        _segment("B", 20.0, 25.0),
    ]
    assert speakers_overlap("A", "B", segments) is False


def test_extract_speaker_embeddings_skips_zero_duration() -> None:
    diar_segments = [
        _segment("A", 0.0, 0.0),
        _segment("B", 1.0, 4.0),
    ]

    def _model(audio_path: Path, start: float, end: float) -> list[float]:
        return [1.0, 1.0]

    centroids = extract_speaker_embeddings(
        _AUDIO_PATH,
        diar_segments,
        embedding_model=_model,
    )
    assert set(centroids) == {"B"}


def test_extract_speaker_embeddings_returns_empty_for_none_model() -> None:
    diar_segments = [_segment("A", 0.0, 3.0)]
    assert extract_speaker_embeddings(
        _AUDIO_PATH,
        diar_segments,
        embedding_model=None,
    ) == {}


def test_extract_speaker_embeddings_skips_nonfinite_and_empty_vectors() -> None:
    diar_segments = [
        _segment("A", 0.0, 4.0),
        _segment("A", 10.0, 14.0),
        _segment("A", 20.0, 24.0),
        _segment("B", 30.0, 34.0),
    ]

    def _model(audio_path: Path, start: float, end: float):
        if start < 10.0:
            return [float("nan"), 0.0]
        if start < 20.0:
            return []
        if start < 30.0:
            return [1.0, 0.0]
        return None

    centroids = extract_speaker_embeddings(
        _AUDIO_PATH,
        diar_segments,
        embedding_model=_model,
    )
    # A keeps the one valid vector; B is dropped because model returned None.
    assert set(centroids) == {"A"}
    assert centroids["A"] == [1.0, 0.0]


def test_extract_speaker_embeddings_crops_long_segment_middle() -> None:
    received_ranges: list[tuple[float, float]] = []

    def _model(audio_path: Path, start: float, end: float) -> list[float]:
        received_ranges.append((start, end))
        return [1.0, 0.0]

    diar_segments = [_segment("A", 0.0, 10.0)]
    extract_speaker_embeddings(
        _AUDIO_PATH,
        diar_segments,
        embedding_model=_model,
        segment_duration_sec=3.0,
    )
    assert received_ranges == [(3.5, 6.5)]


def test_extract_speaker_embeddings_caps_at_max_segments_per_speaker() -> None:
    call_count = {"n": 0}

    def _model(audio_path: Path, start: float, end: float) -> list[float]:
        call_count["n"] += 1
        return [1.0, 0.0]

    diar_segments = [_segment("A", float(i), float(i) + 0.5) for i in range(10)]
    extract_speaker_embeddings(
        _AUDIO_PATH,
        diar_segments,
        embedding_model=_model,
        max_segments_per_speaker=3,
    )
    assert call_count["n"] == 3


def test_extract_speaker_embeddings_drops_dim_mismatch_between_segments() -> None:
    call_count = {"n": 0}

    def _model(audio_path: Path, start: float, end: float):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [1.0, 0.0]
        return [1.0, 0.0, 0.5]

    diar_segments = [
        _segment("A", 0.0, 1.0),
        _segment("A", 2.0, 3.0),
    ]
    centroids = extract_speaker_embeddings(
        _AUDIO_PATH,
        diar_segments,
        embedding_model=_model,
    )
    assert centroids == {"A": [1.0, 0.0]}


def test_extract_speaker_embeddings_ignores_empty_speaker_label() -> None:
    diar_segments = [
        {"speaker": "", "start": 0.0, "end": 3.0},
        _segment("A", 4.0, 7.0),
    ]

    def _model(audio_path: Path, start: float, end: float):
        return [1.0, 0.0]

    centroids = extract_speaker_embeddings(
        _AUDIO_PATH,
        diar_segments,
        embedding_model=_model,
    )
    assert set(centroids) == {"A"}


def test_merge_identical_embeddings_no_overlap() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    embedding = [1.0, 0.0]
    model = _make_embedding_model(
        {"A": embedding, "B": embedding},
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    assert merge_map  # something was merged
    assert len({row["speaker"] for row in updated}) == 1


def test_no_merge_different_embeddings() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    model = _make_embedding_model(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}


def test_no_merge_overlapping_speakers() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 2.0, 7.0),  # overlaps with A
    ]
    embedding = [1.0, 0.0]
    model = _make_embedding_model(
        {"A": embedding, "B": embedding},
        {"A": [(0.0, 5.0)], "B": [(2.0, 7.0)]},
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}


def test_no_merge_below_threshold() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    # similarity 0.75 between A and B, below both thresholds when set equal.
    sim = 0.75
    complement = math.sqrt(1.0 - sim * sim)
    model = _make_embedding_model(
        {
            "A": [1.0, 0.0],
            "B": [sim, complement],
        },
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
        no_overlap_similarity_threshold=DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
    )
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}


def test_three_speakers_two_merge() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
        _segment("C", 12.0, 17.0),
    ]
    model = _make_embedding_model(
        {
            "A": [1.0, 0.0],
            "B": [1.0, 0.0],
            "C": [0.0, 1.0],
        },
        {
            "A": [(0.0, 5.0)],
            "B": [(6.0, 11.0)],
            "C": [(12.0, 17.0)],
        },
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    labels = {row["speaker"] for row in updated}
    assert "C" in labels
    assert len(labels) == 2
    assert len(merge_map) == 1
    merged_label, kept_label = next(iter(merge_map.items()))
    assert merged_label in {"A", "B"}
    assert kept_label in {"A", "B"}
    assert merged_label != kept_label


def test_transitive_merge_collapses_chain() -> None:
    diar_segments = [
        # Make A the biggest speaker so everything collapses to A
        _segment("A", 0.0, 20.0),
        _segment("B", 21.0, 26.0),
        _segment("C", 27.0, 32.0),
    ]
    # A similar to B; B similar to C; no overlaps.
    model = _make_embedding_model(
        {
            "A": [1.0, 0.0],
            "B": [0.95, math.sqrt(1 - 0.95 * 0.95)],
            "C": [0.9025, 0.95 * math.sqrt(1 - 0.95 * 0.95)],
        },
        {
            "A": [(0.0, 20.0)],
            "B": [(21.0, 26.0)],
            "C": [(27.0, 32.0)],
        },
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=0.80,
    )
    assert {row["speaker"] for row in updated} == {"A"}
    assert set(merge_map) == {"B", "C"}
    assert set(merge_map.values()) == {"A"}


def test_merge_disabled_via_none_model() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=None,
    )
    assert merge_map == {}
    assert updated == diar_segments
    # Returned list must be a copy, not the same object.
    assert updated is not diar_segments


def test_merge_ignores_empty_speaker_labels_in_totals() -> None:
    diar_segments = [
        {"speaker": "", "start": 0.0, "end": 5.0},
        _segment("A", 6.0, 11.0),
        _segment("B", 12.0, 17.0),
    ]
    model = _make_embedding_model(
        {"A": [1.0, 0.0], "B": [1.0, 0.0]},
        {"A": [(6.0, 11.0)], "B": [(12.0, 17.0)]},
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    # Speakers A and B merge; the empty-label row is passed through untouched.
    assert merge_map == {"B": "A"}
    labels = [row["speaker"] for row in updated]
    assert "" in labels
    assert labels.count("A") == 2


def test_merge_empty_segments_noop() -> None:
    updated, merge_map, _diag = merge_similar_speakers(
        [],
        audio_path=_AUDIO_PATH,
        embedding_model=_make_embedding_model({}, {}),
    )
    assert updated == []
    assert merge_map == {}


def test_merge_single_speaker_noop() -> None:
    diar_segments = [_segment("A", 0.0, 5.0)]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_make_embedding_model(
            {"A": [1.0, 0.0]}, {"A": [(0.0, 5.0)]}
        ),
    )
    assert merge_map == {}
    assert updated == diar_segments


def test_merge_ties_break_lexicographically() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    model = _make_embedding_model(
        {
            "A": [1.0, 0.0],
            "B": [1.0, 0.0],
        },
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    # Equal totals -> lexicographically smaller label wins.
    assert merge_map == {"B": "A"}
    assert {row["speaker"] for row in updated} == {"A"}


def test_merge_larger_speaker_wins() -> None:
    diar_segments = [
        _segment("A", 0.0, 1.0),
        _segment("B", 2.0, 20.0),
    ]
    model = _make_embedding_model(
        {
            "A": [1.0, 0.0],
            "B": [1.0, 0.0],
        },
        {"A": [(0.0, 1.0)], "B": [(2.0, 20.0)]},
    )
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    assert merge_map == {"A": "B"}
    assert {row["speaker"] for row in updated} == {"B"}


def test_merge_accepts_objects_with_tolist() -> None:
    class _FakeArray:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    def _model(audio_path: Path, start: float, end: float):
        return _FakeArray([1.0, 0.0])

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {"B": "A"}
    assert {row["speaker"] for row in updated} == {"A"}


def test_merge_rejects_unconvertible_values() -> None:
    def _model(audio_path: Path, start: float, end: float):
        return object()

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    # Non-iterable vectors are dropped -> no centroids -> no merge.
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}


def test_merge_rejects_vectors_with_non_numeric_entries() -> None:
    def _model(audio_path: Path, start: float, end: float):
        return [1.0, "bad"]

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}


def test_merge_accepts_two_d_single_row_embeddings() -> None:
    """Pyannote ``Inference(window="whole").crop`` returns shape (1, dim)."""

    def _model(audio_path: Path, start: float, end: float):
        # Simulate a numpy-style (1, 2) array via nested list.
        return [[1.0, 0.0]]

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {"B": "A"}
    assert {row["speaker"] for row in updated} == {"A"}


def test_merge_mean_pools_two_d_multi_row_embeddings() -> None:
    """When inference returns multiple frames we mean-pool along the first axis."""

    def _model(audio_path: Path, start: float, end: float):
        # Two frames each; mean is [1.0, 0.0] so both speakers share a centroid.
        return [[1.0, 0.0], [1.0, 0.0]]

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {"B": "A"}
    assert {row["speaker"] for row in updated} == {"A"}


def test_merge_rejects_two_d_vectors_with_shape_mismatch() -> None:
    def _model(audio_path: Path, start: float, end: float):
        return [[1.0, 0.0], [1.0, 0.0, 0.5]]  # jagged rows

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {}


def test_merge_rejects_two_d_vectors_with_non_numeric_entries() -> None:
    def _model(audio_path: Path, start: float, end: float):
        return [[1.0, "bad"]]

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {}


def test_merge_rejects_two_d_vectors_with_empty_rows() -> None:
    def _model(audio_path: Path, start: float, end: float):
        return [[], []]

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {}


def test_merge_rejects_two_d_vectors_with_nonfinite_pool_result() -> None:
    def _model(audio_path: Path, start: float, end: float):
        return [[float("nan"), 0.0]]

    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, _diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=_model,
    )
    assert merge_map == {}


def test_diagnostics_low_similarity() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    sim = 0.75
    complement = math.sqrt(1.0 - sim * sim)
    model = _make_embedding_model(
        {
            "A": [1.0, 0.0],
            "B": [sim, complement],
        },
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    _updated, merge_map, diagnostics = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
        no_overlap_similarity_threshold=DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
    )
    assert merge_map == {}
    assert diagnostics["embedding_model_available"] is True
    assert diagnostics["speakers_found"] == ["A", "B"]
    assert diagnostics["centroids_computed"] == ["A", "B"]
    actions = [entry["action"] for entry in diagnostics["pairwise_scores"]]
    assert "skipped_low_similarity" in actions
    low = next(
        entry
        for entry in diagnostics["pairwise_scores"]
        if entry["action"] == "skipped_low_similarity"
    )
    assert low["similarity"] == pytest.approx(sim)
    assert low["overlap"] is False
    assert {low["speaker_a"], low["speaker_b"]} == {"A", "B"}
    assert low["effective_threshold"] == pytest.approx(
        DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD
    )


def test_diagnostics_overlap() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 2.0, 7.0),  # overlaps with A
    ]
    embedding = [1.0, 0.0]
    model = _make_embedding_model(
        {"A": embedding, "B": embedding},
        {"A": [(0.0, 5.0)], "B": [(2.0, 7.0)]},
    )
    _updated, merge_map, diagnostics = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    assert merge_map == {}
    actions = [entry["action"] for entry in diagnostics["pairwise_scores"]]
    assert "skipped_overlap" in actions
    overlap_entry = next(
        entry
        for entry in diagnostics["pairwise_scores"]
        if entry["action"] == "skipped_overlap"
    )
    assert overlap_entry["overlap"] is True
    assert overlap_entry["similarity"] == pytest.approx(1.0)
    assert {overlap_entry["speaker_a"], overlap_entry["speaker_b"]} == {"A", "B"}
    assert overlap_entry["effective_threshold"] == pytest.approx(
        DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD
    )


def test_diagnostics_merged() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    embedding = [1.0, 0.0]
    model = _make_embedding_model(
        {"A": embedding, "B": embedding},
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    _updated, merge_map, diagnostics = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
    )
    assert merge_map  # something was merged
    merged_entries = [
        entry
        for entry in diagnostics["pairwise_scores"]
        if entry["action"] == "merged"
    ]
    assert len(merged_entries) == 1
    entry = merged_entries[0]
    assert entry["overlap"] is False
    assert entry["similarity"] == pytest.approx(1.0)
    assert {entry["speaker_a"], entry["speaker_b"]} == {"A", "B"}
    assert entry["effective_threshold"] == pytest.approx(
        DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD
    )
    assert diagnostics["merges_applied"] == merge_map
    assert diagnostics["centroids_computed"] == ["A", "B"]
    assert diagnostics["speakers_found"] == ["A", "B"]


def test_merge_relaxed_threshold_no_overlap() -> None:
    """similarity=0.75, no overlap, strict=0.80, relaxed=0.70 -> MERGE."""
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    sim = 0.75
    complement = math.sqrt(1.0 - sim * sim)
    model = _make_embedding_model(
        {"A": [1.0, 0.0], "B": [sim, complement]},
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    updated, merge_map, diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=0.80,
        no_overlap_similarity_threshold=0.70,
    )
    assert merge_map  # merged
    assert len({row["speaker"] for row in updated}) == 1
    merged_entry = next(
        e for e in diag["pairwise_scores"] if e["action"] == "merged"
    )
    assert merged_entry["effective_threshold"] == pytest.approx(0.70)


def test_no_merge_strict_threshold_with_overlap() -> None:
    """similarity=0.85, overlap=true -> NOT MERGED (overlap blocks)."""
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 2.0, 7.0),  # overlaps with A
    ]
    sim = 0.85
    complement = math.sqrt(1.0 - sim * sim)
    model = _make_embedding_model(
        {"A": [1.0, 0.0], "B": [sim, complement]},
        {"A": [(0.0, 5.0)], "B": [(2.0, 7.0)]},
    )
    updated, merge_map, diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=0.80,
        no_overlap_similarity_threshold=0.70,
    )
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}
    overlap_entry = next(
        e for e in diag["pairwise_scores"] if e["action"] == "skipped_overlap"
    )
    assert overlap_entry["overlap"] is True
    assert overlap_entry["effective_threshold"] == pytest.approx(0.80)


def test_no_merge_between_thresholds_with_overlap() -> None:
    """similarity=0.75, overlap=true, strict=0.80, relaxed=0.70.

    The pair passes the relaxed guard (0.75 >= 0.70), but overlap pushes
    the effective threshold to 0.80, and 0.75 < 0.80 -> skipped.
    """
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 2.0, 7.0),  # overlaps with A
    ]
    sim = 0.75
    complement = math.sqrt(1.0 - sim * sim)
    model = _make_embedding_model(
        {"A": [1.0, 0.0], "B": [sim, complement]},
        {"A": [(0.0, 5.0)], "B": [(2.0, 7.0)]},
    )
    updated, merge_map, diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=0.80,
        no_overlap_similarity_threshold=0.70,
    )
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}
    low_entry = next(
        e for e in diag["pairwise_scores"] if e["action"] == "skipped_low_similarity"
    )
    assert low_entry["overlap"] is True
    assert low_entry["effective_threshold"] == pytest.approx(0.80)


def test_no_merge_below_relaxed_threshold() -> None:
    """similarity=0.65, no overlap, relaxed=0.70 -> NOT MERGED."""
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    sim = 0.65
    complement = math.sqrt(1.0 - sim * sim)
    model = _make_embedding_model(
        {"A": [1.0, 0.0], "B": [sim, complement]},
        {"A": [(0.0, 5.0)], "B": [(6.0, 11.0)]},
    )
    updated, merge_map, diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=0.80,
        no_overlap_similarity_threshold=0.70,
    )
    assert merge_map == {}
    assert {row["speaker"] for row in updated} == {"A", "B"}
    low_entry = next(
        e for e in diag["pairwise_scores"] if e["action"] == "skipped_low_similarity"
    )
    assert low_entry["effective_threshold"] == pytest.approx(0.70)


def test_real_case_0786_merges_with_dual_threshold() -> None:
    """Exact Plaud case: similarity=0.786, no overlap, strict=0.80, relaxed=0.70.

    Before this change the pair was skipped (0.786 < 0.80). With the relaxed
    no-overlap threshold (0.70) and no temporal overlap, it must now merge.
    """
    diar_segments = [
        _segment("SPEAKER_00", 0.0, 120.0),
        _segment("SPEAKER_01", 130.0, 250.0),
    ]
    sim = 0.786
    complement = math.sqrt(1.0 - sim * sim)
    model = _make_embedding_model(
        {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [sim, complement]},
        {"SPEAKER_00": [(0.0, 120.0)], "SPEAKER_01": [(130.0, 250.0)]},
    )
    updated, merge_map, diag = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=model,
        similarity_threshold=0.80,
        no_overlap_similarity_threshold=0.70,
    )
    assert merge_map  # merged
    assert len({row["speaker"] for row in updated}) == 1
    merged_entry = next(
        e for e in diag["pairwise_scores"] if e["action"] == "merged"
    )
    assert merged_entry["similarity"] == pytest.approx(sim, abs=0.01)
    assert merged_entry["effective_threshold"] == pytest.approx(0.70)


def test_diagnostics_no_model() -> None:
    diar_segments = [
        _segment("A", 0.0, 5.0),
        _segment("B", 6.0, 11.0),
    ]
    updated, merge_map, diagnostics = merge_similar_speakers(
        diar_segments,
        audio_path=_AUDIO_PATH,
        embedding_model=None,
    )
    assert merge_map == {}
    assert updated == diar_segments
    assert diagnostics["embedding_model_available"] is False
    assert diagnostics["speakers_found"] == []
    assert diagnostics["centroids_computed"] == []
    assert diagnostics["pairwise_scores"] == []
    assert diagnostics["merges_applied"] == {}


class _StubDiariser:
    """Simple diariser stub used by orchestrator helper tests."""

    def __init__(self, *, pipeline_model=None, pipeline_attr: str = "_pipeline_model"):
        if pipeline_model is not None:
            setattr(self, pipeline_attr, pipeline_model)


def _patched_module(name: str, module: object):
    class _Guard:
        def __enter__(self_inner):
            self_inner._previous = sys.modules.get(name)
            sys.modules[name] = module  # type: ignore[assignment]
            return module

        def __exit__(self_inner, exc_type, exc, tb):
            if self_inner._previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = self_inner._previous
            return False

    return _Guard()


def test_diariser_pipeline_model_prefers_private_attribute() -> None:
    model = object()
    diariser = _StubDiariser(pipeline_model=model, pipeline_attr="_pipeline_model")
    assert pipeline_orchestrator._diariser_pipeline_model(diariser) is model


def test_diariser_pipeline_model_falls_through_to_pipeline() -> None:
    model = object()
    diariser = _StubDiariser(pipeline_model=model, pipeline_attr="pipeline")
    assert pipeline_orchestrator._diariser_pipeline_model(diariser) is model


def test_diariser_pipeline_model_returns_none_when_missing() -> None:
    diariser = _StubDiariser()
    assert pipeline_orchestrator._diariser_pipeline_model(diariser) is None


def test_diariser_pipeline_model_skips_self_reference() -> None:
    class _SelfDiariser:
        def __init__(self) -> None:
            self._pipeline_model = self

    diariser = _SelfDiariser()
    assert pipeline_orchestrator._diariser_pipeline_model(diariser) is None


def test_resolve_pyannote_embedding_model_returns_cached_instance() -> None:
    diariser = _StubDiariser()
    sentinel = object()
    diariser._lan_speaker_embedding_model = sentinel  # type: ignore[attr-defined]
    assert (
        pipeline_orchestrator._resolve_pyannote_embedding_model(diariser) is sentinel
    )


def test_resolve_pyannote_embedding_model_respects_unavailable_flag() -> None:
    diariser = _StubDiariser()
    diariser._lan_speaker_embedding_unavailable = True  # type: ignore[attr-defined]
    assert pipeline_orchestrator._resolve_pyannote_embedding_model(diariser) is None


def test_resolve_pyannote_embedding_model_marks_unavailable_without_pipeline() -> None:
    diariser = _StubDiariser()
    assert pipeline_orchestrator._resolve_pyannote_embedding_model(diariser) is None
    assert getattr(diariser, "_lan_speaker_embedding_unavailable", False) is True


def test_resolve_pyannote_embedding_model_uses_existing_private_embedding(
    tmp_path: Path,
) -> None:
    class _FakeSegment:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    fake_core = types.SimpleNamespace(Segment=_FakeSegment)

    class _FakeEmbedding:
        def __init__(self) -> None:
            self.calls: list[tuple[str, _FakeSegment]] = []

        def crop(self, path: str, segment: _FakeSegment):
            self.calls.append((path, segment))
            return [1.0, 0.0]

    pipeline_model = types.SimpleNamespace(_embedding=_FakeEmbedding())
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )

    with _patched_module("pyannote.core", fake_core):
        model_callable = pipeline_orchestrator._resolve_pyannote_embedding_model(
            diariser
        )
        assert callable(model_callable)
        result = model_callable(tmp_path / "audio.wav", 1.0, 2.0)
    assert result == [1.0, 0.0]
    assert pipeline_model._embedding.calls
    # Subsequent calls hit the cache on the diariser.
    assert (
        pipeline_orchestrator._resolve_pyannote_embedding_model(diariser)
        is model_callable
    )


def test_resolve_pyannote_embedding_model_uses_public_embedding_attr() -> None:
    class _FakeEmbedding:
        def crop(self, *_args, **_kwargs):
            raise RuntimeError("boom")  # triggers debug-log branch

    fake_core = types.SimpleNamespace(Segment=lambda *_a, **_k: object())
    pipeline_model = types.SimpleNamespace(embedding=_FakeEmbedding())
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )
    with _patched_module("pyannote.core", fake_core):
        model_callable = pipeline_orchestrator._resolve_pyannote_embedding_model(
            diariser
        )
        assert callable(model_callable)
        assert model_callable(Path("/x.wav"), 0.0, 1.0) is None


def test_resolve_pyannote_embedding_model_falls_back_to_inference_with_pipeline_name() -> None:
    # Simulates pyannote ``SpeakerDiarization`` where ``pipeline.embedding``
    # stores the model identifier string (not an Inference-like object).
    pipeline_model = types.SimpleNamespace(
        embedding="pyannote/custom-embed",
    )
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )

    class _FakeInference:
        def __init__(self, name: str, *, window: str) -> None:
            self.name = name
            self.window = window

        def crop(self, path: str, segment):
            return [0.25, 0.75]

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)
    fake_core = types.SimpleNamespace(Segment=lambda *a, **k: (a, k))

    with _patched_module("pyannote.audio", fake_audio), _patched_module(
        "pyannote.core", fake_core
    ):
        model_callable = pipeline_orchestrator._resolve_pyannote_embedding_model(
            diariser
        )
        assert callable(model_callable)
        vector = model_callable(Path("/y.wav"), 3.0, 4.0)
    assert vector == [0.25, 0.75]


def test_resolve_pyannote_embedding_model_falls_back_when_private_embedding_lacks_crop() -> None:
    """Covers the real pyannote-audio >= 3.1 case.

    ``SpeakerDiarization._embedding`` is a ``PretrainedSpeakerEmbedding``
    callable — it takes waveforms, not (file, segment) pairs. Using it as an
    Inference substitute causes every ``crop()`` call to raise, silently
    disabling the merge step. The resolver must detect the missing ``crop``
    method and wrap the pipeline's embedding identifier with a fresh
    ``Inference`` instead.
    """

    class _PretrainedSpeakerEmbedding:
        """Callable with no ``crop()`` — mirrors pyannote's internal object."""

        def __call__(self, waveforms):
            return [[1.0, 0.0]]

    pipeline_model = types.SimpleNamespace(
        _embedding=_PretrainedSpeakerEmbedding(),
        embedding="pyannote/wespeaker-from-pipeline",
    )
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )

    build_calls: list[Any] = []

    class _FakeInference:
        def __init__(self, name, *, window: str) -> None:
            build_calls.append((name, window))
            self.name = name

        def crop(self, path: str, segment):
            return [0.5, 0.5]

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)
    fake_core = types.SimpleNamespace(Segment=lambda *a, **k: (a, k))

    with _patched_module("pyannote.audio", fake_audio), _patched_module(
        "pyannote.core", fake_core
    ):
        model_callable = pipeline_orchestrator._resolve_pyannote_embedding_model(
            diariser
        )
        assert callable(model_callable)
        vector = model_callable(Path("/z.wav"), 0.0, 1.0)
    assert vector == [0.5, 0.5]
    # The embedding attr was ignored; we built a fresh Inference by name.
    assert build_calls == [("pyannote/wespeaker-from-pipeline", "whole")]


def test_resolve_pyannote_embedding_model_uses_default_when_embedding_attr_not_a_string() -> None:
    """When pipeline.embedding is neither crop-capable nor a string, use the default."""

    class _OpaqueEmbedding:
        def __call__(self, *_a, **_k):  # pragma: no cover - not invoked
            return None

    pipeline_model = types.SimpleNamespace(_embedding=_OpaqueEmbedding())
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )

    build_calls: list[Any] = []

    class _FakeInference:
        def __init__(self, name, *, window: str) -> None:
            build_calls.append((name, window))

        def crop(self, *_a, **_k):  # pragma: no cover - not invoked
            return [1.0]

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)
    with _patched_module("pyannote.audio", fake_audio):
        pipeline_orchestrator._resolve_pyannote_embedding_model(diariser)
    assert build_calls == [
        (pipeline_orchestrator._DEFAULT_SPEAKER_EMBEDDING_MODEL, "whole")
    ]


def test_resolve_pyannote_embedding_model_handles_inference_import_error() -> None:
    pipeline_model = types.SimpleNamespace()
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )
    # Replace pyannote.audio with a module missing Inference to simulate an
    # ImportError inside the helper's lazy import block.
    empty_module = types.SimpleNamespace()
    with _patched_module("pyannote.audio", empty_module):
        result = pipeline_orchestrator._resolve_pyannote_embedding_model(diariser)
    assert result is None
    assert getattr(diariser, "_lan_speaker_embedding_unavailable", False) is True


def test_resolve_pyannote_embedding_model_handles_inference_constructor_error() -> None:
    pipeline_model = types.SimpleNamespace()

    class _FailingInference:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("no weights")

    fake_audio = types.SimpleNamespace(Inference=_FailingInference)
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )
    with _patched_module("pyannote.audio", fake_audio):
        result = pipeline_orchestrator._resolve_pyannote_embedding_model(diariser)
    assert result is None
    assert getattr(diariser, "_lan_speaker_embedding_unavailable", False) is True


def test_build_pyannote_inference_uses_diarization_safe_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wespeaker checkpoint must be loaded under the trusted safe-globals
    context to survive ``torch.load(weights_only=True)``."""

    contexts: list[str] = []

    class _FakeInference:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.kwargs = kwargs

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)

    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _fake_ctx(extra_fqns=None):
        contexts.append("entered")
        yield

    monkeypatch.setattr(
        pipeline_orchestrator,
        "diarization_safe_globals_for_torch_load",
        _fake_ctx,
    )

    with _patched_module("pyannote.audio", fake_audio):
        inference = pipeline_orchestrator._build_pyannote_inference(
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        )

    assert isinstance(inference, _FakeInference)
    assert inference.kwargs == {"window": "whole"}
    assert contexts == ["entered"]


def test_build_pyannote_inference_passes_torch_device_when_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a non-CPU device is provided, it must be forwarded as a
    ``torch.device`` object so the embedding model runs on GPU."""

    class _FakeInference:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.kwargs = kwargs

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)

    fake_torch = types.SimpleNamespace(device=lambda spec: ("torch.device", spec))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with _patched_module("pyannote.audio", fake_audio):
        inference = pipeline_orchestrator._build_pyannote_inference(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            device="cuda:0",
        )

    assert isinstance(inference, _FakeInference)
    assert inference.kwargs.get("window") == "whole"
    assert inference.kwargs.get("device") == ("torch.device", "cuda:0")


def test_build_pyannote_inference_skips_device_for_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``device='cpu'`` should not pass a torch.device kwarg (matches
    pre-existing CPU-only behaviour)."""

    class _FakeInference:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.kwargs = kwargs

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)

    with _patched_module("pyannote.audio", fake_audio):
        inference = pipeline_orchestrator._build_pyannote_inference(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            device="cpu",
        )
    assert isinstance(inference, _FakeInference)
    assert "device" not in inference.kwargs


def test_build_pyannote_inference_retries_on_unsupported_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the first load attempt raises ``Unsupported global``, the helper
    must extract the FQN, allowlist it, and retry — mirroring the diarization
    pipeline loader's safe-globals retry."""

    attempts: list[list[str]] = []

    class _FakeInference:
        def __init__(self, name: str, **kwargs) -> None:
            extras = list(_current_extras["value"])
            attempts.append(extras)
            if not extras:
                raise RuntimeError(
                    "Unsupported global: GLOBAL omegaconf.base.ContainerMetadata"
                )
            self.name = name
            self.kwargs = kwargs

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)

    _current_extras: dict[str, list[str]] = {"value": []}

    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _fake_ctx(extra_fqns=None):
        _current_extras["value"] = list(extra_fqns or [])
        yield

    monkeypatch.setattr(
        pipeline_orchestrator,
        "diarization_safe_globals_for_torch_load",
        _fake_ctx,
    )
    monkeypatch.setattr(
        pipeline_orchestrator,
        "import_trusted_diarization_symbol",
        lambda fqn: object(),
    )

    with _patched_module("pyannote.audio", fake_audio):
        inference = pipeline_orchestrator._build_pyannote_inference(
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        )

    assert isinstance(inference, _FakeInference)
    assert attempts[0] == []
    assert "omegaconf.base.ContainerMetadata" in attempts[1]


def test_build_pyannote_inference_breaks_when_retry_fqn_untrusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the unsupported-global FQN cannot be imported via the trusted
    diarization allowlist, the helper must stop retrying and return None."""

    class _FakeInference:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError(
                "Unsupported global: GLOBAL omegaconf.base.ContainerMetadata"
            )

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)

    monkeypatch.setattr(
        pipeline_orchestrator,
        "import_trusted_diarization_symbol",
        lambda fqn: None,
    )

    with _patched_module("pyannote.audio", fake_audio):
        result = pipeline_orchestrator._build_pyannote_inference(
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        )
    assert result is None


def test_build_pyannote_inference_exhausts_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every attempt raises with a fresh retryable FQN, the helper must
    bound retries via the safe-global attempt budget and eventually return
    None instead of looping forever."""

    suffixes = ["A", "B", "C", "D", "E"]
    counter = {"i": 0}

    class _FakeInference:
        def __init__(self, *_args, **_kwargs) -> None:
            i = counter["i"]
            counter["i"] += 1
            sym = suffixes[i]
            raise RuntimeError(f"Unsupported global: GLOBAL omegaconf.test.{sym}")

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)

    monkeypatch.setattr(
        pipeline_orchestrator,
        "import_trusted_diarization_symbol",
        lambda fqn: object(),
    )

    with _patched_module("pyannote.audio", fake_audio):
        result = pipeline_orchestrator._build_pyannote_inference(
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        )
    assert result is None
    # The helper must respect the retry budget (3) and not loop indefinitely.
    assert counter["i"] == 3


def test_resolve_pyannote_embedding_model_forwards_pipeline_device() -> None:
    """The embedding model should run on the same device as the diarization
    pipeline (set via ``_lan_effective_device`` on the pipeline_model)."""

    pipeline_model = types.SimpleNamespace(
        embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
        _lan_effective_device="cuda:0",
    )
    diariser = _StubDiariser(
        pipeline_model=pipeline_model, pipeline_attr="_pipeline_model"
    )

    constructed: list["_FakeInference"] = []

    class _FakeInference:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.kwargs = kwargs
            self.device = kwargs.get("device")
            constructed.append(self)

        def crop(self, path: str, segment):
            return [0.1, 0.9]

    fake_audio = types.SimpleNamespace(Inference=_FakeInference)
    fake_torch = types.SimpleNamespace(device=lambda spec: ("torch.device", spec))

    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch  # type: ignore[assignment]
    try:
        with _patched_module("pyannote.audio", fake_audio):
            model_callable = pipeline_orchestrator._resolve_pyannote_embedding_model(
                diariser
            )
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
    assert callable(model_callable)
    assert len(constructed) == 1
    assert constructed[0].kwargs.get("device") == ("torch.device", "cuda:0")
