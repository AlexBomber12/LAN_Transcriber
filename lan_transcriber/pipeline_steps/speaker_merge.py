"""Merge diarization speakers that share a voice centroid but never overlap.

This module implements an optional post-diarization pass that detects when the
pyannote pipeline has split a single speaker into two labels. The heuristic is:

- Extract a voice embedding centroid for every diarization speaker from a few
  of their longest segments.
- For every pair of speakers whose centroid cosine similarity is above a
  configurable threshold, check whether their diarization segments ever
  overlap in time. If they never overlap, merge the smaller-speech speaker
  into the larger one.

The contextual "no overlap" constraint protects real multi-speaker meetings
where two people happen to have similar voices but talk simultaneously.

All functions are pure and take an injectable ``embedding_model`` callable so
tests can use deterministic fake embeddings without requiring GPU or model
downloads. The callable signature is::

    embedding_model(audio_path: Path, start: float, end: float) -> Sequence[float]

Callers that do not have an embedding model available should pass ``None`` to
:func:`merge_similar_speakers`; the function returns the input segments
unchanged (graceful fallback).

The module intentionally avoids importing numpy so it can be imported in CI
environments where heavy scientific Python deps are stubbed. The vectors are
small (speaker embeddings are typically 192-256 floats) so plain Python math
is perfectly adequate.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from lan_transcriber.utils import safe_float

#: Callable signature for the embedding model. It must return a sequence of
#: floats (or any object convertible via :func:`_to_vector`).
EmbeddingModel = Callable[[Path, float, float], "Any"]

DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD = 0.80
DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD = 0.70
DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS = 5
DEFAULT_SPEAKER_MERGE_SEGMENT_DURATION_SEC = 3.0
DEFAULT_SPEAKER_MERGE_OVERLAP_TOLERANCE_SEC = 0.1

_logger = logging.getLogger(__name__)


def _segment_duration(row: dict[str, Any]) -> float:
    start = safe_float(row.get("start"), default=0.0)
    end = safe_float(row.get("end"), default=start)
    return max(end - start, 0.0)


def _speaker_total_seconds(
    diar_segments: Sequence[dict[str, Any]],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in diar_segments:
        speaker = str(row.get("speaker") or "")
        if not speaker:
            continue
        totals[speaker] = totals.get(speaker, 0.0) + _segment_duration(row)
    return totals


def _to_vector(value: Any) -> list[float] | None:
    """Best-effort coerce ``value`` into a list of finite floats.

    Returns ``None`` when the input is empty, contains non-finite entries, or
    cannot be converted. Accepts plain sequences and any object that supports
    ``tolist()`` (numpy arrays etc).

    Pyannote's ``Inference.crop`` commonly returns a 2-D array of shape
    ``(n_frames, dim)`` — in ``window="whole"`` mode typically ``(1, dim)``.
    This function transparently handles that case by mean-pooling along the
    leading axis so callers always get a single ``(dim,)`` centroid.
    """

    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(row, (list, tuple)) for row in value)
    ):
        rows: list[list[float]] = []
        for row in value:
            row_floats: list[float] = []
            for item in row:
                try:
                    row_floats.append(float(item))
                except (TypeError, ValueError):
                    return None
            rows.append(row_floats)
        dim = len(rows[0])
        if dim == 0:
            return None
        if not all(len(row) == dim for row in rows):
            return None
        averaged = [sum(row[i] for row in rows) / len(rows) for i in range(dim)]
        if not all(math.isfinite(x) for x in averaged):
            return None
        return averaged
    try:
        iterator = iter(value)
    except TypeError:
        return None
    flat: list[float] = []
    for item in iterator:
        try:
            flat.append(float(item))
        except (TypeError, ValueError):
            return None
    if not flat:
        return None
    if not all(math.isfinite(x) for x in flat):
        return None
    return flat


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    count = len(vectors)
    dim = len(vectors[0])
    totals = [0.0] * dim
    for vec in vectors:
        for idx in range(dim):
            totals[idx] += float(vec[idx])
    return [value / count for value in totals]


def extract_speaker_embeddings(
    audio_path: Path,
    diar_segments: Sequence[dict[str, Any]],
    *,
    embedding_model: EmbeddingModel | None,
    max_segments_per_speaker: int = DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS,
    segment_duration_sec: float = DEFAULT_SPEAKER_MERGE_SEGMENT_DURATION_SEC,
) -> dict[str, list[float]]:
    """Compute a centroid embedding for each diarization speaker.

    For every unique speaker the function selects up to
    ``max_segments_per_speaker`` of their longest segments, asks
    ``embedding_model`` for an embedding vector of each selected segment, and
    averages the vectors into a centroid. Speakers whose segments all fail to
    produce a usable embedding are skipped.
    """

    if embedding_model is None:
        return {}

    per_speaker_segments: dict[str, list[dict[str, Any]]] = {}
    for row in diar_segments:
        speaker = str(row.get("speaker") or "")
        if not speaker:
            continue
        duration = _segment_duration(row)
        if duration <= 0.0:
            continue
        per_speaker_segments.setdefault(speaker, []).append(
            {
                "start": safe_float(row.get("start"), default=0.0),
                "end": safe_float(row.get("end"), default=0.0),
                "duration": duration,
            }
        )

    safe_max = max(int(max_segments_per_speaker), 1)
    safe_target_duration = max(float(segment_duration_sec), 0.0)

    centroids: dict[str, list[float]] = {}
    for speaker, segments in per_speaker_segments.items():
        segments.sort(
            key=lambda item: (
                -float(item["duration"]),
                float(item["start"]),
            )
        )
        vectors: list[list[float]] = []
        for seg in segments[:safe_max]:
            start = float(seg["start"])
            end = float(seg["end"])
            if safe_target_duration > 0.0 and (end - start) > safe_target_duration:
                # Use the middle ``segment_duration_sec`` window of a long
                # segment so that the embedding captures speech rather than
                # silence at the edges.
                midpoint = (start + end) / 2.0
                half = safe_target_duration / 2.0
                start = midpoint - half
                end = midpoint + half
            try:
                raw_vector = embedding_model(audio_path, start, end)
            except Exception as exc:  # pragma: no cover - logged and skipped
                _logger.warning(
                    "speaker_merge embedding extraction failed: speaker=%s "
                    "start=%.3f end=%.3f error=%s",
                    speaker,
                    start,
                    end,
                    exc,
                )
                continue
            vector = _to_vector(raw_vector)
            if vector is None:
                continue
            if vectors and len(vector) != len(vectors[0]):
                continue
            vectors.append(vector)
        if not vectors:
            continue
        centroids[speaker] = _mean_vector(vectors)
    return centroids


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
    *,
    left_norm: float,
    right_norm: float,
) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def compute_pairwise_similarity(
    embeddings: dict[str, Sequence[float]],
) -> list[tuple[str, str, float]]:
    """Compute cosine similarity for every unordered pair of speakers."""

    pairs: list[tuple[str, str, float]] = []
    speakers: list[tuple[str, list[float], float]] = []
    for speaker in sorted(embeddings.keys()):
        vector = list(embeddings[speaker])
        norm = _vector_norm(vector)
        if norm <= 0.0:
            continue
        speakers.append((speaker, vector, norm))
    for i, (left_speaker, left_vec, left_norm) in enumerate(speakers):
        for right_speaker, right_vec, right_norm in speakers[i + 1 :]:
            if len(left_vec) != len(right_vec):
                continue
            pairs.append(
                (
                    left_speaker,
                    right_speaker,
                    _cosine_similarity(
                        left_vec,
                        right_vec,
                        left_norm=left_norm,
                        right_norm=right_norm,
                    ),
                )
            )
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs


def speakers_overlap(
    speaker_a: str,
    speaker_b: str,
    diar_segments: Sequence[dict[str, Any]],
    *,
    tolerance_sec: float = DEFAULT_SPEAKER_MERGE_OVERLAP_TOLERANCE_SEC,
) -> bool:
    """Return True when the two speakers ever share overlapping segments.

    ``tolerance_sec`` is subtracted from the overlap length so that tiny
    diarization boundary jitter is not interpreted as simultaneous speech.
    """

    if speaker_a == speaker_b:
        return False
    safe_tolerance = max(float(tolerance_sec), 0.0)
    a_segments = [
        (
            safe_float(row.get("start"), default=0.0),
            safe_float(row.get("end"), default=0.0),
        )
        for row in diar_segments
        if str(row.get("speaker") or "") == speaker_a
    ]
    b_segments = [
        (
            safe_float(row.get("start"), default=0.0),
            safe_float(row.get("end"), default=0.0),
        )
        for row in diar_segments
        if str(row.get("speaker") or "") == speaker_b
    ]
    for a_start, a_end in a_segments:
        if a_end <= a_start:
            continue
        for b_start, b_end in b_segments:
            if b_end <= b_start:
                continue
            overlap = min(a_end, b_end) - max(a_start, b_start)
            if overlap > safe_tolerance:
                return True
    return False


def _resolve_merge_target(merge_map: dict[str, str], label: str) -> str:
    current = label
    seen: set[str] = set()
    while current in merge_map and current not in seen:
        seen.add(current)
        current = merge_map[current]
    return current


def _empty_diagnostics(
    *,
    embedding_model_available: bool,
) -> dict[str, Any]:
    return {
        "embedding_model_available": embedding_model_available,
        "speakers_found": [],
        "centroids_computed": [],
        "pairwise_scores": [],
        "merges_applied": {},
    }


def merge_similar_speakers(
    diar_segments: Sequence[dict[str, Any]],
    *,
    audio_path: Path,
    embedding_model: EmbeddingModel | None,
    similarity_threshold: float = DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD,
    no_overlap_similarity_threshold: float = DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD,
    max_segments_per_speaker: int = DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS,
    segment_duration_sec: float = DEFAULT_SPEAKER_MERGE_SEGMENT_DURATION_SEC,
    overlap_tolerance_sec: float = DEFAULT_SPEAKER_MERGE_OVERLAP_TOLERANCE_SEC,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Merge diarization speakers with similar voices that never overlap.

    Returns a tuple ``(updated_segments, merge_map, diagnostics)`` where
    ``merge_map`` is a mapping from the merged label to the kept label after
    transitive merges and ``diagnostics`` is a dict describing the inputs and
    per-pair outcomes so operators can understand why speakers were or were
    not merged. When ``embedding_model`` is ``None`` the function is a no-op:
    it returns the original segments (as a new list), an empty ``merge_map``,
    and a diagnostics dict with ``embedding_model_available=False``.
    """

    original_segments = [dict(row) for row in diar_segments]
    if embedding_model is None:
        return (
            original_segments,
            {},
            _empty_diagnostics(embedding_model_available=False),
        )

    speakers_found = sorted(
        {
            str(row.get("speaker") or "")
            for row in original_segments
            if str(row.get("speaker") or "")
        }
    )
    diagnostics: dict[str, Any] = {
        "embedding_model_available": True,
        "speakers_found": speakers_found,
        "centroids_computed": [],
        "pairwise_scores": [],
        "merges_applied": {},
    }

    if not original_segments:
        return original_segments, {}, diagnostics

    centroids = extract_speaker_embeddings(
        audio_path,
        original_segments,
        embedding_model=embedding_model,
        max_segments_per_speaker=max_segments_per_speaker,
        segment_duration_sec=segment_duration_sec,
    )
    diagnostics["centroids_computed"] = sorted(centroids.keys())
    if len(centroids) < 2:
        return original_segments, {}, diagnostics

    pairs = compute_pairwise_similarity(centroids)
    original_totals = _speaker_total_seconds(original_segments)
    totals = dict(original_totals)
    merge_map: dict[str, str] = {}

    for left_speaker, right_speaker, similarity in pairs:
        left_target = _resolve_merge_target(merge_map, left_speaker)
        right_target = _resolve_merge_target(merge_map, right_speaker)
        if left_target == right_target:
            diagnostics["pairwise_scores"].append(
                {
                    "speaker_a": left_speaker,
                    "speaker_b": right_speaker,
                    "similarity": float(similarity),
                    "overlap": False,
                    "action": "skipped_already_merged",
                    "effective_threshold": None,
                }
            )
            continue
        # Pairs are sorted by similarity descending.  If below the relaxed
        # (lower) threshold, all remaining pairs will also be below it.
        if similarity < no_overlap_similarity_threshold:
            _logger.info(
                "speaker_merge: skip %s<->%s similarity=%.3f < "
                "no_overlap_threshold=%.3f",
                left_speaker,
                right_speaker,
                similarity,
                no_overlap_similarity_threshold,
            )
            diagnostics["pairwise_scores"].append(
                {
                    "speaker_a": left_speaker,
                    "speaker_b": right_speaker,
                    "similarity": float(similarity),
                    "overlap": False,
                    "action": "skipped_low_similarity",
                    "effective_threshold": float(no_overlap_similarity_threshold),
                }
            )
            break
        # Use a fresh copy of the original segments but with previously decided
        # merges applied so overlap detection sees the post-merge world.
        overlap_segments: list[dict[str, Any]] = []
        for row in original_segments:
            speaker = _resolve_merge_target(
                merge_map, str(row.get("speaker") or "")
            )
            overlap_segments.append({**row, "speaker": speaker})
        has_overlap = speakers_overlap(
            left_target,
            right_target,
            overlap_segments,
            tolerance_sec=overlap_tolerance_sec,
        )
        effective_threshold = (
            similarity_threshold if has_overlap else no_overlap_similarity_threshold
        )
        if similarity < effective_threshold:
            _logger.info(
                "speaker_merge: skip %s<->%s similarity=%.3f < "
                "effective_threshold=%.3f (overlap=%s)",
                left_speaker,
                right_speaker,
                similarity,
                effective_threshold,
                has_overlap,
            )
            diagnostics["pairwise_scores"].append(
                {
                    "speaker_a": left_speaker,
                    "speaker_b": right_speaker,
                    "similarity": float(similarity),
                    "overlap": has_overlap,
                    "action": "skipped_low_similarity",
                    "effective_threshold": float(effective_threshold),
                }
            )
            continue
        if has_overlap:
            _logger.info(
                "speaker_merge: skip %s<->%s similarity=%.3f overlap=True",
                left_speaker,
                right_speaker,
                similarity,
            )
            diagnostics["pairwise_scores"].append(
                {
                    "speaker_a": left_speaker,
                    "speaker_b": right_speaker,
                    "similarity": float(similarity),
                    "overlap": True,
                    "action": "skipped_overlap",
                    "effective_threshold": float(effective_threshold),
                }
            )
            continue
        left_total = totals.get(left_target, 0.0)
        right_total = totals.get(right_target, 0.0)
        if right_total > left_total:
            kept, merged = right_target, left_target
        elif left_total > right_total:
            kept, merged = left_target, right_target
        else:
            # Deterministic tie-breaker: lexicographically smaller label wins.
            kept, merged = sorted((left_target, right_target))
        merge_map[merged] = kept
        totals[kept] = totals.get(kept, 0.0) + totals.get(merged, 0.0)
        totals[merged] = 0.0
        diagnostics["pairwise_scores"].append(
            {
                "speaker_a": left_speaker,
                "speaker_b": right_speaker,
                "similarity": float(similarity),
                "overlap": False,
                "action": "merged",
                "effective_threshold": float(effective_threshold),
            }
        )
        _logger.info(
            "Merged speaker %s into %s: similarity=%.3f, overlap=False, "
            "%s_seconds=%.3f, %s_seconds=%.3f",
            merged,
            kept,
            similarity,
            merged,
            original_totals.get(merged, 0.0),
            kept,
            original_totals.get(kept, 0.0),
        )

    if not merge_map:
        return original_segments, {}, diagnostics

    # Apply merges transitively to every segment.
    updated_segments: list[dict[str, Any]] = []
    for row in original_segments:
        original_speaker = str(row.get("speaker") or "")
        final_speaker = _resolve_merge_target(merge_map, original_speaker)
        if final_speaker != original_speaker:
            updated_segments.append({**row, "speaker": final_speaker})
        else:
            updated_segments.append(dict(row))

    # Return a merge_map limited to labels that were actually replaced.
    flattened_map: dict[str, str] = {}
    for label in {str(row.get("speaker") or "") for row in original_segments}:
        final = _resolve_merge_target(merge_map, label)
        if final != label:
            flattened_map[label] = final

    diagnostics["merges_applied"] = dict(flattened_map)
    return updated_segments, flattened_map, diagnostics


__all__ = [
    "DEFAULT_SPEAKER_MERGE_MAX_SEGMENTS",
    "DEFAULT_SPEAKER_MERGE_NO_OVERLAP_SIMILARITY_THRESHOLD",
    "DEFAULT_SPEAKER_MERGE_OVERLAP_TOLERANCE_SEC",
    "DEFAULT_SPEAKER_MERGE_SEGMENT_DURATION_SEC",
    "DEFAULT_SPEAKER_MERGE_SIMILARITY_THRESHOLD",
    "EmbeddingModel",
    "compute_pairwise_similarity",
    "extract_speaker_embeddings",
    "merge_similar_speakers",
    "speakers_overlap",
]
