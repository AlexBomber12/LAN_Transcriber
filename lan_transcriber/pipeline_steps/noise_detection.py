"""Detect diarization speakers whose snippets contain no real speech.

After ``export_speaker_snippets`` writes accepted snippets and the manifest,
:func:`apply_noise_flags_to_manifest` runs a lightweight VAD pass on each
accepted snippet, annotates manifest entries with ``speech_ratio`` and
``noise_suspected``, and surfaces a top-level ``noise_speakers`` summary.
The diarization metadata file is updated in place by
:func:`update_diarization_metadata_with_noise` so downstream consumers
(UI, transcript export) see a single source of truth.
"""

from __future__ import annotations

import audioop
import json
import wave
from pathlib import Path
from typing import Any

from lan_transcriber.artifacts import atomic_write_json

DEFAULT_NOISE_SPEECH_RATIO_THRESHOLD = 0.3

_RMS_THRESHOLD = 350
_FRAME_SECONDS = 0.03


def compute_wav_speech_ratio(audio_path: Path) -> float | None:
    """Return ratio of voiced frames in ``audio_path`` (PCM WAV).

    Uses the same RMS-based 30 ms windowing as the upstream precheck so that
    silence/noise-only snippets and real speech are evaluated consistently.
    Returns ``None`` for unreadable inputs and ``0.0`` for empty WAVs.
    """

    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            sample_width = src.getsampwidth()
            frame_samples = max(int(rate * _FRAME_SECONDS), 1)
            frame_bytes = frame_samples * channels * sample_width
            voiced = 0
            total = 0
            while True:
                chunk = src.readframes(frame_samples)
                if not chunk:
                    break
                if len(chunk) < frame_bytes // 2:
                    break
                total += 1
                if audioop.rms(chunk, sample_width) >= _RMS_THRESHOLD:
                    voiced += 1
        if total == 0:
            return 0.0
        return round(voiced / float(total), 4)
    except Exception:
        return None


def _accepted_entries(speaker_entries: list[Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in speaker_entries
        if isinstance(entry, dict) and str(entry.get("status") or "") == "accepted"
    ]


def _speaker_speech_ratio(
    accepted: list[dict[str, Any]],
    *,
    snippets_dir: Path,
) -> tuple[float | None, int, int]:
    """Compute aggregate speech ratio across accepted snippets for one speaker.

    Returns ``(ratio, evaluated_count, missing_count)`` where ``ratio`` is the
    average of per-snippet ratios that could be computed. ``None`` is returned
    when no snippet could be evaluated.
    """

    ratios: list[float] = []
    missing = 0
    for entry in accepted:
        relative_path = str(entry.get("relative_path") or "").strip()
        if not relative_path:
            missing += 1
            continue
        snippet_path = snippets_dir / relative_path
        ratio = compute_wav_speech_ratio(snippet_path)
        if ratio is None:
            missing += 1
            continue
        entry["speech_ratio"] = ratio
        ratios.append(ratio)
    if not ratios:
        return None, 0, missing
    avg = round(sum(ratios) / len(ratios), 4)
    return avg, len(ratios), missing


def analyze_snippet_noise(
    manifest: dict[str, Any],
    *,
    snippets_dir: Path,
    threshold: float = DEFAULT_NOISE_SPEECH_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """Mutate ``manifest`` with per-speaker noise flags and return a summary.

    For each speaker whose accepted snippets average a speech ratio below
    ``threshold``, the speaker is appended to the returned ``noise_speakers``
    list and every accepted entry in that speaker's manifest list gains
    ``noise_suspected=True``. Speakers without evaluable snippets are left
    untouched and not flagged.
    """

    speakers_payload = manifest.get("speakers")
    summary: dict[str, Any] = {
        "noise_speakers": [],
        "speaker_metrics": {},
        "threshold": round(max(0.0, min(float(threshold), 1.0)), 4),
    }
    if not isinstance(speakers_payload, dict):
        manifest["noise_speakers"] = []
        manifest["noise_speech_ratio_threshold"] = summary["threshold"]
        return summary
    noise_speakers: list[str] = []
    metrics: dict[str, Any] = {}
    for speaker_label in sorted(speakers_payload):
        entries = speakers_payload.get(speaker_label)
        if not isinstance(entries, list):
            continue
        accepted = _accepted_entries(entries)
        if not accepted:
            continue
        ratio, evaluated, missing = _speaker_speech_ratio(
            accepted, snippets_dir=snippets_dir
        )
        if ratio is None:
            metrics[speaker_label] = {
                "speech_ratio": None,
                "evaluated_snippets": 0,
                "missing_snippets": missing,
                "flagged": False,
            }
            continue
        flagged = ratio < summary["threshold"]
        metrics[speaker_label] = {
            "speech_ratio": ratio,
            "evaluated_snippets": evaluated,
            "missing_snippets": missing,
            "flagged": bool(flagged),
        }
        if flagged:
            noise_speakers.append(speaker_label)
            for entry in accepted:
                entry["noise_suspected"] = True
    manifest["noise_speakers"] = list(noise_speakers)
    manifest["noise_speech_ratio_threshold"] = summary["threshold"]
    summary["noise_speakers"] = list(noise_speakers)
    summary["speaker_metrics"] = metrics
    return summary


def apply_noise_flags_to_manifest(
    manifest_path: Path,
    *,
    snippets_dir: Path,
    threshold: float = DEFAULT_NOISE_SPEECH_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """Load the snippets manifest, annotate it with noise flags, and persist it."""

    if not manifest_path.exists():
        return {
            "noise_speakers": [],
            "speaker_metrics": {},
            "threshold": round(max(0.0, min(float(threshold), 1.0)), 4),
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "noise_speakers": [],
            "speaker_metrics": {},
            "threshold": round(max(0.0, min(float(threshold), 1.0)), 4),
        }
    if not isinstance(manifest, dict):
        return {
            "noise_speakers": [],
            "speaker_metrics": {},
            "threshold": round(max(0.0, min(float(threshold), 1.0)), 4),
        }
    summary = analyze_snippet_noise(
        manifest,
        snippets_dir=snippets_dir,
        threshold=threshold,
    )
    atomic_write_json(manifest_path, manifest)
    return summary


def update_diarization_metadata_with_noise(
    metadata_path: Path,
    *,
    summary: dict[str, Any],
) -> None:
    """Persist the noise-speaker summary onto ``diarization_metadata.json``."""

    if not metadata_path.exists():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload["noise_speakers"] = list(summary.get("noise_speakers") or [])
    payload["noise_speaker_metrics"] = dict(summary.get("speaker_metrics") or {})
    payload["noise_speech_ratio_threshold"] = summary.get(
        "threshold", DEFAULT_NOISE_SPEECH_RATIO_THRESHOLD
    )
    atomic_write_json(metadata_path, payload)


__all__ = [
    "DEFAULT_NOISE_SPEECH_RATIO_THRESHOLD",
    "analyze_snippet_noise",
    "apply_noise_flags_to_manifest",
    "compute_wav_speech_ratio",
    "update_diarization_metadata_with_noise",
]
