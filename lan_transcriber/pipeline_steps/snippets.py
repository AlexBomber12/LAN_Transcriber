from __future__ import annotations

from collections import defaultdict
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.utils import safe_float

DEFAULT_SNIPPET_PAD_SECONDS = 0.25
DEFAULT_SNIPPET_MAX_DURATION_SECONDS = 8.0
DEFAULT_SNIPPET_MIN_DURATION_SECONDS = 0.8
DEFAULT_SNIPPETS_PER_SPEAKER = 3
MAX_ALLOWED_OVERLAP_SECONDS = 0.05
MAX_ALLOWED_OVERLAP_RATIO = 0.02


@dataclass(frozen=True)
class SnippetExportRequest:
    audio_path: Path
    diar_segments: Sequence[dict[str, Any]]
    snippets_dir: Path
    duration_sec: float | None
    speaker_turns: Sequence[dict[str, Any]] = ()
    degraded_diarization: bool = False
    pad_seconds: float = DEFAULT_SNIPPET_PAD_SECONDS
    max_clip_duration_sec: float = DEFAULT_SNIPPET_MAX_DURATION_SECONDS
    min_clip_duration_sec: float = DEFAULT_SNIPPET_MIN_DURATION_SECONDS
    max_snippets_per_speaker: int = DEFAULT_SNIPPETS_PER_SPEAKER


def _speaker_slug(label: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    slug = slug.strip("_")
    return slug or "speaker"


def _clear_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _candidate_sources(request: SnippetExportRequest) -> list[dict[str, Any]]:
    primary = request.speaker_turns if request.speaker_turns else request.diar_segments
    source_kind = "turn" if request.speaker_turns else "segment"
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(primary):
        speaker = str(row.get("speaker") or "S1").strip() or "S1"
        start = safe_float(row.get("start"), default=0.0)
        end = safe_float(row.get("end"), default=start)
        if end < start:
            end = start
        out.append(
            {
                "speaker": speaker,
                "source_start": round(start, 3),
                "source_end": round(end, 3),
                "source_kind": source_kind,
                "source_index": idx,
            }
        )
    out.sort(
        key=lambda item: (
            str(item["speaker"]),
            float(item["source_start"]),
            float(item["source_end"]),
            int(item["source_index"]),
        )
    )
    return out


def _snippet_window(
    start: float,
    end: float,
    *,
    pad_seconds: float,
    max_duration_sec: float,
    duration_sec: float | None,
) -> tuple[float, float]:
    base_start = max(0.0, start - max(pad_seconds, 0.0))
    base_end = max(base_start, end + max(pad_seconds, 0.0))
    if duration_sec is not None and duration_sec > 0:
        base_end = min(base_end, duration_sec)
        base_start = min(base_start, base_end)
    target = min(max_duration_sec, max(0.0, base_end - base_start))
    if target > 0.0 and base_end - base_start > target:
        center = start + ((end - start) / 2.0 if end > start else 0.0)
        clip_start = center - (target / 2.0)
        clip_end = clip_start + target
        if duration_sec is not None and duration_sec > 0:
            if clip_end > duration_sec:
                clip_end = duration_sec
                clip_start = max(0.0, clip_end - target)
        if clip_start < 0.0:
            clip_end = max(0.0, clip_end - clip_start)
            clip_start = 0.0
    else:
        clip_start = base_start
        clip_end = base_end
    return round(clip_start, 3), round(clip_end, 3)


def _extract_wav_snippet_with_wave(
    audio_path: Path,
    out_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> bool:
    if audio_path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            start_frame = max(0, int(start_sec * rate))
            end_frame = max(start_frame + 1, int(end_sec * rate))
            src.setpos(min(start_frame, src.getnframes()))
            frames = src.readframes(max(0, end_frame - start_frame))
        if not frames:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(rate)
            dst.writeframes(frames)
        return True
    except Exception:
        return False


def _extract_wav_snippet_with_ffmpeg(
    audio_path: Path,
    out_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    duration = max(0.1, end_sec - start_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return False
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 44


def _overlap_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _overlap_against_other_speakers(
    speaker: str,
    *,
    clip_start: float,
    clip_end: float,
    diar_segments: Sequence[dict[str, Any]],
) -> float:
    overlaps: list[tuple[float, float]] = []
    for segment in diar_segments:
        other_speaker = str(segment.get("speaker") or "S1").strip() or "S1"
        if other_speaker == speaker:
            continue
        seg_start = safe_float(segment.get("start"), default=0.0)
        seg_end = safe_float(segment.get("end"), default=seg_start)
        if seg_end <= seg_start:
            continue
        overlap = _overlap_seconds(clip_start, clip_end, seg_start, seg_end)
        if overlap <= 0.0:
            continue
        overlaps.append((max(clip_start, seg_start), min(clip_end, seg_end)))
    return round(sum(end - start for start, end in _merge_intervals(overlaps)), 3)


def _trim_padding_to_clean_region(
    speaker: str,
    *,
    clip_start: float,
    clip_end: float,
    diar_segments: Sequence[dict[str, Any]],
) -> tuple[float, float]:
    trimmed_start = clip_start
    trimmed_end = clip_end
    other_segments: list[tuple[float, float]] = []
    for segment in diar_segments:
        other_speaker = str(segment.get("speaker") or "S1").strip() or "S1"
        if other_speaker == speaker:
            continue
        seg_start = safe_float(segment.get("start"), default=0.0)
        seg_end = safe_float(segment.get("end"), default=seg_start)
        if seg_end <= seg_start:
            continue
        other_segments.append((seg_start, seg_end))

    for seg_start, seg_end in sorted(other_segments):
        if seg_start <= trimmed_start < seg_end:
            trimmed_start = seg_end
    for seg_start, seg_end in sorted(other_segments, reverse=True):
        if seg_start < trimmed_end <= seg_end:
            trimmed_end = seg_start
    if trimmed_end < trimmed_start:
        trimmed_end = trimmed_start
    return round(trimmed_start, 3), round(trimmed_end, 3)


def _purity_score(
    *,
    clip_duration: float,
    overlap_ratio: float,
    max_clip_duration_sec: float,
    degraded: bool,
) -> float:
    duration_score = min(clip_duration / max(max_clip_duration_sec, 0.001), 1.0)
    score = ((1.0 - min(max(overlap_ratio, 0.0), 1.0)) * 0.85) + (duration_score * 0.15)
    if degraded:
        score -= 0.5
    return round(max(score, 0.0), 4)


def _candidate_status(
    *,
    clip_duration: float,
    overlap_seconds: float,
    overlap_ratio: float,
    min_clip_duration_sec: float,
    degraded_diarization: bool,
) -> str | None:
    if degraded_diarization:
        return "rejected_degraded"
    if clip_duration < max(min_clip_duration_sec, 0.0):
        return "rejected_short"
    if (
        overlap_seconds > MAX_ALLOWED_OVERLAP_SECONDS
        or overlap_ratio > MAX_ALLOWED_OVERLAP_RATIO
    ):
        return "rejected_overlap"
    return None


def _extract_snippet(
    audio_path: Path,
    out_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> str | None:
    if _extract_wav_snippet_with_wave(
        audio_path,
        out_path,
        start_sec=start_sec,
        end_sec=end_sec,
    ):
        return "wave"
    if _extract_wav_snippet_with_ffmpeg(
        audio_path,
        out_path,
        start_sec=start_sec,
        end_sec=end_sec,
    ):
        return "ffmpeg"
    return None


def _manifest_path(snippets_dir: Path) -> Path:
    return snippets_dir.parent / "snippets_manifest.json"


def _write_manifest(
    *,
    snippets_dir: Path,
    source_kind: str,
    degraded_diarization: bool,
    pad_seconds: float,
    max_clip_duration_sec: float,
    min_clip_duration_sec: float,
    max_snippets_per_speaker: int,
    speakers: dict[str, list[dict[str, Any]]],
) -> None:
    atomic_write_json(
        _manifest_path(snippets_dir),
        {
            "version": 1,
            "source_kind": source_kind,
            "degraded_diarization": bool(degraded_diarization),
            "pad_seconds": round(max(pad_seconds, 0.0), 3),
            "max_clip_duration_seconds": round(max(max_clip_duration_sec, 0.0), 3),
            "min_clip_duration_seconds": round(max(min_clip_duration_sec, 0.0), 3),
            "max_snippets_per_speaker": max(max_snippets_per_speaker, 0),
            "speakers": speakers,
        },
    )


def write_empty_snippets_manifest(
    *,
    snippets_dir: Path,
    pad_seconds: float = DEFAULT_SNIPPET_PAD_SECONDS,
    max_clip_duration_sec: float = DEFAULT_SNIPPET_MAX_DURATION_SECONDS,
    min_clip_duration_sec: float = DEFAULT_SNIPPET_MIN_DURATION_SECONDS,
    max_snippets_per_speaker: int = DEFAULT_SNIPPETS_PER_SPEAKER,
) -> None:
    _clear_dir(snippets_dir)
    _write_manifest(
        snippets_dir=snippets_dir,
        source_kind="segment",
        degraded_diarization=False,
        pad_seconds=pad_seconds,
        max_clip_duration_sec=max_clip_duration_sec,
        min_clip_duration_sec=min_clip_duration_sec,
        max_snippets_per_speaker=max_snippets_per_speaker,
        speakers={},
    )


def export_speaker_snippets(request: SnippetExportRequest) -> list[Path]:
    _clear_dir(request.snippets_dir)

    candidates_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in _candidate_sources(request):
        candidates_by_speaker[str(candidate["speaker"])].append(candidate)

    all_outputs: list[Path] = []
    manifest_speakers: dict[str, list[dict[str, Any]]] = {}
    for speaker in sorted(candidates_by_speaker):
        speaker_dir = request.snippets_dir / _speaker_slug(speaker)
        ranked_candidates: list[dict[str, Any]] = []
        for candidate in candidates_by_speaker[speaker]:
            seg_start = safe_float(candidate.get("source_start"), default=0.0)
            seg_end = safe_float(candidate.get("source_end"), default=seg_start)
            clip_start, clip_end = _snippet_window(
                seg_start,
                seg_end,
                pad_seconds=request.pad_seconds,
                max_duration_sec=request.max_clip_duration_sec,
                duration_sec=request.duration_sec,
            )
            clip_start, clip_end = _trim_padding_to_clean_region(
                speaker,
                clip_start=clip_start,
                clip_end=clip_end,
                diar_segments=request.diar_segments,
            )
            duration = round(max(0.0, clip_end - clip_start), 3)
            overlap_seconds = _overlap_against_other_speakers(
                speaker,
                clip_start=clip_start,
                clip_end=clip_end,
                diar_segments=request.diar_segments,
            )
            overlap_ratio = round(overlap_seconds / duration, 4) if duration > 0.0 else 1.0
            ranked_candidates.append(
                {
                    "speaker": speaker,
                    "source_kind": str(candidate["source_kind"]),
                    "source_start": round(seg_start, 3),
                    "source_end": round(seg_end, 3),
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "duration_seconds": duration,
                    "overlap_seconds": overlap_seconds,
                    "overlap_ratio": overlap_ratio,
                    "purity_score": _purity_score(
                        clip_duration=duration,
                        overlap_ratio=overlap_ratio,
                        max_clip_duration_sec=request.max_clip_duration_sec,
                        degraded=request.degraded_diarization,
                    ),
                    "status": _candidate_status(
                        clip_duration=duration,
                        overlap_seconds=overlap_seconds,
                        overlap_ratio=overlap_ratio,
                        min_clip_duration_sec=request.min_clip_duration_sec,
                        degraded_diarization=request.degraded_diarization,
                    ),
                    "recommended": False,
                    "extraction_backend": "none",
                }
            )

        ranked_candidates.sort(
            key=lambda item: (
                -float(item["purity_score"]),
                float(item["overlap_ratio"]),
                -float(item["duration_seconds"]),
                float(item["source_start"]),
                float(item["source_end"]),
                str(item["source_kind"]),
            )
        )

        accepted_count = 0
        manifest_entries: list[dict[str, Any]] = []
        for position, candidate in enumerate(ranked_candidates, start=1):
            entry = dict(candidate)
            entry["snippet_id"] = f"{_speaker_slug(speaker)}-{position:02d}"
            entry["ranking_position"] = position
            if entry["status"] is None and accepted_count >= max(request.max_snippets_per_speaker, 0):
                entry["status"] = "rejected_rank_limit"
            if entry["status"] is None:
                out_path = speaker_dir / f"{accepted_count + 1}.wav"
                backend = _extract_snippet(
                    request.audio_path,
                    out_path,
                    start_sec=float(entry["clip_start"]),
                    end_sec=float(entry["clip_end"]),
                )
                if backend is None:
                    entry["status"] = "rejected_failed_extract"
                else:
                    entry["status"] = "accepted"
                    entry["extraction_backend"] = backend
                    entry["relative_path"] = f"{_speaker_slug(speaker)}/{out_path.name}"
                    accepted_count += 1
                    all_outputs.append(out_path)
            manifest_entries.append(entry)

        for entry in manifest_entries:
            if entry["status"] == "accepted":
                entry["recommended"] = True
                break
        manifest_speakers[speaker] = manifest_entries

    _write_manifest(
        snippets_dir=request.snippets_dir,
        source_kind="turn" if request.speaker_turns else "segment",
        degraded_diarization=request.degraded_diarization,
        pad_seconds=request.pad_seconds,
        max_clip_duration_sec=request.max_clip_duration_sec,
        min_clip_duration_sec=request.min_clip_duration_sec,
        max_snippets_per_speaker=request.max_snippets_per_speaker,
        speakers=manifest_speakers,
    )
    all_outputs.sort()
    return all_outputs


__all__ = ["SnippetExportRequest", "export_speaker_snippets", "write_empty_snippets_manifest"]
