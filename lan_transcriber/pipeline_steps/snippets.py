from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from lan_transcriber.utils import safe_float


@dataclass(frozen=True)
class SnippetExportRequest:
    audio_path: Path
    diar_segments: Sequence[dict[str, Any]]
    snippets_dir: Path
    duration_sec: float | None


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


def _snippet_window(
    start: float,
    end: float,
    *,
    duration_sec: float | None,
) -> tuple[float, float]:
    seg_duration = max(0.0, end - start)
    target = min(20.0, max(10.0, seg_duration if seg_duration > 0 else 10.0))
    center = start + (seg_duration / 2.0 if seg_duration > 0 else 0.0)
    clip_start = max(0.0, center - (target / 2.0))
    clip_end = clip_start + target
    if duration_sec is not None and duration_sec > 0:
        if clip_end > duration_sec:
            clip_end = duration_sec
            clip_start = max(0.0, clip_end - target)
    if clip_end <= clip_start:
        clip_end = clip_start + min(target, 1.0)
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


def _write_silence_wav(path: Path, duration_sec: float = 1.0) -> None:
    samples = max(int(16000 * max(duration_sec, 0.1)), 1)
    payload = b"\x00\x00" * samples
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(16000)
        wav_out.writeframes(payload)


def _overlap_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def export_speaker_snippets(request: SnippetExportRequest) -> list[Path]:
    _clear_dir(request.snippets_dir)

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in request.diar_segments:
        speaker = str(segment.get("speaker", "S1"))
        by_speaker.setdefault(speaker, []).append(segment)

    all_outputs: list[Path] = []
    for speaker in sorted(by_speaker):
        ranked = sorted(
            by_speaker[speaker],
            key=lambda row: (
                -(safe_float(row.get("end")) - safe_float(row.get("start"))),
                safe_float(row.get("start")),
                safe_float(row.get("end")),
            ),
        )
        chosen: list[dict[str, Any]] = []
        for candidate in ranked:
            start = safe_float(candidate.get("start"))
            end = safe_float(candidate.get("end"), default=start)
            if end <= start:
                continue
            overlaps = any(
                _overlap_seconds(
                    start,
                    end,
                    safe_float(existing.get("start")),
                    safe_float(existing.get("end")),
                )
                > 0.5
                for existing in chosen
            )
            if overlaps:
                continue
            chosen.append(candidate)
            if len(chosen) == 3:
                break
        if len(chosen) < 2:
            for candidate in ranked:
                if candidate in chosen:
                    continue
                chosen.append(candidate)
                if len(chosen) == 2:
                    break

        speaker_dir = request.snippets_dir / _speaker_slug(speaker)
        for idx, segment in enumerate(
            sorted(chosen, key=lambda row: safe_float(row.get("start"))), start=1
        ):
            seg_start = safe_float(segment.get("start"))
            seg_end = safe_float(segment.get("end"), default=seg_start)
            clip_start, clip_end = _snippet_window(
                seg_start,
                seg_end,
                duration_sec=request.duration_sec,
            )
            out_path = speaker_dir / f"{idx}.wav"
            written = _extract_wav_snippet_with_wave(
                request.audio_path,
                out_path,
                start_sec=clip_start,
                end_sec=clip_end,
            )
            if not written:
                written = _extract_wav_snippet_with_ffmpeg(
                    request.audio_path,
                    out_path,
                    start_sec=clip_start,
                    end_sec=clip_end,
                )
            if not written:
                _write_silence_wav(out_path, duration_sec=min(max(clip_end - clip_start, 1.0), 2.0))
            all_outputs.append(out_path)
    all_outputs.sort()
    return all_outputs


__all__ = ["SnippetExportRequest", "export_speaker_snippets"]
