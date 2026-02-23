from __future__ import annotations

import audioop
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrecheckResult:
    duration_sec: float | None
    speech_ratio: float | None
    quarantine_reason: str | None = None


def _audio_duration_from_wave(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            frames = src.getnframes()
        if rate <= 0:
            return None
        return frames / float(rate)
    except Exception:
        return None


def _audio_duration_from_ffprobe(audio_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _speech_ratio_from_wave(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            sample_width = src.getsampwidth()
            frame_samples = max(int(rate * 0.03), 1)
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
                if audioop.rms(chunk, sample_width) >= 350:
                    voiced += 1
            if total == 0:
                return 0.0
            return voiced / float(total)
    except Exception:
        return None


def _speech_ratio_from_ffmpeg(audio_path: Path) -> float | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            if proc.stdout is None:
                return None
            frame_bytes = 960  # 30ms @ 16kHz * 2 bytes sample
            voiced = 0
            total = 0
            while True:
                chunk = proc.stdout.read(frame_bytes)
                if not chunk:
                    break
                if len(chunk) < frame_bytes:
                    break
                total += 1
                if audioop.rms(chunk, 2) >= 300:
                    voiced += 1
            proc.wait()
            if proc.returncode != 0:
                return None
            if total == 0:
                return 0.0
            return voiced / float(total)
    except Exception:
        return None


def run_precheck(
    audio_path: Path,
    *,
    min_duration_sec: float,
    min_speech_ratio: float,
) -> PrecheckResult:
    """Compute duration + VAD speech ratio and decide quarantine status."""
    duration_sec = _audio_duration_from_wave(audio_path)
    if duration_sec is None:
        duration_sec = _audio_duration_from_ffprobe(audio_path)

    speech_ratio = _speech_ratio_from_wave(audio_path)
    if speech_ratio is None:
        speech_ratio = _speech_ratio_from_ffmpeg(audio_path)

    quarantine_reason: str | None = None
    if duration_sec is None or speech_ratio is None:
        quarantine_reason = "precheck_metrics_unavailable"
    elif duration_sec < min_duration_sec:
        quarantine_reason = f"duration_lt_{min_duration_sec:.0f}s"
    elif speech_ratio < min_speech_ratio:
        quarantine_reason = f"speech_ratio_lt_{min_speech_ratio:.2f}"

    return PrecheckResult(
        duration_sec=round(duration_sec, 3) if duration_sec is not None else None,
        speech_ratio=round(speech_ratio, 4) if speech_ratio is not None else None,
        quarantine_reason=quarantine_reason,
    )


__all__ = ["PrecheckResult", "run_precheck"]
