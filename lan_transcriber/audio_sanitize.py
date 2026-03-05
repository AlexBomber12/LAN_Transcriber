from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
import wave

DEFAULT_AUDIO_SANITIZE_TIMEOUT_SECONDS = 120.0


class AudioSanitizeError(Exception):
    """Raised when ffmpeg-based audio sanitization fails."""


def _is_target_pcm_wav(audio_path: Path) -> bool:
    if audio_path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            fmt = (
                wav_file.getframerate(),
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getcomptype(),
            )
            if fmt != (16000, 1, 2, "NONE"):
                return False

            frame_count = wav_file.getnframes()
            if frame_count <= 0:
                return False

            expected_bytes = frame_count * 2  # mono * sampwidth(2)
            read_bytes = 0
            while read_bytes < expected_bytes:
                remaining_frames = (expected_bytes - read_bytes + 1) // 2
                chunk = wav_file.readframes(min(4096, remaining_frames))
                if not chunk:
                    return False
                read_bytes += len(chunk)
            return read_bytes == expected_bytes
    except Exception:
        return False


def sanitize_audio_for_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    timeout_seconds: float = DEFAULT_AUDIO_SANITIZE_TIMEOUT_SECONDS,
) -> Path:
    input_audio = Path(input_path)
    output_audio = Path(output_path)

    # Deterministic short-circuit only for already normalized PCM WAV.
    if _is_target_pcm_wav(input_audio):
        return input_audio

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioSanitizeError(
            "ffmpeg is required for audio sanitization but was not found in PATH"
        )

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_audio),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_audio),
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioSanitizeError(
            f"ffmpeg timed out after {timeout_seconds:.1f}s: {shlex.join(cmd)}"
        ) from exc

    if proc.returncode != 0:
        diagnostics = (proc.stderr or "").strip()
        if not diagnostics:
            diagnostics = (proc.stdout or "").strip()
        diagnostics = diagnostics.replace("\n", " ")[:400]
        raise AudioSanitizeError(
            f"ffmpeg failed with exit code {proc.returncode}: {shlex.join(cmd)}; "
            f"stderr={diagnostics}"
        )

    return output_audio


__all__ = [
    "AudioSanitizeError",
    "DEFAULT_AUDIO_SANITIZE_TIMEOUT_SECONDS",
    "sanitize_audio_for_pipeline",
]
