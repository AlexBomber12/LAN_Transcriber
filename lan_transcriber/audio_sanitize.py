from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

DEFAULT_AUDIO_SANITIZE_TIMEOUT_SECONDS = 120.0


class AudioSanitizeError(Exception):
    """Raised when ffmpeg-based audio sanitization fails."""


def sanitize_audio_for_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    timeout_seconds: float = DEFAULT_AUDIO_SANITIZE_TIMEOUT_SECONDS,
) -> Path:
    input_audio = Path(input_path)
    output_audio = Path(output_path)

    # Deterministic short-circuit for WAV uploads.
    if input_audio.suffix.lower() == ".wav":
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
        stderr_snippet = ((proc.stderr or proc.stdout or "").strip().replace("\n", " "))[
            :400
        ]
        raise AudioSanitizeError(
            f"ffmpeg failed with exit code {proc.returncode}: {shlex.join(cmd)}; "
            f"stderr={stderr_snippet}"
        )

    return output_audio


__all__ = [
    "AudioSanitizeError",
    "DEFAULT_AUDIO_SANITIZE_TIMEOUT_SECONDS",
    "sanitize_audio_for_pipeline",
]
