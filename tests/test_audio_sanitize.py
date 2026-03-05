from __future__ import annotations

from pathlib import Path
import subprocess
import wave

import pytest

import lan_transcriber.audio_sanitize as audio_sanitize


def _write_wav(path: Path, *, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setframerate(sample_rate)
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.writeframes(b"\x00\x00" * sample_rate)


def test_sanitize_audio_for_pipeline_short_circuits_wav(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    _write_wav(source, sample_rate=16000, channels=1)

    result = audio_sanitize.sanitize_audio_for_pipeline(
        source,
        tmp_path / "derived.wav",
    )

    assert result == source


def test_sanitize_audio_for_pipeline_invalid_wav_does_not_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "broken.wav"
    source.write_bytes(b"not-a-wave")
    monkeypatch.setattr(audio_sanitize.shutil, "which", lambda _name: None)

    with pytest.raises(audio_sanitize.AudioSanitizeError, match="ffmpeg is required"):
        audio_sanitize.sanitize_audio_for_pipeline(source, tmp_path / "output.wav")


def test_sanitize_audio_for_pipeline_non_target_wav_is_transcoded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.wav"
    _write_wav(source, sample_rate=44100, channels=2)
    output = tmp_path / "derived" / "audio_sanitized.wav"
    monkeypatch.setattr(audio_sanitize.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Proc:
        returncode = 0
        stderr = ""
        stdout = ""

    monkeypatch.setattr(audio_sanitize.subprocess, "run", lambda *_a, **_k: _Proc())

    result = audio_sanitize.sanitize_audio_for_pipeline(source, output)
    assert result == output


def test_sanitize_audio_for_pipeline_requires_ffmpeg_for_non_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"fake")
    monkeypatch.setattr(audio_sanitize.shutil, "which", lambda _name: None)

    with pytest.raises(audio_sanitize.AudioSanitizeError, match="ffmpeg is required"):
        audio_sanitize.sanitize_audio_for_pipeline(source, tmp_path / "output.wav")


def test_sanitize_audio_for_pipeline_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"fake")
    output = tmp_path / "derived" / "audio_sanitized.wav"
    monkeypatch.setattr(audio_sanitize.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    seen: dict[str, object] = {}

    class _Proc:
        returncode = 0
        stderr = ""
        stdout = ""

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(audio_sanitize.subprocess, "run", _fake_run)

    result = audio_sanitize.sanitize_audio_for_pipeline(
        source,
        output,
        timeout_seconds=42.0,
    )

    assert result == output
    assert seen["cmd"] == [
        "/usr/bin/ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    assert seen["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 42.0,
    }


def test_sanitize_audio_for_pipeline_raises_on_ffmpeg_non_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"fake")
    output = tmp_path / "audio_sanitized.wav"
    monkeypatch.setattr(audio_sanitize.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Proc:
        returncode = 2
        stderr = "decode failed\nbitrate mismatch"
        stdout = ""

    monkeypatch.setattr(audio_sanitize.subprocess, "run", lambda *_a, **_k: _Proc())

    with pytest.raises(audio_sanitize.AudioSanitizeError) as exc_info:
        audio_sanitize.sanitize_audio_for_pipeline(source, output)

    message = str(exc_info.value)
    assert "exit code 2" in message
    assert "/usr/bin/ffmpeg" in message
    assert "decode failed bitrate mismatch" in message


def test_sanitize_audio_for_pipeline_uses_stdout_when_stderr_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"fake")
    output = tmp_path / "audio_sanitized.wav"
    monkeypatch.setattr(audio_sanitize.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Proc:
        returncode = 1
        stderr = ""
        stdout = "bad stream\ncannot decode"

    monkeypatch.setattr(audio_sanitize.subprocess, "run", lambda *_a, **_k: _Proc())

    with pytest.raises(audio_sanitize.AudioSanitizeError) as exc_info:
        audio_sanitize.sanitize_audio_for_pipeline(source, output)

    assert "bad stream cannot decode" in str(exc_info.value)


def test_sanitize_audio_for_pipeline_raises_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"fake")
    output = tmp_path / "audio_sanitized.wav"
    monkeypatch.setattr(audio_sanitize.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["ffmpeg", "-i"], timeout=7)

    monkeypatch.setattr(audio_sanitize.subprocess, "run", _raise_timeout)

    with pytest.raises(audio_sanitize.AudioSanitizeError) as exc_info:
        audio_sanitize.sanitize_audio_for_pipeline(source, output, timeout_seconds=7.0)

    message = str(exc_info.value)
    assert "timed out after 7.0s" in message
    assert "/usr/bin/ffmpeg" in message
