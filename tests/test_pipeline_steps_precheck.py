from __future__ import annotations

from pathlib import Path

from lan_transcriber.pipeline_steps import precheck


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "a.wav"
    path.write_bytes(b"fake")
    return path


def test_run_precheck_duration_threshold(tmp_path: Path, monkeypatch):
    audio = _audio(tmp_path)
    monkeypatch.setattr(precheck, "_audio_duration_from_wave", lambda _p: 10.0)
    monkeypatch.setattr(precheck, "_speech_ratio_from_wave", lambda _p: 0.8)

    result = precheck.run_precheck(audio, min_duration_sec=20.0, min_speech_ratio=0.1)

    assert result.quarantine_reason == "duration_lt_20s"


def test_run_precheck_vad_threshold(tmp_path: Path, monkeypatch):
    audio = _audio(tmp_path)
    monkeypatch.setattr(precheck, "_audio_duration_from_wave", lambda _p: 30.0)
    monkeypatch.setattr(precheck, "_speech_ratio_from_wave", lambda _p: 0.02)

    result = precheck.run_precheck(audio, min_duration_sec=20.0, min_speech_ratio=0.1)

    assert result.quarantine_reason == "speech_ratio_lt_0.10"


def test_run_precheck_metrics_unavailable(tmp_path: Path, monkeypatch):
    audio = _audio(tmp_path)
    monkeypatch.setattr(precheck, "_audio_duration_from_wave", lambda _p: None)
    monkeypatch.setattr(precheck, "_audio_duration_from_ffprobe", lambda _p: None)
    monkeypatch.setattr(precheck, "_speech_ratio_from_wave", lambda _p: None)
    monkeypatch.setattr(precheck, "_speech_ratio_from_ffmpeg", lambda _p: None)

    result = precheck.run_precheck(audio, min_duration_sec=20.0, min_speech_ratio=0.1)

    assert result.quarantine_reason == "precheck_metrics_unavailable"
    assert result.duration_sec is None
    assert result.speech_ratio is None


def test_audio_duration_from_ffprobe_parses_successful_output(tmp_path: Path, monkeypatch):
    audio = _audio(tmp_path)
    monkeypatch.setattr(precheck.shutil, "which", lambda _name: "/usr/bin/ffprobe")

    class _Proc:
        returncode = 0
        stdout = "12.5\n"

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc())
    assert precheck._audio_duration_from_ffprobe(audio) == 12.5


def test_audio_duration_from_wave_zero_rate_and_exception_paths(tmp_path: Path, monkeypatch):
    audio = _audio(tmp_path)
    wav = tmp_path / "a.wav"
    wav.write_bytes(audio.read_bytes())

    class _Wave:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getframerate(self):
            return 0

        def getnframes(self):
            return 100

    monkeypatch.setattr(precheck.wave, "open", lambda *_a, **_k: _Wave())
    assert precheck._audio_duration_from_wave(wav) is None

    monkeypatch.setattr(precheck.wave, "open", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    assert precheck._audio_duration_from_wave(wav) is None


def test_speech_ratio_from_wave_short_chunk_returns_zero(tmp_path: Path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")

    class _Wave:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getframerate(self):
            return 16000

        def getnchannels(self):
            return 1

        def getsampwidth(self):
            return 2

        def readframes(self, _n):
            return b"\x00"  # shorter than required half frame

    monkeypatch.setattr(precheck.wave, "open", lambda *_a, **_k: _Wave())
    assert precheck._speech_ratio_from_wave(wav) == 0.0


def test_speech_ratio_from_ffmpeg_voiced_and_error_paths(tmp_path: Path, monkeypatch):
    audio = _audio(tmp_path)
    monkeypatch.setattr(precheck.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Stdout:
        def __init__(self, chunks: list[bytes]):
            self._chunks = chunks

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    class _Proc:
        def __init__(self, chunks: list[bytes], returncode: int):
            self.stdout = _Stdout(chunks)
            self.returncode = returncode

        def wait(self, timeout=None):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(precheck.audioop, "rms", lambda _chunk, _width: 400)
    monkeypatch.setattr(
        precheck.subprocess,
        "Popen",
        lambda *_a, **_k: _Proc([b"\x01" * 960, b""], 0),
    )
    assert precheck._speech_ratio_from_ffmpeg(audio) == 1.0

    monkeypatch.setattr(
        precheck.subprocess,
        "Popen",
        lambda *_a, **_k: _Proc([b"\x01" * 100, b""], 1),
    )
    assert precheck._speech_ratio_from_ffmpeg(audio) is None
