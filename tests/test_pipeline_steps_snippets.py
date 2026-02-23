from __future__ import annotations

import wave
from pathlib import Path

from lan_transcriber.pipeline_steps import snippets
from lan_transcriber.pipeline_steps.snippets import SnippetExportRequest, export_speaker_snippets


def _wav(path: Path, *, duration_sec: float = 1.0) -> Path:
    rate = 16000
    samples = int(rate * duration_sec)
    payload = b"\x00\x00" * samples
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(rate)
        wav_out.writeframes(payload)
    return path


def test_clear_dir_creates_missing_directory(tmp_path: Path):
    target = tmp_path / "snippets"
    snippets._clear_dir(target)
    assert target.exists()
    assert target.is_dir()


def test_snippet_window_clamps_to_recording_duration():
    start, end = snippets._snippet_window(9.0, 13.0, duration_sec=10.0)
    assert 0.0 <= start <= end <= 10.0


def test_extract_wav_snippet_with_ffmpeg_success_and_failure(tmp_path: Path, monkeypatch):
    audio = _wav(tmp_path / "in.wav", duration_sec=2.0)
    out_path = tmp_path / "out" / "1.wav"

    monkeypatch.setattr(snippets.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class _Proc:
        def __init__(self, code: int):
            self.returncode = code

    def _run_ok(*_a, **_k):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"R" * 100)
        return _Proc(0)

    monkeypatch.setattr(snippets.subprocess, "run", _run_ok)
    assert snippets._extract_wav_snippet_with_ffmpeg(audio, out_path, start_sec=0.0, end_sec=0.8)

    monkeypatch.setattr(snippets.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert not snippets._extract_wav_snippet_with_ffmpeg(audio, out_path, start_sec=0.0, end_sec=0.8)


def test_export_speaker_snippets_falls_back_to_silence(tmp_path: Path, monkeypatch):
    audio = _wav(tmp_path / "src.wav", duration_sec=4.0)
    out_dir = tmp_path / "derived" / "snippets"
    stale = out_dir / "S1" / "old.wav"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")

    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: False)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: False)

    outputs = export_speaker_snippets(
        SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 1.5, "speaker": "S1"},
                {"start": 2.0, "end": 3.5, "speaker": "S1"},
            ],
            snippets_dir=out_dir,
            duration_sec=4.0,
        )
    )

    assert len(outputs) == 2
    assert all(path.exists() and path.stat().st_size > 44 for path in outputs)
    assert not stale.exists()
