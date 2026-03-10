from __future__ import annotations

import json
import wave
from pathlib import Path

from lan_transcriber.pipeline_steps import snippets
from lan_transcriber.pipeline_steps.snippets import SnippetExportRequest, export_speaker_snippets


def _wav(path: Path, *, duration_sec: float = 12.0) -> Path:
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


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_clear_dir_creates_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / "snippets"
    snippets._clear_dir(target)  # noqa: SLF001
    assert target.exists()
    assert target.is_dir()


def test_snippet_window_clamps_to_recording_duration_and_caps_long_clip() -> None:
    start, end = snippets._snippet_window(2.0, 12.0, pad_seconds=0.25, max_duration_sec=4.0, duration_sec=10.0)  # noqa: SLF001
    assert (start, end) == (5.0, 9.0)

    clamped_start, clamped_end = snippets._snippet_window(9.0, 13.0, pad_seconds=0.25, max_duration_sec=8.0, duration_sec=10.0)  # noqa: SLF001
    assert 0.0 <= clamped_start <= clamped_end <= 10.0


def test_extract_snippet_prefers_wave_then_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    audio = _wav(tmp_path / "in.wav", duration_sec=2.0)
    out_path = tmp_path / "out" / "1.wav"

    assert snippets._extract_snippet(audio, out_path, start_sec=0.0, end_sec=0.8) == "wave"  # noqa: SLF001

    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: False)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: True)
    assert snippets._extract_snippet(audio, out_path, start_sec=0.0, end_sec=0.8) == "ffmpeg"  # noqa: SLF001

    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: False)
    assert snippets._extract_snippet(audio, out_path, start_sec=0.0, end_sec=0.8) is None  # noqa: SLF001


def test_extract_wav_snippet_with_ffmpeg_success_and_failure(tmp_path: Path, monkeypatch) -> None:
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
    assert snippets._extract_wav_snippet_with_ffmpeg(audio, out_path, start_sec=0.0, end_sec=0.8)  # noqa: SLF001

    monkeypatch.setattr(snippets.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert not snippets._extract_wav_snippet_with_ffmpeg(audio, out_path, start_sec=0.0, end_sec=0.8)  # noqa: SLF001


def test_snippet_helper_edges_cover_candidate_and_trim_branches(tmp_path: Path, monkeypatch) -> None:
    request = SnippetExportRequest(
        audio_path=tmp_path / "dummy.wav",
        diar_segments=[],
        snippets_dir=tmp_path / "snips",
        duration_sec=None,
        speaker_turns=[{"start": 2.0, "end": 1.0, "speaker": "S1"}],
    )
    assert snippets._candidate_sources(request)[0]["source_end"] == 2.0  # noqa: SLF001

    clamped_start, clamped_end = snippets._snippet_window(5.0, 16.0, pad_seconds=0.0, max_duration_sec=4.0, duration_sec=10.0)  # noqa: SLF001
    assert (clamped_start, clamped_end) == (6.0, 10.0)

    zero_start, zero_end = snippets._snippet_window(-4.0, 6.0, pad_seconds=0.0, max_duration_sec=4.0, duration_sec=None)  # noqa: SLF001
    assert (zero_start, zero_end) == (0.0, 4.0)

    trimmed = snippets._trim_padding_to_clean_region(  # noqa: SLF001
        "S1",
        clip_start=0.9,
        clip_end=1.05,
        diar_segments=[
            {"start": 0.9, "end": 1.2, "speaker": "S2"},
            {"start": 1.3, "end": 1.3, "speaker": "S2"},
        ],
    )
    assert trimmed == (1.2, 1.2)
    assert snippets._merge_intervals([(0.0, 1.0), (0.5, 1.5)]) == [(0.0, 1.5)]  # noqa: SLF001
    assert snippets._overlap_against_other_speakers(  # noqa: SLF001
        "S1",
        clip_start=0.0,
        clip_end=1.0,
        diar_segments=[
            {"start": 0.2, "end": 0.6, "speaker": "S2"},
            {"start": 0.6, "end": 0.6, "speaker": "S2"},
        ],
    ) == 0.4

    assert not snippets._extract_wav_snippet_with_wave(  # noqa: SLF001
        tmp_path / "in.mp3",
        tmp_path / "out.wav",
        start_sec=0.0,
        end_sec=0.8,
    )
    monkeypatch.setattr(snippets.shutil, "which", lambda _name: None)
    assert not snippets._extract_wav_snippet_with_ffmpeg(  # noqa: SLF001
        tmp_path / "in.wav",
        tmp_path / "out.wav",
        start_sec=0.0,
        end_sec=0.8,
    )


def test_export_speaker_snippets_prefers_clean_turns_and_writes_manifest(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "src.wav")
    out_dir = tmp_path / "derived" / "snippets"

    outputs = export_speaker_snippets(
        SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 2.0, "speaker": "S1"},
                {"start": 4.0, "end": 7.0, "speaker": "S1"},
                {"start": 5.5, "end": 6.2, "speaker": "S2"},
            ],
            speaker_turns=[
                {"start": 0.0, "end": 2.0, "speaker": "S1"},
                {"start": 4.0, "end": 7.0, "speaker": "S1"},
            ],
            snippets_dir=out_dir,
            duration_sec=12.0,
        )
    )

    assert [path.relative_to(out_dir).as_posix() for path in outputs] == ["S1/1.wav"]
    manifest = _manifest(tmp_path / "derived" / "snippets_manifest.json")
    entries = manifest["speakers"]["S1"]
    assert [entry["status"] for entry in entries] == ["accepted", "rejected_overlap"]
    assert entries[0]["recommended"] is True
    assert entries[0]["relative_path"] == "S1/1.wav"
    assert entries[0]["source_kind"] == "turn"
    assert entries[0]["purity_score"] > entries[1]["purity_score"]


def test_export_speaker_snippets_records_extract_failure_without_silence(tmp_path: Path, monkeypatch) -> None:
    audio = _wav(tmp_path / "src.wav")
    out_dir = tmp_path / "derived" / "snippets"
    stale = out_dir / "S1" / "old.wav"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")

    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: False)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: False)

    outputs = export_speaker_snippets(
        SnippetExportRequest(
            audio_path=audio,
            diar_segments=[{"start": 0.0, "end": 1.5, "speaker": "S1"}],
            snippets_dir=out_dir,
            duration_sec=12.0,
        )
    )

    assert outputs == []
    assert not stale.exists()
    assert list(out_dir.rglob("*.wav")) == []
    manifest = _manifest(tmp_path / "derived" / "snippets_manifest.json")
    assert manifest["speakers"]["S1"][0]["status"] == "rejected_failed_extract"
    assert "relative_path" not in manifest["speakers"]["S1"][0]


def test_export_speaker_snippets_writes_stable_manifest_and_limits_to_three(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "src.wav", duration_sec=20.0)
    out_dir = tmp_path / "derived" / "snippets"

    outputs = export_speaker_snippets(
        SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 1.2, "speaker": "S1"},
                {"start": 2.0, "end": 3.2, "speaker": "S1"},
                {"start": 4.0, "end": 5.2, "speaker": "S1"},
                {"start": 6.0, "end": 7.2, "speaker": "S1"},
            ],
            snippets_dir=out_dir,
            duration_sec=20.0,
        )
    )

    assert [path.relative_to(out_dir).as_posix() for path in outputs] == [
        "S1/1.wav",
        "S1/2.wav",
        "S1/3.wav",
    ]
    manifest_path = tmp_path / "derived" / "snippets_manifest.json"
    first_manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(first_manifest_text)
    entries = manifest["speakers"]["S1"]
    assert [entry["ranking_position"] for entry in entries] == [1, 2, 3, 4]
    assert [entry["status"] for entry in entries] == [
        "accepted",
        "accepted",
        "accepted",
        "rejected_rank_limit",
    ]

    export_speaker_snippets(
        SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 1.2, "speaker": "S1"},
                {"start": 2.0, "end": 3.2, "speaker": "S1"},
                {"start": 4.0, "end": 5.2, "speaker": "S1"},
                {"start": 6.0, "end": 7.2, "speaker": "S1"},
            ],
            snippets_dir=out_dir,
            duration_sec=20.0,
        )
    )
    assert manifest_path.read_text(encoding="utf-8") == first_manifest_text


def test_export_speaker_snippets_rejects_short_and_degraded_candidates(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "src.wav", duration_sec=4.0)

    short_outputs = export_speaker_snippets(
        SnippetExportRequest(
            audio_path=audio,
            diar_segments=[{"start": 1.0, "end": 1.0, "speaker": "S1"}],
            snippets_dir=tmp_path / "short" / "snippets",
            duration_sec=4.0,
            pad_seconds=0.0,
            min_clip_duration_sec=0.5,
        )
    )
    assert short_outputs == []
    short_manifest = _manifest(tmp_path / "short" / "snippets_manifest.json")
    assert short_manifest["speakers"]["S1"][0]["status"] == "rejected_short"

    degraded_outputs = export_speaker_snippets(
        SnippetExportRequest(
            audio_path=audio,
            diar_segments=[{"start": 0.0, "end": 1.0, "speaker": "S1"}],
            snippets_dir=tmp_path / "degraded" / "snippets",
            duration_sec=4.0,
            degraded_diarization=True,
        )
    )
    assert degraded_outputs == []
    degraded_manifest = _manifest(tmp_path / "degraded" / "snippets_manifest.json")
    assert degraded_manifest["degraded_diarization"] is True
    assert degraded_manifest["speakers"]["S1"][0]["status"] == "rejected_degraded"
