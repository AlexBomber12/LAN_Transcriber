from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from lan_transcriber.pipeline_steps import noise_detection


def _silent_wav(path: Path, *, duration_sec: float = 4.0) -> Path:
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


def _tone_wav(
    path: Path,
    *,
    duration_sec: float = 4.0,
    amplitude: int = 12000,
    freq: float = 220.0,
) -> Path:
    rate = 16000
    samples = int(rate * duration_sec)
    chunk = bytearray()
    for index in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * freq * (index / rate)))
        chunk += struct.pack("<h", value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(rate)
        wav_out.writeframes(bytes(chunk))
    return path


def _mixed_wav(
    path: Path,
    *,
    speech_ratio: float,
    duration_sec: float = 4.0,
    amplitude: int = 12000,
    freq: float = 220.0,
) -> Path:
    """Produce a WAV where the first ``speech_ratio`` fraction is a tone and the rest is silence."""

    rate = 16000
    total_samples = int(rate * duration_sec)
    voiced_samples = int(total_samples * speech_ratio)
    chunk = bytearray()
    for index in range(voiced_samples):
        value = int(amplitude * math.sin(2 * math.pi * freq * (index / rate)))
        chunk += struct.pack("<h", value)
    chunk += b"\x00\x00" * (total_samples - voiced_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(rate)
        wav_out.writeframes(bytes(chunk))
    return path


def test_compute_speech_ratio_returns_zero_for_silence(tmp_path: Path) -> None:
    audio = _silent_wav(tmp_path / "silent.wav")
    assert noise_detection.compute_wav_speech_ratio(audio) == 0.0


def test_compute_speech_ratio_returns_high_value_for_tone(tmp_path: Path) -> None:
    audio = _tone_wav(tmp_path / "tone.wav")
    ratio = noise_detection.compute_wav_speech_ratio(audio)
    assert ratio is not None
    assert ratio > 0.9


def test_compute_speech_ratio_rejects_non_wav_path(tmp_path: Path) -> None:
    bogus = tmp_path / "x.mp3"
    bogus.write_bytes(b"not wav")
    assert noise_detection.compute_wav_speech_ratio(bogus) is None


def test_compute_speech_ratio_handles_unreadable_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"\x00")
    assert noise_detection.compute_wav_speech_ratio(broken) is None


def _eight_bit_tone_wav(path: Path, *, duration_sec: float = 4.0) -> Path:
    rate = 16000
    samples = int(rate * duration_sec)
    chunk = bytearray()
    for index in range(samples):
        # Centred unsigned 8-bit PCM around 128, swinging by 100 to mimic tone.
        value = int(128 + 100 * math.sin(2 * math.pi * 220 * (index / rate)))
        chunk.append(max(0, min(255, value)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(1)
        wav_out.setframerate(rate)
        wav_out.writeframes(bytes(chunk))
    return path


def test_compute_speech_ratio_scales_threshold_for_8bit_pcm(tmp_path: Path) -> None:
    """Eight-bit PCM tones must read as speech, not noise (RMS caps at ~128)."""

    audio = _eight_bit_tone_wav(tmp_path / "tone8.wav")
    ratio = noise_detection.compute_wav_speech_ratio(audio)
    assert ratio is not None
    assert ratio > 0.5


def test_rms_threshold_scales_with_sample_width() -> None:
    assert noise_detection._rms_threshold_for_width(2) == pytest.approx(350.0)
    assert noise_detection._rms_threshold_for_width(1) < 2.0
    assert noise_detection._rms_threshold_for_width(4) > 350.0
    assert noise_detection._rms_threshold_for_width(0) == 1.0


def test_compute_speech_ratio_returns_zero_when_no_full_frames(tmp_path: Path) -> None:
    rate = 16000
    path = tmp_path / "tiny.wav"
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(rate)
        wav_out.writeframes(b"")
    assert noise_detection.compute_wav_speech_ratio(path) == 0.0


def _build_manifest(
    *,
    snippets_dir: Path,
    speakers: dict[str, list[Path | None]],
) -> dict[str, object]:
    speakers_payload: dict[str, list[dict[str, object]]] = {}
    for speaker_label, snippet_paths in speakers.items():
        entries: list[dict[str, object]] = []
        for index, snippet in enumerate(snippet_paths, start=1):
            if snippet is None:
                entries.append(
                    {
                        "status": "rejected_short",
                        "ranking_position": index,
                    }
                )
                continue
            relative = snippet.relative_to(snippets_dir).as_posix()
            entries.append(
                {
                    "status": "accepted",
                    "ranking_position": index,
                    "relative_path": relative,
                }
            )
        speakers_payload[speaker_label] = entries
    return {"version": 1, "speakers": speakers_payload}


def test_analyze_snippet_noise_flags_silence_speakers_and_skips_real_ones(
    tmp_path: Path,
) -> None:
    snippets_dir = tmp_path / "snippets"
    silent_dir = snippets_dir / "SPEAKER_00"
    voiced_dir = snippets_dir / "SPEAKER_01"
    silent = _silent_wav(silent_dir / "1.wav")
    voiced = _tone_wav(voiced_dir / "1.wav")
    manifest = _build_manifest(
        snippets_dir=snippets_dir,
        speakers={"SPEAKER_00": [silent], "SPEAKER_01": [voiced]},
    )
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=snippets_dir, threshold=0.3
    )
    assert summary["noise_speakers"] == ["SPEAKER_00"]
    assert manifest["noise_speakers"] == ["SPEAKER_00"]
    silent_entry = manifest["speakers"]["SPEAKER_00"][0]
    voiced_entry = manifest["speakers"]["SPEAKER_01"][0]
    assert silent_entry["noise_suspected"] is True
    assert silent_entry["speech_ratio"] == 0.0
    assert "noise_suspected" not in voiced_entry
    assert voiced_entry["speech_ratio"] > 0.3


def test_analyze_snippet_noise_threshold_boundary_just_below(tmp_path: Path) -> None:
    snippets_dir = tmp_path / "snippets"
    speaker_dir = snippets_dir / "SPEAKER_00"
    snippet = _mixed_wav(speaker_dir / "1.wav", speech_ratio=0.20)
    manifest = _build_manifest(
        snippets_dir=snippets_dir,
        speakers={"SPEAKER_00": [snippet]},
    )
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=snippets_dir, threshold=0.3
    )
    assert summary["noise_speakers"] == ["SPEAKER_00"]
    assert manifest["speakers"]["SPEAKER_00"][0]["noise_suspected"] is True


def test_analyze_snippet_noise_threshold_boundary_just_above(tmp_path: Path) -> None:
    snippets_dir = tmp_path / "snippets"
    speaker_dir = snippets_dir / "SPEAKER_00"
    snippet = _mixed_wav(speaker_dir / "1.wav", speech_ratio=0.55)
    manifest = _build_manifest(
        snippets_dir=snippets_dir,
        speakers={"SPEAKER_00": [snippet]},
    )
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=snippets_dir, threshold=0.3
    )
    assert summary["noise_speakers"] == []
    assert "noise_suspected" not in manifest["speakers"]["SPEAKER_00"][0]


def test_analyze_snippet_noise_clears_stale_per_entry_flags(tmp_path: Path) -> None:
    """Re-running on an already-annotated manifest must not leave stale flags."""

    snippets_dir = tmp_path / "snippets"
    voiced_dir = snippets_dir / "SPEAKER_REAL"
    voiced = _tone_wav(voiced_dir / "1.wav")
    manifest = _build_manifest(
        snippets_dir=snippets_dir,
        speakers={"SPEAKER_REAL": [voiced]},
    )
    # Seed stale per-entry flags from a prior run.
    manifest["speakers"]["SPEAKER_REAL"][0]["noise_suspected"] = True
    manifest["speakers"]["SPEAKER_REAL"][0]["speech_ratio"] = 0.0

    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=snippets_dir, threshold=0.3
    )
    assert summary["noise_speakers"] == []
    entry = manifest["speakers"]["SPEAKER_REAL"][0]
    assert "noise_suspected" not in entry
    assert entry["speech_ratio"] > 0.3


def test_analyze_snippet_noise_handles_missing_speakers_payload() -> None:
    manifest: dict[str, object] = {"version": 1}
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=Path("/tmp/missing"), threshold=0.3
    )
    assert summary["noise_speakers"] == []
    assert manifest["noise_speakers"] == []


def test_analyze_snippet_noise_skips_non_dict_speaker_entries(tmp_path: Path) -> None:
    manifest: dict[str, object] = {
        "version": 1,
        "speakers": {
            "SPEAKER_00": "not-a-list",
            "SPEAKER_01": [
                "stringy",
                {"status": "rejected_short"},
                {"status": "accepted"},
            ],
        },
    }
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=tmp_path, threshold=0.3
    )
    assert summary["noise_speakers"] == []
    metrics = summary["speaker_metrics"]["SPEAKER_01"]
    assert metrics["evaluated_snippets"] == 0
    assert metrics["missing_snippets"] == 1


def test_apply_noise_flags_to_manifest_persists_changes(tmp_path: Path) -> None:
    snippets_dir = tmp_path / "snippets"
    silent = _silent_wav(snippets_dir / "SPEAKER_00" / "1.wav")
    manifest_path = tmp_path / "snippets_manifest.json"
    manifest = _build_manifest(
        snippets_dir=snippets_dir,
        speakers={"SPEAKER_00": [silent]},
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = noise_detection.apply_noise_flags_to_manifest(
        manifest_path,
        snippets_dir=snippets_dir,
        threshold=0.3,
    )
    assert summary["noise_speakers"] == ["SPEAKER_00"]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["noise_speakers"] == ["SPEAKER_00"]
    assert persisted["speakers"]["SPEAKER_00"][0]["noise_suspected"] is True


def test_apply_noise_flags_to_manifest_returns_empty_when_missing(tmp_path: Path) -> None:
    summary = noise_detection.apply_noise_flags_to_manifest(
        tmp_path / "missing.json",
        snippets_dir=tmp_path,
        threshold=0.3,
    )
    assert summary["noise_speakers"] == []
    assert summary["threshold"] == 0.3


def test_apply_noise_flags_to_manifest_handles_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "snippets_manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")
    summary = noise_detection.apply_noise_flags_to_manifest(
        manifest_path, snippets_dir=tmp_path, threshold=0.3
    )
    assert summary["noise_speakers"] == []


def test_apply_noise_flags_to_manifest_handles_non_dict_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "snippets_manifest.json"
    manifest_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    summary = noise_detection.apply_noise_flags_to_manifest(
        manifest_path, snippets_dir=tmp_path, threshold=0.3
    )
    assert summary["noise_speakers"] == []


def test_update_diarization_metadata_with_noise_persists_summary(tmp_path: Path) -> None:
    metadata_path = tmp_path / "diarization_metadata.json"
    metadata_path.write_text(json.dumps({"version": 1, "mode": "pyannote"}), encoding="utf-8")
    summary = {
        "noise_speakers": ["SPEAKER_00"],
        "speaker_metrics": {
            "SPEAKER_00": {"speech_ratio": 0.0, "evaluated_snippets": 1, "flagged": True},
        },
        "threshold": 0.3,
    }
    noise_detection.update_diarization_metadata_with_noise(
        metadata_path, summary=summary
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["noise_speakers"] == ["SPEAKER_00"]
    assert payload["noise_speech_ratio_threshold"] == 0.3
    assert payload["noise_speaker_metrics"]["SPEAKER_00"]["flagged"] is True


def test_update_diarization_metadata_handles_missing_path(tmp_path: Path) -> None:
    noise_detection.update_diarization_metadata_with_noise(
        tmp_path / "missing.json",
        summary={"noise_speakers": [], "speaker_metrics": {}, "threshold": 0.3},
    )


def test_update_diarization_metadata_handles_invalid_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("not-json", encoding="utf-8")
    noise_detection.update_diarization_metadata_with_noise(
        metadata_path,
        summary={"noise_speakers": ["SPEAKER_00"], "speaker_metrics": {}, "threshold": 0.3},
    )
    # File is left untouched on parse failure
    assert metadata_path.read_text(encoding="utf-8") == "not-json"


def test_update_diarization_metadata_handles_non_dict_payload(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    noise_detection.update_diarization_metadata_with_noise(
        metadata_path,
        summary={"noise_speakers": [], "speaker_metrics": {}, "threshold": 0.3},
    )
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == [1, 2]


def test_speaker_with_only_missing_relative_path_is_not_flagged(tmp_path: Path) -> None:
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    manifest = {
        "version": 1,
        "speakers": {
            "SPEAKER_00": [
                {"status": "accepted", "relative_path": ""},
            ],
        },
    }
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=snippets_dir, threshold=0.3
    )
    assert summary["noise_speakers"] == []
    metrics = summary["speaker_metrics"]["SPEAKER_00"]
    assert metrics["missing_snippets"] == 1
    assert metrics["evaluated_snippets"] == 0
    assert metrics["flagged"] is False


def test_analyze_snippet_noise_clamps_threshold(tmp_path: Path) -> None:
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    manifest = {"version": 1, "speakers": {}}
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=snippets_dir, threshold=2.0
    )
    assert summary["threshold"] == 1.0
    assert manifest["noise_speech_ratio_threshold"] == 1.0


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_analyze_snippet_noise_round_trip_thresholds(tmp_path: Path, threshold: float) -> None:
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    manifest = {"version": 1, "speakers": {}}
    summary = noise_detection.analyze_snippet_noise(
        manifest, snippets_dir=snippets_dir, threshold=threshold
    )
    assert summary["threshold"] == threshold
