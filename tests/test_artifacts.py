from pathlib import Path
import json

from lan_transcriber.artifacts import (
    atomic_write_json,
    atomic_write_text,
    build_recording_artifacts,
    stage_raw_audio,
)


def test_recording_artifact_layout_and_writes(tmp_path: Path) -> None:
    artifacts = build_recording_artifacts(
        recordings_root=tmp_path / "recordings",
        recording_id="Meeting 42",
        audio_ext=".mp3",
    )

    src_audio = tmp_path / "src.mp3"
    src_audio.write_bytes(b"\x00\x01")
    stage_raw_audio(src_audio, artifacts.raw_audio_path)

    atomic_write_text(artifacts.transcript_txt_path, "hello world")
    atomic_write_json(artifacts.summary_json_path, {"summary": "- one"})
    atomic_write_json(artifacts.segments_json_path, [{"speaker": "S1", "text": "hello"}])

    assert artifacts.recording_id == "meeting-42"
    assert artifacts.raw_audio_path.exists()
    assert artifacts.transcript_txt_path.read_text(encoding="utf-8") == "hello world"
    assert json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))["summary"] == "- one"
    assert json.loads(artifacts.segments_json_path.read_text(encoding="utf-8"))[0]["speaker"] == "S1"


def test_stage_raw_audio_is_noop_when_source_equals_destination(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"abc")

    out = stage_raw_audio(path, path)
    assert out == path
    assert path.read_bytes() == b"abc"
