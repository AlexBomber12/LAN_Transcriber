from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from lan_transcriber.pipeline_steps import orchestrator as pipeline


def _write_wav(path: Path, *, duration_sec: float) -> Path:
    sample_rate = 16000
    samples = max(int(sample_rate * duration_sec), 1)
    frames = bytearray()
    for idx in range(samples):
        value = int(9000 * math.sin((2.0 * math.pi * 220.0 * idx) / sample_rate))
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
    return path


class _FakeDiariser:
    mode = "pyannote"

    async def __call__(self, _audio_path: Path):
        class _Annotation:
            def itertracks(self, yield_label: bool = False):
                first = SimpleNamespace(start=0.0, end=0.8)
                second = SimpleNamespace(start=0.8, end=1.5)
                if yield_label:
                    yield first, "S1"
                    yield second, "S2"
                else:
                    yield (first,)
                    yield (second,)

        return _Annotation()


class _FakeLLM:
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        response_format: dict[str, object] | None = None,
    ) -> dict[str, str]:
        del system_prompt, user_prompt, model, response_format
        payload = {
            "topic": "E2E-lite summary",
            "summary_bullets": ["Pipeline completed with deterministic mocked components."],
            "decisions": [],
            "action_items": [],
            "tone_score": 68,
            "emotional_summary": "Neutral.",
            "questions": {
                "total_count": 0,
                "types": {
                    "open": 0,
                    "yes_no": 0,
                    "clarification": 0,
                    "status": 0,
                    "decision_seeking": 0,
                },
                "extracted": [],
            },
        }
        return {"content": json.dumps(payload)}


@pytest.mark.asyncio
async def test_e2e_lite_processing_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio_path = _write_wav(tmp_path / "e2e-lite.wav", duration_sec=1.5)
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "aliases.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        precheck_min_duration_sec=0.1,
        precheck_min_speech_ratio=0.0,
    )

    def _fake_asr(
        _audio_path: Path,
        *,
        override_lang: str | None,
        cfg: pipeline.Settings,
        step_log_callback=None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        del override_lang, cfg, step_log_callback
        return (
            [
                {
                    "start": 0.0,
                    "end": 0.7,
                    "text": "hello team",
                    "words": [
                        {"start": 0.0, "end": 0.3, "word": "hello"},
                        {"start": 0.31, "end": 0.7, "word": "team"},
                    ],
                },
                {
                    "start": 0.75,
                    "end": 1.45,
                    "text": "quick update",
                    "words": [
                        {"start": 0.75, "end": 1.05, "word": "quick"},
                        {"start": 1.06, "end": 1.45, "word": "update"},
                    ],
                },
            ],
            {"language": "en", "language_probability": 0.99},
        )

    monkeypatch.setattr(pipeline, "_whisperx_asr", _fake_asr)

    progress_events: list[tuple[str, float]] = []
    result = await asyncio.wait_for(
        pipeline.run_pipeline(
            audio_path=audio_path,
            cfg=cfg,
            llm=_FakeLLM(),
            diariser=_FakeDiariser(),
            recording_id="rec-e2e-lite",
            progress_callback=lambda stage, progress: progress_events.append((stage, progress)),
        ),
        timeout=20,
    )

    derived = cfg.recordings_root / "rec-e2e-lite" / "derived"
    transcript_txt = derived / "transcript.txt"
    transcript_json = derived / "transcript.json"
    summary_json = derived / "summary.json"
    metrics_json = derived / "metrics.json"
    diarization_json = derived / "segments.json"

    assert result.summary != "Quarantined"
    assert result.body.strip() != ""
    assert transcript_txt.exists()
    assert transcript_txt.read_text(encoding="utf-8").strip() != ""

    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary_payload["summary_bullets"]
    assert summary_payload["topic"] == "E2E-lite summary"

    transcript_payload = json.loads(transcript_json.read_text(encoding="utf-8"))
    assert transcript_payload["text"].strip() != ""
    assert transcript_payload["speakers"] == ["S1", "S2"]

    diarization_payload = json.loads(diarization_json.read_text(encoding="utf-8"))
    assert len(diarization_payload) == 2
    assert all(str(item.get("speaker")).startswith("S") for item in diarization_payload)

    metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert metrics_payload["status"] == "ok"

    snippets = sorted((derived / "snippets").glob("*/*.wav"))
    assert snippets
    assert all(path.stat().st_size > 44 for path in snippets)

    assert [stage for stage, _ in progress_events] == [
        "precheck",
        "stt",
        "stt",
        "diarize",
        "align",
        "language",
        "llm",
        "metrics",
    ]


@pytest.mark.asyncio
async def test_e2e_lite_processing_failure_writes_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    audio_path = _write_wav(tmp_path / "e2e-lite-failure.wav", duration_sec=1.2)
    cfg = pipeline.Settings(
        speaker_db=tmp_path / "aliases.yaml",
        tmp_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        precheck_min_duration_sec=0.1,
        precheck_min_speech_ratio=0.0,
    )

    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mocked asr failure")),
    )
    with pytest.raises(RuntimeError, match="mocked asr failure"):
        await asyncio.wait_for(
            pipeline.run_pipeline(
                audio_path=audio_path,
                cfg=cfg,
                llm=_FakeLLM(),
                diariser=_FakeDiariser(),
                recording_id="rec-e2e-lite-fail",
            ),
            timeout=20,
        )

    derived = cfg.recordings_root / "rec-e2e-lite-fail" / "derived"
    summary_payload = json.loads((derived / "summary.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((derived / "metrics.json").read_text(encoding="utf-8"))

    assert summary_payload["status"] == "failed"
    assert metrics_payload["status"] == "failed"
    assert "mocked asr failure" in str(metrics_payload.get("error", ""))
