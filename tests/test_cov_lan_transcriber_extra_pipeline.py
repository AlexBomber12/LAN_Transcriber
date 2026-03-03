from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from lan_transcriber import llm_client
from lan_transcriber.pipeline_steps import language, orchestrator as pipeline, precheck, snippets, speaker_turns, summary_builder


def _settings(tmp_path: Path, **overrides: Any) -> pipeline.Settings:
    defaults = {
        "speaker_db": tmp_path / "db.yaml",
        "tmp_root": tmp_path / "tmp",
        "recordings_root": tmp_path / "recordings",
    }
    defaults.update(overrides)
    return pipeline.Settings(**defaults)


def _audio_file(tmp_path: Path, name: str = "audio.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00")
    return path


class _NoTracksDiariser:
    async def __call__(self, _audio_path: Path):
        return object()


class _FakeLLM:
    async def generate(self, **_kwargs: Any) -> dict[str, str]:
        return {
            "content": json.dumps(
                {
                    "topic": "T",
                    "summary_bullets": ["ok"],
                    "decisions": [],
                    "action_items": [],
                    "emotional_summary": "Neutral",
                    "questions": {"total_count": 0, "types": {}, "extracted": []},
                }
            )
        }


def test_timeout_seconds_and_retryable_status_paths() -> None:
    assert llm_client._timeout_seconds("bad", default=7.5) == 7.5
    assert llm_client._timeout_seconds("-1", default=7.5) == 7.5
    assert llm_client._int_setting("bad", default=1024, minimum=256) == 1024
    assert llm_client._int_setting("128", default=1024, minimum=256) == 1024
    assert llm_client._int_setting("512", default=1024, minimum=256) == 512
    assert llm_client._resolve_retry_max_tokens("2000", base_max_tokens=1000) == 2000
    assert llm_client._resolve_retry_max_tokens("800", base_max_tokens=1000) == 2000
    assert llm_client._resolve_retry_max_tokens(None, base_max_tokens=5000) == 5000
    assert (
        llm_client._resolve_retry_max_tokens(
            "5000",
            base_max_tokens=4096,
        )
        == 5000
    )
    assert llm_client._base_url_host("http://example.test:8000/path") == "example.test"
    assert llm_client._base_url_host("not-a-url") == "not-a-url"

    req = httpx.Request("POST", "http://example.test")
    retry_resp = httpx.Response(503, request=req)
    non_retry_resp = httpx.Response(404, request=req)
    assert llm_client._is_retryable_exception(
        httpx.HTTPStatusError("boom", request=req, response=retry_resp)
    )
    assert not llm_client._is_retryable_exception(
        httpx.HTTPStatusError("boom", request=req, response=non_retry_resp)
    )


def test_resolve_base_url_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    assert llm_client._resolve_base_url(" http://custom ") == "http://custom"

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LAN_ENV", "staging")
    with pytest.raises(ValueError, match="Missing required environment variable"):
        llm_client._resolve_base_url(None)


@pytest.mark.asyncio
async def test_post_chat_completion_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[str]:
            return ["bad-shape"]

    class _Client:
        def __init__(self, **_kwargs: Any):
            return None

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def post(self, *_args: Any, **_kwargs: Any) -> _Resp:
            return _Resp()

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _Client)
    client = llm_client.LLMClient(base_url="http://example.test", timeout=0.1)
    with pytest.raises(ValueError, match="JSON object"):
        await client._post_chat_completion(
            url="http://example.test/v1/chat/completions",
            payload={"messages": []},
            headers={},
        )


@pytest.mark.asyncio
async def test_load_mock_message_edge_cases(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("   ", encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=empty_path)._load_mock_message() == {
        "role": "assistant",
        "content": "",
    }

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json", encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=invalid_path)._load_mock_message() == {
        "role": "assistant",
        "content": "not-json",
    }

    string_path = tmp_path / "string.json"
    string_path.write_text(json.dumps("hello"), encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=string_path)._load_mock_message() == {
        "role": "assistant",
        "content": "hello",
    }

    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert llm_client.LLMClient(mock_response_path=list_path)._load_mock_message() == {
        "role": "assistant",
        "content": '["a", "b"]',
    }

    choices_non_dict_path = tmp_path / "choices-non-dict.json"
    choices_non_dict_path.write_text(
        json.dumps({"choices": ["x"], "content": "fallback"}),
        encoding="utf-8",
    )
    assert llm_client.LLMClient(mock_response_path=choices_non_dict_path)._load_mock_message() == {
        "role": "assistant",
        "content": "fallback",
    }

    choices_message_path = tmp_path / "choices-message.json"
    choices_message_path.write_text(
        json.dumps({"choices": [{"message": {"role": "assistant", "content": "from-choices"}}]}),
        encoding="utf-8",
    )
    assert llm_client.LLMClient(mock_response_path=choices_message_path)._load_mock_message() == {
        "role": "assistant",
        "content": "from-choices",
    }

    choices_bad_message_path = tmp_path / "choices-bad-message.json"
    choices_bad_message_path.write_text(
        json.dumps({"choices": [{"message": "not-a-dict"}], "content": "fallback-after-bad-message"}),
        encoding="utf-8",
    )
    assert llm_client.LLMClient(mock_response_path=choices_bad_message_path)._load_mock_message() == {
        "role": "assistant",
        "content": "fallback-after-bad-message",
    }


@pytest.mark.asyncio
async def test_generate_headers_and_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    client = llm_client.LLMClient(base_url="http://example.test", api_key="secret-key")

    async def _fake_post(
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        attempt_number: int | None = None,
    ) -> dict[str, Any]:
        del attempt_number
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(client, "_post_chat_completion", _fake_post)
    result = await client.generate(
        "sys",
        "usr",
        model="x",
        response_format={"type": "json_object"},
    )
    assert result["content"] == "ok"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["max_tokens"] == client.max_tokens

    fallback_client = llm_client.LLMClient(base_url="http://example.test")

    async def _fallback_post(**_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": "bad-shape"}], "content": "fallback-content"}

    monkeypatch.setattr(fallback_client, "_post_chat_completion", _fallback_post)
    fallback = await fallback_client.generate("sys", "usr")
    assert fallback["content"] == "fallback-content"

    async def _fallback_non_dict_choice(**_kwargs: Any) -> dict[str, Any]:
        return {"choices": [123], "content": "fallback-non-dict"}

    monkeypatch.setattr(fallback_client, "_post_chat_completion", _fallback_non_dict_choice)
    fallback2 = await fallback_client.generate("sys", "usr")
    assert fallback2["content"] == "fallback-non-dict"

    async def _missing_content_post(**_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{}]}

    monkeypatch.setattr(fallback_client, "_post_chat_completion", _missing_content_post)
    with pytest.raises(ValueError, match="missing choices"):
        await fallback_client.generate("sys", "usr")


def test_guess_language_and_analysis_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    assert language._guess_language_from_text("") is None
    assert language._guess_language_from_text("!!! 123 ???") is None
    assert language._guess_language_from_text("hola señor, gracias") == "es"
    assert language._guess_language_from_text("hello team and thanks") == "en"
    assert language._guess_language_from_text("hello hola") is None
    assert language._guess_language_from_text("alpha beta gamma") is None

    assert (
        language.segment_language(
            {"text": "???", "language": None},
            detected_language=None,
            transcript_language_override=None,
        )
        == "unknown"
    )

    analysis = language.analyse_languages(
        [
            {"start": 0.0, "end": 0.1, "text": "???"},
            {"start": 5.0, "end": 1.0, "text": "backwards", "language": "en"},
            {"start": 6.0, "end": 12.0, "text": "hello team", "language": "en"},
        ],
        detected_language=None,
        transcript_language_override=None,
    )
    assert analysis.dominant_language == "en"
    backwards_span = next(span for span in analysis.spans if span["start"] == 5.0)
    assert backwards_span["end"] == 5.0

    unknown_only = language.analyse_languages(
        [{"start": 0.0, "end": 1.0, "text": "???"}],
        detected_language=None,
        transcript_language_override=None,
    )
    assert unknown_only.dominant_language == "unknown"

    monkeypatch.setattr(language, "_duration_weight", lambda *_args, **_kwargs: 0.0)
    zero_weight = language.analyse_languages(
        [{"start": 0.0, "end": 1.0, "text": "hello team"}],
        detected_language="en",
        transcript_language_override=None,
    )
    assert zero_weight.distribution == {}


def test_ffprobe_and_speech_ratio_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _audio_file(tmp_path, "a.wav")
    monkeypatch.setattr(precheck.shutil, "which", lambda _name: "/usr/bin/tool")

    def _run_raises(*_args: Any, **_kwargs: Any):
        raise RuntimeError("run failed")

    monkeypatch.setattr(precheck.subprocess, "run", _run_raises)
    assert precheck._audio_duration_from_ffprobe(audio) is None

    class _Proc:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(1, "12"))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(0, "   "))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(0, "not-a-float"))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.subprocess, "run", lambda *_a, **_k: _Proc(0, "-1.0"))
    assert precheck._audio_duration_from_ffprobe(audio) is None

    monkeypatch.setattr(precheck.wave, "open", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    assert precheck._speech_ratio_from_wave(audio) is None

    class _ProcStdoutNone:
        stdout = None
        returncode = 0

        def wait(self, timeout=None):
            del timeout
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(precheck.subprocess, "Popen", lambda *_a, **_k: _ProcStdoutNone())
    assert precheck._speech_ratio_from_ffmpeg(audio) is None

    class _StdoutEmpty:
        def read(self, _size: int) -> bytes:
            return b""

    class _ProcEmpty:
        def __init__(self) -> None:
            self.stdout = _StdoutEmpty()
            self.returncode = 0

        def wait(self, timeout=None):
            del timeout
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(precheck.subprocess, "Popen", lambda *_a, **_k: _ProcEmpty())
    assert precheck._speech_ratio_from_ffmpeg(audio) == 0.0

    monkeypatch.setattr(
        precheck.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("popen-failed")),
    )
    assert precheck._speech_ratio_from_ffmpeg(audio) is None


def test_snippets_edge_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "snips"
    target.mkdir(parents=True, exist_ok=True)
    (target / "stale.wav").write_bytes(b"old")
    snippets._clear_dir(target)
    assert list(target.iterdir()) == []

    # Force the defensive `clip_end <= clip_start` branch.
    monkeypatch.setattr(snippets, "min", lambda *_args: 0.0, raising=False)
    start, end = snippets._snippet_window(0.0, 0.0, duration_sec=None)
    assert start == 0.0
    assert end == 0.0

    class _WaveNoFrames:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getframerate(self) -> int:
            return 16000

        def getnchannels(self) -> int:
            return 1

        def getsampwidth(self) -> int:
            return 2

        def getnframes(self) -> int:
            return 100

        def setpos(self, _value: int) -> None:
            return None

        def readframes(self, _value: int) -> bytes:
            return b""

    monkeypatch.setattr(snippets.wave, "open", lambda *_a, **_k: _WaveNoFrames())
    assert not snippets._extract_wav_snippet_with_wave(
        tmp_path / "in.wav",
        tmp_path / "out.wav",
        start_sec=0.0,
        end_sec=1.0,
    )

    monkeypatch.setattr(snippets.wave, "open", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom")))
    assert not snippets._extract_wav_snippet_with_wave(
        tmp_path / "in.wav",
        tmp_path / "out.wav",
        start_sec=0.0,
        end_sec=1.0,
    )


def test_export_speaker_snippets_overlap_and_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _audio_file(tmp_path, "src.wav")
    out_dir = tmp_path / "derived" / "snippets"
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: True)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: True)

    outputs = snippets.export_speaker_snippets(
        snippets.SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 5.0, "speaker": "S1"},
                {"start": 1.0, "end": 4.0, "speaker": "S1"},
                {"start": 9.0, "end": 9.0, "speaker": "S1"},
            ],
            snippets_dir=out_dir,
            duration_sec=10.0,
        )
    )
    assert len(outputs) == 2


def test_export_speaker_snippets_fallback_continue_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _audio_file(tmp_path, "src3.wav")
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: True)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: True)
    outputs = snippets.export_speaker_snippets(
        snippets.SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 2.0, "end": 2.0, "speaker": "S3"},
                {"start": 3.0, "end": 3.0, "speaker": "S3"},
            ],
            snippets_dir=tmp_path / "snips3",
            duration_sec=10.0,
        )
    )
    assert len(outputs) == 2


def test_export_speaker_snippets_stops_after_three_non_overlapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _audio_file(tmp_path, "src2.wav")
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_wave", lambda *_a, **_k: True)
    monkeypatch.setattr(snippets, "_extract_wav_snippet_with_ffmpeg", lambda *_a, **_k: True)
    outputs = snippets.export_speaker_snippets(
        snippets.SnippetExportRequest(
            audio_path=audio,
            diar_segments=[
                {"start": 0.0, "end": 6.0, "speaker": "S2"},
                {"start": 7.0, "end": 8.0, "speaker": "S2"},
                {"start": 9.0, "end": 10.0, "speaker": "S2"},
                {"start": 11.0, "end": 12.0, "speaker": "S2"},
            ],
            snippets_dir=tmp_path / "snips2",
            duration_sec=12.0,
        )
    )
    assert len(outputs) == 3


def test_speaker_turn_helpers_cover_remaining_branches() -> None:
    assert speaker_turns._normalise_word({"word": " "}, 0.0, 1.0) is None
    assert speaker_turns._normalise_word({"word": "ok", "start": 2.0, "end": 1.0}, 0.0, 1.0)["end"] == 2.0

    normalised = speaker_turns.normalise_asr_segments(
        [
            {
                "start": 2.0,
                "end": 1.0,
                "text": "segment",
                "words": ["bad", {"word": ""}, {"word": "ok", "start": 1.0, "end": 0.5}],
            }
        ]
    )
    assert normalised[0]["end"] == 2.0
    assert normalised[0]["words"][0]["word"] == "ok"

    assert speaker_turns._diarization_segments(None) == []

    class _Diar:
        def itertracks(self, yield_label: bool = False):
            del yield_label
            yield "bad-item"
            yield (SimpleNamespace(start=4.0, end=1.0), "S9")

    diar_rows = speaker_turns._diarization_segments(_Diar())
    assert diar_rows[0]["end"] == 4.0

    assert speaker_turns._pick_speaker(0.0, 1.0, []) == "S1"
    assert (
        speaker_turns._pick_speaker(
            1.0,
            1.0,
            [{"start": 1.0, "end": 1.0, "speaker": "S2"}],
        )
        == "S2"
    )

    words = speaker_turns._words_from_segments(
        [
            {"start": 0.0, "end": 1.0, "text": "fallback", "words": "not-a-list"},
            {"start": 1.0, "end": 2.0, "words": ["bad", {"word": ""}, {"word": "x", "start": 2.0, "end": 1.0}]},
        ],
        default_language="en",
    )
    assert any(item["word"] == "fallback" for item in words)
    assert any(item["word"] == "x" and item["end"] == 2.0 for item in words)
    words_no_lang = speaker_turns._words_from_segments(
        [{"start": 0.0, "end": 1.0, "text": "nolanguage"}],
        default_language=None,
    )
    assert "language" not in words_no_lang[0]

    normalised_turns = speaker_turns._normalise_turns(
        [
            "bad",
            {"text": "   "},
            {"start": 2.0, "end": 1.0, "speaker": "S1", "text": "ok", "language": "EN-us"},
        ]
    )
    assert normalised_turns[0]["end"] == 2.0
    assert normalised_turns[0]["language"] == "en"

    stats = speaker_turns.count_interruptions(
        [
            {"start": 0.0, "end": 0.0, "speaker": "S1", "text": "zero"},
            {"start": 5.0, "end": 8.0, "speaker": "S2", "text": "base"},
            {"start": 4.0, "end": 6.0, "speaker": "S3", "text": "too-early"},
            {"start": 7.9, "end": 8.1, "speaker": "S4", "text": "tiny-overlap"},
        ],
        overlap_threshold=2.0,
    )
    assert stats["total"] == 0

    turns = speaker_turns.build_speaker_turns(
        [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"start": 0.0, "end": 0.4, "word": "hello"}],
                "language": "en",
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "team",
                "words": [{"start": 2.0, "end": 2.4, "word": "team"}],
                "language": "en",
            },
        ],
        [{"start": 0.0, "end": 4.0, "speaker": "S1"}],
        default_language=None,
    )
    assert len(turns) == 2

    turns_no_language = speaker_turns.build_speaker_turns(
        [
            {"start": 0.0, "end": 0.5, "text": "one", "words": [{"start": 0.0, "end": 0.5, "word": "one"}]},
            {"start": 2.0, "end": 2.5, "text": "two", "words": [{"start": 2.0, "end": 2.5, "word": "two"}]},
        ],
        [{"start": 0.0, "end": 4.0, "speaker": "S1"}],
        default_language=None,
    )
    assert len(turns_no_language) == 2

    equal_start_stats = speaker_turns.count_interruptions(
        [
            {"start": 1.0, "end": 3.0, "speaker": "S1", "text": "first"},
            {"start": 1.0, "end": 2.0, "speaker": "S2", "text": "second"},
        ],
        overlap_threshold=0.0,
    )
    assert equal_start_stats["total"] == 0


def test_summary_builder_helper_edge_paths(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="task is required"):
        summary_builder.ActionItem(task="  ")

    item = summary_builder.ActionItem(task="x", owner=None, deadline=None, confidence=0.3)
    assert item.owner is None
    assert item.deadline is None

    q = summary_builder.Question(types="bad")
    assert set(q.types.keys()) == {"open", "yes_no", "clarification", "status", "decision_seeking"}

    resp = summary_builder.SummaryResponse(
        topic="T",
        summary_bullets=["one"],
        decisions=[],
        action_items=[],
        emotional_summary=[],
        questions={"total_count": 0, "types": {}, "extracted": []},
    )
    assert resp.emotional_summary == "Neutral and focused discussion."

    assert summary_builder._chunk_text_for_prompt("   ") == []
    chunks = summary_builder._chunk_text_for_prompt("x" * 25, max_chars=10)
    assert len(chunks) == 3

    limited_turns = summary_builder._normalise_prompt_speaker_turns(
        [{"start": 0, "end": 1, "speaker": "S1", "text": "a b c"}],
        max_turns=0,
    )
    assert limited_turns == []

    _sys_prompt, user_prompt = summary_builder.build_summary_prompts("hello", "en")
    payload = json.loads(user_prompt)
    assert payload["speaker_turns"][0]["text"] == "hello"
    _sys_prompt_empty, user_prompt_empty = summary_builder.build_summary_prompts("   ", "en")
    assert json.loads(user_prompt_empty)["speaker_turns"] == []

    assert summary_builder._extract_json_dict("{not-json}{not-json}") is None

    original_findall = summary_builder.re.findall
    summary_builder.re.findall = lambda *_args, **_kwargs: ['{"topic":"x"}']  # type: ignore[assignment]
    try:
        extracted = summary_builder._extract_json_dict("[1]")
    finally:
        summary_builder.re.findall = original_findall  # type: ignore[assignment]
    assert extracted == {"topic": "x"}

    many = summary_builder._normalise_action_items_fallback([{"task": f"t{i}"} for i in range(40)])
    assert len(many) == 30
    mixed = summary_builder._normalise_action_items_fallback(["task", "", {"task": " "}])
    assert mixed[0]["task"] == "task"
    scalar = summary_builder._normalise_action_items_fallback("single task")
    assert scalar[0]["task"] == "single task"

    class _FakeValidationError:
        def errors(self) -> list[dict[str, Any]]:
            return []

    assert summary_builder._validation_reason(_FakeValidationError()) == "validation_failed"

    payload_no_topic = summary_builder.build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "summary_bullets": ["one"],
                "decisions": [],
                "action_items": [],
                "emotional_summary": "ok",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="Default topic",
    )
    assert payload_no_topic["topic"] == "Default topic"

    payload_invalid_no_artifacts = summary_builder.build_summary_payload(
        raw_llm_content=json.dumps(
            {
                "topic": "bad",
                "summary_bullets": [],
                "decisions": [],
                "action_items": [],
                "emotional_summary": "ok",
                "questions": {"total_count": 0, "types": {}, "extracted": []},
            }
        ),
        model="m",
        target_summary_language="en",
        friendly=0,
    )
    assert payload_invalid_no_artifacts["parse_error"] is True

    payload_no_json_no_artifacts = summary_builder.build_summary_payload(
        raw_llm_content="plain text",
        model="m",
        target_summary_language="en",
        friendly=0,
    )
    assert payload_no_json_no_artifacts["parse_error_reason"] == "json_object_not_found"

    fallback_from_summary = summary_builder._fallback_payload(
        raw_llm_content="raw summary fallback",
        extracted={"summary": "- one\n- two"},
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="topic",
        parse_error_reason="reason",
    )
    assert fallback_from_summary["summary_bullets"] == ["one", "two"]
    fallback_from_bullets = summary_builder._fallback_payload(
        raw_llm_content="raw summary fallback",
        extracted={"summary_bullets": ["already"], "topic": "A"},
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="topic",
        parse_error_reason="reason",
    )
    assert fallback_from_bullets["summary_bullets"] == ["already"]

    fallback_from_raw = summary_builder._fallback_payload(
        raw_llm_content="- from raw",
        extracted={},
        model="m",
        target_summary_language="en",
        friendly=0,
        default_topic="topic",
        parse_error_reason="reason",
    )
    assert fallback_from_raw["summary_bullets"] == ["from raw"]


def test_orchestrator_helpers_cover_remaining_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert pipeline._merge_similar(["abc", "xyz", "pqr"], threshold=0.9) == ["abc", "xyz", "pqr"]
    assert pipeline._merge_similar(["abc", "abd", "xyz"], threshold=0.6) == ["abc", "xyz"]

    ann = pipeline._fallback_diarization(None)
    tracks = list(ann.itertracks(yield_label=True))
    assert tracks[0][1] == "S1"

    cfg = _settings(tmp_path, asr_device="auto")
    original_import = builtins.__import__

    def _import_fail_torch(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "torch":
            raise ImportError("no torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_fail_torch)
    assert pipeline._select_asr_device(cfg) == "cpu"
    monkeypatch.setattr(builtins, "__import__", original_import)

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    fake_torch.version = SimpleNamespace(cuda="12.4")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert pipeline._select_asr_device(cfg) == "cuda"

    configured = _settings(tmp_path, asr_compute_type="float32")
    assert pipeline._select_compute_type(configured, "cpu") == "float32"

    target_dir = tmp_path / "clear"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "file.txt").write_text("x", encoding="utf-8")
    pipeline._clear_dir(target_dir)
    assert list(target_dir.iterdir()) == []


def test_sentiment_score_uses_truncation_and_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def _pipeline(task: str, **kwargs: Any):
        calls["task"] = task
        calls["factory_kwargs"] = dict(kwargs)

        def _infer(text: str, **infer_kwargs: Any):
            calls["text"] = text
            calls["infer_kwargs"] = dict(infer_kwargs)
            return [{"label": "positive", "score": 0.91}]

        return _infer

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    score = pipeline._sentiment_score("x" * 5000)
    assert score == 91
    assert calls["task"] == "sentiment-analysis"
    assert calls["factory_kwargs"] == {
        "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "device": -1,
    }
    assert calls["text"] == "x" * 4000
    assert calls["infer_kwargs"] == {"truncation": True, "max_length": 512}


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [
        ("negative", 0.2, 80),
        ("neutral", 0.7, 50),
    ],
)
def test_sentiment_score_negative_and_fallback_labels(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    value: float,
    expected: int,
) -> None:
    def _pipeline(_task: str, **_kwargs: Any):
        return lambda *_a, **_k: [{"label": label, "score": value}]

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert pipeline._sentiment_score("hello") == expected


def test_sentiment_score_returns_neutral_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []

    def _pipeline(_task: str, **_kwargs: Any):
        def _boom(*_args: Any, **_infer_kwargs: Any):
            raise RuntimeError("model failed")

        return _boom

    fake_transformers = ModuleType("transformers")
    fake_transformers.pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        pipeline._logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    assert pipeline._sentiment_score("hello") == 0.0
    assert warnings == ["Sentiment scoring failed (RuntimeError); using neutral score"]


@pytest.mark.asyncio
async def test_emit_progress_awaitable_and_error_branches() -> None:
    events: list[tuple[str, float]] = []

    async def _async_cb(stage: str, progress: float) -> None:
        events.append((stage, progress))

    await pipeline._emit_progress(_async_cb, stage="x", progress=0.5)
    assert events == [("x", 0.5)]

    def _bad_cb(_stage: str, _progress: float) -> None:
        raise RuntimeError("boom")

    await pipeline._emit_progress(_bad_cb, stage="y", progress=0.9)


def test_whisperx_asr_callback_and_retry_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_whisperx = ModuleType("whisperx")

    def _legacy(_path: str, **_kwargs: Any):
        return [{"start": 0.0, "end": 1.0, "text": "hi"}], {"language": "en"}

    fake_whisperx.transcribe = _legacy
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(pipeline, "ensure_ctranslate2_no_execstack", lambda: ["/tmp/lib.so"])
    monkeypatch.setattr(pipeline, "patch_pyannote_inference_ignore_use_auth_token", lambda: True)

    def _bad_step_log(_msg: str) -> None:
        raise RuntimeError("cannot log")

    segments, info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "a.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_enable_align=False, vad_method="pyannote"),
        step_log_callback=_bad_step_log,
    )
    assert segments and info["language"] == "en"

    calls: list[dict[str, Any]] = []

    def _fake_call(_fn: Any, *_args: Any, **kwargs: Any):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise TypeError("synthetic type error")
        return ([{"start": 0.0, "end": 1.0, "text": "ok"}], {"language": "en"})

    monkeypatch.setattr(pipeline, "call_with_supported_kwargs", _fake_call)
    monkeypatch.setattr(pipeline, "ensure_ctranslate2_no_execstack", lambda: [])
    monkeypatch.setattr(pipeline, "patch_pyannote_inference_ignore_use_auth_token", lambda: False)
    segments, info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "b.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_enable_align=False),
        step_log_callback=list.append,
    )
    assert segments and info["language"] == "en"
    assert "word_timestamps" in calls[0]
    assert "word_timestamps" not in calls[1]


def test_whisperx_asr_align_typeerror_and_exception_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_whisperx = ModuleType("whisperx")
    fake_whisperx.transcribe = None

    class _Model:
        vad_model = staticmethod(lambda _payload: [])

        def transcribe(self, audio: str, *, batch_size: int, vad_filter: bool, language: str | None):
            del audio, batch_size, vad_filter, language
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "a"}], "language": "en"}

    fake_whisperx.load_audio = lambda _p: "audio"
    fake_whisperx.load_model = lambda *_a, **_k: _Model()
    fake_whisperx.load_align_model = lambda **_k: ("model", {"lang": "en"})

    def _align_typeerror(*args: Any, **kwargs: Any):
        if kwargs:
            raise TypeError("unexpected keyword")
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "aligned"}]}

    fake_whisperx.align = _align_typeerror
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    segments, _info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "modern.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_device="cpu", asr_enable_align=True),
    )
    assert segments[0]["text"] == "aligned"

    def _align_error(*_args: Any, **_kwargs: Any):
        raise RuntimeError("align failed")

    fake_whisperx.align = _align_error
    segments, _info = pipeline._whisperx_asr(
        _audio_file(tmp_path, "modern2.wav"),
        override_lang=None,
        cfg=_settings(tmp_path, asr_device="cpu", asr_enable_align=True),
    )
    assert segments[0]["text"] == "a"


@pytest.mark.asyncio
async def test_run_pipeline_backfills_detected_language_and_uses_fallback_diarization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_whisperx_asr",
        lambda *_a, **_k: (
            [{"start": 0.0, "end": 1.0, "text": "hello team and thanks"}],
            {"language": "unknown", "language_probability": 0.2},
        ),
    )
    monkeypatch.setattr(pipeline, "_sentiment_score", lambda _text: 50)
    monkeypatch.setattr(
        pipeline,
        "export_speaker_snippets",
        lambda _req: [],
    )
    monkeypatch.setattr(pipeline, "_save_aliases", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_load_aliases", lambda *_a, **_k: {})

    cfg = _settings(tmp_path)
    result = await pipeline.run_pipeline(
        audio_path=_audio_file(tmp_path, "pipeline.mp3"),
        cfg=cfg,
        llm=_FakeLLM(),
        diariser=_NoTracksDiariser(),
        recording_id="rec-fallback-diar",
        precheck=pipeline.PrecheckResult(duration_sec=30.0, speech_ratio=0.8, quarantine_reason=None),
    )
    assert result.summary.strip() == "- ok"

    derived = cfg.recordings_root / "rec-fallback-diar" / "derived"
    transcript_data = json.loads((derived / "transcript.json").read_text(encoding="utf-8"))
    diar_data = json.loads((derived / "segments.json").read_text(encoding="utf-8"))
    assert transcript_data["language"]["detected"] == "en"
    assert diar_data and diar_data[0]["speaker"] == "S1"


@pytest.mark.asyncio
async def test_protocol_stub_call_executes() -> None:
    out = await pipeline.Diariser.__call__(object(), Path("x"))
    assert out is None


def test_build_speaker_turns_false_final_current_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TruthyEmptyWords:
        def __bool__(self) -> bool:
            return True

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(speaker_turns, "_words_from_segments", lambda *_a, **_k: _TruthyEmptyWords())
    turns = speaker_turns.build_speaker_turns([], [], default_language=None)
    assert turns == []
