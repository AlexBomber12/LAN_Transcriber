from __future__ import annotations

import wave
from pathlib import Path

import pytest

from lan_transcriber.pipeline_steps import multilingual_asr
from lan_transcriber.pipeline_steps.language import (
    LanguageAnalysis,
    analyse_languages,
    resolve_target_summary_language,
    segment_language,
)


def _write_pcm_wav(path: Path, *, duration_sec: float) -> Path:
    frame_rate = 16000
    frame_count = int(frame_rate * duration_sec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(b"\x00\x01" * frame_count)
    return path


def test_segment_language_and_summary_resolution_paths() -> None:
    assert (
        segment_language(
            {"text": "the and to of in"},
            detected_language="fr",
            transcript_language_override=None,
        )
        == "fr"
    )
    assert (
        resolve_target_summary_language(
            "de",
            dominant_language="en",
            detected_language="es",
        )
        == "de"
    )
    assert (
        resolve_target_summary_language(
            None,
            dominant_language="en",
            detected_language="es",
        )
        == "en"
    )
    assert (
        resolve_target_summary_language(
            None,
            dominant_language="unknown",
            detected_language="es",
        )
        == "es"
    )
    assert (
        resolve_target_summary_language(
            None,
            dominant_language="unknown",
            detected_language=None,
        )
        == "en"
    )


def test_analyse_languages_conflict_and_empty_paths() -> None:
    analysis = analyse_languages(
        [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "hola equipo gracias",
                "language": "en",
                "language_confidence": 0.95,
            },
            {
                "start": 4.0,
                "end": 8.0,
                "text": "hello team thanks",
            },
        ],
        detected_language=None,
        transcript_language_override=None,
    )

    assert analysis.review_required is True
    assert analysis.review_reason_code == "multilingual_uncertain"
    assert analysis.conflict_segment_count == 1
    assert analysis.segments[0]["language_conflict"] is True
    assert analysis.spans[0]["uncertain"] is True

    empty = analyse_languages(
        [],
        detected_language="es",
        transcript_language_override=None,
    )
    assert empty.dominant_language == "es"
    assert empty.review_required is False


def test_analyse_languages_marks_low_confidence_multilingual_segments() -> None:
    analysis = analyse_languages(
        [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "hello team thanks",
                "language": "en",
                "language_confidence": 0.4,
            },
            {
                "start": 4.0,
                "end": 8.0,
                "text": "hola equipo gracias",
                "language": "es",
                "language_confidence": 0.45,
            },
        ],
        detected_language=None,
        transcript_language_override=None,
    )

    assert analysis.review_required is True
    assert analysis.review_reason_code == "multilingual_uncertain"
    assert analysis.uncertain_segment_count == 2
    assert analysis.segments[0]["language_uncertain"] is True
    assert analysis.segments[1]["language_uncertain"] is True


def test_should_use_multilingual_path_respects_modes_and_switches() -> None:
    empty = LanguageAnalysis([], "unknown", {}, [])
    assert multilingual_asr.should_use_multilingual_path(
        empty,
        configured_mode="force_single_language",
    ) == (False, "forced_single_language")
    assert multilingual_asr.should_use_multilingual_path(
        empty,
        configured_mode="auto",
    ) == (False, "no_segments")

    mixed = LanguageAnalysis(
        segments=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}],
        dominant_language="en",
        distribution={"en": 50.0, "es": 50.0},
        spans=[
            {"start": 0.0, "end": 4.0, "lang": "en"},
            {"start": 4.0, "end": 8.0, "lang": "es"},
        ],
    )
    assert multilingual_asr.should_use_multilingual_path(
        mixed,
        configured_mode="force_multilingual",
    ) == (True, "forced_multilingual")
    assert multilingual_asr.should_use_multilingual_path(
        mixed,
        configured_mode="auto",
    ) == (True, "credible_language_switches")

    single = LanguageAnalysis(
        segments=[{"start": 0.0, "end": 8.0}],
        dominant_language="en",
        distribution={"en": 100.0},
        spans=[{"start": 0.0, "end": 8.0, "lang": "en"}],
    )
    assert multilingual_asr.should_use_multilingual_path(
        single,
        configured_mode="auto",
    ) == (False, "dominant_single_language")

    weak_switch = LanguageAnalysis(
        segments=[{"start": 0.0, "end": 2.0}],
        dominant_language="en",
        distribution={"en": 90.0, "es": 10.0},
        spans=[
            {"start": 0.0, "end": 2.0, "lang": "en"},
            {"start": 2.0, "end": 3.0, "lang": "es"},
        ],
    )
    assert multilingual_asr.should_use_multilingual_path(
        weak_switch,
        configured_mode="auto",
    ) == (False, "dominant_single_language")


def test_plan_multilingual_chunks_merges_matching_segments_and_preserves_uncertain() -> None:
    analysis = LanguageAnalysis(
        segments=[
            {
                "start": 0.0,
                "end": 2.0,
                "language": "en",
                "language_confidence": 0.95,
            },
            {
                "start": 2.3,
                "end": 4.0,
                "language": "en",
                "language_confidence": 0.9,
            },
            {
                "start": 4.5,
                "end": 7.0,
                "language": "es",
                "language_confidence": 0.5,
                "language_uncertain": True,
            },
        ],
        dominant_language="en",
        distribution={"en": 60.0, "es": 40.0},
        spans=[],
    )

    chunks = multilingual_asr.plan_multilingual_chunks(analysis)
    assert len(chunks) == 2
    assert chunks[0].start == 0.0
    assert chunks[0].end == 4.0
    assert chunks[0].language_hint == "en"
    assert chunks[1].language_hint is None
    assert chunks[1].uncertain is True

    backwards = LanguageAnalysis(
        segments=[
            {
                "start": 5.0,
                "end": 1.0,
                "language": "eng",
                "language_confidence": None,
            }
        ],
        dominant_language="en",
        distribution={"en": 100.0},
        spans=[],
    )
    backwards_chunks = multilingual_asr.plan_multilingual_chunks(backwards)
    assert backwards_chunks[0].start == 5.0
    assert backwards_chunks[0].end == 5.0
    assert backwards_chunks[0].language == "en"


def test_private_multilingual_helpers_cover_edge_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert multilingual_asr.ChunkPlan(  # noqa: SLF001
        start=0.0,
        end=1.0,
        language="en",
        confidence=None,
        language_hint=None,
        uncertain=False,
        conflict=False,
        segment_count=1,
    ).to_payload(index=1, total=1)["hint_applied"] is False

    seen: list[str] = []

    def _raising_callback(message: str) -> None:
        seen.append(message)
        raise RuntimeError("ignore me")

    multilingual_asr._log(_raising_callback, "synthetic")  # noqa: SLF001
    assert seen == ["synthetic"]
    assert multilingual_asr._language_from_info(  # noqa: SLF001
        {"detected_language": "es"}
    ) == ("es", None)

    shifted_words = multilingual_asr._offset_segment_words(  # noqa: SLF001
        [{"start": 0.5, "end": 0.2}, "skip", {"start": 1.0}, {"end": 2.0}],
        offset_seconds=1.0,
    )
    assert shifted_words[0]["start"] == 1.5
    assert shifted_words[0]["end"] == 1.2
    assert shifted_words[1]["start"] == 2.0
    assert shifted_words[2]["end"] == 3.0

    annotated = multilingual_asr._annotate_chunk_segments(  # noqa: SLF001
        [{"start": 2.0, "end": 1.0, "words": "not-a-list"}],
        offset_seconds=1.0,
        language="unknown",
        confidence=None,
        language_hint=None,
        uncertain=False,
        conflict=False,
    )
    assert annotated[0]["start"] == 3.0
    assert annotated[0]["end"] == 3.0
    assert annotated[0]["words"] == []
    assert annotated[0]["language"] == "unknown"
    assert "language_confidence" not in annotated[0]
    assert annotated[0]["language_source"] == "chunk_detected"

    no_language = multilingual_asr._annotate_chunk_segments(  # noqa: SLF001
        [{"start": 0.0, "end": 1.0}],
        offset_seconds=0.0,
        language="",
        confidence=None,
        language_hint=None,
        uncertain=False,
        conflict=False,
    )
    assert "language" not in no_language[0]

    fallback_only = multilingual_asr._merge_result_language_info(  # noqa: SLF001
        [{"result_language": None, "language": None, "language_hint": None}],
        fallback_info={"language": "es"},
    )
    assert fallback_only == {"language": "es"}

    selected = multilingual_asr._select_segments_for_range(  # noqa: SLF001
        [
            "skip",
            {"start": 2.0, "end": 1.0, "text": "boundary"},
            {"start": 1.0, "end": 3.0, "text": "inside"},
            {"start": 3.0, "end": 4.0, "text": "outside"},
        ],
        start_seconds=1.5,
        end_seconds=2.5,
    )
    assert [row["text"] for row in selected] == ["boundary", "inside"]
    assert multilingual_asr._select_segment_indexes_for_range(  # noqa: SLF001
        [
            {"start": 2.0, "end": 1.0, "text": "boundary"},
            {"start": 1.0, "end": 3.0, "text": "inside"},
        ],
        start_seconds=1.5,
        end_seconds=2.5,
        excluded_indexes={0},
    ) == [1]
    monkeypatch.setattr(
        multilingual_asr,
        "_select_segment_indexes_for_range",
        lambda *_a, **_k: [0],
    )
    assert (
        multilingual_asr._select_segments_for_range(  # noqa: SLF001
            ["skip"],
            start_seconds=0.0,
            end_seconds=1.0,
        )
        == []
    )

    merged = multilingual_asr._merge_result_language_info(  # noqa: SLF001
        [{"result_language": "en", "start": 1.0, "end": 1.0}],
        fallback_info={"language": "es"},
    )
    assert merged["language"] == "en"
    assert merged["language_probability"] == 1.0


def test_write_wav_chunk_clamps_zero_length_ranges(tmp_path: Path) -> None:
    source = _write_pcm_wav(tmp_path / "source.wav", duration_sec=1.0)
    output = multilingual_asr._write_wav_chunk(  # noqa: SLF001
        source,
        start_sec=0.4,
        end_sec=0.4,
        output_path=tmp_path / "chunk.wav",
    )

    with wave.open(str(output), "rb") as handle:
        assert handle.getnframes() == 1


def test_run_language_aware_asr_override_and_empty_chunk_plan_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _write_pcm_wav(tmp_path / "override.wav", duration_sec=2.0)
    calls: list[str | None] = []

    def _transcribe(
        path: Path,
        override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert path.exists()
        calls.append(override_lang)
        return (
            [{"start": 0.0, "end": 1.0, "text": "hola"}],
            {"language": override_lang or "en"},
        )

    segments, info, payload = multilingual_asr.run_language_aware_asr(
        audio,
        override_lang="es",
        configured_mode="auto",
        tmp_root=tmp_path / "tmp",
        transcribe_fn=_transcribe,
    )
    assert calls == ["es"]
    assert segments[0]["text"] == "hola"
    assert info["language"] == "es"
    assert payload["selection_reason"] == "transcript_language_override"

    monkeypatch.setattr(multilingual_asr, "plan_multilingual_chunks", lambda *_a, **_k: [])
    calls.clear()
    segments, info, payload = multilingual_asr.run_language_aware_asr(
        audio,
        override_lang=None,
        configured_mode="force_multilingual",
        tmp_root=tmp_path / "tmp",
        transcribe_fn=_transcribe,
    )
    assert calls == [None]
    assert segments[0]["text"] == "hola"
    assert info["language"] == "en"
    assert payload["selection_reason"] == "empty_chunk_plan"


def test_run_language_aware_asr_force_multilingual_requires_wav_input(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mixed.mp3"
    audio.write_bytes(b"synthetic")

    def _transcribe(
        path: Path,
        override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert path == audio
        assert override_lang is None
        return (
            [
                {"start": 0.0, "end": 4.0, "text": "hello team thanks"},
                {"start": 4.0, "end": 8.0, "text": "hola equipo gracias"},
            ],
            {"language": "en", "language_probability": 0.92},
        )

    with pytest.raises(
        ValueError,
        match="force_multilingual requires a WAV input",
    ):
        multilingual_asr.run_language_aware_asr(
            audio,
            override_lang=None,
            configured_mode="force_multilingual",
            tmp_root=tmp_path / "tmp",
            transcribe_fn=_transcribe,
        )


def test_run_language_aware_asr_multilingual_executes_chunk_hints_and_conflicts(
    tmp_path: Path,
) -> None:
    audio = _write_pcm_wav(tmp_path / "mixed.wav", duration_sec=8.0)
    calls: list[str | None] = []

    def _transcribe(
        path: Path,
        override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        calls.append(override_lang)
        if path == audio:
            return (
                [
                    {"start": 0.0, "end": 4.0, "text": "hello team thanks"},
                    {"start": 4.0, "end": 8.0, "text": "hola equipo gracias"},
                ],
                {"language": "en", "language_probability": 0.92},
            )
        if override_lang == "en":
            return (
                [{"start": 0.0, "end": 4.0, "text": "hello team thanks"}],
                {"language": "en", "language_probability": 0.98},
            )
        return (
            [{"start": 0.0, "end": 4.0, "text": "hello team thanks"}],
            {"language": "en", "language_probability": 0.99},
        )

    segments, info, payload = multilingual_asr.run_language_aware_asr(
        audio,
        override_lang=None,
        configured_mode="auto",
        tmp_root=tmp_path / "tmp",
        transcribe_fn=_transcribe,
    )

    assert calls == [None, "en", "es"]
    assert payload["used_multilingual_path"] is True
    assert payload["selected_mode"] == "multilingual"
    assert payload["chunks"][0]["hint_applied"] is True
    assert payload["chunks"][1]["conflict"] is True
    assert payload["chunks"][1]["result_uncertain"] is True
    assert segments[0]["start"] == 0.0
    assert segments[1]["start"] == 4.0
    assert segments[1]["language_conflict"] is True
    assert segments[1]["language_hint"] == "es"
    assert info["language"] == "en"


def test_run_language_aware_asr_multilingual_preserves_initial_segments_when_chunk_is_empty(
    tmp_path: Path,
) -> None:
    audio = _write_pcm_wav(tmp_path / "mixed-empty-chunk.wav", duration_sec=8.0)

    def _transcribe(
        path: Path,
        override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if path == audio:
            return (
                [
                    {"start": 0.0, "end": 4.0, "text": "hello team thanks"},
                    {"start": 4.0, "end": 8.0, "text": "hola equipo gracias"},
                ],
                {"language": "en", "language_probability": 0.92},
            )
        if override_lang == "en":
            return (
                [{"start": 0.0, "end": 4.0, "text": "hello team thanks"}],
                {"language": "en", "language_probability": 0.98},
            )
        return ([], {})

    segments, info, payload = multilingual_asr.run_language_aware_asr(
        audio,
        override_lang=None,
        configured_mode="auto",
        tmp_root=tmp_path / "tmp",
        transcribe_fn=_transcribe,
    )

    assert payload["used_multilingual_path"] is True
    assert segments[0]["text"] == "hello team thanks"
    assert segments[1]["text"] == "hola equipo gracias"
    assert segments[1]["start"] == 4.0
    assert segments[1]["language"] == "es"
    assert info["language"] == "en"


def test_run_language_aware_asr_multilingual_empty_chunks_do_not_duplicate_fallback_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _write_pcm_wav(tmp_path / "mixed-empty-duplicate.wav", duration_sec=8.0)

    monkeypatch.setattr(
        multilingual_asr,
        "plan_multilingual_chunks",
        lambda *_a, **_k: [
            multilingual_asr.ChunkPlan(
                start=0.0,
                end=4.0,
                language="en",
                confidence=0.95,
                language_hint="en",
                uncertain=False,
                conflict=False,
                segment_count=1,
            ),
            multilingual_asr.ChunkPlan(
                start=4.0,
                end=8.0,
                language="en",
                confidence=0.95,
                language_hint="en",
                uncertain=False,
                conflict=False,
                segment_count=1,
            ),
        ],
    )

    def _transcribe(
        path: Path,
        override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if path == audio:
            return (
                [{"start": 3.0, "end": 5.0, "text": "boundary overlap"}],
                {"language": "en", "language_probability": 0.92},
            )
        assert override_lang == "en"
        return ([], {"language": "en", "language_probability": 0.92})

    segments, _info, payload = multilingual_asr.run_language_aware_asr(
        audio,
        override_lang=None,
        configured_mode="force_multilingual",
        tmp_root=tmp_path / "tmp",
        transcribe_fn=_transcribe,
    )

    assert payload["used_multilingual_path"] is True
    assert [segment["text"] for segment in segments] == ["boundary overlap"]
    assert segments[0]["start"] == 3.0
    assert segments[0]["end"] == 5.0


def test_run_language_aware_asr_multilingual_without_confidence_payload(
    tmp_path: Path,
) -> None:
    audio = _write_pcm_wav(tmp_path / "mixed-no-confidence.wav", duration_sec=8.0)

    def _transcribe(
        path: Path,
        override_lang: str | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if path == audio:
            return (
                [
                    {"start": 0.0, "end": 4.0, "text": "alpha beta gamma"},
                    {"start": 4.0, "end": 8.0, "text": "delta epsilon zeta"},
                ],
                {},
            )
        return (
            [{"start": 0.0, "end": 4.0, "text": "alpha beta gamma"}],
            {},
        )

    _segments, _info, payload = multilingual_asr.run_language_aware_asr(
        audio,
        override_lang=None,
        configured_mode="force_multilingual",
        tmp_root=tmp_path / "tmp",
        transcribe_fn=_transcribe,
    )

    assert payload["used_multilingual_path"] is True
    assert "result_confidence" not in payload["chunks"][0]
