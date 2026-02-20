from __future__ import annotations

import asyncio
import audioop
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List, Protocol, Sequence

from pydantic_settings import BaseSettings

from . import normalizer
from .aliases import ALIAS_PATH, load_aliases as _load_aliases, save_aliases as _save_aliases
from .artifacts import (
    atomic_write_json,
    atomic_write_text,
    build_recording_artifacts,
    stage_raw_audio,
)
from .llm_client import LLMClient
from .metrics import error_rate_total, p95_latency_seconds
from .models import SpeakerSegment, TranscriptResult
from .runtime_paths import (
    default_recordings_root,
    default_tmp_root,
    default_unknown_dir,
    default_voices_dir,
)


@dataclass(frozen=True)
class PrecheckResult:
    duration_sec: float | None
    speech_ratio: float | None
    quarantine_reason: str | None = None


class Diariser(Protocol):
    """Minimal interface for speaker diarisation."""

    async def __call__(self, audio_path: Path): ...


class Settings(BaseSettings):
    """Runtime configuration for the transcription pipeline."""

    speaker_db: Path = ALIAS_PATH
    recordings_root: Path = default_recordings_root()
    voices_dir: Path = default_voices_dir()
    unknown_dir: Path = default_unknown_dir()
    tmp_root: Path = default_tmp_root()
    llm_model: str = "llama3:8b"
    embed_threshold: float = 0.65
    merge_similar: float = 0.9
    precheck_min_duration_sec: float = 20.0
    precheck_min_speech_ratio: float = 0.10

    class Config:
        env_prefix = "LAN_"


def _merge_similar(
    lines: Iterable[str], threshold: float
) -> List[str]:  # pragma: no cover - simple heuristic
    out: List[str] = []
    for line in lines:
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        sim = sum(a == b for a, b in zip(prev, line)) / max(len(prev), len(line))
        if sim >= threshold:
            continue
        out.append(line)
    return out


def _sentiment_score(text: str) -> int:  # pragma: no cover - trivial wrapper
    from transformers import pipeline as hf_pipeline

    sent = hf_pipeline("sentiment-analysis")(text[:4000])[0]
    if sent["label"] == "positive":
        return int(sent["score"] * 100)
    if sent["label"] == "negative":
        return int((1 - sent["score"]) * 100)
    return 50


def refresh_aliases(result: TranscriptResult, alias_path: Path = ALIAS_PATH) -> None:
    """Reload aliases from disk and update ``result`` in-place."""
    aliases = _load_aliases(alias_path)
    result.speakers = sorted({aliases.get(s.speaker, s.speaker) for s in result.segments})


def _default_recording_id(audio_path: Path) -> str:
    stem = audio_path.stem.strip()
    return stem or "recording"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_word(word: dict[str, Any], seg_start: float, seg_end: float) -> dict[str, Any] | None:
    text = str(word.get("word") or word.get("text") or "").strip()
    if not text:
        return None
    start = _safe_float(word.get("start"), default=seg_start)
    end = _safe_float(word.get("end"), default=max(start, seg_end))
    if end < start:
        end = start
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "word": text,
    }


def _normalise_asr_segments(raw_segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_segments):
        start = _safe_float(raw.get("start"), default=float(idx))
        end = _safe_float(raw.get("end"), default=start)
        if end < start:
            end = start
        text = str(raw.get("text") or "").strip()
        words_raw = raw.get("words")
        words: list[dict[str, Any]] = []
        if isinstance(words_raw, list):
            for word in words_raw:
                if not isinstance(word, dict):
                    continue
                normalised = _normalise_word(word, start, end)
                if normalised is not None:
                    words.append(normalised)
        if not words and text:
            words = [{"start": round(start, 3), "end": round(end, 3), "word": text}]
        payload: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "words": words,
        }
        language = raw.get("language")
        if isinstance(language, str) and language.strip():
            payload["language"] = language.strip()
        out.append(payload)
    return out


def _language_payload(info: dict[str, Any]) -> dict[str, Any]:
    detected = str(
        info.get("language")
        or info.get("detected_language")
        or info.get("lang")
        or "unknown"
    )
    confidence_raw = None
    for key in (
        "language_probability",
        "language_confidence",
        "language_score",
        "probability",
    ):
        if key in info and info[key] is not None:
            confidence_raw = info[key]
            break
    confidence = None
    if confidence_raw is not None:
        confidence = round(_safe_float(confidence_raw, default=0.0), 4)
    return {"detected": detected, "confidence": confidence}


def _safe_diarization_segments(diarization: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if diarization is None or not hasattr(diarization, "itertracks"):
        return out
    try:
        tracks = diarization.itertracks(yield_label=True)
    except Exception:
        return out
    for item in tracks:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        seg, label = item
        start = _safe_float(getattr(seg, "start", 0.0), default=0.0)
        end = _safe_float(getattr(seg, "end", start), default=start)
        if end < start:
            end = start
        out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": str(label),
            }
        )
    out.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    return out


def _overlap_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _pick_speaker(start: float, end: float, diar_segments: Sequence[dict[str, Any]]) -> str:
    if not diar_segments:
        return "S1"
    best_speaker = str(diar_segments[0]["speaker"])
    best_overlap = -1.0
    midpoint = (start + end) / 2.0
    best_distance = float("inf")
    for seg in diar_segments:
        d_start = _safe_float(seg.get("start"), default=0.0)
        d_end = _safe_float(seg.get("end"), default=d_start)
        overlap = _overlap_seconds(start, end, d_start, d_end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(seg.get("speaker", "S1"))
        if overlap == 0.0:
            if midpoint < d_start:
                distance = d_start - midpoint
            elif midpoint > d_end:
                distance = midpoint - d_end
            else:
                distance = 0.0
            if best_overlap <= 0.0 and distance < best_distance:
                best_distance = distance
                best_speaker = str(seg.get("speaker", "S1"))
    return best_speaker


def _words_from_segments(
    asr_segments: Sequence[dict[str, Any]],
    *,
    default_language: str | None,
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for seg in asr_segments:
        seg_start = _safe_float(seg.get("start"), default=0.0)
        seg_end = _safe_float(seg.get("end"), default=seg_start)
        seg_language = seg.get("language")
        language = (
            str(seg_language)
            if isinstance(seg_language, str) and seg_language.strip()
            else default_language
        )
        seg_words = seg.get("words")
        if not isinstance(seg_words, list):
            seg_words = []
        if not seg_words and seg.get("text"):
            seg_words = [
                {
                    "start": seg_start,
                    "end": seg_end,
                    "word": str(seg.get("text")),
                }
            ]
        for raw_word in seg_words:
            if not isinstance(raw_word, dict):
                continue
            text = str(raw_word.get("word") or "").strip()
            if not text:
                continue
            start = _safe_float(raw_word.get("start"), default=seg_start)
            end = _safe_float(raw_word.get("end"), default=max(start, seg_end))
            if end < start:
                end = start
            payload: dict[str, Any] = {
                "start": round(start, 3),
                "end": round(end, 3),
                "word": text,
            }
            if language:
                payload["language"] = language
            words.append(payload)
    words.sort(key=lambda row: (row["start"], row["end"], row["word"]))
    return words


def _build_speaker_turns(
    asr_segments: Sequence[dict[str, Any]],
    diar_segments: Sequence[dict[str, Any]],
    *,
    default_language: str | None,
) -> list[dict[str, Any]]:
    words = _words_from_segments(asr_segments, default_language=default_language)
    if not words:
        return []

    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for word in words:
        start = _safe_float(word.get("start"), default=0.0)
        end = _safe_float(word.get("end"), default=start)
        speaker = _pick_speaker(start, end, diar_segments)
        language = word.get("language")
        if (
            current is not None
            and current["speaker"] == speaker
            and start - _safe_float(current["end"], default=start) <= 1.0
        ):
            current["end"] = round(max(_safe_float(current["end"]), end), 3)
            current["text"] = f"{current['text']} {word['word']}".strip()
        else:
            if current is not None:
                turns.append(current)
            current = {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "text": str(word["word"]),
            }
            if isinstance(language, str) and language:
                current["language"] = language
    if current is not None:
        turns.append(current)
    return turns


def _speaker_slug(label: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    slug = slug.strip("_")
    return slug or "speaker"


def _clear_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _snippet_window(
    start: float,
    end: float,
    *,
    duration_sec: float | None,
) -> tuple[float, float]:
    seg_duration = max(0.0, end - start)
    target = min(20.0, max(10.0, seg_duration if seg_duration > 0 else 10.0))
    center = start + (seg_duration / 2.0 if seg_duration > 0 else 0.0)
    clip_start = max(0.0, center - (target / 2.0))
    clip_end = clip_start + target
    if duration_sec is not None and duration_sec > 0:
        if clip_end > duration_sec:
            clip_end = duration_sec
            clip_start = max(0.0, clip_end - target)
    if clip_end <= clip_start:
        clip_end = clip_start + min(target, 1.0)
    return round(clip_start, 3), round(clip_end, 3)


def _extract_wav_snippet_with_wave(
    audio_path: Path,
    out_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> bool:
    if audio_path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            start_frame = max(0, int(start_sec * rate))
            end_frame = max(start_frame + 1, int(end_sec * rate))
            src.setpos(min(start_frame, src.getnframes()))
            frames = src.readframes(max(0, end_frame - start_frame))
        if not frames:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(rate)
            dst.writeframes(frames)
        return True
    except Exception:
        return False


def _extract_wav_snippet_with_ffmpeg(
    audio_path: Path,
    out_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    duration = max(0.1, end_sec - start_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return False
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 44


def _write_silence_wav(path: Path, duration_sec: float = 1.0) -> None:
    samples = max(int(16000 * max(duration_sec, 0.1)), 1)
    payload = b"\x00\x00" * samples
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(16000)
        wav_out.writeframes(payload)


def _export_speaker_snippets(
    *,
    audio_path: Path,
    diar_segments: Sequence[dict[str, Any]],
    snippets_dir: Path,
    duration_sec: float | None,
) -> list[Path]:
    _clear_dir(snippets_dir)

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in diar_segments:
        speaker = str(segment.get("speaker", "S1"))
        by_speaker.setdefault(speaker, []).append(segment)

    all_outputs: list[Path] = []
    for speaker in sorted(by_speaker):
        ranked = sorted(
            by_speaker[speaker],
            key=lambda row: (
                -(_safe_float(row.get("end")) - _safe_float(row.get("start"))),
                _safe_float(row.get("start")),
                _safe_float(row.get("end")),
            ),
        )
        chosen: list[dict[str, Any]] = []
        for candidate in ranked:
            start = _safe_float(candidate.get("start"))
            end = _safe_float(candidate.get("end"), default=start)
            if end <= start:
                continue
            overlaps = any(
                _overlap_seconds(
                    start,
                    end,
                    _safe_float(existing.get("start")),
                    _safe_float(existing.get("end")),
                )
                > 0.5
                for existing in chosen
            )
            if overlaps:
                continue
            chosen.append(candidate)
            if len(chosen) == 3:
                break
        if len(chosen) < 2:
            for candidate in ranked:
                if candidate in chosen:
                    continue
                chosen.append(candidate)
                if len(chosen) == 2:
                    break

        speaker_dir = snippets_dir / _speaker_slug(speaker)
        for idx, segment in enumerate(
            sorted(chosen, key=lambda row: _safe_float(row.get("start"))), start=1
        ):
            seg_start = _safe_float(segment.get("start"))
            seg_end = _safe_float(segment.get("end"), default=seg_start)
            clip_start, clip_end = _snippet_window(
                seg_start,
                seg_end,
                duration_sec=duration_sec,
            )
            out_path = speaker_dir / f"{idx}.wav"
            written = _extract_wav_snippet_with_wave(
                audio_path,
                out_path,
                start_sec=clip_start,
                end_sec=clip_end,
            )
            if not written:
                written = _extract_wav_snippet_with_ffmpeg(
                    audio_path,
                    out_path,
                    start_sec=clip_start,
                    end_sec=clip_end,
                )
            if not written:
                _write_silence_wav(out_path, duration_sec=min(max(clip_end - clip_start, 1.0), 2.0))
            all_outputs.append(out_path)
    all_outputs.sort()
    return all_outputs


def _audio_duration_from_wave(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            frames = src.getnframes()
        if rate <= 0:
            return None
        return frames / float(rate)
    except Exception:
        return None


def _audio_duration_from_ffprobe(audio_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _speech_ratio_from_wave(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            sample_width = src.getsampwidth()
            frame_samples = max(int(rate * 0.03), 1)
            frame_bytes = frame_samples * channels * sample_width
            voiced = 0
            total = 0
            while True:
                chunk = src.readframes(frame_samples)
                if not chunk:
                    break
                if len(chunk) < frame_bytes // 2:
                    break
                total += 1
                if audioop.rms(chunk, sample_width) >= 350:
                    voiced += 1
            if total == 0:
                return 0.0
            return voiced / float(total)
    except Exception:
        return None


def _speech_ratio_from_ffmpeg(audio_path: Path) -> float | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            if proc.stdout is None:
                return None
            frame_bytes = 960  # 30ms @ 16kHz * 2 bytes sample
            voiced = 0
            total = 0
            while True:
                chunk = proc.stdout.read(frame_bytes)
                if not chunk:
                    break
                if len(chunk) < frame_bytes:
                    break
                total += 1
                if audioop.rms(chunk, 2) >= 300:
                    voiced += 1
            proc.wait(timeout=60)
            if proc.returncode != 0:
                return None
            if total == 0:
                return 0.0
            return voiced / float(total)
    except Exception:
        return None


def run_precheck(audio_path: Path, cfg: Settings | None = None) -> PrecheckResult:
    """Compute duration + VAD speech ratio and decide quarantine status."""
    settings = cfg or Settings()
    duration_sec = _audio_duration_from_wave(audio_path)
    if duration_sec is None:
        duration_sec = _audio_duration_from_ffprobe(audio_path)

    speech_ratio = _speech_ratio_from_wave(audio_path)
    if speech_ratio is None:
        speech_ratio = _speech_ratio_from_ffmpeg(audio_path)

    quarantine_reason: str | None = None
    if duration_sec is not None and duration_sec < settings.precheck_min_duration_sec:
        quarantine_reason = f"duration_lt_{settings.precheck_min_duration_sec:.0f}s"
    elif (
        speech_ratio is not None
        and speech_ratio < settings.precheck_min_speech_ratio
    ):
        quarantine_reason = (
            f"speech_ratio_lt_{settings.precheck_min_speech_ratio:.2f}"
        )

    return PrecheckResult(
        duration_sec=round(duration_sec, 3) if duration_sec is not None else None,
        speech_ratio=round(speech_ratio, 4) if speech_ratio is not None else None,
        quarantine_reason=quarantine_reason,
    )


def _fallback_diarization(duration_sec: float | None) -> Any:
    duration = max(duration_sec or 0.0, 0.1)

    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            if yield_label:
                yield SimpleNamespace(start=0.0, end=duration), "S1"
            else:  # pragma: no cover - legacy compatibility branch
                yield (SimpleNamespace(start=0.0, end=duration),)

    return _Annotation()


async def run_pipeline(
    audio_path: Path,
    cfg: Settings,
    llm: LLMClient,
    diariser: Diariser,
    recording_id: str | None = None,
    precheck: PrecheckResult | None = None,
) -> TranscriptResult:
    """Transcribe ``audio_path`` and return a structured result."""
    start = time.perf_counter()
    import whisperx

    artifact_paths = build_recording_artifacts(
        cfg.recordings_root,
        recording_id=recording_id or _default_recording_id(audio_path),
        audio_ext=audio_path.suffix,
    )
    stage_raw_audio(audio_path, artifact_paths.raw_audio_path)

    precheck_result = precheck or run_precheck(audio_path, cfg)

    atomic_write_json(
        artifact_paths.metrics_json_path,
        {
            "status": "running",
            "version": 1,
            "precheck": {
                "duration_sec": precheck_result.duration_sec,
                "speech_ratio": precheck_result.speech_ratio,
                "quarantine_reason": precheck_result.quarantine_reason,
            },
        },
    )

    if precheck_result.quarantine_reason:
        atomic_write_text(artifact_paths.transcript_txt_path, "")
        atomic_write_json(
            artifact_paths.transcript_json_path,
            {
                "recording_id": artifact_paths.recording_id,
                "language": {"detected": "unknown", "confidence": None},
                "segments": [],
                "speakers": [],
                "text": "",
            },
        )
        atomic_write_json(artifact_paths.segments_json_path, [])
        atomic_write_json(artifact_paths.speaker_turns_json_path, [])
        atomic_write_json(
            artifact_paths.summary_json_path,
            {
                "friendly": 0,
                "model": cfg.llm_model,
                "summary": "",
                "status": "quarantined",
                "reason": precheck_result.quarantine_reason,
            },
        )
        atomic_write_json(
            artifact_paths.metrics_json_path,
            {
                "status": "quarantined",
                "version": 1,
                "precheck": {
                    "duration_sec": precheck_result.duration_sec,
                    "speech_ratio": precheck_result.speech_ratio,
                    "quarantine_reason": precheck_result.quarantine_reason,
                },
            },
        )
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(
            summary="Quarantined",
            body="",
            friendly=0,
            speakers=[],
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=[],
            segments=[],
        )

    def _asr() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        kwargs: dict[str, Any] = {"vad_filter": True, "language": "auto"}
        try:
            segments, info = whisperx.transcribe(
                str(audio_path),
                word_timestamps=True,
                **kwargs,
            )
        except TypeError:
            segments, info = whisperx.transcribe(str(audio_path), **kwargs)
        return list(segments), dict(info or {})

    async def _safe_diarise() -> Any:
        try:
            return await diariser(audio_path)
        except Exception:
            return _fallback_diarization(precheck_result.duration_sec)

    asr_task = asyncio.to_thread(_asr)
    diar_task = _safe_diarise()
    (raw_segments, info), diarization = await asyncio.gather(asr_task, diar_task)

    asr_segments = _normalise_asr_segments(raw_segments)
    language_info = _language_payload(info)
    detected_language = language_info["detected"] if language_info["detected"] != "unknown" else None

    asr_text = " ".join(seg.get("text", "").strip() for seg in asr_segments).strip()
    clean_text = normalizer.dedup(asr_text)
    diar_segments = _safe_diarization_segments(diarization)
    if not diar_segments and asr_segments:
        fallback_end = max(_safe_float(seg.get("end")) for seg in asr_segments)
        diar_segments = [
            {"start": 0.0, "end": round(max(fallback_end, 0.1), 3), "speaker": "S1"}
        ]

    speaker_turns = _build_speaker_turns(
        asr_segments,
        diar_segments,
        default_language=detected_language,
    )

    aliases = _load_aliases(cfg.speaker_db)
    for row in diar_segments:
        label = str(row["speaker"])
        aliases.setdefault(label, label)
    _save_aliases(aliases, cfg.speaker_db)

    if not clean_text:
        atomic_write_text(artifact_paths.transcript_txt_path, "")
        atomic_write_json(
            artifact_paths.transcript_json_path,
            {
                "recording_id": artifact_paths.recording_id,
                "language": language_info,
                "segments": asr_segments,
                "speakers": sorted({aliases.get(row["speaker"], row["speaker"]) for row in diar_segments}),
                "text": "",
            },
        )
        atomic_write_json(artifact_paths.segments_json_path, diar_segments)
        atomic_write_json(artifact_paths.speaker_turns_json_path, speaker_turns)
        atomic_write_json(
            artifact_paths.summary_json_path,
            {
                "friendly": 0,
                "model": cfg.llm_model,
                "summary": "No speech detected",
            },
        )
        atomic_write_json(
            artifact_paths.metrics_json_path,
            {
                "status": "no_speech",
                "version": 1,
                "precheck": {
                    "duration_sec": precheck_result.duration_sec,
                    "speech_ratio": precheck_result.speech_ratio,
                    "quarantine_reason": None,
                },
                "language": language_info,
                "asr_segments": len(asr_segments),
                "diar_segments": len(diar_segments),
                "speaker_turns": len(speaker_turns),
            },
        )
        p95_latency_seconds.observe(time.perf_counter() - start)
        return TranscriptResult(
            summary="No speech detected",
            body="",
            friendly=0,
            speakers=sorted({aliases.get(row["speaker"], row["speaker"]) for row in diar_segments}),
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=[],
            segments=[],
        )

    snippet_paths = _export_speaker_snippets(
        audio_path=audio_path,
        diar_segments=diar_segments,
        snippets_dir=artifact_paths.snippets_dir,
        duration_sec=precheck_result.duration_sec,
    )

    speaker_lines = [
        f"[{turn['start']:.2f}-{turn['end']:.2f}] **{aliases.get(turn['speaker'], turn['speaker'])}:** {turn['text']}"
        for turn in speaker_turns
    ]
    speaker_lines = _merge_similar(speaker_lines, cfg.merge_similar)

    friendly = _sentiment_score(clean_text)
    sys_prompt = (
        "You are an assistant who writes concise 5-8 bullet summaries of any audio transcript. "
        "Return only the list without extra explanation."
    )
    user_prompt = f"{sys_prompt}\n\nTRANSCRIPT:\n{clean_text}\n\nSUMMARY:"

    try:
        msg = await llm.generate(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            model=cfg.llm_model,
        )
        summary = msg.get("content", "") if isinstance(msg, dict) else str(msg)

        serialised_segments = [
            SpeakerSegment(
                start=_safe_float(turn["start"]),
                end=_safe_float(turn["end"]),
                speaker=aliases.get(turn["speaker"], turn["speaker"]),
                text=str(turn["text"]),
            )
            for turn in speaker_turns
        ]
        atomic_write_text(artifact_paths.transcript_txt_path, clean_text)
        atomic_write_json(
            artifact_paths.transcript_json_path,
            {
                "recording_id": artifact_paths.recording_id,
                "language": language_info,
                "segments": asr_segments,
                "speaker_lines": speaker_lines,
                "speakers": sorted(set(aliases.get(turn["speaker"], turn["speaker"]) for turn in speaker_turns)),
                "text": clean_text,
            },
        )
        atomic_write_json(artifact_paths.segments_json_path, diar_segments)
        atomic_write_json(artifact_paths.speaker_turns_json_path, speaker_turns)
        atomic_write_json(
            artifact_paths.summary_json_path,
            {
                "friendly": friendly,
                "model": cfg.llm_model,
                "summary": summary,
            },
        )
        atomic_write_json(
            artifact_paths.metrics_json_path,
            {
                "status": "ok",
                "version": 1,
                "precheck": {
                    "duration_sec": precheck_result.duration_sec,
                    "speech_ratio": precheck_result.speech_ratio,
                    "quarantine_reason": None,
                },
                "language": language_info,
                "asr_segments": len(asr_segments),
                "diar_segments": len(diar_segments),
                "speaker_turns": len(speaker_turns),
                "snippets": len(snippet_paths),
            },
        )

        result = TranscriptResult(
            summary=summary,
            body=clean_text,
            friendly=friendly,
            speakers=sorted(set(aliases.get(turn["speaker"], turn["speaker"]) for turn in speaker_turns)),
            summary_path=artifact_paths.summary_json_path,
            body_path=artifact_paths.transcript_txt_path,
            unknown_chunks=snippet_paths,
            segments=serialised_segments,
        )
    except Exception:
        error_rate_total.inc()
        atomic_write_json(
            artifact_paths.summary_json_path,
            {
                "friendly": friendly,
                "model": cfg.llm_model,
                "summary": "",
                "status": "failed",
            },
        )
        raise
    finally:
        p95_latency_seconds.observe(time.perf_counter() - start)

    return result


__all__ = [
    "run_pipeline",
    "run_precheck",
    "PrecheckResult",
    "Settings",
    "Diariser",
    "refresh_aliases",
]
