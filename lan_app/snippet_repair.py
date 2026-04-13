from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence
from uuid import uuid4

from lan_transcriber.artifacts import atomic_write_json
from lan_transcriber.pipeline import Settings as PipelineSettings
from lan_transcriber.pipeline_steps.noise_detection import (
    apply_noise_flags_to_manifest,
    update_diarization_metadata_with_noise,
)
from lan_transcriber.pipeline_steps.snippets import (
    SnippetExportRequest,
    export_speaker_snippets,
    write_empty_snippets_manifest,
)

from .config import AppSettings
from .constants import (
    RECORDING_STATUSES,
    RECORDING_STATUS_PROCESSING,
    RECORDING_STATUS_QUEUED,
    RECORDING_STATUS_STOPPING,
)
from .db import (
    get_recording,
    has_started_job_for_recording,
    list_recording_pipeline_stages,
    list_recordings,
    mark_recording_pipeline_stage_completed,
)

_LOG = logging.getLogger(__name__)
_ACTIVE_RECORDING_STATUSES = frozenset(
    {
        RECORDING_STATUS_QUEUED,
        RECORDING_STATUS_PROCESSING,
        RECORDING_STATUS_STOPPING,
    }
)
_TERMINAL_RECORDING_STATUSES = frozenset(RECORDING_STATUSES) - _ACTIVE_RECORDING_STATUSES


class SnippetRepairError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "snippet_repair_error").strip() or "snippet_repair_error"


class SnippetRepairPreconditionError(SnippetRepairError):
    pass


class SnippetRepairExecutionError(SnippetRepairError):
    pass


@dataclass(frozen=True)
class SnippetRepairEligibility:
    recording_id: str
    available: bool
    artifact_state: str
    reason_code: str | None = None
    reason_text: str | None = None
    recording_status: str | None = None
    audio_path: Path | None = None
    audio_source: str | None = None
    duration_sec: float | None = None
    diarization_segments: tuple[dict[str, Any], ...] = ()
    speaker_turns: tuple[dict[str, Any], ...] = ()
    degraded_diarization: bool = False


@dataclass(frozen=True)
class SnippetRepairResult:
    recording_id: str
    manifest_status: str
    accepted_snippets: int
    speaker_count: int
    warning_count: int
    degraded_diarization: bool
    audio_source: str
    duration_sec: float
    artifact_state_before: str


@dataclass(frozen=True)
class SnippetRepairBatchItem:
    recording_id: str
    outcome: str
    detail: str
    manifest_status: str | None = None
    accepted_snippets: int = 0


@dataclass(frozen=True)
class SnippetRepairBatchSummary:
    regenerated: int
    skipped: int
    failed: int
    items: tuple[SnippetRepairBatchItem, ...]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _recording_root(recording_id: str, *, settings: AppSettings) -> Path:
    return settings.recordings_root / recording_id


def _derived_dir(recording_id: str, *, settings: AppSettings) -> Path:
    return _recording_root(recording_id, settings=settings) / "derived"


def _snippets_dir(recording_id: str, *, settings: AppSettings) -> Path:
    return _derived_dir(recording_id, settings=settings) / "snippets"


def _snippets_manifest_path(recording_id: str, *, settings: AppSettings) -> Path:
    return _derived_dir(recording_id, settings=settings) / "snippets_manifest.json"


def _repair_log_path(recording_id: str, *, settings: AppSettings) -> Path:
    return _recording_root(recording_id, settings=settings) / "logs" / "step-snippet-repair.log"


def _append_repair_log(recording_id: str, message: str, *, settings: AppSettings) -> None:
    path = _repair_log_path(recording_id, settings=settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{_utc_now()}] {message}\n")


def _safe_path(candidate: Path, *, root: Path) -> Path | None:
    root_resolved = root.resolve()
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return None
    return resolved


def _resolve_raw_audio_path(recording_id: str, *, settings: AppSettings) -> Path | None:
    raw_dir = _recording_root(recording_id, settings=settings) / "raw"
    safe_raw_dir = _safe_path(raw_dir, root=_recording_root(recording_id, settings=settings))
    if safe_raw_dir is None or not safe_raw_dir.exists() or not safe_raw_dir.is_dir():
        return None
    for candidate in sorted(safe_raw_dir.glob("audio.*")):
        safe_candidate = _safe_path(candidate, root=safe_raw_dir)
        if safe_candidate is None or not safe_candidate.is_file():
            continue
        return safe_candidate
    return None


def _resolve_audio_path(
    recording_id: str,
    *,
    settings: AppSettings,
) -> tuple[Path | None, str | None]:
    sanitized_audio_path = _derived_dir(recording_id, settings=settings) / "audio_sanitized.wav"
    safe_sanitized = _safe_path(
        sanitized_audio_path,
        root=_recording_root(recording_id, settings=settings),
    )
    if safe_sanitized is not None and safe_sanitized.exists() and safe_sanitized.is_file():
        return safe_sanitized, "sanitized_audio"
    raw_audio_path = _resolve_raw_audio_path(recording_id, settings=settings)
    if raw_audio_path is not None:
        return raw_audio_path, "raw_audio"
    return None, None


def _manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    speakers_payload = manifest.get("speakers")
    if not isinstance(speakers_payload, dict):
        return []
    entries: list[dict[str, Any]] = []
    for rows in speakers_payload.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                entries.append(dict(row))
    return entries


def snippet_manifest_counts(manifest: dict[str, Any]) -> dict[str, int]:
    accepted_snippets = 0
    speaker_count = 0
    warning_count = 0
    speakers_payload = manifest.get("speakers")
    if isinstance(speakers_payload, dict):
        for entries in speakers_payload.values():
            if not isinstance(entries, list):
                continue
            speaker_count += 1
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status") or "").strip()
                if status == "accepted":
                    accepted_snippets += 1
                elif status:
                    warning_count += 1
    warnings_payload = manifest.get("warnings")
    if isinstance(warnings_payload, list):
        warning_count += sum(1 for item in warnings_payload if isinstance(item, dict))
    return {
        "accepted_snippets": accepted_snippets,
        "speaker_count": speaker_count,
        "warning_count": warning_count,
    }


def snippet_export_result_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    counts = snippet_manifest_counts(manifest)
    return {
        "manifest_status": str(manifest.get("manifest_status") or "ok"),
        "accepted_snippets": counts["accepted_snippets"],
        "speaker_count": counts["speaker_count"],
        "warning_count": counts["warning_count"],
        "degraded_diarization": bool(manifest.get("degraded_diarization")),
    }


def _normalise_warnings(warnings: Sequence[dict[str, str]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for warning in warnings or ():
        if not isinstance(warning, dict):
            continue
        code = str(warning.get("code") or "").strip()
        message = str(warning.get("message") or "").strip()
        payload: dict[str, str] = {}
        if code:
            payload["code"] = code
        if message:
            payload["message"] = message
        if payload:
            normalized.append(payload)
    return normalized


def finalize_snippets_manifest(
    manifest_path: Path,
    *,
    manifest_status: str,
    degraded_diarization: bool | None = None,
    warnings: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    manifest = _load_json_dict(manifest_path)
    if not manifest:
        raise SnippetRepairExecutionError(
            "missing_snippets_manifest",
            "Snippet regeneration finished without writing snippets_manifest.json.",
        )
    normalized_warnings = _normalise_warnings(warnings)
    if normalized_warnings:
        manifest["warnings"] = normalized_warnings
    else:
        manifest.pop("warnings", None)
    if degraded_diarization is not None:
        manifest["degraded_diarization"] = bool(degraded_diarization)
    manifest["manifest_status"] = str(manifest_status or "ok")
    manifest.update(snippet_manifest_counts(manifest))
    atomic_write_json(manifest_path, manifest)
    return manifest


def _write_empty_snippet_manifest(
    snippets_dir: Path,
    *,
    pipeline_settings: PipelineSettings,
    manifest_status: str,
    degraded_diarization: bool = False,
    warnings: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    write_empty_snippets_manifest(
        snippets_dir=snippets_dir,
        pad_seconds=pipeline_settings.snippet_pad_seconds,
        max_clip_duration_sec=pipeline_settings.snippet_max_duration_seconds,
        min_clip_duration_sec=pipeline_settings.snippet_min_duration_seconds,
        max_snippets_per_speaker=pipeline_settings.snippet_max_per_speaker,
    )
    return finalize_snippets_manifest(
        snippets_dir.parent / "snippets_manifest.json",
        manifest_status=manifest_status,
        degraded_diarization=degraded_diarization,
        warnings=warnings,
    )


def snippet_artifact_state(recording_id: str, *, settings: AppSettings) -> str:
    snippets_dir = _snippets_dir(recording_id, settings=settings)
    manifest_path = _snippets_manifest_path(recording_id, settings=settings)
    manifest_exists = manifest_path.exists()
    snippet_files = (
        snippets_dir.exists()
        and snippets_dir.is_dir()
        and any(path.is_file() for path in snippets_dir.rglob("*.wav"))
    )
    if not manifest_exists:
        return "stale" if snippet_files else "missing"

    manifest = _load_json_dict(manifest_path)
    if not manifest:
        return "stale"

    safe_root = _safe_path(snippets_dir, root=_recording_root(recording_id, settings=settings))
    if safe_root is None:
        return "stale"

    for entry in _manifest_entries(manifest):
        if str(entry.get("status") or "").strip() != "accepted":
            continue
        relative_path = str(entry.get("relative_path") or "").strip()
        if not relative_path:
            return "stale"
        safe_entry_path = _safe_path(snippets_dir / relative_path, root=safe_root)
        if safe_entry_path is None or not safe_entry_path.exists() or not safe_entry_path.is_file():
            return "stale"
    return "present"


def assess_snippet_repair(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
) -> SnippetRepairEligibility:
    cfg = settings or AppSettings()
    artifact_state = snippet_artifact_state(recording_id, settings=cfg)
    recording = get_recording(recording_id, settings=cfg)
    if recording is None:
        return SnippetRepairEligibility(
            recording_id=recording_id,
            available=False,
            artifact_state=artifact_state,
            reason_code="recording_not_found",
            reason_text="Recording not found.",
        )

    recording_status = str(recording.get("status") or "").strip() or None
    if (
        recording_status in _ACTIVE_RECORDING_STATUSES
        or has_started_job_for_recording(recording_id, settings=cfg)
    ):
        return SnippetRepairEligibility(
            recording_id=recording_id,
            available=False,
            artifact_state=artifact_state,
            reason_code="recording_active",
            reason_text=(
                "Snippet regeneration is disabled while the recording is queued or processing."
            ),
            recording_status=recording_status,
        )

    audio_path, audio_source = _resolve_audio_path(recording_id, settings=cfg)
    if audio_path is None or audio_source is None:
        return SnippetRepairEligibility(
            recording_id=recording_id,
            available=False,
            artifact_state=artifact_state,
            reason_code="missing_audio",
            reason_text="Missing sanitized audio and no raw audio fallback is available.",
            recording_status=recording_status,
        )

    derived_dir = _derived_dir(recording_id, settings=cfg)
    precheck_payload = _load_json(derived_dir / "precheck.json")
    duration_sec = None
    if isinstance(precheck_payload, dict):
        duration_sec = _safe_float(precheck_payload.get("duration_sec"))
    if duration_sec is None or duration_sec <= 0.0:
        return SnippetRepairEligibility(
            recording_id=recording_id,
            available=False,
            artifact_state=artifact_state,
            reason_code="missing_precheck_duration",
            reason_text="Missing precheck duration in derived/precheck.json.",
            recording_status=recording_status,
        )

    diarization_segments_payload = _load_json(derived_dir / "diarization_segments.json")
    if not isinstance(diarization_segments_payload, list):
        return SnippetRepairEligibility(
            recording_id=recording_id,
            available=False,
            artifact_state=artifact_state,
            reason_code="missing_diarization_segments",
            reason_text="Missing or unreadable derived/diarization_segments.json.",
            recording_status=recording_status,
        )

    speaker_turns_payload = _load_json(derived_dir / "speaker_turns.json")
    if not isinstance(speaker_turns_payload, list):
        return SnippetRepairEligibility(
            recording_id=recording_id,
            available=False,
            artifact_state=artifact_state,
            reason_code="missing_speaker_turns",
            reason_text="Missing or unreadable derived/speaker_turns.json.",
            recording_status=recording_status,
        )

    diarization_metadata_payload = _load_json(derived_dir / "diarization_metadata.json")
    if not isinstance(diarization_metadata_payload, dict):
        return SnippetRepairEligibility(
            recording_id=recording_id,
            available=False,
            artifact_state=artifact_state,
            reason_code="missing_diarization_metadata",
            reason_text="Missing or unreadable derived/diarization_metadata.json.",
            recording_status=recording_status,
        )

    diarization_segments = tuple(
        row for row in diarization_segments_payload if isinstance(row, dict)
    )
    speaker_turns = tuple(row for row in speaker_turns_payload if isinstance(row, dict))
    return SnippetRepairEligibility(
        recording_id=recording_id,
        available=True,
        artifact_state=artifact_state,
        recording_status=recording_status,
        audio_path=audio_path,
        audio_source=audio_source,
        duration_sec=duration_sec,
        diarization_segments=diarization_segments,
        speaker_turns=speaker_turns,
        degraded_diarization=bool(diarization_metadata_payload.get("degraded")),
    )


def _staged_snippet_output_root(recording_id: str, *, settings: AppSettings) -> Path:
    return Path(
        tempfile.mkdtemp(
            prefix=".snippet-repair-",
            dir=str(_derived_dir(recording_id, settings=settings)),
        )
    )


def _build_staged_snippet_outputs(
    eligibility: SnippetRepairEligibility,
    *,
    settings: AppSettings,
    pipeline_settings: PipelineSettings,
) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    temp_root = _staged_snippet_output_root(eligibility.recording_id, settings=settings)
    staged_snippets_dir = temp_root / "snippets"
    try:
        if not eligibility.speaker_turns:
            manifest = _write_empty_snippet_manifest(
                staged_snippets_dir,
                pipeline_settings=pipeline_settings,
                manifest_status="no_usable_speech",
                degraded_diarization=eligibility.degraded_diarization,
                warnings=[
                    {
                        "code": "no_speaker_turns",
                        "message": (
                            "No speaker turns were available, so snippet regeneration "
                            "produced no clips."
                        ),
                    }
                ],
            )
            return temp_root, manifest, None

        export_speaker_snippets(
            SnippetExportRequest(
                audio_path=eligibility.audio_path or Path(""),
                diar_segments=list(eligibility.diarization_segments),
                snippets_dir=staged_snippets_dir,
                duration_sec=eligibility.duration_sec,
                speaker_turns=list(eligibility.speaker_turns),
                degraded_diarization=eligibility.degraded_diarization,
                pad_seconds=pipeline_settings.snippet_pad_seconds,
                max_clip_duration_sec=pipeline_settings.snippet_max_duration_seconds,
                min_clip_duration_sec=pipeline_settings.snippet_min_duration_seconds,
                max_snippets_per_speaker=pipeline_settings.snippet_max_per_speaker,
            )
        )
        manifest_path = staged_snippets_dir.parent / "snippets_manifest.json"
        noise_summary: dict[str, Any] | None = None
        if pipeline_settings.noise_detection_enabled:
            noise_summary = apply_noise_flags_to_manifest(
                manifest_path,
                snippets_dir=staged_snippets_dir,
                threshold=pipeline_settings.noise_speech_ratio_threshold,
            )
        manifest = _load_json_dict(manifest_path)
        counts = snippet_manifest_counts(manifest)
        manifest_status = "ok"
        if counts["accepted_snippets"] == 0:
            manifest_status = (
                "degraded"
                if eligibility.degraded_diarization
                else "no_clean_snippets"
            )
        elif counts["warning_count"] > 0:
            manifest_status = "partial"
        manifest = finalize_snippets_manifest(
            manifest_path,
            manifest_status=manifest_status,
            degraded_diarization=eligibility.degraded_diarization,
        )
        return temp_root, manifest, noise_summary
    except SnippetRepairError:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise SnippetRepairExecutionError(
            type(exc).__name__,
            str(exc) or type(exc).__name__,
        ) from exc


def _replace_staged_snippet_outputs(
    recording_id: str,
    staged_root: Path,
    *,
    settings: AppSettings,
) -> None:
    derived_dir = _derived_dir(recording_id, settings=settings)
    target_snippets_dir = derived_dir / "snippets"
    target_manifest_path = derived_dir / "snippets_manifest.json"
    staged_snippets_dir = staged_root / "snippets"
    staged_manifest_path = staged_root / "snippets_manifest.json"
    backup_token = uuid4().hex[:8]
    backup_snippets_dir = derived_dir / f".snippets-backup-{backup_token}"
    backup_manifest_path = (
        derived_dir / f".snippets_manifest-backup-{backup_token}.json"
    )
    moved_snippets = False
    moved_manifest = False
    staged_snippets_moved = False
    try:
        if target_snippets_dir.exists():
            target_snippets_dir.rename(backup_snippets_dir)
            moved_snippets = True
        if target_manifest_path.exists():
            target_manifest_path.rename(backup_manifest_path)
            moved_manifest = True
        staged_snippets_dir.rename(target_snippets_dir)
        staged_snippets_moved = True
        staged_manifest_path.rename(target_manifest_path)
    except Exception as exc:
        if moved_manifest and backup_manifest_path.exists():
            target_manifest_path.unlink(missing_ok=True)
            backup_manifest_path.rename(target_manifest_path)
        if staged_snippets_moved and target_snippets_dir.exists():
            shutil.rmtree(target_snippets_dir, ignore_errors=True)
        if moved_snippets and backup_snippets_dir.exists():
            backup_snippets_dir.rename(target_snippets_dir)
        raise SnippetRepairExecutionError(
            "replace_failed",
            f"Failed to replace snippet artifacts: {exc}",
        ) from exc
    finally:
        shutil.rmtree(staged_root, ignore_errors=True)
        shutil.rmtree(backup_snippets_dir, ignore_errors=True)
        backup_manifest_path.unlink(missing_ok=True)


def _snippet_stage_metadata(
    recording_id: str,
    *,
    settings: AppSettings,
    result: SnippetRepairResult,
    origin: str,
) -> dict[str, Any]:
    existing_row = next(
        (
            row
            for row in list_recording_pipeline_stages(recording_id, settings=settings)
            if str(row.get("stage_name") or "").strip() == "snippet_export"
        ),
        {},
    )
    existing_metadata = (
        dict(existing_row.get("metadata_json"))
        if isinstance(existing_row.get("metadata_json"), dict)
        else {}
    )
    metadata = dict(existing_metadata)
    metadata.update(
        {
            "manifest_status": result.manifest_status,
            "accepted_snippets": result.accepted_snippets,
            "speaker_count": result.speaker_count,
            "warning_count": result.warning_count,
            "degraded_diarization": result.degraded_diarization,
            "repair_origin": origin,
            "repair_at": _utc_now(),
            "repair_audio_source": result.audio_source,
            "repair_duration_sec": round(result.duration_sec, 3),
            "repair_artifact_state_before": result.artifact_state_before,
        }
    )
    return metadata


def repair_recording_snippets(
    recording_id: str,
    *,
    settings: AppSettings | None = None,
    origin: str = "manual",
) -> SnippetRepairResult:
    cfg = settings or AppSettings()
    eligibility = assess_snippet_repair(recording_id, settings=cfg)
    if not eligibility.available:
        raise SnippetRepairPreconditionError(
            str(eligibility.reason_code or "snippet_repair_unavailable"),
            str(eligibility.reason_text or "Snippet regeneration is unavailable."),
        )

    # Honor caller-supplied AppSettings overrides (especially the noise-detection
    # knobs) so callers that disable detection or change the threshold via
    # `settings=` actually see that take effect during repair, rather than
    # silently reading process env vars via PipelineSettings().
    pipeline_settings = PipelineSettings(
        noise_detection_enabled=cfg.noise_detection_enabled,
        noise_speech_ratio_threshold=cfg.noise_speech_ratio_threshold,
        exclude_noise_speakers_from_transcript=cfg.exclude_noise_speakers_from_transcript,
    )
    _LOG.info(
        "snippet repair start recording_id=%s origin=%s audio_source=%s artifact_state=%s",
        recording_id,
        origin,
        eligibility.audio_source,
        eligibility.artifact_state,
    )
    _append_repair_log(
        recording_id,
        (
            "snippet repair start "
            f"origin={origin} "
            f"audio_source={eligibility.audio_source} "
            f"artifact_state={eligibility.artifact_state}"
        ),
        settings=cfg,
    )
    staged_root, manifest, noise_summary = _build_staged_snippet_outputs(
        eligibility,
        settings=cfg,
        pipeline_settings=pipeline_settings,
    )
    _replace_staged_snippet_outputs(recording_id, staged_root, settings=cfg)
    if pipeline_settings.noise_detection_enabled:
        summary_source = noise_summary or {}
        update_diarization_metadata_with_noise(
            _derived_dir(recording_id, settings=cfg) / "diarization_metadata.json",
            summary={
                "noise_speakers": list(summary_source.get("noise_speakers") or []),
                "speaker_metrics": dict(summary_source.get("speaker_metrics") or {}),
                "threshold": summary_source.get(
                    "threshold", pipeline_settings.noise_speech_ratio_threshold
                ),
            },
        )
    metadata = snippet_export_result_metadata(manifest)
    result = SnippetRepairResult(
        recording_id=recording_id,
        manifest_status=str(metadata["manifest_status"]),
        accepted_snippets=int(metadata["accepted_snippets"]),
        speaker_count=int(metadata["speaker_count"]),
        warning_count=int(metadata["warning_count"]),
        degraded_diarization=bool(metadata["degraded_diarization"]),
        audio_source=str(eligibility.audio_source or "unknown"),
        duration_sec=float(eligibility.duration_sec or 0.0),
        artifact_state_before=eligibility.artifact_state,
    )
    mark_recording_pipeline_stage_completed(
        recording_id,
        stage_name="snippet_export",
        metadata=_snippet_stage_metadata(
            recording_id,
            settings=cfg,
            result=result,
            origin=origin,
        ),
        settings=cfg,
    )
    _LOG.info(
        "snippet repair success recording_id=%s origin=%s status=%s accepted=%s speakers=%s warnings=%s",
        recording_id,
        origin,
        result.manifest_status,
        result.accepted_snippets,
        result.speaker_count,
        result.warning_count,
    )
    _append_repair_log(
        recording_id,
        (
            "snippet repair success "
            f"origin={origin} "
            f"status={result.manifest_status} "
            f"accepted={result.accepted_snippets} "
            f"speakers={result.speaker_count} "
            f"warnings={result.warning_count}"
        ),
        settings=cfg,
    )
    return result


def _iter_terminal_recordings(*, settings: AppSettings) -> Sequence[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total = 1
    while offset < total:
        batch, total = list_recordings(
            settings=settings,
            limit=500,
            offset=offset,
        )
        rows.extend(batch)
        offset += len(batch)
        if not batch:
            break
    return [
        row
        for row in rows
        if str(row.get("status") or "").strip() in _TERMINAL_RECORDING_STATUSES
    ]


def backfill_missing_snippets(
    *,
    settings: AppSettings | None = None,
    origin: str = "batch",
) -> SnippetRepairBatchSummary:
    cfg = settings or AppSettings()
    items: list[SnippetRepairBatchItem] = []
    regenerated = 0
    skipped = 0
    failed = 0
    for row in _iter_terminal_recordings(settings=cfg):
        recording_id = str(row.get("id") or "").strip()
        if not recording_id:
            continue
        artifact_state = snippet_artifact_state(recording_id, settings=cfg)
        if artifact_state == "present":
            skipped += 1
            items.append(
                SnippetRepairBatchItem(
                    recording_id=recording_id,
                    outcome="skipped",
                    detail="snippets already exist",
                )
            )
            continue
        try:
            result = repair_recording_snippets(
                recording_id,
                settings=cfg,
                origin=origin,
            )
        except SnippetRepairPreconditionError as exc:
            skipped += 1
            items.append(
                SnippetRepairBatchItem(
                    recording_id=recording_id,
                    outcome="skipped",
                    detail=str(exc),
                )
            )
        except SnippetRepairExecutionError as exc:
            failed += 1
            items.append(
                SnippetRepairBatchItem(
                    recording_id=recording_id,
                    outcome="failed",
                    detail=f"{exc.code}: {exc}",
                )
            )
        else:
            regenerated += 1
            items.append(
                SnippetRepairBatchItem(
                    recording_id=recording_id,
                    outcome="regenerated",
                    detail=(
                        f"{result.accepted_snippets} clean snippets across "
                        f"{result.speaker_count} speakers"
                    ),
                    manifest_status=result.manifest_status,
                    accepted_snippets=result.accepted_snippets,
                )
            )
    return SnippetRepairBatchSummary(
        regenerated=regenerated,
        skipped=skipped,
        failed=failed,
        items=tuple(items),
    )


__all__ = [
    "SnippetRepairBatchItem",
    "SnippetRepairBatchSummary",
    "SnippetRepairEligibility",
    "SnippetRepairError",
    "SnippetRepairExecutionError",
    "SnippetRepairPreconditionError",
    "SnippetRepairResult",
    "assess_snippet_repair",
    "backfill_missing_snippets",
    "finalize_snippets_manifest",
    "repair_recording_snippets",
    "snippet_artifact_state",
    "snippet_export_result_metadata",
    "snippet_manifest_counts",
]
