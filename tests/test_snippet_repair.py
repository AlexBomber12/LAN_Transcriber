from __future__ import annotations

import json
from pathlib import Path
import runpy
import shutil
import sys
import wave

import pytest

from lan_app import snippet_repair
from lan_app.config import AppSettings
from lan_app.constants import RECORDING_STATUS_READY
from lan_app.db import (
    create_recording,
    init_db,
    list_recording_pipeline_stages,
    mark_recording_pipeline_stage_completed,
)
from lan_app.tools import repair_snippets as repair_snippets_tool


def _cfg(tmp_path: Path) -> AppSettings:
    cfg = AppSettings(
        data_root=tmp_path,
        recordings_root=tmp_path / "recordings",
        db_path=tmp_path / "db" / "app.db",
    )
    cfg.metrics_snapshot_path = tmp_path / "metrics.snap"
    return cfg


def _write_pcm_wav(path: Path, *, duration_sec: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(sample_rate * duration_sec), 1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def _seed_repair_artifacts(
    cfg: AppSettings,
    recording_id: str,
    *,
    raw_only: bool = False,
    include_speaker_turns: bool = True,
) -> None:
    recording_root = cfg.recordings_root / recording_id
    derived = recording_root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    if raw_only:
        _write_pcm_wav(recording_root / "raw" / "audio.wav", duration_sec=3.0)
    else:
        _write_pcm_wav(derived / "audio_sanitized.wav", duration_sec=3.0)
    (derived / "precheck.json").write_text(
        json.dumps({"duration_sec": 3.0, "speech_ratio": 0.8, "quarantine_reason": None}),
        encoding="utf-8",
    )
    (derived / "diarization_segments.json").write_text(
        json.dumps(
            [
                {"speaker": "S1", "start": 0.0, "end": 1.2},
                {"speaker": "S2", "start": 1.3, "end": 2.1},
            ]
        ),
        encoding="utf-8",
    )
    (derived / "diarization_metadata.json").write_text(
        json.dumps({"degraded": False}),
        encoding="utf-8",
    )
    if include_speaker_turns:
        (derived / "speaker_turns.json").write_text(
            json.dumps(
                [
                    {"speaker": "S1", "start": 0.0, "end": 1.2, "text": "hello team"},
                    {"speaker": "S2", "start": 1.3, "end": 2.1, "text": "hi there"},
                    {"speaker": "S1", "start": 2.2, "end": 2.9, "text": "follow up"},
                ]
            ),
            encoding="utf-8",
        )


def _write_existing_manifest(cfg: AppSettings, recording_id: str) -> tuple[Path, Path]:
    derived = cfg.recordings_root / recording_id / "derived"
    snippets = derived / "snippets" / "S1"
    snippets.mkdir(parents=True, exist_ok=True)
    snippet_path = snippets / "1.wav"
    snippet_path.write_bytes(b"old-clip")
    manifest_path = derived / "snippets_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_kind": "turn",
                "degraded_diarization": False,
                "manifest_status": "ok",
                "pad_seconds": 0.25,
                "max_clip_duration_seconds": 8.0,
                "min_clip_duration_seconds": 0.8,
                "max_snippets_per_speaker": 3,
                "accepted_snippets": 1,
                "speaker_count": 1,
                "warning_count": 0,
                "speakers": {
                    "S1": [
                        {
                            "snippet_id": "S1-01",
                            "speaker": "S1",
                            "status": "accepted",
                            "recommended": True,
                            "relative_path": "S1/1.wav",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, snippet_path


def test_repair_recording_snippets_succeeds_and_only_updates_snippet_stage(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-repair-1",
        source="upload",
        source_filename="repair.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-repair-1")
    mark_recording_pipeline_stage_completed(
        "rec-repair-1",
        stage_name="speaker_turns",
        metadata={"keep": True},
        settings=cfg,
    )
    mark_recording_pipeline_stage_completed(
        "rec-repair-1",
        stage_name="llm_extract",
        metadata={"preserved": "yes"},
        settings=cfg,
    )

    result = snippet_repair.repair_recording_snippets(
        "rec-repair-1",
        settings=cfg,
        origin="pytest",
    )

    assert result.accepted_snippets >= 1
    manifest = json.loads(
        (cfg.recordings_root / "rec-repair-1" / "derived" / "snippets_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["manifest_status"] in {"ok", "partial"}
    stages = {
        row["stage_name"]: row
        for row in list_recording_pipeline_stages("rec-repair-1", settings=cfg)
    }
    assert stages["llm_extract"]["metadata_json"] == {"preserved": "yes"}
    assert stages["snippet_export"]["status"] == "completed"
    assert stages["snippet_export"]["metadata_json"]["repair_origin"] == "pytest"


def test_assess_snippet_repair_uses_safe_raw_audio_fallback(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-repair-raw-1",
        source="upload",
        source_filename="repair-raw.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-repair-raw-1", raw_only=True)

    eligibility = snippet_repair.assess_snippet_repair(
        "rec-repair-raw-1",
        settings=cfg,
    )

    assert eligibility.available is True
    assert eligibility.audio_source == "raw_audio"


def test_repair_recording_snippets_rejects_missing_speaker_turns(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-repair-missing-1",
        source="upload",
        source_filename="repair-missing.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(
        cfg,
        "rec-repair-missing-1",
        include_speaker_turns=False,
    )

    eligibility = snippet_repair.assess_snippet_repair(
        "rec-repair-missing-1",
        settings=cfg,
    )

    assert eligibility.available is False
    assert eligibility.reason_code == "missing_speaker_turns"
    with pytest.raises(
        snippet_repair.SnippetRepairPreconditionError,
        match="speaker_turns.json",
    ):
        snippet_repair.repair_recording_snippets(
            "rec-repair-missing-1",
            settings=cfg,
            origin="pytest",
        )


def test_repair_recording_snippets_keeps_existing_artifacts_when_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-repair-safe-1",
        source="upload",
        source_filename="repair-safe.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-repair-safe-1")
    manifest_path, snippet_path = _write_existing_manifest(cfg, "rec-repair-safe-1")
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_bytes = snippet_path.read_bytes()
    monkeypatch.setattr(
        snippet_repair,
        "export_speaker_snippets",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("snippet boom")),
    )

    with pytest.raises(snippet_repair.SnippetRepairExecutionError, match="snippet boom"):
        snippet_repair.repair_recording_snippets(
            "rec-repair-safe-1",
            settings=cfg,
            origin="pytest",
        )

    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    assert snippet_path.read_bytes() == original_bytes


def test_backfill_missing_snippets_repairs_missing_and_stale_artifacts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    for recording_id in (
        "rec-batch-missing-1",
        "rec-batch-stale-1",
        "rec-batch-ineligible-1",
        "rec-batch-present-1",
    ):
        create_recording(
            recording_id,
            source="upload",
            source_filename=f"{recording_id}.wav",
            status=RECORDING_STATUS_READY,
            settings=cfg,
        )
    _seed_repair_artifacts(cfg, "rec-batch-missing-1")
    _seed_repair_artifacts(cfg, "rec-batch-stale-1")
    _seed_repair_artifacts(
        cfg,
        "rec-batch-ineligible-1",
        include_speaker_turns=False,
    )
    _seed_repair_artifacts(cfg, "rec-batch-present-1")
    stale_snippets = cfg.recordings_root / "rec-batch-stale-1" / "derived" / "snippets" / "S1"
    stale_snippets.mkdir(parents=True, exist_ok=True)
    (stale_snippets / "1.wav").write_bytes(b"stale")
    _write_existing_manifest(cfg, "rec-batch-present-1")

    summary = snippet_repair.backfill_missing_snippets(
        settings=cfg,
        origin="pytest_batch",
    )

    assert summary.regenerated == 2
    assert summary.skipped == 2
    assert summary.failed == 0
    items = {item.recording_id: item for item in summary.items}
    assert items["rec-batch-missing-1"].outcome == "regenerated"
    assert items["rec-batch-stale-1"].outcome == "regenerated"
    assert items["rec-batch-ineligible-1"].outcome == "skipped"
    assert "speaker_turns.json" in items["rec-batch-ineligible-1"].detail
    assert items["rec-batch-present-1"].detail == "snippets already exist"
    assert (
        cfg.recordings_root / "rec-batch-stale-1" / "derived" / "snippets_manifest.json"
    ).exists()


def test_snippet_repair_helpers_cover_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")

    assert snippet_repair._load_json(tmp_path / "missing.json") is None  # noqa: SLF001
    assert snippet_repair._load_json(bad_json) is None  # noqa: SLF001
    assert snippet_repair._safe_float(None) is None  # noqa: SLF001
    assert snippet_repair._safe_float("bad") is None  # noqa: SLF001
    assert snippet_repair._safe_float("1.5") == 1.5  # noqa: SLF001
    assert snippet_repair._safe_path(outside, root=root) is None  # noqa: SLF001

    original_resolve = Path.resolve

    def _boom_resolve(self: Path, *args, **kwargs):
        if self == root / "boom":
            raise OSError("boom")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _boom_resolve)
    assert snippet_repair._safe_path(root / "boom", root=root) is None  # noqa: SLF001

    manifest_path = tmp_path / "snippets_manifest.json"
    with pytest.raises(
        snippet_repair.SnippetRepairExecutionError,
        match="without writing snippets_manifest.json",
    ):
        snippet_repair.finalize_snippets_manifest(manifest_path, manifest_status="ok")

    manifest_path.write_text(
        json.dumps(
            {
                "speakers": {
                    "S1": [{"status": "accepted"}, {"status": "rejected_overlap"}, "bad"],
                    "S2": "bad",
                },
                "warnings": ["bad"],
            }
        ),
        encoding="utf-8",
    )
    manifest = snippet_repair.finalize_snippets_manifest(  # noqa: SLF001
        manifest_path,
        manifest_status="",
        degraded_diarization=True,
        warnings=[{}, {"code": "warn_code"}, {"message": "warn_message"}],
    )
    assert manifest["manifest_status"] == "ok"
    assert manifest["degraded_diarization"] is True
    assert manifest["warnings"] == [{"code": "warn_code"}, {"message": "warn_message"}]
    assert snippet_repair._manifest_entries({"speakers": {"S1": [1, {"status": "accepted"}]}}) == [  # noqa: SLF001
        {"status": "accepted"}
    ]
    assert snippet_repair.snippet_manifest_counts({"speakers": []}) == {
        "accepted_snippets": 0,
        "speaker_count": 0,
        "warning_count": 0,
    }

    create_recording(
        "rec-artifacts-1",
        source="upload",
        source_filename="artifacts.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    snippets_root = cfg.recordings_root / "rec-artifacts-1" / "derived" / "snippets" / "S1"
    snippets_root.mkdir(parents=True, exist_ok=True)
    (snippets_root / "1.wav").write_bytes(b"clip")
    assert snippet_repair.snippet_artifact_state("rec-artifacts-1", settings=cfg) == "stale"
    manifest_state_path = cfg.recordings_root / "rec-artifacts-1" / "derived" / "snippets_manifest.json"
    manifest_state_path.write_text("{", encoding="utf-8")
    assert snippet_repair.snippet_artifact_state("rec-artifacts-1", settings=cfg) == "stale"
    manifest_state_path.write_text(
        json.dumps(
            {
                "speakers": {
                    "S1": [{"status": "accepted", "relative_path": ""}],
                }
            }
        ),
        encoding="utf-8",
    )
    assert snippet_repair.snippet_artifact_state("rec-artifacts-1", settings=cfg) == "stale"
    manifest_state_path.write_text(
        json.dumps(
            {
                "speakers": {
                    "S1": [{"status": "accepted", "relative_path": "S1/1.wav"}],
                }
            }
        ),
        encoding="utf-8",
    )
    assert snippet_repair.snippet_artifact_state("rec-artifacts-1", settings=cfg) == "present"


def test_raw_audio_and_manifest_helper_edge_cases(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-raw-helper-1",
        source="upload",
        source_filename="raw-helper.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    raw_dir = cfg.recordings_root / "rec-raw-helper-1" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "audio.fake").mkdir()
    assert snippet_repair._resolve_raw_audio_path("rec-raw-helper-1", settings=cfg) is None  # noqa: SLF001
    (raw_dir / "audio.fake").rmdir()
    assert snippet_repair._resolve_raw_audio_path("rec-raw-helper-1", settings=cfg) is None  # noqa: SLF001

    assert snippet_repair._manifest_entries({"speakers": []}) == []  # noqa: SLF001
    assert snippet_repair._manifest_entries({"speakers": {"S1": "bad"}}) == []  # noqa: SLF001
    assert snippet_repair._normalise_warnings([  # noqa: SLF001
        "bad",
        {"code": ""},
        {"message": "warn"},
    ]) == [{"message": "warn"}]

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"speakers": {}, "warnings": [{"message": "old"}]}), encoding="utf-8")
    manifest = snippet_repair.finalize_snippets_manifest(  # noqa: SLF001
        manifest_path,
        manifest_status="custom",
        degraded_diarization=None,
        warnings=None,
    )
    assert manifest["manifest_status"] == "custom"
    assert "warnings" not in manifest
    assert snippet_repair.snippet_manifest_counts(  # noqa: SLF001
        {"speakers": {"S1": [{"status": "rejected_overlap"}]}}
    )["warning_count"] == 1
    assert snippet_repair.snippet_manifest_counts(  # noqa: SLF001
        {"speakers": {"S1": [{"status": ""}]}}
    ) == {"accepted_snippets": 0, "speaker_count": 1, "warning_count": 0}


def test_assess_snippet_repair_covers_remaining_preconditions(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    assert snippet_repair.assess_snippet_repair("missing", settings=cfg).reason_code == (
        "recording_not_found"
    )

    create_recording(
        "rec-active-1",
        source="upload",
        source_filename="active.wav",
        status="Processing",
        settings=cfg,
    )
    assert snippet_repair.assess_snippet_repair("rec-active-1", settings=cfg).reason_code == (
        "recording_active"
    )

    create_recording(
        "rec-audio-missing-1",
        source="upload",
        source_filename="audio-missing.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    assert snippet_repair.assess_snippet_repair(
        "rec-audio-missing-1",
        settings=cfg,
    ).reason_code == "missing_audio"

    create_recording(
        "rec-precheck-missing-1",
        source="upload",
        source_filename="precheck-missing.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-precheck-missing-1")
    (cfg.recordings_root / "rec-precheck-missing-1" / "derived" / "precheck.json").write_text(
        json.dumps({"duration_sec": 0}),
        encoding="utf-8",
    )
    assert snippet_repair.assess_snippet_repair(
        "rec-precheck-missing-1",
        settings=cfg,
    ).reason_code == "missing_precheck_duration"

    create_recording(
        "rec-diarization-missing-1",
        source="upload",
        source_filename="diarization-missing.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-diarization-missing-1")
    (cfg.recordings_root / "rec-diarization-missing-1" / "derived" / "diarization_segments.json").unlink()
    assert snippet_repair.assess_snippet_repair(
        "rec-diarization-missing-1",
        settings=cfg,
    ).reason_code == "missing_diarization_segments"

    create_recording(
        "rec-metadata-missing-1",
        source="upload",
        source_filename="metadata-missing.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-metadata-missing-1")
    (cfg.recordings_root / "rec-metadata-missing-1" / "derived" / "diarization_metadata.json").unlink()
    assert snippet_repair.assess_snippet_repair(
        "rec-metadata-missing-1",
        settings=cfg,
    ).reason_code == "missing_diarization_metadata"

    create_recording(
        "rec-precheck-invalid-1",
        source="upload",
        source_filename="precheck-invalid.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-precheck-invalid-1")
    (cfg.recordings_root / "rec-precheck-invalid-1" / "derived" / "precheck.json").write_text(
        json.dumps([]),
        encoding="utf-8",
    )
    assert snippet_repair.assess_snippet_repair(
        "rec-precheck-invalid-1",
        settings=cfg,
    ).reason_code == "missing_precheck_duration"


def test_snippet_artifact_state_safe_path_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-safe-path-1",
        source="upload",
        source_filename="safe-path.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-safe-path-1")
    manifest_path, _snippet_path = _write_existing_manifest(cfg, "rec-safe-path-1")
    original_safe_path = snippet_repair._safe_path  # noqa: SLF001

    def _safe_root_none(candidate: Path, *, root: Path):
        if candidate == cfg.recordings_root / "rec-safe-path-1" / "derived" / "snippets":
            return None
        return original_safe_path(candidate, root=root)

    monkeypatch.setattr(snippet_repair, "_safe_path", _safe_root_none)
    assert snippet_repair.snippet_artifact_state("rec-safe-path-1", settings=cfg) == "stale"

    monkeypatch.setattr(snippet_repair, "_safe_path", original_safe_path)
    manifest_path.write_text(
        json.dumps(
            {
                "speakers": {
                    "S1": [
                        {"status": "rejected_overlap", "relative_path": "S1/1.wav"},
                        {"status": "accepted", "relative_path": "../bad.wav"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert snippet_repair.snippet_artifact_state("rec-safe-path-1", settings=cfg) == "stale"


def test_staged_output_helpers_cover_no_speech_partial_degraded_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    create_recording(
        "rec-staged-1",
        source="upload",
        source_filename="staged.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-staged-1")
    pipeline_settings = snippet_repair.PipelineSettings()  # noqa: SLF001
    audio_path = cfg.recordings_root / "rec-staged-1" / "derived" / "audio_sanitized.wav"

    empty_eligibility = snippet_repair.SnippetRepairEligibility(
        recording_id="rec-staged-1",
        available=True,
        artifact_state="missing",
        audio_path=audio_path,
        audio_source="sanitized_audio",
        duration_sec=3.0,
        diarization_segments=({"speaker": "S1", "start": 0.0, "end": 1.0},),
        speaker_turns=(),
    )
    temp_root, manifest = snippet_repair._build_staged_snippet_outputs(  # noqa: SLF001
        empty_eligibility,
        settings=cfg,
        pipeline_settings=pipeline_settings,
    )
    assert manifest["manifest_status"] == "no_usable_speech"
    shutil.rmtree(temp_root, ignore_errors=True)

    def _partial_export(request):
        request.snippets_dir.mkdir(parents=True, exist_ok=True)
        (request.snippets_dir / "S1").mkdir(parents=True, exist_ok=True)
        (request.snippets_dir / "S1" / "1.wav").write_bytes(b"clip")
        (request.snippets_dir.parent / "snippets_manifest.json").write_text(
            json.dumps(
                {
                    "speakers": {
                        "S1": [
                            {"status": "accepted", "relative_path": "S1/1.wav"},
                            {"status": "rejected_overlap"},
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(snippet_repair, "export_speaker_snippets", _partial_export)
    partial_eligibility = snippet_repair.assess_snippet_repair("rec-staged-1", settings=cfg)
    temp_root, manifest = snippet_repair._build_staged_snippet_outputs(  # noqa: SLF001
        partial_eligibility,
        settings=cfg,
        pipeline_settings=pipeline_settings,
    )
    assert manifest["manifest_status"] == "partial"
    shutil.rmtree(temp_root, ignore_errors=True)

    def _degraded_export(request):
        request.snippets_dir.mkdir(parents=True, exist_ok=True)
        (request.snippets_dir.parent / "snippets_manifest.json").write_text(
            json.dumps({"speakers": {"S1": [{"status": "rejected_degraded"}]}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(snippet_repair, "export_speaker_snippets", _degraded_export)
    degraded_eligibility = snippet_repair.SnippetRepairEligibility(
        **{
            **partial_eligibility.__dict__,
            "degraded_diarization": True,
        }
    )
    temp_root, manifest = snippet_repair._build_staged_snippet_outputs(  # noqa: SLF001
        degraded_eligibility,
        settings=cfg,
        pipeline_settings=pipeline_settings,
    )
    assert manifest["manifest_status"] == "degraded"
    shutil.rmtree(temp_root, ignore_errors=True)

    monkeypatch.setattr(
        snippet_repair,
        "export_speaker_snippets",
        lambda *_a, **_k: (_ for _ in ()).throw(
            snippet_repair.SnippetRepairExecutionError("boom", "bad export")
        ),
    )
    with pytest.raises(snippet_repair.SnippetRepairExecutionError, match="bad export"):
        snippet_repair._build_staged_snippet_outputs(  # noqa: SLF001
            partial_eligibility,
            settings=cfg,
            pipeline_settings=pipeline_settings,
        )


def test_replace_staged_outputs_rolls_back_after_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    create_recording(
        "rec-rollback-1",
        source="upload",
        source_filename="rollback.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-rollback-1")
    manifest_path, snippet_path = _write_existing_manifest(cfg, "rec-rollback-1")
    staged_root = cfg.recordings_root / "rec-rollback-1" / "derived" / ".stage"
    staged_root.mkdir(parents=True, exist_ok=True)
    (staged_root / "snippets" / "S1").mkdir(parents=True, exist_ok=True)
    (staged_root / "snippets" / "S1" / "1.wav").write_bytes(b"new-clip")
    staged_manifest = staged_root / "snippets_manifest.json"
    staged_manifest.write_text(json.dumps({"speakers": {}}), encoding="utf-8")

    original_rename = Path.rename

    def _rename_with_failure(self: Path, target: Path):
        if self == staged_manifest:
            raise OSError("rename boom")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", _rename_with_failure)

    with pytest.raises(snippet_repair.SnippetRepairExecutionError, match="replace snippet artifacts"):
        snippet_repair._replace_staged_snippet_outputs(  # noqa: SLF001
            "rec-rollback-1",
            staged_root,
            settings=cfg,
        )

    assert manifest_path.exists()
    assert snippet_path.exists()
    assert snippet_path.read_bytes() == b"old-clip"


def test_replace_staged_outputs_failure_without_existing_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    create_recording(
        "rec-rollback-empty-1",
        source="upload",
        source_filename="rollback-empty.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    staged_root = cfg.recordings_root / "rec-rollback-empty-1" / "derived" / ".stage"
    staged_root.mkdir(parents=True, exist_ok=True)
    staged_snippets = staged_root / "snippets"
    staged_snippets.mkdir(parents=True, exist_ok=True)
    (staged_root / "snippets_manifest.json").write_text(json.dumps({"speakers": {}}), encoding="utf-8")

    original_rename = Path.rename

    def _rename_fail_on_snippets(self: Path, target: Path):
        if self == staged_snippets:
            raise OSError("snippets rename boom")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", _rename_fail_on_snippets)

    with pytest.raises(snippet_repair.SnippetRepairExecutionError, match="replace snippet artifacts"):
        snippet_repair._replace_staged_snippet_outputs(  # noqa: SLF001
            "rec-rollback-empty-1",
            staged_root,
            settings=cfg,
        )

    assert not (cfg.recordings_root / "rec-rollback-empty-1" / "derived" / "snippets").exists()


def test_repair_recording_snippets_skips_noise_detection_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-repair-no-noise",
        source="upload",
        source_filename="repair-no-noise.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-repair-no-noise")
    monkeypatch.setenv("LAN_NOISE_DETECTION_ENABLED", "false")
    monkeypatch.setattr(
        snippet_repair,
        "apply_noise_flags_to_manifest",
        lambda *_a, **_k: pytest.fail("noise detection should be disabled"),
    )

    snippet_repair.repair_recording_snippets(
        "rec-repair-no-noise",
        settings=cfg,
        origin="pytest",
    )
    manifest = json.loads(
        (cfg.recordings_root / "rec-repair-no-noise" / "derived" / "snippets_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "noise_speakers" not in manifest


def test_repair_recording_snippets_is_idempotent(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-repair-repeat-1",
        source="upload",
        source_filename="repair-repeat.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-repair-repeat-1")

    first = snippet_repair.repair_recording_snippets(
        "rec-repair-repeat-1",
        settings=cfg,
        origin="pytest_first",
    )
    first_manifest = json.loads(
        (cfg.recordings_root / "rec-repair-repeat-1" / "derived" / "snippets_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    second = snippet_repair.repair_recording_snippets(
        "rec-repair-repeat-1",
        settings=cfg,
        origin="pytest_second",
    )
    second_manifest = json.loads(
        (cfg.recordings_root / "rec-repair-repeat-1" / "derived" / "snippets_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert first.accepted_snippets == second.accepted_snippets
    assert first_manifest == second_manifest


def test_iter_terminal_recordings_and_backfill_failure_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-terminal-1",
        source="upload",
        source_filename="terminal.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-terminal-1")
    create_recording(
        "rec-stale-1",
        source="upload",
        source_filename="stale.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    stale_snippets = cfg.recordings_root / "rec-stale-1" / "derived" / "snippets" / "S1"
    stale_snippets.mkdir(parents=True, exist_ok=True)
    (stale_snippets / "1.wav").write_bytes(b"stale")

    rows = snippet_repair._iter_terminal_recordings(settings=cfg)  # noqa: SLF001
    assert [row["id"] for row in rows] == ["rec-terminal-1", "rec-stale-1"]

    monkeypatch.setattr(
        snippet_repair,
        "_iter_terminal_recordings",
        lambda **_kwargs: [
            {"id": ""},
            {"id": "rec-stale-1", "status": RECORDING_STATUS_READY},
            {"id": "rec-terminal-1", "status": RECORDING_STATUS_READY},
        ],
    )

    monkeypatch.setattr(
        snippet_repair,
        "repair_recording_snippets",
        lambda *_a, **_k: (_ for _ in ()).throw(
            snippet_repair.SnippetRepairExecutionError("boom", "failed repair")
        ),
    )
    summary = snippet_repair.backfill_missing_snippets(settings=cfg, origin="pytest")
    assert summary.skipped == 0
    assert summary.failed == 2
    assert [(item.recording_id, item.detail) for item in summary.items] == [
        ("rec-stale-1", "boom: failed repair"),
        ("rec-terminal-1", "boom: failed repair"),
    ]


def test_iter_terminal_recordings_breaks_on_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_list_recordings(*, settings, limit, offset):
        calls["count"] += 1
        if calls["count"] == 1:
            return ([{"id": "rec-one", "status": RECORDING_STATUS_READY}], 2)
        return ([], 2)

    monkeypatch.setattr(snippet_repair, "list_recordings", _fake_list_recordings)
    rows = snippet_repair._iter_terminal_recordings(settings=_cfg(Path("/tmp")))  # noqa: SLF001
    assert rows == [{"id": "rec-one", "status": RECORDING_STATUS_READY}]


def test_repair_snippets_cli_single_and_scan_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    create_recording(
        "rec-cli-1",
        source="upload",
        source_filename="cli.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    create_recording(
        "rec-cli-scan-1",
        source="upload",
        source_filename="cli-scan.wav",
        status=RECORDING_STATUS_READY,
        settings=cfg,
    )
    _seed_repair_artifacts(cfg, "rec-cli-1")
    _seed_repair_artifacts(cfg, "rec-cli-scan-1")
    monkeypatch.setattr(
        repair_snippets_tool,
        "repair_recording_snippets",
        lambda recording_id, origin: snippet_repair.repair_recording_snippets(
            recording_id,
            settings=cfg,
            origin=origin,
        ),
    )
    monkeypatch.setattr(
        repair_snippets_tool,
        "backfill_missing_snippets",
        lambda origin: snippet_repair.backfill_missing_snippets(
            settings=cfg,
            origin=origin,
        ),
    )

    assert repair_snippets_tool.main(["--recording-id", "rec-cli-1"]) == 0
    single_output = capsys.readouterr().out
    assert "REGENERATED rec-cli-1" in single_output

    assert repair_snippets_tool.main(["--scan-missing"]) == 0
    batch_output = capsys.readouterr().out
    assert "REGENERATED rec-cli-scan-1" in batch_output
    assert "SUMMARY regenerated=1 skipped=1 failed=0" in batch_output


def test_repair_snippets_cli_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        repair_snippets_tool,
        "repair_recording_snippets",
        lambda *_a, **_k: (_ for _ in ()).throw(
            snippet_repair.SnippetRepairPreconditionError("blocked", "nope")
        ),
    )
    assert repair_snippets_tool.main(["--recording-id", "rec-fail"]) == 1
    assert "FAILED rec-fail blocked: nope" in capsys.readouterr().out

    monkeypatch.setattr(
        repair_snippets_tool,
        "backfill_missing_snippets",
        lambda *_a, **_k: snippet_repair.SnippetRepairBatchSummary(
            regenerated=0,
            skipped=0,
            failed=1,
            items=(
                snippet_repair.SnippetRepairBatchItem(
                    recording_id="rec-fail-batch",
                    outcome="failed",
                    detail="boom",
                ),
            ),
        ),
    )
    assert repair_snippets_tool.main(["--scan-missing"]) == 1
    assert "FAILED rec-fail-batch boom" in capsys.readouterr().out


def test_repair_snippets_cli_rejects_blank_recording_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = {"batch": False}

    def _unexpected_backfill(*_args, **_kwargs):
        called["batch"] = True
        raise AssertionError("batch backfill should not run for blank recording ids")

    monkeypatch.setattr(repair_snippets_tool, "backfill_missing_snippets", _unexpected_backfill)

    assert repair_snippets_tool.main(["--recording-id", ""]) == 1
    assert called["batch"] is False
    assert "FAILED recording-id invalid: blank recording id" in capsys.readouterr().out


def test_backfill_missing_snippets_uses_default_settings_and_regenerated_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    init_db(cfg)
    monkeypatch.setattr(snippet_repair, "AppSettings", lambda: cfg)
    monkeypatch.setattr(
        snippet_repair,
        "_iter_terminal_recordings",
        lambda **_kwargs: [{"id": "rec-default-1", "status": RECORDING_STATUS_READY}],
    )
    monkeypatch.setattr(
        snippet_repair,
        "snippet_artifact_state",
        lambda *_a, **_k: "missing",
    )
    monkeypatch.setattr(
        snippet_repair,
        "repair_recording_snippets",
        lambda *_a, **_k: snippet_repair.SnippetRepairResult(
            recording_id="rec-default-1",
            manifest_status="ok",
            accepted_snippets=2,
            speaker_count=1,
            warning_count=0,
            degraded_diarization=False,
            audio_source="sanitized_audio",
            duration_sec=3.0,
            artifact_state_before="missing",
        ),
    )

    summary = snippet_repair.backfill_missing_snippets(origin="default_settings")
    assert summary.regenerated == 1
    assert summary.items[0].detail == "2 clean snippets across 1 speakers"


def test_repair_snippets_tool_module_main_branch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "lan_app.snippet_repair.repair_recording_snippets",
        lambda recording_id, origin="cli_single", **_kwargs: snippet_repair.SnippetRepairResult(
            recording_id=recording_id,
            manifest_status="ok",
            accepted_snippets=1,
            speaker_count=1,
            warning_count=0,
            degraded_diarization=False,
            audio_source="sanitized_audio",
            duration_sec=1.0,
            artifact_state_before="missing",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["repair_snippets.py", "--recording-id", "rec-main-1"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("lan_app.tools.repair_snippets", run_name="__main__")
    assert exc.value.code == 0
    assert "REGENERATED rec-main-1" in capsys.readouterr().out
