Run PLANNED PR

PR_ID: PR-SPEAKER-MERGE-TORCH-LOAD-FIX-01
Branch: pr-speaker-merge-torch-load-fix-01
Title: Fix speaker merge embedding model: torch.load weights_only failure and CPU-only device

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
Two issues prevent speaker merge from working:
1. The embedding model (pyannote/wespeaker-voxceleb-resnet34-LM) fails to load with "Weights only load failed" because PyTorch 2.6+ defaults torch.load() to weights_only=True. The diarization pipeline already solved this with safe globals wrappers, but _build_pyannote_inference does not use them.
2. When loading succeeds (after fix 1), the Inference is created without a device parameter, so it runs on CPU even when the diarization pipeline runs on GPU. This makes embedding extraction slower and wastes the available GPU.

Phase 1 - Inspect
Read these files:
- lan_transcriber/pipeline_steps/orchestrator.py: _build_pyannote_inference (line ~949), _resolve_pyannote_embedding_model (line ~971), how omegaconf_safe_globals_for_torch_load is used (line ~1336)
- lan_transcriber/torch_safe_globals.py: omegaconf_safe_globals_for_torch_load, diarization_safe_globals_for_torch_load
- lan_app/diarization_loader.py: _from_pretrained_with_safe_globals for the retry pattern, _move_pipeline_to_best_device for device handling

Phase 2 - Implement

FIX 1: Wrap embedding model loading with safe globals
In _build_pyannote_inference, wrap the Inference() call:

Change from:
    try:
        return Inference(model_or_name, window="whole")
    except Exception as exc:

To:
    try:
        with diarization_safe_globals_for_torch_load():
            return Inference(model_or_name, window="whole")
    except Exception as exc:

Import diarization_safe_globals_for_torch_load from lan_transcriber.torch_safe_globals at the top of the file if not already imported. Use diarization_safe_globals_for_torch_load (not omegaconf) because the wespeaker model uses the same pyannote ecosystem classes.

If the first attempt still fails with an unsupported global error, apply the same retry pattern from diarization_loader.py _from_pretrained_with_safe_globals: catch the error, extract the FQN via unsupported_global_diarization_fqn_from_error, import it, retry with extra_fqns.

FIX 2: Pass device to Inference
In _build_pyannote_inference, add a device parameter:

def _build_pyannote_inference(model_or_name: Any, *, device: str | None = None) -> Any | None:

Pass it to Inference:
    kwargs = {"window": "whole"}
    if device and device != "cpu":
        import torch
        kwargs["device"] = torch.device(device)
    with diarization_safe_globals_for_torch_load():
        return Inference(model_or_name, **kwargs)

In _resolve_pyannote_embedding_model, when calling _build_pyannote_inference (line ~1014), pass the device from the diariser:
    effective_device = getattr(diariser, "_lan_effective_device", None)
    inference = _build_pyannote_inference(model_name, device=effective_device)

Log the device:
    _logger.info(
        "speaker_merge: embedding model ready (source=%s, device=%s)",
        resolution_source,
        getattr(inference, "device", "unknown"),
    )

Phase 3 - Test and verify
- Run full CI.
- After deploy, full reprocess a recording.
- Verify speaker_merge_diagnostics shows pairwise_scores with actual similarity values (not skipped_reason=embedding_model_unavailable).
- Verify worker logs show "speaker_merge: embedding model ready" with device=cuda (not cpu).

Success criteria:
- Embedding model loads successfully with torch safe globals.
- Embedding model runs on the same device as diarization (GPU when available).
- speaker_merge_diagnostics contains pairwise similarity scores.
- No existing tests break.
