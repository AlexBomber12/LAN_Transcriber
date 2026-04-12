Run PLANNED PR

PR_ID: PR-GPU-STATUS-FIX-01
Branch: pr-gpu-status-fix-01
Title: Fix system bar GPU status showing CPU only when worker uses GPU

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

Context
The system bar shows "GPU runtime: CPU only" even when the worker is actively using GPU for diarization and ASR. The root cause is architectural: system_status.py checks cuda_facts in the API container process, but GPU is attached to the worker container. The API process has no CUDA visibility, so it always falls through to the "CPU only" branch.

Phase 1 - Inspect and map
Read these files:
- lan_app/system_status.py: _gpu_runtime_item (line ~430), how cuda_facts is used, where "CPU only" is returned
- lan_app/templates/partials/control_center/system_bar_items.html: how GPU status is rendered
- lan_app/diagnostics.py: existing diagnostics collection patterns
- docker-compose.yml: which containers have GPU access (deploy.resources.reservations.devices)
- lan_app/worker_tasks.py: check if worker reports device info anywhere (diarization_metadata, asr_execution)

Phase 2 - Implement
The worker already writes effective_device to diarization_metadata.json and asr_execution.json for each recording. Use this as the source of truth.

CHANGE 1: Collect GPU status from recent worker artifacts
In system_status.py, add a function that checks the most recent recording's diarization_metadata.json or asr_execution.json for effective_device. If the last processed recording used "cuda", report GPU as available.

Alternatively, store worker GPU capability in a lightweight shared state:
- Worker writes a "worker_status.json" to /data/ on startup and periodically with: {"gpu_available": true, "device": "cuda", "cuda_version": "12.1", "device_name": "...", "last_heartbeat": "..."}
- API reads this file in _gpu_runtime_item instead of checking local cuda_facts.

Option B (worker_status.json) is cleaner because it survives across recordings and shows status even when no recordings are processed.

CHANGE 2: Update _gpu_runtime_item
Modify the function to:
1. First check worker_status.json for GPU info.
2. Fall back to local cuda_facts if worker_status.json is unavailable.
3. Show "GPU ready (worker)" or "GPU busy (worker)" based on worker heartbeat + queue state.

CHANGE 3: Worker heartbeat
In worker startup (worker.py or worker_tasks.py), write worker_status.json with GPU facts. Update it periodically (every 60s) or on each job completion.

Phase 3 - Test and verify
- Run full CI.
- Verify system bar shows "GPU ready" when worker has GPU access.
- Verify system bar shows "CPU only" when worker has no GPU.
- Verify status updates when worker is processing (GPU busy).

Success criteria:
- System bar accurately reflects worker GPU status, not API container GPU status.
- No existing tests break.
