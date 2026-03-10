PR-GPU-SCHEDULER-01
===================

Branch: pr-gpu-scheduler-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Implement a long-term GPU scheduling foundation that keeps the pipeline fast on a single NVIDIA GPU without random CUDA OOM failures. The current code loads pyannote diarization onto CUDA in lan_app/diarization_loader.py, starts diarization construction early from lan_app/worker_tasks.py, then runs ASR and diarization together with asyncio.gather(...) in lan_transcriber/pipeline_steps/orchestrator.py. This creates an unstable VRAM peak and jobs fail with RuntimeError: CUDA out of memory while loading faster-whisper / WhisperX. The fix must maximize useful GPU usage, not simply move everything to CPU. On a single GPU the correct design is controlled sequential ownership of the GPU, with ASR kept warm when beneficial and diarization loaded lazily only for its stage. Also fix the UI duration format to HH:MM:SS and keep existing local recording timestamps unchanged.

Constraints
- Scope is GPU scheduling, device selection, lazy diarization loading, retry/review handling for GPU OOM, targeted UI duration formatting, and the minimal supporting docs/tests.
- Do not redesign the overall recording pipeline, job model, or speaker/glossary/calendar features.
- Preserve existing upload, export, transcript, routing, snippets, and speaker-bank behavior.
- Preserve the current recording-detail local timestamp display. Do not change Captured at timezone behavior for recording detail pages.
- Maintain deterministic offline tests and keep 100% statement and branch coverage for every changed/new project module.
- Keep the implementation production-safe for LAN Docker deployment.

Implementation requirements

1) Introduce an explicit GPU scheduling and device policy
- Add explicit configuration for ASR and diarization devices instead of silently forcing pyannote to CUDA whenever torch.cuda.is_available() is true.
- Add settings/env support for at least:
  - LAN_ASR_DEVICE with values like auto, cpu, cuda, cuda:0, cuda:1
  - LAN_DIARIZATION_DEVICE with values like auto, cpu, cuda, cuda:0, cuda:1
  - LAN_GPU_SCHEDULER_MODE with values auto, sequential, parallel
- Auto policy must be safe:
  - on a single visible GPU, default to sequential scheduling when both heavy stages target the same GPU
  - allow parallel execution only when ASR and diarization are on different devices in a way that is actually safe, for example cuda:0 vs cuda:1, or one stage on CPU and the other on GPU
- Add small helpers that resolve effective devices, normalize aliases, and expose the chosen scheduling mode for logs and tests.
- Log the effective ASR device, diarization device, scheduler mode, and relevant CUDA visibility facts once per job.

2) Stop preloading diarization onto the GPU before ASR starts
- In lan_app.worker_tasks the precheck phase must no longer eagerly construct a heavy pyannote pipeline that can occupy VRAM before ASR model loading.
- Refactor the precheck/build path so that precheck returns only the lightweight diarization plan, profile, or callable factory metadata needed by the main pipeline.
- Actual pyannote pipeline loading must happen lazily at the diarization stage, not during precheck, unless the selected policy/device makes it explicitly safe.
- Preserve current profile-selection behavior and degraded fallback metadata. This PR is not allowed to regress the existing dialog vs meeting profile logic.

3) Make single-GPU execution sequential by design
- In lan_transcriber/pipeline_steps/orchestrator.py remove the unconditional asyncio.gather(...) between ASR and diarization when both stages would contend for the same GPU.
- Implement a scheduler decision point:
  - same GPU or single-GPU auto mode -> run sequentially
  - safe different-device mode -> parallel execution may remain allowed
- Sequential order on one GPU must be:
  - ASR stage
  - release or trim unneeded ASR temporary memory
  - diarization stage
  - downstream alignment/turn-building as today
- Before the second heavy stage starts, perform explicit memory cleanup where appropriate, such as gc.collect() and torch.cuda.empty_cache(), guarded so tests remain offline and CPU-safe.
- Do not keep pyannote resident on the same GPU across the whole job when that increases failure risk.

4) Keep ASR warm across jobs when safe and beneficial
- Add a process-local ASR model cache keyed by the effective ASR loading configuration, for example model name, device, compute type, vad method, and any other load-time parameter that materially changes the loaded object.
- Reuse the loaded WhisperX / faster-whisper model across worker jobs in the same process so repeat jobs do not pay full load cost every time.
- The cache must be explicit, testable, and releasable. Provide a small helper for tests to clear the cache.
- Do not couple the warm cache to per-recording glossary content. Glossary remains a transcribe-time concern, not a model-load key.
- Keep diarization lazy by default. This PR should prefer warm ASR plus ephemeral diarization on the same GPU.

5) Respect explicit device choices in the diarization loader
- Update lan_app/diarization_loader.py so _move_pipeline_to_best_device and related load functions respect the configured diarization device and scheduler policy.
- Do not silently move diarization to CUDA when LAN_DIARIZATION_DEVICE resolves to cpu.
- Support explicit device strings like cuda:0 when torch supports them.
- Log the effective diarization device that was actually used.
- If diarization was requested on CUDA but the move fails, downgrade safely only when policy is auto. If the user explicitly forced a CUDA device, fail with a clear error rather than silently changing execution semantics.

6) Add VRAM awareness and bounded fallback behavior
- Add a small helper around torch.cuda.mem_get_info() or equivalent, guarded so it returns None cleanly when CUDA is unavailable or the API is missing.
- Log free/total VRAM before heavy model load/move operations when CUDA is in play.
- In auto mode, use this information to avoid obviously unsafe choices. For example, do not attempt parallel same-device execution when available memory is below a conservative threshold.
- Add a bounded fallback only where it makes sense and keeps behavior understandable. Acceptable fallback examples:
  - after a GPU OOM during ASR model load, clear cache and retry once with a smaller compute type such as int8_float16, but only in auto mode
  - after a GPU OOM during diarization CUDA move, retry on CPU only in auto mode
- Do not create hidden multi-step magic. Every fallback must be logged clearly and be deterministic.

7) Treat GPU OOM as a first-class non-retryable failure
- In lan_app/worker_tasks.py refine _is_retryable_exception(...) so generic RuntimeError is not enough.
- GPU memory failures such as CUDA out of memory, CUDA failed with error out of memory, CUBLAS_STATUS_ALLOC_FAILED, or equivalent CUDA allocation errors must be classified as non-retryable.
- Add a dedicated review-reason mapping in _review_reason_from_exception(...) for gpu_oom with clear user-facing text explaining that the worker ran out of GPU memory while loading or running a heavy model.
- Ensure the recording ends in a meaningful NeedsReview state with the specific reason instead of burning retries and surfacing only Processing hit the retry limit after repeated errors.
- Keep existing retry behavior for actual transient failures like timeouts and network issues.

8) Fix UI duration formatting to HH:MM:SS
- Update lan_app/ui_routes.py so _format_duration_seconds(...) renders durations as fixed-width HH:MM:SS.
- Examples:
  - 2 seconds -> 00:00:02
  - 4699.26 seconds -> 01:18:19
- Use sensible rounding or truncation consistently and document the choice in tests.
- Keep the existing local timestamp display unchanged.

9) Tests with full coverage
Add deterministic offline tests for at least these cases:
- scheduler mode resolves to sequential on a single GPU when ASR and diarization target the same device
- scheduler mode allows parallel execution only when devices are genuinely separate and safe
- precheck no longer eagerly loads a heavy diarization pipeline onto CUDA before ASR stage
- ASR warm cache reuses the loaded model across repeated jobs/configurations and can be cleared in tests
- diarization loader respects forced cpu, forced cuda device, and auto fallback semantics
- GPU OOM classification is non-retryable and maps to a dedicated gpu_oom review reason
- retry-limit masking regression is prevented by asserting the specific review reason/text path
- duration formatting returns HH:MM:SS for None, zero, integer, and fractional inputs according to the chosen rule
- any new scheduler helper branches are fully covered
Mock torch, whisperx, pyannote, and CUDA memory functions as needed so tests remain offline and deterministic.

10) Documentation and operational notes
- Update README and/or docs/runbook.md with a short operator section covering:
  - the new device env vars and scheduler mode
  - recommended single-GPU configuration
  - the fact that ASR is cached warm while diarization is lazy/ephemeral by default on one GPU
  - how gpu_oom is surfaced in the UI/logs
- Keep the instructions practical for Docker Compose deployment on the LAN server.

Verification steps (must be included in PR description)
- On a single-GPU environment, process a recording that previously failed with CUDA OOM and confirm ASR and diarization no longer overlap on the same GPU.
- Confirm the worker logs show effective ASR device, diarization device, scheduler mode, and any fallback that occurred.
- Confirm repeated jobs reuse the ASR model instead of reloading it each time.
- Confirm a forced GPU OOM path results in a specific gpu_oom review reason and does not waste retries.
- Confirm the recording detail page shows duration in HH:MM:SS while Captured at remains unchanged.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Explicit GPU scheduler/device policy
- Lazy diarization loading instead of eager precheck GPU residency
- Sequential single-GPU execution with safe optional parallel mode only when devices differ
- Warm reusable ASR cache
- Non-retryable gpu_oom classification and review reason
- HH:MM:SS duration formatting
- Updated docs and tests with 100% statement and branch coverage for changed/new modules

Success criteria
- Single-GPU processing no longer fails because pyannote and WhisperX compete for the same VRAM by default.
- GPU usage remains aggressive and efficient, but coordinated rather than concurrent on one device.
- Repeated jobs are faster because ASR stays warm when configuration is unchanged.
- CUDA OOM surfaces as a specific actionable review reason, not as a generic retry-limit failure.
- Duration is displayed as HH:MM:SS.
- CI remains green.
```
