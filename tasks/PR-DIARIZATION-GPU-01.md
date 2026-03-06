PR-DIARIZATION-GPU-01
=====================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Move the pyannote diarization pipeline onto GPU when CUDA is available, so the diarization step stops running on CPU and no longer takes excessively long on longer recordings.

Context
We already verified inside the worker container:
- torch.cuda.is_available() == True
- torch.cuda.device_count() >= 1
- torch.version.cuda is present
We also inspected the code and found that lan_app/diarization_loader.py creates the pyannote pipeline with Pipeline.from_pretrained(...), but does not move it to CUDA.
As a result, diarization likely runs on CPU even though the container has GPU access.

Scope
This PR is only about device placement for the pyannote diarization pipeline and the related diagnostics/tests.
Do not redesign the pipeline. Do not change WhisperX device logic here. Do not remove CPU fallback behavior.

Implementation requirements

1) Update lan_app/diarization_loader.py
- After a successful Pipeline.from_pretrained(...) call, detect whether CUDA is available:
  - import torch
  - if torch.cuda.is_available(): pipeline.to(torch.device("cuda"))
  - else leave it on CPU
- Make this work in the real loader path that is used by production, not in a test-only branch.

2) Add clear device logging
- Add one concise INFO log line when the diarization pipeline is created:
  - "Pyannote diarization device: cuda"
  - or "Pyannote diarization device: cpu"
- If moving to CUDA fails for any reason, log a warning and continue on CPU rather than crashing the worker.
- Do not spam logs.

3) Keep behavior safe and deterministic
- If torch import fails or CUDA is unavailable, keep the current CPU behavior.
- Do not assume GPU always exists in CI.
- Do not make diarization mandatory-GPU; this must remain a graceful optimization.

4) Add regression tests (100% statement + branch coverage for changed code)
- Add tests for the loader in a dedicated test module or the existing diarization loader tests.
- Use monkeypatch/fakes for Pipeline.from_pretrained and torch:
  a) CUDA available:
     - fake torch.cuda.is_available returns True
     - ensure pipeline.to(torch.device("cuda")) is called exactly once
     - ensure log/message indicates cuda
  b) CUDA unavailable:
     - fake torch.cuda.is_available returns False
     - ensure pipeline.to(...) is not called
     - ensure log/message indicates cpu
  c) CUDA move failure:
     - fake pipeline.to(...) raises an exception
     - ensure loader logs a warning and still returns the pipeline object instead of crashing
- Keep tests offline. No real pyannote downloads. No real GPU required.
- Maintain 100% statement and branch coverage for any modified/new code paths.

5) Optional small verification helper (acceptable if minimal)
- If it helps diagnostics, add a tiny helper function that returns the selected diarization device string, and cover it fully in tests.
- Keep it minimal and only if it improves code clarity.

6) Verification steps (include in PR description)
- Rebuild and restart worker:
  - docker compose build --no-cache worker
  - docker compose up -d --force-recreate --no-deps worker
- Verify in logs that diarization reports device=cuda.
- Process a short recording and confirm diarization step is materially faster than before (or at minimum no longer clearly CPU-bound).
- Confirm fallback/CPU path still works when GPU is not available.

Deliverables
- lan_app/diarization_loader.py updated to move pyannote pipeline to CUDA when available
- concise device logging
- regression tests with 100% statement and branch coverage for the modified code

Success criteria
- On the GPU host, pyannote diarization pipeline is explicitly moved to CUDA.
- Worker logs clearly state the diarization device.
- CI remains green without requiring GPU.
- CPU remains a safe fallback if CUDA is unavailable or the move fails.
```
