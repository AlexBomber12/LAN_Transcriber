PR-CUDA-CUDNN9-ALIGN-01
=======================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Fix the worker crash during processing caused by cuDNN mismatch:
- Runtime error: "Could not load library libcudnn_ops_infer.so.8: No such file or directory"
This happens because the current environment ships cuDNN 9 (libcudnn*.so.9) but ctranslate2 4.4.0 expects cuDNN 8 (libcudnn_ops_infer.so.8).
Implement a minimal, safe alignment to cuDNN 9 by moving the CUDA stack to cu124 and upgrading ctranslate2 to a cuDNN 9 compatible version.

Constraints
- GPU host driver supports CUDA 12.8 (verified), so using cu124 wheels is compatible.
- Keep changes minimal (micro PR), focused on dependency alignment + deterministic detection of regressions.
- Do not redesign pipeline logic. Do not change API behavior. Keep existing 100% coverage enforcement intact.
- CPU support is not required.

Implementation plan

1) Upgrade ctranslate2 and move PyTorch wheels from cu121 to cu124
- In requirements-cu121.in (or rename to requirements-cu124.in if you choose, but keep it micro):
  - Change ctranslate2 pin:
    - from: ctranslate2==4.4.0
    - to: ctranslate2==4.5.0 (or 4.5.0+ if exact pin used elsewhere)
  - Change torch stack pins to cu124:
    - torch==2.5.1+cu121 -> torch==2.5.1+cu124
    - torchaudio==2.5.1+cu121 -> torchaudio==2.5.1+cu124
    - torchvision==0.20.1+cu121 -> torchvision==0.20.1+cu124
- Update the pip-compile command / documentation to use:
  - --extra-index-url https://download.pytorch.org/whl/cu124
- Re-run pip-compile to regenerate the lockfile (requirements-cu121.txt) deterministically.
  - Ensure the lockfile includes nvidia-cudnn-cu12==9.x and no cuDNN 8 artifacts.
- Run `pip check` in a clean venv to confirm no broken requirements.

2) Ensure Docker builds use the updated lockfile
- Confirm Dockerfile runtime targets install from requirements-cu121.txt (the lockfile you just regenerated).
- Add/strengthen a build-time dependency smoke step in the Dockerfile to catch this class of mismatch early:
  - `python -c "import ctranslate2; import faster_whisper; import torch; print('ctranslate2', ctranslate2.__version__, 'torch cuda', torch.version.cuda)"`
  - Also keep the existing whisperx import checks if present (matplotlib + whisperx.asr), but ensure this step runs for the worker target that is actually used.
- The Docker build must fail fast if ctranslate2 cannot be imported due to missing cuDNN libraries.

3) Strengthen docker smoke test to detect cuDNN mismatch (CI)
- Update tests/test_docker_smoke.py to include an in-container import check for:
  - `import ctranslate2`
  - `import torch`
- Do not require a GPU for the smoke test; import-only is sufficient because the mismatch fails on import/dlopen.

4) Optional but recommended: runtime diagnostic command (no functional changes)
- Add a small CLI helper or make a documented one-liner to print:
  - ctranslate2 version
  - torch.version.cuda
  - whether torch.cuda.is_available()
This helps confirm the runtime stack quickly when debugging.

5) Update documentation (small)
- In README/runbook, update any mention of cu121 to cu124 if present in build instructions.
- Add a short troubleshooting note:
  - cuDNN 8 vs 9 mismatch manifests as missing libcudnn_ops_infer.so.8
  - fixed by cu124 + ctranslate2>=4.5

Verification steps (must be included in PR description)
- Rebuild worker image without cache:
  - docker compose build --no-cache worker
  - docker compose up -d --force-recreate --no-deps worker
- In the worker container, confirm:
  - python -c "import ctranslate2; import torch; print(ctranslate2.__version__, torch.version.cuda)"
- Re-run processing of a short recording; the worker must not crash with libcudnn_ops_infer.so.8 error.

Success criteria
- Importing ctranslate2 inside the worker container succeeds (no missing libcudnn_ops_infer.so.8).
- Worker no longer crashes during processing with ret_val=134 and libcudnn_ops_infer.so.8 error.
- Docker build fails early if a future dependency change reintroduces cuDNN mismatch (via strengthened build-time check).
- CI docker smoke test fails if ctranslate2 cannot be imported due to runtime CUDA/cuDNN issues.
```
