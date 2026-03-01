    PR-DIARIZATION-ROBUST-02
    ========================

    Prompt (copy as-is into Codex)
    ------------------------------

    ```text
    You are Codex Agent working on the LAN-Transcriber repository.

Goal
Implement Variant 1 only: persistent Hugging Face cache + explicit warmup command + robust diarization model loading.
Fix diarization startup/runtime fragility by making pyannote diarization loading version-safe, token-aware, and non-fatal.
Also fix the current runtime failure where whisperx/pyannote import chain crashes due to missing matplotlib in the worker image.

Context
Observed failures:
- Pipeline.from_pretrained("pyannote/speaker-diarization@3.1") fails with RevisionNotFoundError / HTTP 404. The correct pipeline repo for pyannote v3 is "pyannote/speaker-diarization-3.1" (repo name), not a revision tag on "pyannote/speaker-diarization".
- Jobs may crash later with TypeError: 'NoneType' object is not callable when an uninitialised diarization pipeline ends up being called.
- whisperx.load_model can fail at import time with ModuleNotFoundError: No module named 'matplotlib' because whisperx imports pyannote VoiceActivityDetection, which imports pyannote tasks that require matplotlib.

Non-goals
- Do not implement Variant 2 (baking models into the Docker image). No BuildKit secrets. No "download at build time".
- Do not redesign the whole pipeline. Do not change diarization semantics beyond loading and resilience.
- Do not weaken the existing 100% statement and branch coverage gate.

Implementation requirements

1) Single configurable diarization model id (correct default)
- Replace any hardcoded diarization model id with a single config setting:
  - env var: LAN_DIARIZATION_MODEL_ID
  - default: pyannote/speaker-diarization-3.1
- Keep accepted formats:
  - "org/repo" (no revision)
  - "org/repo@rev" (optional revision)
- Document clearly:
  - For pyannote v3, the recommended id is the repo name "pyannote/speaker-diarization-3.1" (no "@").
  - If users use "pyannote/speaker-diarization@3.1" by habit, we will attempt to translate it.

2) Robust loader with deterministic fallbacks
Implement a robust loader for pyannote.audio.Pipeline that tries candidates in order:
- Parse model id into repo_id and optional revision.
- If revision exists:
  - Try Pipeline.from_pretrained(repo_id, revision=rev)
  - Try Pipeline.from_pretrained(f"{repo_id}@{rev}")  (compat for older pyannote)
  - If rev looks like a version string (e.g., "3.1", "3.0", "3.1.0") and these fail with RevisionNotFoundError/HTTP 404,
    also try repo-name fallback: Pipeline.from_pretrained(f"{repo_id}-{rev}") with NO revision.
- Then try:
  - Pipeline.from_pretrained(repo_id) (no revision)
- If any load error occurs (RevisionNotFoundError, gated access 401/403, network failure, missing deps),
  the system must NOT crash the job. It must fall back to the existing _FallbackDiariser for that recording and continue.

3) Never allow None to reach callable path
- Ensure _PyannoteDiariser is only constructed with a non-None, callable pipeline.
- Add explicit guards so None can never produce a 'NoneType is not callable' later.
- In _PyannoteDiariser.__call__:
  - First attempt dict form: model({"audio": str(audio_path)})
  - If that raises TypeError due to signature mismatch, then try string path: model(str(audio_path))
  - Do not catch broad Exception for retry. Catch only the signature mismatch case.

4) Fix missing matplotlib in worker/runtime
- Add matplotlib as a runtime dependency for the worker image (and any place whisperx runs) so whisperx/pyannote imports do not crash.
- Ensure headless-safe backend by setting MPLBACKEND=Agg in the worker environment (compose env or app default).
- Add a regression test that imports whisperx.load_model (or the minimal import path that used to crash) without failing.

5) Persistent HF cache in Docker Compose (Variant 1)
- Ensure worker and API containers share a persistent Hugging Face cache across restarts and image rebuilds.
- Use HF_HOME pointing under /data.
- Add a named docker volume and mount it:
  - volume name: hf-cache
  - mount: hf-cache:/data/hf
  - env: HF_HOME=/data/hf
- Update docker-compose.yml and docker-compose.dev.yml (if present). Keep current layouts consistent.

6) Warmup command (mandatory)
- Add a CLI warmup tool that downloads and validates the diarization pipeline into the cache and then exits.
- Command target:
  - python -m lan_app.tools.warm_models --models diarization
- It must:
  - Load diarization using the same loader as production
  - Exit code 0 on success
  - Exit code 2 if token is missing
  - Exit code 3 if gated access is not granted (401/403)
  - Exit code 4 if revision not found (and repo-name fallback also missing)
  - Exit code 5 for other errors
- It must print concise, actionable errors and the exact repo id that requires acceptance.

7) Documentation: "Before first start"
Update README/runbook with:
- Create Hugging Face access token with read scope.
- Accept gated terms for:
  - pyannote/speaker-diarization-3.1
  - pyannote/segmentation-3.0
  - any additional repos the warmup reports
- Set HF_TOKEN in .env and ensure it is passed into the worker container.
- Run warmup once to populate the hf-cache volume.
- Verify:
  - Pipeline.from_pretrained("pyannote/speaker-diarization-3.1") inside the worker container
  - import whisperx.load_model inside the worker container (matplotlib regression)

8) Tests and 100% coverage for touched code
- Add unit tests that cover:
  - loader falls back to _FallbackDiariser when revision is missing (simulate RevisionNotFoundError/404)
  - loader tries repo-name fallback "repo-rev" when "repo@rev" revision is missing
  - loader uses env override LAN_DIARIZATION_MODEL_ID
  - _PyannoteDiariser never accepts None and never throws NoneType callable errors
  - warm_models exit codes/messages for missing token, gated denial, revision missing, success (mock Pipeline.from_pretrained)
- Keep statement and branch coverage at 100% for changed/new modules.

Deliverables
- Robust diarization loader and default model id set to pyannote/speaker-diarization-3.1
- matplotlib dependency added + MPLBACKEND configured
- Persistent HF cache volume wired into compose
- Warmup tool with clear exit codes
- Updated docs with pre-start checklist and verify steps
- Tests with 100% coverage for new/changed code

Success criteria
- Pipeline.from_pretrained("pyannote/speaker-diarization-3.1") succeeds in the worker container (given token + accepted terms).
- whisperx import path no longer crashes due to missing matplotlib.
- A diarization version typo/missing revision no longer breaks processing; recording completes via fallback diarization.
- Restarting containers does not re-download diarization assets if hf-cache volume exists.
- CI passes, including 100% coverage enforcement and docker smoke.
    ```
