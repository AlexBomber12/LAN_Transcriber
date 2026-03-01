PR-DIARIZATION-ROBUST-02
======================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Implement Variant 1 only: persistent Hugging Face cache + explicit warmup command + robust diarization model loading.
Fix diarization startup/runtime fragility by making pyannote diarization loading revision-safe, token-aware, and non-fatal.
Add a pre-start checklist (token, gated access, warmup) so the system does not download during first job processing.

Context
Current failures are caused by an invalid hardcoded revision (example: pyannote/speaker-diarization@3.2 returning RevisionNotFoundError) and by allowing an uninitialised pipeline to be called later, which crashes with TypeError: 'NoneType' object is not callable.
Workers may also start without HF_TOKEN in env, or without having accepted gated model terms, making from_pretrained fail.

Non-goals
Do not implement Variant 2 (baking models into the Docker image). Do not add BuildKit secrets. Do not increase image size.
Do not redesign the whole pipeline. Do not change diarization semantics beyond loading and resilience.
Do not weaken the existing 100% statement and branch coverage gate.

Implementation requirements

1) Single configurable diarization model id
- Replace any hardcoded diarization model id and revision with a single config setting:
  - env var: LAN_DIARIZATION_MODEL_ID
  - default: pyannote/speaker-diarization@3.1
- Keep the value as a repo id optionally with revision, like "org/repo@rev".

2) Robust loader with deterministic fallback
- Implement a robust loader for pyannote Pipeline that tries candidates in order:
  - If model id includes revision:
    - Pipeline.from_pretrained(repo_id, revision=rev)
    - Pipeline.from_pretrained(f"{repo_id}@{rev}")
  - Then:
    - Pipeline.from_pretrained(repo_id) (no revision)
- If any load error occurs (including RevisionNotFoundError, 401/403 gated access, network failure, missing deps), the system must NOT crash the job.
- Instead, fall back to the existing _FallbackDiariser for that recording and continue.

3) Never allow None to reach callable path
- Ensure _PyannoteDiariser is only constructed with a non-None, callable pipeline.
- Add explicit guards so None can never produce a 'NoneType is not callable' later.
- In _PyannoteDiariser.__call__:
  - First attempt dict form: model({"audio": str(audio_path)})
  - If that raises TypeError due to signature mismatch, then try string path: model(str(audio_path))
  - Do not catch broad Exception for the retry. Catch only the signature mismatch case, otherwise re-raise.

4) Persistent HF cache in Docker Compose
- Ensure worker and API containers share a persistent Hugging Face cache across restarts and image rebuilds.
- Use HF_HOME (and optionally TRANSFORMERS_CACHE) pointing under /data.
- Add a named docker volume and mount it:
  - example volume name: hf-cache
  - example mount: hf-cache:/data/hf
  - example env: HF_HOME=/data/hf
- Update docker-compose.yml and docker-compose.dev.yml (if present). Ensure existing volumes and paths remain consistent.

5) Warmup command (mandatory)
- Add a CLI warmup tool that downloads and validates diarization pipeline into the cache and then exits.
- Command target:
  - python -m lan_app.tools.warm_models --models diarization
- It must:
  - Load the diarization pipeline using the same loader as production code
  - Exit code 0 on success
  - Exit code 2 if token is missing
  - Exit code 3 if gated access is not granted (401/403)
  - Exit code 4 if revision not found
  - Exit code 5 for other errors
- It must print concise, actionable messages, including which Hugging Face repo requires gated acceptance if applicable.
- It must respect HF_TOKEN (primary) and HUGGINGFACE_HUB_TOKEN (fallback).

6) Documentation: "Before first start"
Update README or a runbook doc to include:
- Create Hugging Face access token with read scope.
- Accept gated terms for:
  - pyannote/speaker-diarization (pipeline repo)
  - any additional repos referenced by the pipeline (warmup will tell you which ones if missing)
- Set HF_TOKEN in .env and ensure it is passed into the worker container.
- Run warmup command once (include docker exec or docker compose run example).
- Start docker compose.
- Add a Verify section with a one-liner that loads the pipeline inside the running worker.

7) Tests and 100% coverage for touched code
- Add unit tests that cover:
  - loader falls back to _FallbackDiariser when revision is missing (simulate RevisionNotFoundError)
  - loader uses env override LAN_DIARIZATION_MODEL_ID
  - _PyannoteDiariser never accepts None and does not produce NoneType callable errors
  - warm_models exit codes and messages for missing token, gated denial, revision missing, and success (mock pipeline load)
- Keep statement and branch coverage at 100% for changed/new modules.

Deliverables
- Robust diarization loader and default model id updated to 3.1
- Persistent HF cache volume wired into compose
- Warmup tool with clear exit codes
- Updated docs with pre-start checklist and verify step
- Tests with 100% coverage for new/changed code

Success criteria
- A diarization revision typo or missing revision no longer breaks processing. The recording completes using fallback diarization and is not marked failed due to NoneType callable.
- With a valid HF token and accepted gated terms, diarization loads and runs successfully using default model id pyannote/speaker-diarization@3.1.
- Restarting containers does not re-download diarization assets if the cache volume exists.
- CI passes, including 100% coverage enforcement and docker smoke.
```
