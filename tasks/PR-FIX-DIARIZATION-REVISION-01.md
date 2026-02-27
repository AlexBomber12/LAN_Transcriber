PR-FIX-DIARIZATION-REVISION-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-diarization-revision-01
PR title: PR-FIX-DIARIZATION-REVISION-01 Fix pyannote diarization revision handling and prevent recording failures
Base branch: main

Problem
Uploads fail with an exception similar to:
  "Revisions must be passed with `revision=...`"
This happens because the diarization model is loaded using repo_id with "@REV" appended, for example:
  pyannote/speaker-diarization@3.2
Newer huggingface_hub versions reject that syntax and require the revision to be passed separately.

Goals
1) Stop the pipeline from failing due to repo_id "@revision" syntax.
2) If pyannote diarization cannot be initialized (auth gated, missing model, hub error), do not fail the recording.
   Fall back to the existing single-speaker diariser and continue processing.
3) Update tests to reflect the new behavior and keep CI green.

Non-goals
- Do not introduce a new UI workflow.
- Do not change the overall pipeline stages or DB schema.

Implementation plan

A) Introduce a tiny helper to split repo id and revision
- Create a new module: lan_app/hf_repo.py
- Add:
  split_repo_id_and_revision(value: str) -> tuple[str, str | None]
    - Trim whitespace
    - If value contains "@", split on the first "@"
    - Return (repo_id, revision) where revision is None if empty
    - If no "@", return (value, None)

B) Fix pyannote Pipeline.from_pretrained calls to use revision kwarg
1) Update lan_app/worker_tasks.py
- Replace the hardcoded:
    Pipeline.from_pretrained("pyannote/speaker-diarization@3.2")
  with:
    repo_id, revision = split_repo_id_and_revision("pyannote/speaker-diarization@3.2")
    kwargs = {"revision": revision} if revision else {}
    try:
        model = Pipeline.from_pretrained(repo_id, **kwargs)
    except TypeError:
        model = Pipeline.from_pretrained(f"{repo_id}@{revision}" if revision else repo_id)
- Keep the function signature of _build_diariser(duration_sec) unchanged.

2) Update lan_app/ui.py (if the file exists in the repo)
- Replace the same "@3.2" string with the split-and-revision approach.
- Keep behavior identical otherwise.

C) Prevent recordings from failing when diariser init fails
- In lan_app/worker_tasks.py inside _run_precheck_pipeline:
  - Around the non-quarantine diariser selection block, change:
      diariser = _build_diariser(precheck.duration_sec)
    to:
      try:
          diariser = _build_diariser(precheck.duration_sec)
      except Exception as exc:
          _append_step_log(log_path, f"diariser init failed, falling back: {type(exc).__name__}: {exc}")
          diariser = _FallbackDiariser(precheck.duration_sec)
- This must cover auth failures and hub errors, not only ModuleNotFoundError.

D) Tests updates
- Update tests/test_db_queue.py tests around _build_diariser:
  1) test_build_diariser_wraps_sync_pyannote_pipeline
     - Adjust the fake Pipeline.from_pretrained stub to accept **kwargs.
     - Assert that when the model string contains "@3.2", the code calls from_pretrained with:
         name == "pyannote/speaker-diarization"
         kwargs["revision"] == "3.2"
     - Keep the existing assertion that the returned diariser wraps the model call.

  2) Add a new test that _run_precheck_pipeline falls back when _build_diariser raises.
     - Monkeypatch lan_app.worker_tasks._build_diariser to raise RuntimeError("boom")
     - Monkeypatch lan_app.worker_tasks.run_pipeline to capture the diariser argument and do nothing.
     - Provide a non-quarantine PrecheckResult via monkeypatch for run_precheck.
     - Ensure the pipeline uses worker_tasks._FallbackDiariser and process_job returns ok and sets recording status to Ready or NeedsReview depending on routing stubs.
     - Keep the test minimal by stubbing refresh_recording_metrics and refresh_recording_routing to stable values.

Local verification
- scripts/ci.sh
- Manual smoke:
  1) Upload a file in /upload
  2) Confirm the recording no longer fails at diarization init
  3) Confirm a step log line mentions fallback when diarization cannot load

Success criteria
- No use of "repo@revision" is passed into huggingface_hub; revision is passed via revision= when supported.
- A diarization init failure does not fail the recording; the pipeline continues with single-speaker diarization.
- Unit tests covering revision splitting and fallback behavior are green.
- scripts/ci.sh is green.
```
