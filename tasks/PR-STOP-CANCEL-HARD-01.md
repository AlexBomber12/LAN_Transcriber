PR-STOP-CANCEL-HARD-01
======================

Branch: pr-stop-cancel-hard-01

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Extend the new Stop feature so it can interrupt long-running heavy compute more aggressively, not only at soft checkpoints. Soft stop from the previous PR is necessary but insufficient if ASR, diarization, or an LLM call remains busy for a long time. Implement a hard-stop escalation path using child processes for heavy stages, with a short grace period for cooperative exit and then explicit termination if the child does not stop on its own.

Constraints
- This PR builds on the soft-cancel PR and must preserve its user-facing behavior.
- Do not redesign the whole worker architecture into multiple RQ workers or services.
- Keep the public queue/job model unchanged.
- Make termination safe and bounded; do not leave zombie/orphan child processes.
- Maintain 100% statement and branch coverage for every changed/new module.

Implementation requirements

1) Isolate heavy stages into terminable child processes
- Identify the heavy stage families that can block soft cancel for too long. At minimum consider:
  - ASR execution/loading path
  - diarization execution/loading path
  - chunked LLM extraction loop and possibly merge call
- Add a small execution wrapper that can run a stage function in a child process and return structured results/artifacts to the parent.
- Use a standard-library-safe mechanism that works in the Docker/Linux production environment, for example multiprocessing or subprocess-backed wrappers as appropriate.
- Keep the contract explicit and testable.

2) Add hard-stop escalation behavior
- When a recording has a cancel request:
  - first allow the existing soft-cancel path/grace period to work
  - if the current heavy child process does not stop within the bounded grace period, terminate it explicitly
- Add a configurable grace period setting, for example STOP_GRACE_SECONDS / LAN_STOP_GRACE_SECONDS, with a sensible default.
- After termination, mark the recording as Stopped/Cancelled by user, not Failed.
- Log clearly whether a stop finished cooperatively or required forced termination.

3) Keep checkpoint and artifact integrity safe
- Do not mark a stage completed if its child process was terminated mid-stage.
- Preserve already-completed artifacts from earlier stages.
- Clean up obvious temp files for the aborted stage where safe.
- Keep stage/chunk checkpoint rows in a state that allows later manual requeue from scratch rather than accidental partial resume from a corrupt mid-stage state.

4) Parent/child result contract
- The parent worker must receive structured success/failure/cancel information from the child wrapper.
- Avoid pickling huge model objects; child execution should receive only the minimal serializable inputs it needs.
- Keep GPU ownership sane under the existing scheduler logic.
- Ensure child termination does not leave stale CUDA memory allocations longer than necessary.

5) Integrate with existing Stop UI/endpoint
- Reuse the Stop button and cancel endpoint from the previous PR.
- No major UI redesign is needed. Small status-text additions like Force-stopped after grace timeout are acceptable.

6) Logging and observability
- Add clear step-log lines such as:
  - stop requested
  - waiting for cooperative stop grace=...
  - force terminating child stage=asr pid=...
  - stage terminated by user stop
- Preserve operator readability and do not leak huge stack traces into user-facing text.

7) Tests with full coverage
Add deterministic offline tests for at least these cases:
- a cooperative child process exits during the grace period and no forced termination occurs
- a non-cooperative child process is terminated after the grace period
- terminated stages are not marked completed and do not leave the recording in Failed
- earlier completed artifacts remain intact after hard stop
- explicit requeue after hard stop starts cleanly
- the worker logs/status path distinguish soft stop vs forced termination
- grace-period config validation and default handling are covered
Mock/process-isolate the heavy functions so tests remain fast and offline.

8) Documentation and operator notes
- Update README/runbook briefly to explain the two-level stop model:
  - soft stop at safe checkpoints
  - hard-stop escalation for long-running heavy stages after a grace period
- Mention the new grace-period env var.

Verification steps (must be included in PR description)
- Start a long-running heavy stage, request Stop, and confirm the worker first attempts cooperative stop and then force-terminates the child if needed.
- Confirm the recording ends in the stopped/cancelled state, not Failed.
- Confirm manual requeue after hard stop starts from a clean state.
- Run scripts/ci.sh and keep CI green.

Deliverables
- Child-process execution wrapper for heavy stages
- Hard-stop escalation after grace period
- Safe checkpoint/artifact handling around terminated stages
- Updated tests and docs

Success criteria
- Stop can interrupt long-running heavy stages within a bounded time.
- Forced termination does not misclassify the recording as a processing failure.
- Requeue after hard stop still works cleanly.
- CI remains green.
```
