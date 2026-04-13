Run PLANNED PR

PR_ID: PR-LLM-INLINE-EXECUTION-01
Branch: pr-llm-inline-execution-01
Title: Execute LLM calls inline in worker process instead of spawning child processes

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a focused BIG PR, not a MICRO PR. Keep the scope strict.

Context
Every LLM generate call spawns a new child process via _run_child_stage_operation. For chunked summaries this means multiple process spawns per recording. Each spawn creates a new Python interpreter, reinitializes dependencies, creates a new httpx client, and discards all caches. On a recording with 10 LLM chunks, this adds significant orchestration overhead (potentially 30-40% of LLM stage time).

The child process pattern was introduced for GPU-heavy stages (ASR, diarization) to enable memory isolation and OOM recovery. LLM calls are HTTP requests to an external endpoint - they do not use GPU memory in the worker process. There is no memory isolation benefit from running them in child processes.

Phase 1 - Inspect and map
Read these files:
- lan_app/worker_tasks.py: _CancelAwareLLMClient (line ~891), _run_llm_generate_child_operation (line ~2298), _stage_llm_extract (line ~3300)
- lan_app/worker_tasks.py: _run_child_stage_operation (line ~619) to understand the spawn mechanism
- lan_app/worker_tasks.py: how stop/cancel works for LLM stages - identify what must be preserved

Map all code paths where LLM generate goes through child process spawn. Identify:
- Which paths NEED child process (GPU memory isolation) - keep these
- Which paths are pure HTTP calls (LLM) - convert to inline

Phase 2 - Implement

CHANGE 1: Make _CancelAwareLLMClient execute inline
Modify _CancelAwareLLMClient.generate to call LLMClient().generate() directly in the worker process instead of spawning a child process. Keep the cancel/stop awareness:
- Use asyncio timeout for cooperative cancellation
- Check stop request between chunks
- Log chunk progress

CHANGE 2: Keep child process ONLY for ASR and diarization
ASR and diarization legitimately need child process for GPU memory isolation. Keep _run_child_stage_operation for _stage_asr and _stage_diarization only.

CHANGE 3: Preserve stop/cancel behavior
The current cancel mechanism sends signals to child processes. For inline execution, switch to:
- asyncio.wait_for with timeout for each LLM call
- Check _recording_stop_request between chunks
- Raise cancellation exception if stop requested

Phase 3 - Test and verify
- Run full CI.
- Process a recording with chunked summary (long transcript). Measure total LLM stage time before and after.
- Verify stop/cancel still works during LLM processing.
- Verify no GPU memory leaks from inline LLM execution (should be zero since LLM is HTTP-only).

Success criteria:
- LLM calls execute inline in worker process without child process spawn.
- ASR and diarization still use child process for GPU memory isolation.
- Stop/cancel works for LLM stages.
- Measurable reduction in LLM stage time (especially for multi-chunk recordings).
- No existing tests break.
