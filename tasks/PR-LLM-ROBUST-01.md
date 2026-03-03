PR-LLM-ROBUST-01
================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Make the LLM step robust for OpenAI-compatible Ollama endpoints (/v1/chat/completions) by addressing these production issues:
- Responses may arrive with finish_reason="length" when max_tokens is too small, which can produce empty message.content.
- Some responses may include non-empty message.reasoning while message.content is empty. The pipeline requires content to proceed.
- Current behavior can lead to repeated retries and "max attempts exceeded" without a clear diagnosis.

Implement:
1) Configurable LLM max_tokens (env), used consistently for all LLM calls.
2) Fail-fast if message.content is empty after a request (with actionable error and preserved raw response).
3) One explicit retry when finish_reason=="length" or content is empty, with increased max_tokens, before giving up.

Constraints
- Micro PR focused on LLM client robustness only.
- Do not change the LLM prompts or schema logic beyond what is needed to implement the retry/fail-fast.
- Maintain 100% statement and branch coverage for changed/new code.
- CI must not require a running Ollama. All tests must mock HTTP.

Repository context
- LLM client is in lan_transcriber/llm_client.py (async httpx + tenacity).
- Workers call llm.generate(system_prompt=..., user_prompt=..., model=..., response_format={...}).
- Ollama base URL is configured via LLM_BASE_URL and works for POST /v1/chat/completions.
- Current timeouts are configured via LLM_TIMEOUT_SECONDS.

Implementation requirements

A) Add config for max_tokens
1) Add an AppSettings field:
- name: llm_max_tokens (int)
- env var: LLM_MAX_TOKENS
- default: 1024 (choose a safe default for json outputs; must be >= 256)
2) Optionally add llm_max_tokens_retry (int) or compute retry value as min(llm_max_tokens * 2, 4096).
- If you add a setting:
  - env var: LLM_MAX_TOKENS_RETRY
  - default: 2048

B) Ensure payload includes max_tokens
- In lan_transcriber/llm_client.py, ensure the outgoing payload to /v1/chat/completions includes:
  - "max_tokens": settings.llm_max_tokens (or per-call override if present)
- Ensure max_tokens is not silently omitted.
- Keep temperature and response_format behavior as-is.

C) Parse response robustly and fail fast
- After receiving the JSON response:
  - Extract finish_reason (choices[0].finish_reason)
  - Extract content (choices[0].message.content) as a string
  - If content is missing or empty/whitespace:
    - Treat as invalid response, not a successful completion
    - Raise a dedicated exception type, e.g. LLMEmptyContentError, including:
      - finish_reason
      - model name
      - request id if present
      - a short hint that Ollama may have returned reasoning-only output
    - Store the raw response in the existing artifacts/logging mechanism (if the project already stores raw LLM responses).
- Do not accept message.reasoning as a fallback output. Use it only for debugging.

D) Add one explicit retry on finish_reason=length or empty content
- Implement retry logic inside generate(), not only via tenacity:
  1) Attempt 1 with max_tokens = settings.llm_max_tokens
  2) If:
     - finish_reason == "length", or
     - content is empty (LLMEmptyContentError), or
     - response parsing indicates truncation,
     then attempt exactly one more request:
     - max_tokens = retry_max_tokens (either LLM_MAX_TOKENS_RETRY or min(base*2, cap))
  3) If attempt 2 still results in empty content or finish_reason="length":
     - raise a clear exception explaining that max_tokens is still too low and suggesting increasing LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS.
- Ensure this single retry is independent of tenacity retries for network errors.
- Tenacity should continue to handle transport errors (timeouts, connection errors, 5xx), but the content/length retry should not loop indefinitely.

E) Timeouts and observability improvements (small)
- When raising errors, include:
  - base_url host
  - model
  - max_tokens used
  - finish_reason
- Add one INFO or DEBUG log per request attempt with:
  - model, max_tokens, timeout seconds
  Keep it low noise and avoid logging prompts.

F) Tests with 100% coverage for new logic
- Add tests for lan_transcriber/llm_client.py covering:
  1) Payload includes max_tokens from settings.
  2) When response finish_reason="stop" and content non-empty -> returns content.
  3) When finish_reason="length" on attempt 1 -> retries once with larger max_tokens and returns content.
  4) When content is empty on attempt 1 (finish_reason can be "length" or "stop") -> retries once and returns content.
  5) When attempt 2 still returns finish_reason="length" or empty content -> raises LLMEmptyContentError (or a dedicated LLMTruncatedError) with actionable message.
- Use httpx mocking (respx) or monkeypatch the internal _post_chat_completion method to return canned JSON responses.
- Ensure branch coverage for:
  - content empty path
  - length retry path
  - no-retry path
  - retry-fails path

G) Update env example and docs
- Add to .env.example:
  - LLM_MAX_TOKENS=1024
  - LLM_MAX_TOKENS_RETRY=2048 (optional)
  - LLM_TIMEOUT_SECONDS=600 (recommendation for large models)
- In README/runbook, add a short note:
  - If you see finish_reason=length or empty content, increase LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS.

Verification steps (must be included in PR description)
1) In worker container, run a small POST to /v1/chat/completions and confirm content is returned.
2) Run a real recording end-to-end and confirm no "max attempts exceeded" due solely to finish_reason=length or empty content.
3) Confirm that when content is empty, the error message is clear and the raw response is preserved.

Deliverables
- New settings: LLM_MAX_TOKENS (and optional retry setting)
- LLM client updated with max_tokens, fail-fast, and one content/length retry
- Tests covering all branches, 100% statement and branch for modified code
- Updated docs and .env.example

Success criteria
- LLM calls no longer fail with ambiguous "max attempts exceeded" when the true cause is truncation or empty content.
- One controlled retry resolves most truncation cases.
- If still failing, error is actionable (increase max_tokens/timeout) and includes finish_reason and max_tokens used.
- CI is green.
```
