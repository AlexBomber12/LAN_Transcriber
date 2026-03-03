PR-LLM-DIAGNOSTICS-01
=====================

Prompt (copy as-is into Codex)
------------------------------

```text
You are Codex Agent working on the LAN-Transcriber repository.

Goal
Add robust, low-noise LLM diagnostics so intermittent HTTP errors (especially 404) can be understood immediately from logs, without guessing. PR-LLM-ROBUST-01 already improved max_tokens and retry behavior. This PR focuses on:
- Robust URL construction for Ollama OpenAI-compatible endpoints (/v1/chat/completions).
- Actionable error reporting that includes status code and a truncated response body, plus the effective URL and key request metadata.
- No leakage of prompts or user text into logs.

Context
We observed an earlier worker error:
httpx.HTTPStatusError: 404 Not Found for url 'http://<ollama>:11434/v1/chat/completions'
However, manual POSTs from inside the worker to the same URL returned status 200, including when response_format={"type":"json_object"} is present.
This suggests that when a 404 happens, it is caused by a subtle difference in the computed URL (double /v1, trailing slashes, accidental path segments), or by a different model/url being used at runtime, or by stale config.
We need diagnostics that show the computed URL and the response body for any non-2xx HTTP response.

Constraints
- Micro PR: only diagnostics and URL join hardening.
- Do not change prompts, schema, or business logic.
- Do not log prompts or recording text. Log only safe metadata (url host/path, model, max_tokens, timeout).
- Maintain 100% statement and branch coverage for all modified/new project modules.
- CI must not require a running Ollama; tests mock HTTP responses.

Implementation requirements

1) Robust URL join helper
- Add a small helper in lan_transcriber/llm_client.py (or a new small module if you prefer, but keep it minimal) that builds the chat completions URL from LLM_BASE_URL.
- Requirements for build_chat_completions_url(base_url: str) -> str:
  - Strip whitespace.
  - Remove trailing slashes.
  - If base_url already ends with "/v1/chat/completions", return it.
  - If base_url ends with "/v1", append "/chat/completions".
  - If base_url ends with "/v1/", normalize and append "chat/completions".
  - Otherwise append "/v1/chat/completions".
  - Preserve scheme and host, do not alter port.
- Add unit tests covering the above cases.
- Ensure this helper is used everywhere the LLM client constructs the request URL.

2) Safe request metadata logging (no prompt leakage)
- Immediately before sending the HTTP request in _post_chat_completion, log one DEBUG line:
  - llm_url=<effective url>
  - model=<model>
  - max_tokens=<max_tokens>
  - timeout_seconds=<timeout>
  - attempt=<attempt number if available>
  - response_format=<true/false>
- DO NOT include messages, prompts, or user content.

3) Actionable HTTP error messages
- When response status is non-2xx:
  - Read response text (truncate to max 2000 characters).
  - Raise an exception that includes:
    - status_code
    - effective url
    - model
    - max_tokens
    - truncated response body
  - Keep original exception chaining for stack trace clarity.
- If the response body is JSON, include the JSON "error" field if present; otherwise include the raw text.
- Ensure raising still triggers existing tenacity retry logic for network errors, but do not retry indefinitely for deterministic 4xx unless the current policy already does so.

4) Persist safe error artifacts (optional but recommended)
- If the project already writes LLM raw responses as artifacts, add a parallel artifact for failures:
  - store a JSON object containing status, url, model, and truncated body
  - ensure it excludes prompts/messages
- This artifact must not contain PII from the recording text.

5) Tests (100% coverage)
- Add tests mocking HTTPX client behavior:
  a) URL builder produces correct URL for base URLs:
     - http://host:11434
     - http://host:11434/
     - http://host:11434/v1
     - http://host:11434/v1/
     - http://host:11434/v1/chat/completions
  b) On 404 with a JSON error body, the raised exception message includes status, url, model, and a truncated body snippet.
  c) On 404 with non-JSON body, exception includes raw text snippet.
  d) The debug log line does not contain prompts (assert log message contains only the expected fields).
- Keep statement and branch coverage at 100% for modified/new code.

Verification steps (include in PR description)
- Force a controlled 404 by temporarily setting LLM_BASE_URL to an invalid path (e.g., append /bad) and confirm logs show:
  - effective url and safe request metadata
  - error body snippet
- Restore LLM_BASE_URL and confirm successful processing continues.

Deliverables
- Robust URL join helper and its unit tests
- Improved error messages for non-2xx responses including body snippet and request metadata
- Safe debug logging with no prompt leakage
- Optional safe error artifact writing (if supported by current artifact framework)

Success criteria
- If an HTTP 404/400 occurs, the worker logs and exception message clearly show the effective URL and response body without leaking prompts.
- URL construction is stable across base_url variants and prevents accidental double /v1 paths.
- CI remains green with 100% statement and branch coverage.
```
