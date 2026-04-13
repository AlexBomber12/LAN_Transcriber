Run PLANNED PR

PR_ID: PR-LLM-CLIENT-POOLING-01
Branch: pr-llm-client-pooling-01
Title: Reuse httpx.AsyncClient across LLM requests with connection pooling and keep-alive

Follow AGENTS.md exactly for work mode, queue handling, CI, artifacts, MCP usage, and scope control. This is a MICRO PR.

Context
In lan_transcriber/llm_client.py:277, _post_chat_completion creates a new httpx.AsyncClient for every request. This means a new TCP connection, new TLS handshake (if applicable), and new connection pool per LLM call. For chunked summaries with 10+ chunks, this is 10+ unnecessary connection setups.

Phase 1 - Inspect
Read:
- lan_transcriber/llm_client.py: _post_chat_completion, LLMClient class, how client lifecycle is managed
- lan_app/worker_tasks.py: how LLMClient is instantiated (line ~3830 area)

Phase 2 - Implement

CHANGE 1: Make httpx.AsyncClient a class attribute with lazy init
In LLMClient, create the httpx.AsyncClient once on first use and reuse it for all subsequent requests:

class LLMClient:
    _http_client: httpx.AsyncClient | None = None

    @classmethod
    def _get_client(cls) -> httpx.AsyncClient:
        if cls._http_client is None or cls._http_client.is_closed:
            cls._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return cls._http_client

CHANGE 2: Use shared client in _post_chat_completion
Replace:
    async with httpx.AsyncClient(...) as client:
        response = await client.post(...)
With:
    client = self._get_client()
    response = await client.post(...)

CHANGE 3: Add graceful shutdown
Add a classmethod close() that closes the client on worker shutdown. Wire it to worker lifecycle if possible, but make it optional (client can also be garbage collected).

Phase 3 - Test and verify
- Run full CI.
- Verify LLM requests reuse the same TCP connection (check logs or connection count).
- Verify client handles connection errors gracefully (auto-reconnect on next request).
- Verify no resource leaks on long-running worker.

Success criteria:
- Single httpx.AsyncClient reused across all LLM requests in a worker.
- Connection keep-alive and pooling active.
- No TCP connection setup overhead per chunk.
- No existing tests break.
