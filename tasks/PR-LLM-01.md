PR ID: PR-LLM-01
Branch: pr/llm-01

Goal
Spark LLM: topic, summary, decisions, actions (owner/deadline), emotional summary, question typing.

Hard constraints
- Follow AGENTS.md PLANNED PR runbook.
- No secrets in the repo. Any credentials, keys, tokens, or config files must live under /data and be mounted via docker-compose or provided via env vars.
- Implement only what is required for this PR. Do not bundle extra refactors, dependency upgrades, or feature additions.
- Keep changes incremental and keep the application runnable at the end of the PR.
- Preserve Linux-first behavior. Do not add Windows-only steps.
- Maintain backwards compatibility for already committed developer workflows where possible.

Context
We are building a LAN application with a simple DB-like UI to manage meeting recordings. Ingest comes from Google Drive (Service Account + shared folder), processing runs locally, summaries are generated via Spark LLM (OpenAI-compatible API), and publishing goes to OneNote via Microsoft Graph (work account).

Depends on
PR-PIPELINE-01, PR-LANG-01

Work plan
1) Integrate Spark LLM (OpenAI-compatible) for structured outputs
   - Configuration env vars:
     - LLM_BASE_URL
     - LLM_API_KEY (optional, can be blank on LAN)
     - LLM_MODEL
     - LLM_TIMEOUT_SECONDS
   - Implement robust retries and timeouts.

2) Define structured JSON outputs (derived/summary.json)
   - topic: short title
   - summary_bullets: list
   - decisions: list
   - action_items: list of {task, owner?, deadline?, confidence}
   - emotional_summary: 1-3 lines
   - questions:
     - total_count
     - types: {open, yes_no, clarification, status, decision_seeking}
     - optional list of extracted questions (short)

3) Prompt design
   - Input: speaker_turns (with timestamps and speaker labels), calendar title/attendees (if available), target_summary_language.
   - Output must be strict JSON with the schema above.
   - Ensure prompts handle mixed-language: keep content as-is, but produce summaries in target language.

4) UI integration
   - Overview tab shows topic, summary, emotional summary.
   - Metrics tab consumes decisions/action_items/questions.

Local verification
- Provide a mock LLM response path for tests.
- Run a dry run with a small transcript and verify JSON is saved.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- summary.json is produced consistently and validated.
- Action items include owner/deadline when available and a confidence field.
- Emotional summary and question typing are present.
