PR ID: PR-LANG-01
Branch: pr/lang-01

Goal
Multi-language support (2 languages): language spans, dominant language, UI override + reprocess hooks.

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
PR-PIPELINE-01

Work plan
1) Add multi-language detection for mixed-language meetings (2 languages)
   - Produce language spans from transcript:
     - Approach A (MVP): detect per-segment language with a lightweight detector and merge adjacent spans.
     - Approach B: use the STT model language tags if available plus heuristics.
   - Store in transcript.json:
     - dominant_language
     - language_distribution (percentages)
     - language_spans: [{start,end,lang}]

2) UI: Language tab
   - Show auto-detected languages and distribution.
   - Allow override of:
     - target_summary_language (what language summaries are produced in)
     - transcript_language_override (rare, mostly for edge cases)
   - Buttons:
     - Re-summarize (LLM only)
     - Re-transcribe (STT again) with caution

3) Pipeline integration
   - If meeting is mixed-language:
     - summaries should be generated in target_summary_language
     - transcript remains original, but spans are preserved
   - Ensure metrics logic is language-agnostic.

Local verification
- Simulate a mixed-language transcript.json and verify the UI shows spans.
- Re-summarize uses the target language override.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- The system can represent and display 2-language meetings without breaking downstream steps.
- The user can override the target summary language and regenerate summaries.
