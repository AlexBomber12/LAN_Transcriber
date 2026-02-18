PR ID: PR-METRICS-01
Branch: pr/metrics-01

Goal
Conversation analytics metrics per participant and per meeting + UI rendering + exports.

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
PR-PIPELINE-01, PR-LLM-01

Work plan
1) Implement participant-level metrics from diarization + transcript
   - airtime_seconds: sum of speaker turn durations
   - airtime_share: airtime / total_speech_time
   - turns: count of speaker turns after merging tiny gaps (gap_threshold=1.0s)
   - interruptions_done / interruptions_received:
     - interruption if speaker B starts while speaker A is still speaking and overlap >= 0.3s (configurable)
     - count both sides
   - questions_count:
     - use LLM-extracted questions count as primary
     - fallback heuristic: count '?' in transcript for languages where punctuation exists

2) Role hint (simple heuristics + optional LLM tie-break)
   - Leader: high share + many decisions/tasks initiated
   - Facilitator: high turns + high questions
   - Expert: long turns + high terminology density (use LLM tags as proxy)
   - Passive: low share + low turns

3) Meeting-level metrics
   - total_interruptions
   - total_questions + types (from summary.json)
   - decisions_count
   - action_items_count
   - actionability_ratio = (# action items with owner and deadline) / (total action items)
   - emotional_summary (from summary.json)

4) Storage
   - Write derived/metrics.json
   - Also persist into DB tables meeting_metrics and participant_metrics for query and dashboard aggregation.

5) UI
   - Metrics tab:
     - participants table with airtime/share/turns/interruptions/questions/role
     - meeting cards with totals + actionability + emotional summary

Local verification
- Add unit tests for:
  - interruptions counting
  - turn merging logic
  - actionability computation
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- Metrics are computed deterministically and visible in UI.
- Metrics do not depend on language and work for mixed-language meetings.
