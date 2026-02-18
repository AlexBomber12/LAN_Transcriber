PR ID: PR-VOICE-01
Branch: pr/voice-01

Goal
Voice profiles + mapping diarization speakers to known people (human-in-the-loop UI).

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
PR-PIPELINE-01, PR-UI-SHELL-01

Work plan
1) Implement voice profiles (human-in-the-loop) without over-automation
   - CRUD for voice profiles (name, optional notes)
   - Voice samples:
     - link to snippet files produced in PR-PIPELINE-01
     - allow "Add sample from this recording" action

2) Speaker assignment UI
   - In Recording Detail -> Speakers tab:
     - list diarization speakers with snippet playback
     - dropdown to assign to an existing voice profile
     - ability to create a new voice profile and assign immediately
   - Store assignments in DB (speaker_assignments)

3) Optional: candidate matching (very light)
   - If you can compute embeddings reliably:
     - store embeddings for voice samples
     - show top 3 candidate profiles with confidence (do not auto-assign)
   - Keep it optional and behind a feature flag to avoid pipeline fragility.

Local verification
- Create a voice profile and assign a diar speaker label.
- Reload the page and confirm assignment persists.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- Users can map diarization speaker labels to stable people via UI.
- The mapping persists and can be reused for routing and OneNote rendering.
