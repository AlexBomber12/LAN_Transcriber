PR ID: PR-PIPELINE-01
Branch: pr/pipeline-01

Goal
STT + diarization + word timestamps + speaker-attributed transcript + speaker snippets.

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
PR-GDRIVE-INGEST-01, PR-REFRACTOR-CORE-01, PR-DB-QUEUE-01

Work plan
1) Implement STT with word-level timestamps
   - Prefer whisperx or faster-whisper + alignment.
   - Output transcript.json with:
     - segments: [{start,end,text,words:[{start,end,word}]}]
     - language detection output and confidence
   - Output transcript.txt (human-readable)

2) Implement diarization and speaker-attributed transcript
   - Run diarization (pyannote) producing segments.json:
     - [{start,end,speaker}]
   - Combine diarization with transcript words/segments to produce:
     - speaker_turns.json: [{start,end,speaker,text,language(optional)}]
   - Ensure this is deterministic and stored in /data/recordings/<id>/derived/

3) Generate speaker snippets for UI assignment
   - For each diarization speaker label:
     - pick 2-3 high-SNR segments (longest segments, minimal overlap)
     - export 10-20 sec wav snippets into derived/snippets/<speaker>/<n>.wav
   - These snippets are used in UI for voice mapping.

4) Precheck + quarantine integration
   - Compute:
     - duration
     - VAD speech ratio
   - Quarantine rules:
     - duration < 20s, or speech_ratio < 0.10
   - If quarantined:
     - skip heavy pipeline steps and mark status Quarantine with reason.

Local verification
- Process 1 sample audio end-to-end up to speaker_turns + snippets.
- Confirm artifacts are written and UI can play snippets.
- scripts/ci.sh exits 0.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- For a normal recording, transcript.json, segments.json, speaker_turns.json, and snippets are produced.
- For short/empty recordings, quarantine triggers reliably and stops further processing.
