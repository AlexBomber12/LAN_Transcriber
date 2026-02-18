PR ID: PR-REFRACTOR-CORE-01
Branch: pr/refactor-core-01

Goal
Refactor existing code into a stable core pipeline library and /data-backed state.

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
PR-BOOTSTRAP-01

Work plan
1) Separate "core pipeline" from "application layer"
   - Keep lan_transcriber/ as the core library that exposes a stable Python API:
     - core transcription (STT)
     - diarization integration
     - Spark LLM client
     - artifact writer helpers
   - Add a new package lan_app/ (or app/) for:
     - API server
     - UI templates
     - workers / jobs orchestration
     - DB integration (stubbed until PR-DB-QUEUE-01)

2) Remove in-package mutable state
   - Move any speaker alias db, unknown voice chunks, temp dirs, and outputs from lan_transcriber/ to /data.
   - Replace hard-coded paths with configuration:
     - env vars with LAN_ prefix
     - defaults that point to /data/*

3) Define the canonical artifact format (v1)
   - Define a Recording artifact directory structure under /data/recordings/<recording_id>/:
     - raw/audio.<ext> (original)
     - derived/transcript.json
     - derived/transcript.txt
     - derived/segments.json (diarization segments)
     - derived/snippets/ (speaker snippet wav files)
     - derived/summary.json (LLM output placeholder for later PR)
     - derived/metrics.json (placeholder for later PR)
     - logs/step-*.log
   - Implement a small helper module to create this tree and write JSON safely (atomic writes).

4) Ensure existing CLI/UI entry points still work
   - If gradio UI exists, keep it functional (it can remain as a dev tool) but make it consume the new artifact layout and config.

Local verification
- scripts/ci.sh exits 0.
- Minimal smoke: process a small local audio file and produce artifacts under /data/recordings/<id>/.

Artifacts
- scripts/make-review-artifacts.sh

Success criteria
- The codebase has a clear separation between core pipeline and app layer.
- No mutable state is stored inside the Python package directory.
- Artifact layout is documented and produced deterministically.
- Existing workflows do not regress.
