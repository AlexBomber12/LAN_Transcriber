# LAN Transcriber

Offline transcription pipeline with WhisperX and external language model.

## Planned PR workflow

Planned work is queue-driven and defined in `tasks/QUEUE.md`.

- Execute PRs in queue order only.
- Implement only the scope defined in the active `tasks/PR-*.md`.
- Follow `AGENTS.md` for runbook, branch naming, and handoff gates.

## Local CI and review artifacts

Run the same lint/test gates used by CI:

```bash
scripts/ci.sh
```

Generate review artifacts for planned PR handoff:

```bash
scripts/make-review-artifacts.sh
```

This produces:

- `artifacts/ci.log`
- `artifacts/pr.patch`

## Dev mode

Use a compose overlay for fast code iteration without image rebuilds.

Build once:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

After code changes:

```bash
docker compose restart api worker
```

Rebuild images only when dependencies change (for example `requirements.txt` updates).

## Operations runbook

Operational setup, failure handling, backup/restore, and upgrade steps are documented in
[`docs/runbook.md`](docs/runbook.md).

## Workflow (Upload -> Processing -> Export)

1. Open `/upload` and add one or more audio files.
2. Track per-file upload progress and processing progress on the same page.
3. Open the recording detail page at `/recordings/{recording_id}`.
4. Export results:
   - Copy markdown from the export tab for manual OneNote paste.
   - Download ZIP from `/ui/recordings/{recording_id}/export.zip`.

## Runtime data root

Runtime mutable state must live under `/data` in containers (mounted from `./data` in Docker):

- `/data/db/app.db`
- `/data/db/speaker_bank.yaml`
- `/data/recordings/<recording_id>/...`
- `/data/secrets/...` (optional runtime secrets)
- `/data/voices`
- `/data/tmp`

Canonical artifact layout (v1):

```text
/data/recordings/<recording_id>/
  raw/audio.<ext>
  derived/transcript.json
  derived/transcript.txt
  derived/segments.json
  derived/snippets/
  derived/summary.json
  derived/metrics.json
  logs/step-*.log
```

Do not commit secrets or runtime-generated state files.

## LAN deployment notes

- Use a persistent host volume for `/data` (for Docker Compose, `./data:/data` by default).
- Size disk for model cache + uploads + derived artifacts. A practical baseline is:
  - small teams: 50-100 GB
  - medium usage with longer recordings: 200 GB+
- Monitor free space under `/data/recordings` and `/opt/lan_cache/hf`.

## Staging

The staging environment spins up the application and a tiny LLM model using
`docker compose`. Copy the files in `infra/staging` to your server and run:

```bash
cd ~/lan-staging
docker compose up -d --build
```

To use a prebuilt GHCR image instead of a local build, set:

```bash
TRANSCRIBER_IMAGE=ghcr.io/alexbomber12/lan-transcriber:latest
TRANSCRIBER_PULL_POLICY=always
TRANSCRIBER_DOCKER_TARGET=runtime-full
```

The compose file mounts `/opt/lan_cache/hf` into `/root/.cache/huggingface` so
models are cached across runs.

`docker-compose.yml` expects a `.env` file with the following variables:

| Variable | Description |
| --- | --- |
| `LAN_ENV` | Runtime mode: `dev` (default), `staging`, or `prod` |
| `LAN_DB_PATH` | SQLite database path (default `/data/db/app.db`) |
| `LAN_REDIS_URL` | Redis endpoint for the RQ queue |
| `LAN_RQ_QUEUE_NAME` | Queue name consumed by the worker |
| `LAN_API_BEARER_TOKEN` | Optional bearer token for protected POST actions (`/api` and UI POST routes) |
| `UPLOAD_MAX_BYTES` | Optional max size per uploaded file in bytes (`413` when exceeded) |
| `QUARANTINE_RETENTION_DAYS` | Retention period for quarantined recording cleanup (default `7`) |
| `LAN_API_BIND_HOST` | Published API bind host (default `127.0.0.1`) |
| `LAN_API_PORT` | Published API port (default `7860`) |
| `LLM_BASE_URL` | OpenAI-compatible Spark endpoint |
| `LLM_API_KEY` | Optional API key for the LLM |
| `LLM_MODEL` | Model name passed to the OpenAI-compatible endpoint |
| `LLM_TIMEOUT_SECONDS` | Per-request timeout for LLM calls (default `30`) |

`LAN_ENV` controls startup validation:

- `LAN_ENV=dev`: missing `LAN_REDIS_URL` and/or `LLM_BASE_URL` is allowed with warnings; dev defaults are used (`redis://127.0.0.1:6379/0`, `http://127.0.0.1:8000`).
- `LAN_ENV=staging` or `LAN_ENV=prod`: `LAN_REDIS_URL` and `LLM_BASE_URL` are required; startup fails fast if either is missing.

If API auth is enabled, set `LAN_API_BEARER_TOKEN` to a non-empty value in your env file.

`docker compose up` starts:

- `db` (SQLite migration init)
- `redis` (queue broker)
- `api` (FastAPI backend)
- `worker` (RQ worker)

The stack exposes `lan_transcriber_health{env="staging"}` on `/metrics` for
future monitoring.

When `LAN_API_BEARER_TOKEN` is set:

- Protected endpoints accept either `Authorization: Bearer <token>` or the HttpOnly cookie from `POST /ui/login`.
- `GET /healthz`, `GET /healthz/{component}`, `GET /metrics`, and `GET /openapi.json` remain public.
- Upload and recording action POST routes require auth (for example `POST /api/uploads` and `/ui/recordings/{id}/...`).

## Staging deploy secrets

| Secret | Description | Example |
|--------|-------------|---------|
| STAGING_HOST | VPS public IP / DNS | 203.0.113.10 |
| STAGING_USER | SSH user | ubuntu |
| STAGING_SSH_KEY | private key PEM (no passphrase) | multiline |

## Speaker alias API

POST `/alias/{speaker_id}` with JSON `{"alias": "Alice"}` updates
`/data/db/speaker_bank.yaml` (or `LAN_SPEAKER_DB` if overridden).


![demo](docs/demo.gif)

## Release process

Before tagging a new version run the checklist in [docs/release-checklist.md](docs/release-checklist.md).
