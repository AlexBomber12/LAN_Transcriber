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

## Runtime data root

Runtime mutable state must live under `/data` in containers (mounted from `./data` in Docker):

- `/data/db/app.db`
- `/data/db/speaker_bank.yaml`
- `/data/recordings/<recording_id>/...`
- `/data/auth/msal_cache.bin`
- `/data/secrets/gdrive_sa.json`
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
| `LAN_DB_PATH` | SQLite database path (default `/data/db/app.db`) |
| `LAN_REDIS_URL` | Redis endpoint for the RQ queue |
| `LAN_RQ_QUEUE_NAME` | Queue name consumed by the worker |
| `LLM_BASE_URL` | OpenAI-compatible Spark endpoint |
| `LLM_API_KEY` | Optional API key for the LLM |
| `MS_TENANT_ID` | Microsoft Entra tenant ID for delegated Device Code Flow |
| `MS_CLIENT_ID` | Microsoft app registration client ID |
| `MS_SCOPES` | Graph scopes (default: `offline_access User.Read Notes.ReadWrite Calendars.Read`) |

`docker compose up` starts:

- `db` (SQLite migration init)
- `redis` (queue broker)
- `api` (FastAPI backend)
- `worker` (RQ worker)

The stack exposes `lan_transcriber_health{env="staging"}` on `/metrics` for
future monitoring.

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
