# LAN Transcriber Runbook

## Scope

This runbook covers day-2 operations for LAN deployment:

- initial setup
- health verification
- common failure recovery
- backup and restore
- upgrade procedure

## 1) Initial setup

### 1.1 Spark LLM (OpenAI-compatible)

1. Set `LLM_BASE_URL` to the Spark-compatible endpoint.
2. Set `LLM_API_KEY` if the endpoint requires auth.
3. Set `LLM_MODEL` if your endpoint requires explicit model name.
4. Validate connectivity:

```bash
docker compose run --rm api python -m lan_app.healthchecks app
```

### 1.2 Google Drive Service Account ingest

1. Place Service Account JSON under `/data/secrets/gdrive_sa.json`.
2. Share the Drive Inbox folder with the Service Account principal.
3. Configure:
   - `GDRIVE_SA_JSON_PATH=/data/secrets/gdrive_sa.json`
   - `GDRIVE_INBOX_FOLDER_ID=<shared-folder-id>`
4. Trigger one ingest cycle:

```bash
curl -fsS -X POST http://127.0.0.1:7860/api/actions/ingest
```

### 1.3 Microsoft Graph delegated auth (Device Code Flow)

1. Register an Entra app with delegated Graph scopes:
   - `offline_access`
   - `User.Read`
   - `Calendars.Read`
   - `Notes.ReadWrite`
2. Configure `MS_TENANT_ID` and `MS_CLIENT_ID`.
3. Open `/connections` and complete device-code auth.
4. Verify:

```bash
curl -fsS http://127.0.0.1:7860/api/connections/ms/verify
```

## 2) Runtime safety defaults

- Store runtime secrets only under `/data/secrets` or environment variables.
- Do not commit any credentials, key files, or token caches.
- API publish port is loopback-bound by default via:
  - `LAN_API_BIND_HOST=127.0.0.1`
  - `LAN_API_PORT=7860`
- Keep remote access behind SSH tunnel, reverse proxy auth, or a LAN gateway ACL.

## 3) Health checks

Component endpoints:

- `GET /healthz`
- `GET /healthz/app`
- `GET /healthz/db`
- `GET /healthz/redis`
- `GET /healthz/worker`

Container checks:

```bash
docker compose ps
docker compose logs --tail=200 api worker redis
```

## 4) Common failures and fixes

### 4.1 `redis unavailable` / queue failures

Symptoms:
- enqueue/retry returns HTTP 503
- `/healthz/redis` fails

Actions:
1. `docker compose restart redis`
2. Confirm with `docker compose exec redis redis-cli ping`
3. Confirm API check with `curl -fsS http://127.0.0.1:7860/healthz/redis`

### 4.2 Worker missing or stale heartbeat

Symptoms:
- `/healthz/worker` returns 503
- jobs remain `queued`

Actions:
1. `docker compose restart worker`
2. Confirm with `curl -fsS http://127.0.0.1:7860/healthz/worker`
3. Open `/queue` and verify `started`/`finished` transitions resume

### 4.3 Microsoft Graph auth expired

Symptoms:
- publish/calendar calls fail with auth errors

Actions:
1. Re-run Device Code Flow in `/connections`
2. If policy requires hard re-auth, remove cache:

```bash
rm -f /data/auth/msal_cache.bin
```

3. Reconnect in `/connections`

### 4.4 Quarantine growth

Symptoms:
- many recordings in `Quarantine`
- disk growth under `/data/recordings`

Actions:
1. Validate cleanup loop runs (API logs).
2. Confirm retention: `QUARANTINE_RETENTION_DAYS` (default `7`).
3. Manually trigger one pass:

```bash
docker compose exec api python -c "from lan_app.ops import run_retention_cleanup; print(run_retention_cleanup())"
```

## 5) Backup and restore (`/data`)

### 5.1 Backup

Stop writes first:

```bash
docker compose stop api worker
```

Create archive:

```bash
tar -C / -czf lan-transcriber-data-$(date +%Y%m%d-%H%M%S).tgz data
```

Restart services:

```bash
docker compose start api worker
```

### 5.2 Restore

1. Stop stack: `docker compose down`
2. Restore archive to host `/data` mount source (`./data` in this repo)
3. Start stack: `docker compose up -d --build`
4. Verify with:
   - `curl -fsS http://127.0.0.1:7860/healthz`
   - check recordings list in UI

## 6) Upgrade steps

1. Pull new code and review `.env.example` diff.
2. Ensure secrets still resolve from `/data/secrets` or env.
3. Build and restart:

```bash
docker compose up -d --build
```

4. Verify:
   - `scripts/ci.sh` on the branch used for release prep
   - `curl -fsS http://127.0.0.1:7860/healthz`
   - enqueue and process one test recording

## 7) Retry and failure operations

- Failed step retries are available in recording detail, `Log` tab, button `Retry step`.
- `NeedsReview` is not a failure; it indicates manual review workflow.
- `Failed` is terminal after retry policy is exhausted for that step.
