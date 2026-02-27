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

### 1.2 Upload and export flow

1. Open `/upload` and upload one or more files (UI sends multipart to `POST /api/uploads`).
2. Track upload and processing progress per file on `/upload`.
3. Open `/recordings/{recording_id}` for transcript, summary, and export actions.
4. Download export bundle from `GET /ui/recordings/{recording_id}/export.zip`.

### 1.3 Upload size controls

1. Optionally set `UPLOAD_MAX_BYTES` to cap a single uploaded file size.
2. If set, uploads above this limit are rejected with HTTP `413`.
3. Keep app and reverse-proxy limits consistent to avoid mismatched failures.

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

### 4.3 Upload rejected (`413`)

Symptoms:
- upload fails with HTTP `413 Request Entity Too Large`

Actions:
1. Confirm application limit: `UPLOAD_MAX_BYTES` in `.env` (if set).
2. Confirm reverse-proxy size limit (`client_max_body_size`) allows intended file sizes.
3. Re-test with a file below the configured limits.

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

## 5) Nginx reverse proxy for large uploads

Set size and timeout directives high enough for your expected media files.

Example:

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 1024m;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

Notes:
- `client_max_body_size` gates request body size before traffic reaches the app.
- `proxy_read_timeout` and `proxy_send_timeout` should cover long upload/processing responses.

## 6) Backup and restore (`/data`)

### 6.1 Backup

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

### 6.2 Restore

1. Stop stack: `docker compose down`
2. Restore archive to host `/data` mount source (`./data` in this repo)
3. Start stack: `docker compose up -d --build`
4. Verify with:
   - `curl -fsS http://127.0.0.1:7860/healthz`
   - check recordings list in UI

## 7) Upgrade steps

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

## 8) Retry and failure operations

- Failed step retries are available in recording detail, `Log` tab, button `Retry step`.
- `NeedsReview` is not a failure; it indicates manual review workflow.
- `Failed` is terminal after retry policy is exhausted for that step.
