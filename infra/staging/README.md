# Staging deployment files

This directory contains the docker compose stack used by `.github/workflows/staging-deploy.yml`.

## VPS steps

1. Copy the environment template:

```bash
cp .env.staging.example .env
```

2. Edit `.env` and set required runtime values:

- `LAN_ENV=staging`
- `LAN_REDIS_URL` (for this compose stack: `redis://redis:6379/0`)
- `LLM_BASE_URL` (your Spark/OpenAI-compatible endpoint)
- `LAN_API_BEARER_TOKEN` when auth is enabled

3. Ensure the runtime data directory exists:

```bash
mkdir -p data
```

4. Pull the latest images:

```bash
docker compose pull
```

5. Start or update the stack:

```bash
docker compose up -d --remove-orphans
```

6. Run smoke endpoint checks:

```bash
curl -fsS http://127.0.0.1:7860/healthz/app
curl -fsS http://127.0.0.1:7860/openapi.json >/dev/null
```
