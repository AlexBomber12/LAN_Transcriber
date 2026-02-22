# Staging deployment files

This directory contains the docker compose stack used by `.github/workflows/staging-deploy.yml`.

## VPS steps

1. Copy the environment template:

```bash
cp .env.staging.example .env
```

2. Ensure the runtime data directory exists:

```bash
mkdir -p data
```

3. Pull the latest images:

```bash
docker compose pull
```

4. Start or update the stack:

```bash
docker compose up -d --remove-orphans
```

5. Run smoke endpoint checks:

```bash
curl -fsS http://127.0.0.1:7860/healthz/app
curl -fsS http://127.0.0.1:7860/openapi.json >/dev/null
```
