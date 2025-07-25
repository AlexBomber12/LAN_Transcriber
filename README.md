# LAN Transcriber

Offline transcription pipeline with WhisperX and external language model.

## Staging

The staging environment spins up the application and a tiny LLM model using
`docker compose`. Copy the files in `infra/staging` to your server and run:

```bash
cd ~/lan-staging
docker compose pull
docker compose up -d --build
```

The compose file mounts `/opt/lan_cache/hf` into `/root/.cache/huggingface` so
models are cached across runs.

`docker-compose.yml` expects a `.env` file with the following variables:

| Variable | Description |
| --- | --- |
| `PLAUD_EMAIL` | Login for the Plaud fetcher |
| `PLAUD_PASSWORD` | Password for Plaud fetcher |
| `LLM_API_KEY` | Optional API key for the LLM |

The stack exposes `lan_transcriber_health{env="staging"}` on `/metrics` for
future monitoring.

## Staging deploy secrets

| Secret | Description | Example |
|--------|-------------|---------|
| STAGING_HOST | VPS public IP / DNS | 203.0.113.10 |
| STAGING_USER | SSH user | ubuntu |
| STAGING_SSH_KEY | private key PEM (no passphrase) | multiline |

## Speaker alias API

POST `/alias/{speaker_id}` with JSON `{"alias": "Alice"}` updates `speaker_bank.yaml`. Delete the file to reset aliases.


![demo](docs/demo.gif)

