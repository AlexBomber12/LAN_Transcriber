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

`docker-compose.yml` expects a `.env` file with the following variables:

| Variable | Description |
| --- | --- |
| `PLAUD_EMAIL` | Login for the Plaud fetcher |
| `PLAUD_PASSWORD` | Password for Plaud fetcher |
| `LLM_API_KEY` | Optional API key for the LLM |

The stack exposes `lan_transcriber_health{env="staging"}` on `/metrics` for
future monitoring.
