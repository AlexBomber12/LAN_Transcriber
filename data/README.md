# Runtime Data Root

`/data` is the runtime state root inside containers. In local development,
`./data` is mounted to `/data` through Docker.

Expected subdirectories:

- `db/` - SQLite database and speaker alias data (`app.db`, `speaker_bank.yaml`)
- `recordings/<recording_id>/` - canonical per-recording artifacts:
  - `raw/audio.<ext>`
  - `derived/transcript.json`
  - `derived/transcript.txt`
  - `derived/segments.json`
  - `derived/speaker_turns.json`
  - `derived/snippets/`
  - `derived/summary.json`
  - `derived/metrics.json`
  - `logs/step-*.log`
- `auth/` - Microsoft delegated auth token cache (`msal_cache.bin`)
- `secrets/` - mounted secret files (for example `gdrive_sa.json`)
- `voices/` - voice samples and profile assets
- `tmp/` - temporary files used during processing

Do not commit secrets or runtime-generated files from this directory.
