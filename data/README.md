# Runtime Data Root

`/data` is the runtime state root inside containers. In local development,
`./data` is mounted to `/data`.

Expected subdirectories:

- `artifacts/` - transcripts, summaries, snippets, and derived outputs
- `msal/` - Microsoft auth token cache files
- `voices/` - voice samples and profile assets
- `db/` - SQLite database files and migrations state
- `logs/` - runtime and job logs

Do not commit secrets or runtime-generated files from this directory.
