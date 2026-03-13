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

Rebuild images only when dependencies change (for example `requirements-cu121.txt` updates).

## GPU setup

- Worker services in `docker-compose.yml` and `docker-compose.dev.yml` use `gpus: all`.
- Install NVIDIA Container Toolkit on the Docker host so GPU devices are visible inside containers.
- Verify GPU passthrough on the host:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### GPU scheduling

- `LAN_ASR_DEVICE` controls WhisperX/faster-whisper placement: `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`.
- `LAN_DIARIZATION_DEVICE` controls pyannote placement with the same values.
- `LAN_GPU_SCHEDULER_MODE` controls overlap policy: `auto`, `sequential`, `parallel`.
- Recommended single-GPU setup: leave all three at their defaults (`LAN_ASR_DEVICE=auto`, `LAN_DIARIZATION_DEVICE=auto`, `LAN_GPU_SCHEDULER_MODE=auto`).
- In the default single-GPU path, ASR stays warm in-process across jobs when the load config is unchanged, while diarization loads lazily for its stage and is not preloaded during precheck.
- When the worker hits GPU memory exhaustion, the recording now lands in `NeedsReview` with review reason `gpu_oom`, and the worker/step logs include the effective ASR device, diarization device, scheduler mode, and any automatic fallback that occurred.

## CUDA runtime troubleshooting

If the worker crashes with `Could not load library libcudnn_ops_infer.so.8`, the runtime has a cuDNN 8/9 mismatch.
The current lockfile stack uses `ctranslate2>=4.5` with CUDA wheels from `https://download.pytorch.org/whl/cu126`.

Quick runtime diagnostic:

```bash
python -c "import ctranslate2, torch; print('ctranslate2', ctranslate2.__version__, 'torch cuda', torch.version.cuda, 'cuda available', torch.cuda.is_available())"
```

## Operations runbook

Operational setup, failure handling, backup/restore, and upgrade steps are documented in
[`docs/runbook.md`](docs/runbook.md).

## Workflow (Upload -> Processing -> Export)

1. Open `/upload` and add one or more audio files.
2. Uploaded audio is normalized automatically to 16 kHz mono WAV before VAD/ASR/diarization (raw upload is preserved; no user conversion needed).
3. Mixed-language handling now happens automatically in `LAN_ASR_MULTILINGUAL_MODE=auto`.
   - `derived/transcript.json` can include chunk-level `language_spans` and multilingual execution metadata.
   - When chunk language ID remains conflicted or low-confidence, the recording stays in `NeedsReview` with an explicit review reason.
4. Track per-file upload progress and processing progress on the same page.
5. Open the recording detail page at `/recordings/{recording_id}`.
   - `NeedsReview` recordings now show an explicit review reason in both the list and detail UI.
   - Automatic worker retries now resume from the first incomplete or invalidated pipeline stage instead of restarting from raw sanitization every time.
   - Explicit `Requeue` from the UI/API still clears saved stage checkpoints and forces a clean rerun from the beginning.
   - `Stop` on a queued recording removes the queued job immediately and marks the recording `Stopped`.
   - `Stop` on a running recording changes the status to `Stopping`; the worker first waits for the current heavy child process to finish cooperatively, then force-terminates it after `LAN_STOP_GRACE_SECONDS` when needed, and finally marks the recording `Stopped`.
   - A stopped recording stays stopped until you explicitly `Requeue` it.
   - Displayed timestamps are rendered in local Europe/Rome time in the server-rendered UI.
   - Plaud-style filename timestamps are interpreted in `UPLOAD_CAPTURE_TIMEZONE` (default `Europe/Rome`), normalized to UTC in the database, and shown back in local time in the UI.
   - Legacy upload rows created before this fix are backfilled automatically on upgrade once, using the configured upload capture timezone.
   - ICS sync now stores organizer and attendee details per event, and the calendars UI renders those event times in local Europe/Rome time.
   - Uploaded/processed recordings are matched automatically to nearby calendar events using the corrected upload capture time when filename provenance is available.
   - Weak or suspicious capture timestamps keep calendar matching conservative; ambiguous candidates remain reviewable and manually overrideable on the recording `calendar` tab.
   - Duration is taken from `derived/audio_sanitized.wav` when present, then falls back to the raw upload.
   - The app now uses glossary/correction memory instead of ASR model training: manage manual terms and future corrections on `/glossary`.
   - Glossary sources are merged deterministically from stored manual/correction entries, canonical speaker names, selected calendar context, and the current project name/keywords when available.
   - Each processed recording writes `derived/asr_glossary.json`, and the overview page shows which glossary terms were actually forwarded to ASR.
6. Export results:
   - Copy markdown from the export tab for manual OneNote paste.
   - Download ZIP from `/ui/recordings/{recording_id}/export.zip`.
   - Export content appears automatically once the recording reaches a terminal state; no manual refresh is needed.
7. Deleting a recording from the UI/API removes the DB row and the recording directory under `/data/recordings/<recording_id>`. If disk cleanup fails, delete returns an error instead of silently succeeding.

## Speaker bank

- `voice_profiles` now represent canonical speakers: one real person should map to one active record.
- `voice_samples` can store many samples per canonical speaker, including provenance and review metadata for low-confidence matches.
- Duplicate speakers can be merged backend-side without losing samples, assignments, or routing references.
- `/voices` is the canonical speaker management page: inspect linked samples, review likely duplicates, and merge duplicate speakers into the surviving canonical record.
- `/recordings/{recording_id}?tab=speakers` lets you remap diarized labels (`S1`, `S2`, ...) to canonical speakers or leave them unmatched; exports render the corrected canonical names as `Name (Sx)`.
- Speaker snippets are now purity-ranked voice samples built from single-speaker material instead of wide playback windows.
- Snippet export now runs immediately after `speaker_turns`, before LLM summarization, so `derived/snippets_manifest.json` and clean clips can appear while long summaries are still processing.
- The Speakers tab now labels snippet state as pending, generating, ready, failed, or legacy/unavailable instead of collapsing empty states into one generic placeholder.
- Add sample now requires an explicit clean snippet choice from the speakers tab; if no clean snippet exists, the UI explains why and blocks the action.
- Synthetic silence fallback was removed on purpose. Inspect `derived/snippets_manifest.json` to see accepted clips, rejected candidates, overlap, and extraction failures.
- When diarization falls back to degraded mode or a speaker match is low confidence, the recording detail page shows an explicit warning instead of silently trusting the labels.

## ASR glossary and correction memory

- `/glossary` stores manual ASR hints and correction-memory entries as a canonical term plus aliases / observed wrong spellings.
- Use `source=manual` for domain terms you always want available and `source=correction` for previously mis-transcribed names such as `canonical=Sander`, `aliases=[Sandia]`.
- The worker builds a bounded per-recording glossary from stored entries, speaker bank names, selected calendar data, and project context, then forwards that context to WhisperX when the active transcribe callable supports `initial_prompt` and/or `hotwords`.
- The resulting per-recording artifact lives at `derived/asr_glossary.json` for inspection and troubleshooting.

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
  derived/audio_sanitized.wav
  derived/audio_sanitize.json
  derived/asr_glossary.json
  derived/transcript.json
  derived/transcript.txt
  derived/segments.json
  derived/snippets/
  derived/snippets_manifest.json
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
| `LAN_VAD_METHOD` | WhisperX VAD selector for ASR model init: `silero` (default) or `pyannote` |
| `LAN_ASR_MULTILINGUAL_MODE` | Mixed-language ASR mode: `auto` (default), `force_single_language`, or `force_multilingual` |
| `LAN_API_BEARER_TOKEN` | Optional bearer token for protected POST actions (`/api` and UI POST routes) |
| `UPLOAD_CAPTURE_TIMEZONE` | Source timezone used to interpret Plaud-style filename timestamps before storing them in UTC (default `Europe/Rome`) |
| `UPLOAD_MAX_BYTES` | Optional max size per uploaded file in bytes (`413` when exceeded) |
| `QUARANTINE_RETENTION_DAYS` | Retention period for quarantined recording cleanup (default `7`) |
| `LAN_API_BIND_HOST` | Published API bind host (default `127.0.0.1`) |
| `LAN_API_PORT` | Published API port (default `7860`) |
| `LAN_DIARIZATION_PROFILE` | Speaker diarization mode: `auto` (default; first pass uses meeting-like 2..6 hints and retries once with 2 speakers only when the result looks dialog-like), `dialog` (forces 2 speakers and bypasses auto selection), or `meeting` (forces meeting-like 2..6 and bypasses auto selection) |
| `LAN_DIARIZATION_MIN_SPEAKERS` | Optional explicit pyannote `min_speakers` override for the first pass; when set, it overrides the profile default and bypasses auto-profile selection |
| `LAN_DIARIZATION_MAX_SPEAKERS` | Optional explicit pyannote `max_speakers` override for the first pass; when set, it overrides the profile default and bypasses auto-profile selection |
| `LAN_DIARIZATION_MERGE_GAP_SECONDS` | Conservative post-processing gap threshold for merging adjacent same-speaker turns (default `0.5`) |
| `LAN_DIARIZATION_MIN_TURN_SECONDS` | Conservative post-processing threshold for absorbing micro-turns between matching neighbors (default `0.5`) |
| `LLM_BASE_URL` | OpenAI-compatible Spark endpoint |
| `LLM_API_KEY` | Optional API key for the LLM |
| `LLM_MODEL` | Required model name passed to the OpenAI-compatible endpoint (no runtime fallback) |
| `LLM_MAX_TOKENS` | Base `max_tokens` for `/v1/chat/completions` requests (default `1024`) |
| `LLM_MAX_TOKENS_RETRY` | One-shot retry `max_tokens` for truncated/empty LLM output (default `2048`) |
| `LLM_TIMEOUT_SECONDS` | Per-request timeout for LLM calls (default `30`) |
| `LLM_CHUNK_MAX_CHARS` | Transcript chunk size threshold for long-recording map-reduce LLM processing (default `4500`) |
| `LLM_CHUNK_OVERLAP_CHARS` | Deterministic overlap between adjacent transcript chunks (default `300`) |
| `LLM_CHUNK_TIMEOUT_SECONDS` | Wall-clock timeout applied to each chunk extraction call (default `120`) |
| `LLM_CHUNK_SPLIT_MIN_CHARS` | Minimum chunk payload size required before timeout-driven child splits are allowed (default `1200`) |
| `LLM_CHUNK_SPLIT_MAX_DEPTH` | Maximum adaptive split depth for a timed-out chunk before the stage fails (default `2`) |
| `LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS` | Switch to chunked LLM mode when the speaker-attributed transcript exceeds this size (default `4500`) |
| `LLM_MERGE_MAX_TOKENS` | Optional `max_tokens` override for the final merge pass; falls back to `LLM_MAX_TOKENS` when unset |

`LAN_ENV` controls startup validation:

- `LAN_ENV=dev`: missing `LAN_REDIS_URL` and/or `LLM_BASE_URL` is allowed with warnings; dev defaults are used (`redis://127.0.0.1:6379/0`, `http://127.0.0.1:8000`).
- `LAN_ENV=staging` or `LAN_ENV=prod`: `LAN_REDIS_URL` and `LLM_BASE_URL` are required; startup fails fast if either is missing.
- All environments: `LLM_MODEL` is required. Startup fails fast when unset/blank with `LLM_MODEL is required. Set it in .env (e.g., LLM_MODEL=gpt-oss:120b).`

If LLM responses fail with `finish_reason=length` or empty `message.content`, increase `LLM_MAX_TOKENS` and `LLM_TIMEOUT_SECONDS` (and optionally `LLM_MAX_TOKENS_RETRY`).

Long transcripts are processed with a chunked map-reduce LLM flow. Before chunk planning, the worker compacts speaker turns into an LLM-ready transcript with shorter speaker labels, merged adjacent same-speaker turns, and chunk-level time ranges instead of per-line timestamps. During that phase the UI may show progress stages like `llm_chunk_1_of_5` and `llm_merge`.

Chunk extraction now resumes from the failed or incomplete chunk set on the next retry instead of restarting from chunk 1. Completed chunk extracts are validated and reused, and a timed-out chunk can be split into smaller child chunks automatically up to `LLM_CHUNK_SPLIT_MAX_DEPTH`.

Chunk debug artifacts are written under `derived/`, including `llm_compact_transcript.txt`, `llm_compact_transcript.json`, `llm_chunks_plan.json`, `llm_merge_input.json`, per-chunk `llm_chunk_*_{raw,extract,error}.json`, and `llm_merge_error.json` when the merge pass fails. The worker also persists per-chunk state in SQLite so automatic job retries can resume safely.

The recording detail page now includes a compact diagnostics block that shows the primary root cause, current or last stage, chunk `N/M`, elapsed stage or chunk time when available, and any retry-wrapper note separately. For example, a long-transcript failure should now surface `llm_chunk_timeout` or `llm_merge_*` directly instead of only a generic retry-limit message, and stopped recordings show whether the stop stayed soft or escalated to a force-stop.

For diarization quality tuning, keep `LAN_DIARIZATION_PROFILE=auto` for mixed workloads. In `auto`, the worker runs a meeting-oriented first pass, classifies the result from deterministic speaker-share/alternation/overlap heuristics, and retries once with `min_speakers=2` and `max_speakers=2` only when the recording looks dialog-like. Use `dialog` or `meeting` only to force one behavior, and note that explicit `LAN_DIARIZATION_MIN_SPEAKERS` / `LAN_DIARIZATION_MAX_SPEAKERS` overrides bypass auto selection. Each processed recording writes `derived/diarization_metadata.json` with the requested profile, selected profile, initial top-two coverage, retry attempt/winner, applied hints, and smoothing stats.

Mixed-language ASR defaults to `LAN_ASR_MULTILINGUAL_MODE=auto`. When the pipeline sees credible language switches, it retranscribes grouped chunks with per-chunk language hints, writes `language_spans` plus multilingual execution metadata into `derived/transcript.json`, and keeps the recording in `NeedsReview` when chunk-level language ID remains conflicted or low-confidence.

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
