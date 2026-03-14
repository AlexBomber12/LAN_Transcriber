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
3. Set `LLM_MODEL` (required; there is no fallback model).
4. Tune output sizing/timeouts for your model:
   - `LLM_MAX_TOKENS=1024`
   - `LLM_MAX_TOKENS_RETRY=2048`
   - `LLM_TIMEOUT_SECONDS=600` (recommended for larger local models)
   - `LLM_CHUNK_MAX_CHARS=4500`
   - `LLM_CHUNK_OVERLAP_CHARS=300`
   - `LLM_CHUNK_TIMEOUT_SECONDS=120`
   - `LLM_CHUNK_SPLIT_MIN_CHARS=1200`
   - `LLM_CHUNK_SPLIT_MAX_DEPTH=2`
   - `LLM_LONG_TRANSCRIPT_THRESHOLD_CHARS=4500`
   - Optional: `LLM_MERGE_MAX_TOKENS=2048` when the final merge pass needs a larger JSON budget
5. Long transcripts are processed in chunks. Before planning those chunks, the worker compacts the speaker-turn transcript to reduce prompt size while keeping speaker order and chunk-level time ranges. The UI may show `llm_chunk_X_of_Y` followed by `llm_merge`.
6. Stop handling now has two levels:
   - soft stop at the next safe checkpoint
   - hard-stop escalation for the current heavy child process after `LAN_STOP_GRACE_SECONDS` (default `5`)
7. Automatic retries now resume from the failed or incomplete chunk set instead of starting long-transcript processing from chunk 1. Completed chunk extracts are validated before reuse, and a timed-out chunk can split into smaller child chunks automatically when it is still large enough.
8. Debug artifacts for that flow live under `derived/`, including `llm_compact_transcript.txt`, `llm_compact_transcript.json`, `llm_chunks_plan.json`, `llm_merge_input.json`, per-chunk `llm_chunk_*_{raw,extract,error}.json`, and `llm_merge_error.json` when the merge pass fails.
9. If you see `finish_reason=length` or empty `message.content`, increase `LLM_MAX_TOKENS` and `LLM_TIMEOUT_SECONDS` (and optionally `LLM_MERGE_MAX_TOKENS`).
10. Use the recording detail `Diagnostics` block for first-pass triage:
   - primary reason = normalized root cause shown to operators
   - retry wrapper = generic retry-limit context, if present
   - current or last stage + chunk `N/M` + elapsed time = where the pipeline actually got stuck
   - stopped recordings show whether the stop stayed soft or escalated to a force-stop
11. Validate connectivity:

```bash
docker compose run --rm api python -m lan_app.healthchecks app
```

### 1.2 Upload and export flow

1. Open `/` and upload one or more files from the Control Center left pane (fallback: `/upload`). The UI sends multipart uploads to `POST /api/uploads`.
2. Uploaded audio is normalized automatically to 16 kHz mono WAV before VAD/ASR/diarization; no manual conversion is required.
3. Mixed-language handling now runs automatically when `LAN_ASR_MULTILINGUAL_MODE=auto`.
   - `derived/transcript.json` can include `language_spans`, multilingual chunk metadata, and multilingual review metadata.
   - If chunk-level language ID stays conflicted or low-confidence, the recording remains in `NeedsReview` with an explicit review reason.
4. Track upload and processing progress per file from `/`; the Control Center queue updates as new recordings appear and statuses change. `/upload` remains available as a direct fallback.
5. Select the recording on `/` and use the embedded inspector for transcript, summary, routing, speaker review, and export actions.
   - `/recordings/{recording_id}` remains available as a standalone fallback for the same inspector when you need a direct link or full-page view.
   - `NeedsReview` recordings show an explicit review reason in the recordings list and detail page.
   - `Stop` on a queued recording removes the queued job immediately and marks the recording `Stopped`.
   - `Stop` on a running recording changes the status to `Stopping`; the worker first waits for the current heavy child process to exit cooperatively and then force-terminates it after `LAN_STOP_GRACE_SECONDS` when needed, before finalizing the recording as `Stopped`.
   - A stopped recording stays stopped until you explicitly `Requeue` it.
   - The server-rendered UI shows timestamps in local Europe/Rome time.
   - Plaud-style filename timestamps are interpreted in `UPLOAD_CAPTURE_TIMEZONE` (default `Europe/Rome`), then stored in UTC in the database.
   - Legacy upload rows created before this fix are backfilled automatically once after upgrade when the app initializes the database.
   - ICS sync now stores organizer and attendee details per event, and the calendars page renders those event times in local Europe/Rome time.
   - Uploaded and processed recordings are matched automatically to nearby calendar events using the corrected upload capture time when filename provenance is available.
   - Weak or suspicious capture timestamps keep calendar matching conservative; use the recording `calendar` tab to review candidates and manually override the selected event when needed.
   - Duration is sourced from `derived/audio_sanitized.wav` when it exists, otherwise from the raw upload.
   - The app uses glossary/correction memory instead of ASR model training. Manage manual terms and saved corrections on `/glossary`.
   - Glossary sources are merged deterministically from stored manual/correction entries, speaker-bank names, selected calendar context, and project context when available.
   - Each processed recording writes `derived/asr_glossary.json`, and the overview page shows the glossary terms that were actually sent to ASR.
   - Canonical speaker records keep one active person entry with many samples; low-confidence matches stay reviewable instead of auto-merging.
   - Use the `speakers` tab to make an explicit review decision per diarized speaker:
     - `Confirm global match` when the identity should map to a canonical person across recordings.
     - `Keep unknown` when the speaker should stay intentionally unnamed.
     - `Local label only` when the speaker needs a recording-specific name without creating a canonical speaker.
     - `Add sample` only for trusted clean clips; it stays separate from mapping decisions.
   - Exports now show only explicit speaker decisions: confirmed canonical names or local labels render as `Name (Sx)`, while unresolved/system-suggested speakers stay on `Sx`.
   - Speaker snippets are purity-ranked voice samples. Add sample requires selecting an explicit clean clip instead of silently taking the first snippet.
   - Snippet export runs right after `speaker_turns` and before `llm_extract`, so `derived/snippets_manifest.json` and accepted clips can show up while long LLM work is still running.
   - The Speakers tab labels snippet state as pending, generating, ready, failed, or legacy/unavailable so operators can tell whether Add sample should be usable yet.
   - Legacy recordings missing snippet artifacts can be repaired from the Speakers tab without rerunning ASR/LLM. For admin backfill, run `python -m lan_app.tools.repair_snippets --scan-missing`; to repair exactly one recording, run `python -m lan_app.tools.repair_snippets --recording-id <recording_id>`.
   - Silence fallback for failed snippet extraction was removed intentionally. Inspect `derived/snippets_manifest.json` for accepted clips, overlap rejections, degraded-mode blocks, and extraction failures.
   - If diarization fell back to degraded mode or a match stayed low confidence, the speakers tab shows a visible warning badge/message.
6. Download export bundle from `GET /ui/recordings/{recording_id}/export.zip`.
7. Deleting a recording removes the DB row plus `/data/recordings/<recording_id>/raw`, `derived`, `logs`, and other remaining files under that recording root. If cleanup fails, delete returns an error.

### 1.2.2 Glossary and correction-memory workflow

1. Open `/glossary`.
2. Add a canonical term and optional aliases / observed wrong spellings.
3. Use `source=manual` for always-on domain terminology and `source=correction` for future ASR memory.
4. Optional metadata like `Observed in recording` can point back to the recording where you noticed the issue.
5. After processing a recording, inspect `derived/asr_glossary.json` or the recording overview page to confirm the terms that were applied.

### 1.2.1 Canonical speaker merges

1. Treat `/voices` entries as canonical people, not one-off embeddings.
2. Backend merge operations move all samples and references from duplicate speaker A to canonical speaker B.
3. After a merge, routing/training references follow the target canonical speaker automatically.
4. Use `/voices` to review linked samples, inspect duplicate evidence, and run the merge with an explicit target speaker.

### 1.3 Upload size controls

1. Optionally set `UPLOAD_MAX_BYTES` to cap a single uploaded file size.
2. If set, uploads above this limit are rejected with HTTP `413`.
3. Keep app and reverse-proxy limits consistent to avoid mismatched failures.
4. Set `UPLOAD_CAPTURE_TIMEZONE` when the source device names files in a local timezone other than `Europe/Rome`.

### 1.4 Before first start (diarization warmup)

1. Create a Hugging Face token with read scope.
2. Accept gated model terms for `pyannote/speaker-diarization` and any linked models.
3. Set these values in `.env`:
   - `HF_TOKEN=<your token>`
   - `LAN_DIARIZATION_MODEL_ID=pyannote/speaker-diarization-3.1` (or your approved repo id)
   - `LAN_DIARIZATION_PROFILE=auto` (default; first pass uses meeting-like 2..6 hints and retries once with 2 speakers only when the result looks dialog-like)
   - `LAN_VAD_METHOD=silero` (default; set `pyannote` only if explicitly required)
   - `LAN_ASR_MULTILINGUAL_MODE=auto` (default; switches to chunk-level language-aware ASR only when credible language changes are detected)
4. Run warmup once before starting normal traffic:

```bash
docker compose run --rm worker python -m lan_app.tools.warm_models --models diarization
```

Warmup exits with:
- `0`: cache ready
- `2`: token missing
- `3`: gated access not granted
- `4`: revision not found
- `5`: other load errors

Verify in a running worker:

```bash
docker compose exec worker python -c "from lan_app.diarization_loader import load_pyannote_pipeline; print(type(load_pyannote_pipeline()).__name__)"
```

Optional diarization tuning:
- `LAN_DIARIZATION_PROFILE=dialog` forces 2-speaker diarization; `LAN_DIARIZATION_PROFILE=meeting` forces meeting-like 2..6 hints.
- `LAN_DIARIZATION_MIN_SPEAKERS` and `LAN_DIARIZATION_MAX_SPEAKERS` override the first-pass defaults and bypass auto-profile selection when set.
- `LAN_DIARIZATION_MERGE_GAP_SECONDS` and `LAN_DIARIZATION_MIN_TURN_SECONDS` control conservative turn smoothing.
- Each processed recording writes `derived/diarization_metadata.json` with the requested profile, selected profile, initial speaker count, top-two coverage, retry attempt/winner, applied hints, and before/after smoothing counts.
- Mixed-language ASR writes chunk-level `language_spans` and multilingual execution metadata into `derived/transcript.json`; when those chunks stay conflicted or low-confidence, the worker leaves the recording in `NeedsReview`.

## 2) Runtime safety defaults

- Store runtime secrets only under `/data/secrets` or environment variables.
- Do not commit any credentials, key files, or token caches.
- API publish port is loopback-bound by default via:
  - `LAN_API_BIND_HOST=127.0.0.1`
  - `LAN_API_PORT=7860`
- Keep remote access behind SSH tunnel, reverse proxy auth, or a LAN gateway ACL.

## 2.1 GPU scheduler policy

- `LAN_ASR_DEVICE` selects the ASR device: `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`.
- `LAN_DIARIZATION_DEVICE` selects the diarization device with the same values.
- `LAN_GPU_SCHEDULER_MODE` selects execution policy: `auto`, `sequential`, `parallel`.
- Recommended single-GPU LAN server config:
  - `LAN_ASR_DEVICE=auto`
  - `LAN_DIARIZATION_DEVICE=auto`
  - `LAN_GPU_SCHEDULER_MODE=auto`
- In that default mode the worker keeps the ASR model warm across jobs when the model/device/compute-type config is unchanged, but keeps diarization lazy and stage-local to avoid overlapping the two heavy GPU stages on one card.
- Worker step logs now include the effective ASR device, diarization device, scheduler mode, CUDA visibility, and any bounded fallback that was used.
- If the worker still exhausts VRAM, the recording moves to `NeedsReview` with review reason `gpu_oom` instead of burning retries until a generic retry-limit failure.

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
- `NeedsReview` is not a failure; it indicates manual review workflow, and the UI now shows the specific review reason.
- `Failed` is terminal after retry policy is exhausted for that step.
