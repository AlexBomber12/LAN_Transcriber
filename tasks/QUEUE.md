QUEUE

Purpose
- Execute planned PRs in order, without skipping.
- Each PR should be implemented exactly as described in its corresponding tasks file.

Status legend
- TODO: not started
- DOING: in progress
- DONE: merged to main

Queue (in order)

1) PR-BOOTSTRAP-01: Repo bootstrap: tasks system, CI scripts, artifacts, runtime data layout
- Status: DONE
- Tasks file: tasks/PR-BOOTSTRAP-01.md
- Depends on: none

2) PR-REFRACTOR-CORE-01: Refactor existing code into a stable core pipeline library and /data-backed state
- Status: DONE
- Tasks file: tasks/PR-REFRACTOR-CORE-01.md
- Depends on: PR-BOOTSTRAP-01

3) PR-DB-QUEUE-01: Introduce SQLite DB, migrations, job queue, and unified recording/job status model
- Status: DONE
- Tasks file: tasks/PR-DB-QUEUE-01.md
- Depends on: PR-REFRACTOR-CORE-01

4) PR-UI-SHELL-01: Web UI skeleton: dashboard + DB-table recordings list + recording detail shell + connections shell
- Status: DONE
- Tasks file: tasks/PR-UI-SHELL-01.md
- Depends on: PR-DB-QUEUE-01

5) PR-GDRIVE-INGEST-01: Google Drive API ingest using Service Account + shared folder (Inbox)
- Status: DONE
- Tasks file: tasks/PR-GDRIVE-INGEST-01.md
- Depends on: PR-DB-QUEUE-01

6) PR-MS-AUTH-01: Microsoft Graph delegated auth (work) via Device Code Flow + token cache
- Status: DONE
- Tasks file: tasks/PR-MS-AUTH-01.md
- Depends on: PR-UI-SHELL-01

7) PR-CALENDAR-01: Read M365 calendar via Graph and match events to recordings; manual override in UI
- Status: DONE
- Tasks file: tasks/PR-CALENDAR-01.md
- Depends on: PR-MS-AUTH-01 and PR-GDRIVE-INGEST-01

8) PR-PIPELINE-01: STT + diarization + word timestamps + speaker-attributed transcript + speaker snippets
- Status: DONE
- Tasks file: tasks/PR-PIPELINE-01.md
- Depends on: PR-GDRIVE-INGEST-01 and PR-REFRACTOR-CORE-01 and PR-DB-QUEUE-01

9) PR-LANG-01: Multi-language support (2 languages): language spans, dominant language, UI override + reprocess hooks
- Status: DONE
- Tasks file: tasks/PR-LANG-01.md
- Depends on: PR-PIPELINE-01

10) PR-LLM-01: Spark LLM: topic, summary, decisions, actions (owner/deadline), emotional summary, question typing
- Status: DONE
- Tasks file: tasks/PR-LLM-01.md
- Depends on: PR-PIPELINE-01 and PR-LANG-01

11) PR-METRICS-01: Conversation analytics metrics per participant and per meeting + UI rendering + exports
- Status: DONE
- Tasks file: tasks/PR-METRICS-01.md
- Depends on: PR-PIPELINE-01 and PR-LLM-01

12) PR-VOICE-01: Voice profiles + mapping diarization speakers to known people (human-in-the-loop UI)
- Status: DONE
- Tasks file: tasks/PR-VOICE-01.md
- Depends on: PR-PIPELINE-01 and PR-UI-SHELL-01

13) PR-ONENOTE-01: Projects mapping to OneNote sections + Publish to OneNote (work)
- Status: DONE
- Tasks file: tasks/PR-ONENOTE-01.md
- Depends on: PR-MS-AUTH-01 and PR-LLM-01 and PR-METRICS-01

14) PR-ROUTING-01: Project suggestion (calendar + voices + text) with confidence + NeedsReview workflow
- Status: DONE
- Tasks file: tasks/PR-ROUTING-01.md
- Depends on: PR-CALENDAR-01 and PR-VOICE-01 and PR-ONENOTE-01

15) PR-OPS-01: Retention, quarantine cleanup, retries, runbook, and production hardening for LAN deployment
- Status: DONE
- Tasks file: tasks/PR-OPS-01.md
- Depends on: PR-ONENOTE-01 and PR-ROUTING-01

16) PR-JOB-MODEL-01: Single job pipeline model (remove placeholder jobs; restrict requeue/retry)
- Status: DONE
- Tasks file: tasks/PR-JOB-MODEL-01.md
- Depends on: PR-OPS-01

17) PR-ENTRYPOINT-01: Unify entrypoint to lan_app.api:app + worker (Dockerfile CMD, systemd, smoke_test, tests)
- Status: DONE
- Tasks file: tasks/PR-ENTRYPOINT-01.md
- Depends on: PR-JOB-MODEL-01

18) PR-STAGING-01: Fix staging deploy workflow and add infra/staging (compose + env template + real smoke)
- Status: DONE
- Tasks file: tasks/PR-STAGING-01.md
- Depends on: PR-ENTRYPOINT-01

19) PR-SECURITY-01: Optional bearer auth + abuse guards (rate limit, dedupe) for ingest/requeue/delete
- Status: DONE
- Tasks file: tasks/PR-SECURITY-01.md
- Depends on: PR-STAGING-01

20) PR-WORKER-ROBUST-01: Worker robustness: graceful shutdown, timeouts, stuck job recovery, terminal failures
- Status: DONE
- Tasks file: tasks/PR-WORKER-ROBUST-01.md
- Depends on: PR-SECURITY-01

21) PR-PIPELINE-MODULAR-01: Split pipeline.py into testable modules + consolidate utils + robust LLM parsing with schema and raw artifacts
- Status: DONE
- Tasks file: tasks/PR-PIPELINE-MODULAR-01.md
- Depends on: PR-WORKER-ROBUST-01

22) PR-DB-RESILIENCE-01: SQLite resilience: busy timeout, retry-on-locked, migration files, safer connection management
- Status: DONE
- Tasks file: tasks/PR-DB-RESILIENCE-01.md
- Depends on: PR-PIPELINE-MODULAR-01

23) PR-UI-PROGRESS-01: UI feedback: pipeline progress/stage + Connections page real status and Run ingest button
- Status: DONE
- Tasks file: tasks/PR-UI-PROGRESS-01.md
- Depends on: PR-DB-RESILIENCE-01

24) PR-RUNTIME-CONFIG-01: Runtime hardening: fail-fast config for staging/prod + FastAPI lifespan + docs alignment
- Status: DONE
- Tasks file: tasks/PR-RUNTIME-CONFIG-01.md
- Depends on: PR-UI-PROGRESS-01

25) PR-UI-UPLOAD-01: Upload ingest API (multipart) create recordings from UI uploads
- Status: DONE
- Tasks file: tasks/PR-UI-UPLOAD-01.md
- Depends on: PR-RUNTIME-CONFIG-01

26) PR-UI-UPLOAD-02: Upload page UI: multi-file upload + per-file upload progress + processing polling
- Status: DONE
- Tasks file: tasks/PR-UI-UPLOAD-02.md
- Depends on: PR-UI-UPLOAD-01

27) PR-EXPORT-01: Export-only output: OneNote-ready markdown + Download ZIP per recording
- Status: DONE
- Tasks file: tasks/PR-EXPORT-01.md
- Depends on: PR-UI-UPLOAD-02

28) PR-UI-PROGRESS-02: Show pipeline progress on Recordings list table
- Status: DONE
- Tasks file: tasks/PR-UI-PROGRESS-02.md
- Depends on: PR-EXPORT-01

29) PR-REMOVE-MS-01: Remove Microsoft Graph, calendar matching UI, OneNote publish UI and msal dependency
- Status: DONE
- Tasks file: tasks/PR-REMOVE-MS-01.md
- Depends on: PR-UI-PROGRESS-02

30) PR-REMOVE-GDRIVE-01: Remove Google Drive ingest, Connections page, ingest lock, and Google API deps
- Status: DONE
- Tasks file: tasks/PR-REMOVE-GDRIVE-01.md
- Depends on: PR-REMOVE-MS-01

31) PR-DOCS-EXPORT-ONLY-01: Docs update for upload + export-only mode (README, runbook, env example, nginx notes)
- Status: DONE
- Tasks file: tasks/PR-DOCS-EXPORT-ONLY-01.md
- Depends on: PR-REMOVE-GDRIVE-01

32) PR-FIX-DIARIZATION-REVISION-01: Fix pyannote diarization revision handling and fallback (avoid failed recordings)
- Status: DONE
- Tasks file: tasks/PR-FIX-DIARIZATION-REVISION-01.md
- Depends on: PR-DOCS-EXPORT-ONLY-01

33) PR-FIX-WHISPERX-API-01: Fix WhisperX API usage (no whisperx.transcribe) and add modern-path unit test
- Status: DONE
- Tasks file: tasks/PR-FIX-WHISPERX-API-01.md
- Depends on: PR-FIX-DIARIZATION-REVISION-01

34) PR-CI-01: Fix GitHub Actions failures (docker smoke pytest, staging deploy secrets) and remove duplicate CI workflow
- Status: DONE
- Tasks file: tasks/PR-CI-01.md
- Depends on: PR-FIX-WHISPERX-API-01

35) PR-FIX-CTRANSLATE2-EXECSTACK-01: Fix ctranslate2 executable-stack loader failure
- Status: DONE
- Tasks file: tasks/PR-FIX-CTRANSLATE2-EXECSTACK-01.md
- Depends on: PR-FIX-WHISPERX-API-01

36) PR-DEV-01: Add docker-compose.dev.yml for fast iteration without rebuilding images
- Status: DONE
- Tasks file: tasks/PR-DEV-01.md
- Depends on: PR-FIX-CTRANSLATE2-EXECSTACK-01

37) PR-TEST-IMPORTS-01: Expand import smoke tests to cover critical modules
- Status: DONE
- Tasks file: tasks/PR-TEST-IMPORTS-01.md
- Depends on: PR-DEV-01

38) PR-TEST-PIPELINE-RESILIENCE-01: Add worker-level resilience tests for external dependency failures
- Status: DONE
- Tasks file: tasks/PR-TEST-PIPELINE-RESILIENCE-01.md
- Depends on: PR-TEST-IMPORTS-01

39) PR-TEST-API-UI-01: Add end-to-end API and UI route tests for Upload and Export workflow
- Status: DONE
- Tasks file: tasks/PR-TEST-API-UI-01.md
- Depends on: PR-TEST-PIPELINE-RESILIENCE-01

40) PR-SMOKE-DOCKER-01: Strengthen Docker smoke test to catch native dependency loader issues (ctranslate2)
- Status: DONE
- Tasks file: tasks/PR-SMOKE-DOCKER-01.md
- Depends on: PR-TEST-API-UI-01

41) PR-COVERAGE-01: Expand coverage to include lan_app and enforce coverage thresholds
- Status: DONE
- Tasks file: tasks/PR-COVERAGE-01.md
- Depends on: PR-SMOKE-DOCKER-01

42) PR-FIX-OMEGACONF-01: Add missing omegaconf dependency (runtime + CI) and a dependency smoke test
- Status: DONE
- Tasks file: tasks/PR-FIX-OMEGACONF-01.md
- Depends on: PR-COVERAGE-01

43) PR-FIX-CTRANSLATE2-EXECSTACK-02: Ensure runtime images are patched (no execstack) and Docker smoke runs on built image digest (no tag races)
- Status: DONE
- Tasks file: tasks/PR-FIX-CTRANSLATE2-EXECSTACK-02.md
- Depends on: PR-FIX-OMEGACONF-01

44) PR-UI-DELETE-CONFIRM-01: Simplify delete confirmation dialog (Yes/No, remove typing DELETE)
- Status: DONE
- Tasks file: tasks/PR-UI-DELETE-CONFIRM-01.md
- Depends on: PR-FIX-CTRANSLATE2-EXECSTACK-02

45) PR-CALENDAR-ICS-01: Calendar import via ICS (URL subscription and optional file upload)
- Status: DONE
- Tasks file: tasks/PR-CALENDAR-ICS-01.md
- Depends on: PR-UI-DELETE-CONFIRM-01

46) PR-FIX-PYANNOTE-USEAUTH-01: Fix whisperx VAD crash with pyannote (ignore unsupported use_auth_token)
- Status: DONE
- Tasks file: tasks/PR-FIX-PYANNOTE-USEAUTH-01.md
- Depends on: PR-FIX-WHISPERX-API-01

47) PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-01: Make WhisperX transcribe calls signature-compatible (avoid unexpected kwargs like vad_filter)
- Status: DONE
- Tasks file: tasks/PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-01.md
- Depends on: PR-RUNTIME-CONFIG-01

48) PR-TEST-WHISPERX-CONTRACT-01: Add contract tests for WhisperX transcribe API drift (with or without vad_filter)
- Status: DONE
- Tasks file: tasks/PR-TEST-WHISPERX-CONTRACT-01.md
- Depends on: PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-01

49) PR-DEPS-WHISPERX-DETERMINISM-01: Make WhisperX stack deterministic (single resolver pass, pinned versions, pip check) to reduce API drift
- Status: DONE
- Tasks file: tasks/PR-DEPS-WHISPERX-DETERMINISM-01.md
- Depends on: PR-TEST-WHISPERX-CONTRACT-01

50) PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-02: Robustly drop unsupported transcribe kwargs even when signature is uninspectable (fix vad_filter crash)
- Status: DONE
- Tasks file: tasks/PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-02.md
- Depends on: PR-DEPS-WHISPERX-DETERMINISM-01

51) PR-FIX-NONETYPE-CALLABLE-01: Fix startup/runtime crash: 'NoneType' object is not callable + regression test
- Status: DONE
- Tasks file: tasks/PR-FIX-NONETYPE-CALLABLE-01.md
- Depends on: PR-FIX-WHISPERX-TRANSCRIBE-KWARGS-02

52) PR-COVERAGE-INFRA-02: Coverage foundation for 100%: branch coverage, unified local/CI command, htmlcov artifact
- Status: DONE
- Tasks file: tasks/PR-COVERAGE-INFRA-02.md
- Depends on: PR-FIX-NONETYPE-CALLABLE-01

53) PR-COV-LAN_APP-HEALTH-WORKER-01: Raise small lan_app modules to 100% (healthchecks/auth/worker/db_init/uploads/hf_repo)
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_APP-HEALTH-WORKER-01.md
- Depends on: PR-COVERAGE-INFRA-02

54) PR-COV-LAN_APP-CALENDAR-01: Raise calendar ICS parsing + service matching to 100%
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_APP-CALENDAR-01.md
- Depends on: PR-COV-LAN_APP-HEALTH-WORKER-01

55) PR-COV-LAN_APP-EXPORT-OPS-JOBS-01: Raise exporter/jobs/ops/reaper to 100%
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_APP-EXPORT-OPS-JOBS-01.md
- Depends on: PR-COV-LAN_APP-CALENDAR-01

56) PR-COV-LAN_APP-METRICS-ROUTING-01: Raise conversation_metrics and routing to 100%
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_APP-METRICS-ROUTING-01.md
- Depends on: PR-COV-LAN_APP-EXPORT-OPS-JOBS-01

57) PR-COV-LAN_APP-DB-01: Raise lan_app.db to 100% statement and branch coverage
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_APP-DB-01.md
- Depends on: PR-COV-LAN_APP-METRICS-ROUTING-01

58) PR-COV-LAN_APP-API-01: Raise lan_app.api to 100% statement and branch coverage
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_APP-API-01.md
- Depends on: PR-COV-LAN_APP-DB-01

59) PR-COV-LAN_APP-UI-01: Raise lan_app.ui and lan_app.ui_routes to 100% statement and branch coverage
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_APP-UI-01.md
- Depends on: PR-COV-LAN_APP-API-01

60) PR-COV-LAN_TRANSCRIBER-01: Raise lan_transcriber to 100% statement and branch coverage (offline)
- Status: DONE
- Tasks file: tasks/PR-COV-LAN_TRANSCRIBER-01.md
- Depends on: PR-COV-LAN_APP-UI-01

61) PR-COVERAGE-ENFORCE-100-01: Enforce 100% statement and branch coverage in CI
- Status: DONE
- Tasks file: tasks/PR-COVERAGE-ENFORCE-100-01.md
- Depends on: PR-COV-LAN_TRANSCRIBER-01

62) PR-DIARIZATION-ROBUST-02: Robust diarization load, persistent HF cache, warmup command, and pre-start checklist
- Status: DONE
- Tasks file: tasks/PR-DIARIZATION-ROBUST-02.md
- Depends on: PR-COVERAGE-ENFORCE-100-01

63) PR-DIARIZATION-ROBUST-03: Robust diarization load (pyannote v3 repo id), persistent HF cache, warmup command, matplotlib fix, and pre-start checklist
- Status: DONE
- Tasks file: tasks/PR-DIARIZATION-ROBUST-03.md
- Depends on: PR-COVERAGE-ENFORCE-100-01

64) PR-E2E-LITE-01: CI-friendly E2E-lite processing test for a real audio file path (no network, no heavy models)
- Status: DONE
- Tasks file: tasks/PR-E2E-LITE-01.md
- Depends on: PR-DIARIZATION-ROBUST-02

65) PR-CI-DOCKER-SMOKE-PR-01: Run Docker runtime-lite build + docker smoke test on pull_request to catch runtime/native dependency regressions early
- Status: DONE
- Tasks file: tasks/PR-CI-DOCKER-SMOKE-PR-01.md
- Depends on: PR-E2E-LITE-01

66) PR-TEST-RQ-REDIS-INTEGRATION-01: Add Redis+RQ integration test (SimpleWorker burst) to validate enqueue + worker lifecycle in CI
- Status: DONE
- Tasks file: tasks/PR-TEST-RQ-REDIS-INTEGRATION-01.md
- Depends on: PR-CI-DOCKER-SMOKE-PR-01

67) PR-UI-E2E-PLAYWRIGHT-01: Add Playwright UI smoke tests (Upload -> Recordings -> Export ZIP) and run in CI as a separate job
- Status: DONE
- Tasks file: tasks/PR-UI-E2E-PLAYWRIGHT-01.md
- Depends on: PR-TEST-RQ-REDIS-INTEGRATION-01

68) PR-CI-DEPS-STABLE-01: Stable CI lockfiles (pip-compile), CUDA-only runtime, add matplotlib, and strengthen docker smoke imports
- Status: DONE
- Tasks file: tasks/PR-CI-DEPS-STABLE-01.md
- Depends on: PR-COVERAGE-ENFORCE-100-01

69) PR-CUDA-CUDNN9-ALIGN-01: Align CUDA/cuDNN stack (switch to cu124 wheels + ctranslate2>=4.5) to fix missing libcudnn_ops_infer.so.8
- Status: DONE
- Tasks file: tasks/PR-CUDA-CUDNN9-ALIGN-01.md
- Depends on: PR-CI-DEPS-STABLE-01

70) PR-FIX-TORCH-SAFEGLOBALS-01: Fix torch weights_only load crash for pyannote VAD (allowlist OmegaConf ListConfig/DictConfig) + regression tests
- Status: DONE
- Tasks file: tasks/PR-FIX-TORCH-SAFEGLOBALS-01.md
- Depends on: PR-CUDA-CUDNN9-ALIGN-01

71) PR-FIX-TORCH-SAFEGLOBALS-02: Fix torch weights_only load crash for pyannote VAD (allowlist OmegaConf ContainerMetadata + scoped safe_globals + auto-retry) + regression tests
- Status: DONE
- Tasks file: tasks/PR-FIX-TORCH-SAFEGLOBALS-02.md
- Depends on: PR-FIX-TORCH-SAFEGLOBALS-01

72) PR-FIX-TORCH-SAFEGLOBALS-03: Make torch safe-globals allowlisting robust across torch API variants (list vs dict), include OmegaConf ContainerMetadata, and add scoped retry for WhisperX VAD checkpoint load
- Status: DONE
- Tasks file: tasks/PR-FIX-TORCH-SAFEGLOBALS-03.md
- Depends on: PR-FIX-TORCH-SAFEGLOBALS-02

73) PR-VAD-SILERO-01: Switch WhisperX VAD to Silero by default to remove Pyannote VAD checkpoint load failures (torch weights_only / OmegaConf) + tests
- Status: DONE
- Tasks file: tasks/PR-VAD-SILERO-01.md
- Depends on: PR-FIX-TORCH-SAFEGLOBALS-03

74) PR-VAD-SILERO-02: Fix Silero VAD integration with WhisperX (use vad_method, ensure vad_model is callable) + regression tests
- Status: DONE
- Tasks file: tasks/PR-VAD-SILERO-02.md
- Depends on: PR-VAD-SILERO-01

75) PR-FIX-SENTIMENT-GPU-01: Fix sentiment pipeline token-length crash (truncate to model max length) + enable GPU for worker in docker-compose
- Status: DONE
- Tasks file: tasks/PR-FIX-SENTIMENT-GPU-01.md
- Depends on: PR-VAD-SILERO-02

76) PR-LLM-ROBUST-01: LLM robustness: configurable max_tokens, fail-fast on empty content, and 1 retry on finish_reason=length (Ollama /v1/chat/completions)
- Status: DONE
- Tasks file: tasks/PR-LLM-ROBUST-01.md
- Depends on: PR-FIX-SENTIMENT-GPU-01

77) PR-LLM-DIAGNOSTICS-01: LLM diagnostics: robust URL join for /v1/chat/completions + actionable HTTP error bodies/logs (no prompt leakage)
- Status: DONE
- Tasks file: tasks/PR-LLM-DIAGNOSTICS-01.md
- Depends on: PR-LLM-ROBUST-01

78) PR-FIX-LLM-MODEL-DEFAULT-01: Remove hardcoded LLM model default (llama3:8b); require LLM_MODEL env and fail fast with clear error + update tests
- Status: DONE
- Tasks file: tasks/PR-FIX-LLM-MODEL-DEFAULT-01.md
- Depends on: PR-LLM-DIAGNOSTICS-01

79) PR-FIX-DIARIZATION-WEIGHTS-ONLY-01: Fix pyannote diarization init under torch weights_only (scoped safe_globals for omegaconf/pyannote + auto-retry) and prevent fallback-to-single-speaker regressions
- Status: DONE
- Tasks file: tasks/PR-FIX-DIARIZATION-WEIGHTS-ONLY-01.md
- Depends on: PR-FIX-LLM-MODEL-DEFAULT-01

80) PR-AUDIO-SANITIZE-FFMPEG-01: Sanitize uploaded audio with ffmpeg to normalized WAV before VAD/ASR/diarization to avoid broken MP3 decode hangs and partial outputs
- Status: DONE
- Tasks file: tasks/PR-AUDIO-SANITIZE-FFMPEG-01.md
- Depends on: PR-FIX-DIARIZATION-WEIGHTS-ONLY-01

81) PR-DIARIZATION-GPU-01: Move pyannote diarization pipeline to CUDA when available, log effective device, and add regression tests
- Status: DONE
- Tasks file: tasks/PR-DIARIZATION-GPU-01.md
- Depends on: PR-AUDIO-SANITIZE-FFMPEG-01

82) PR-LLM-CHUNKING-01: Chunk long transcripts for LLM (map-reduce summaries, bounded per-chunk timeout, chunk/merge progress, deterministic merge)
- Status: DONE
- Tasks file: tasks/PR-LLM-CHUNKING-01.md
- Depends on: PR-DIARIZATION-GPU-01

83) PR-DIARIZATION-QUALITY-01: Improve speaker quality with diarization hints, dialog retry (2 speakers), and turn smoothing/post-processing
- Status: DONE
- Tasks file: tasks/PR-DIARIZATION-QUALITY-01.md
- Depends on: PR-LLM-CHUNKING-01

84) PR-UI-PIPELINE-UX-01: Pipeline UX polish: explicit review reasons, duration from sanitized WAV, better progress model, timezone fix, auto-refresh on terminal status, and delete removes disk artifacts
- Status: DONE
- Tasks file: tasks/PR-UI-PIPELINE-UX-01.md
- Depends on: PR-DIARIZATION-QUALITY-01

85) PR-DIARIZATION-AUTO-PROFILE-01: Auto-select diarization profile (dialog vs meeting), retry 2-speaker dialogs, and persist profile-selection metadata
- Status: DONE
- Tasks file: tasks/PR-DIARIZATION-AUTO-PROFILE-01.md
- Depends on: PR-DIARIZATION-QUALITY-01

86) PR-SPEAKER-BANK-CANONICAL-01: Canonical speaker bank backend: one person = one record, many embeddings per speaker, one-to-one assignment, duplicate merge operations
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-BANK-CANONICAL-01.md
- Depends on: PR-DIARIZATION-AUTO-PROFILE-01

87) PR-ASR-MULTILINGUAL-01: Mixed-language ASR pipeline: segment-level language ID, per-chunk language hints, language spans artifacts, and review flags for uncertain multilingual chunks
- Status: DONE
- Tasks file: tasks/PR-ASR-MULTILINGUAL-01.md
- Depends on: PR-SPEAKER-BANK-CANONICAL-01

88) PR-SPEAKER-UX-01: Speaker UX: canonical speaker management, duplicate merge UI, per-recording remap UI, degraded diarization visibility
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-UX-01.md
- Depends on: PR-ASR-MULTILINGUAL-01

89) PR-SNIPPETS-PURITY-01: Pure speaker snippets: single-speaker clip selection, snippet manifest, no silence fallback, and explicit snippet choice for Add sample
- Status: DONE
- Tasks file: tasks/PR-SNIPPETS-PURITY-01.md
- Depends on: PR-SPEAKER-UX-01 and PR-SPEAKER-BANK-CANONICAL-01

90) PR-GLOSSARY-CORRECTIONS-01: Multi-source ASR glossary and correction memory: manual glossary UI, speaker/calendar seeds, safe ASR prompt injection, and per-recording glossary artifacts
- Status: DONE
- Tasks file: tasks/PR-GLOSSARY-CORRECTIONS-01.md
- Depends on: PR-SNIPPETS-PURITY-01 and PR-ASR-MULTILINGUAL-01 and PR-CALENDAR-ICS-01

91) PR-CALENDAR-MATCHING-ICS-01: Reliable ICS calendar matching: attendee parsing, sync boundary fixes, runtime calendar_matches population, manual override UI, and downstream summary/glossary context
- Status: DONE
- Tasks file: tasks/PR-CALENDAR-MATCHING-ICS-01.md
- Depends on: PR-GLOSSARY-CORRECTIONS-01 and PR-CALENDAR-ICS-01

92) PR-GPU-SCHEDULER-01: GPU scheduling foundation: lazy diarization, sequential single-GPU execution, warm ASR cache, GPU OOM review reason, and HH:MM:SS duration formatting
- Status: TODO
- Tasks file: tasks/PR-GPU-SCHEDULER-01.md
- Depends on: PR-CALENDAR-MATCHING-ICS-01 and PR-DIARIZATION-GPU-01 and PR-UI-PIPELINE-UX-01
