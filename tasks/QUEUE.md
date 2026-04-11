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

90) PR-GLOSSARY-CORRECTIONS-01: Multi-source ASR glossary and correction memory: manual glossary UI, speaker/calendar seeds, safe ASR prompt injection, and per-recording glossary artifacts
- Status: DONE
- Tasks file: tasks/PR-GLOSSARY-CORRECTIONS-01.md
- Depends on: PR-SNIPPETS-PURITY-01 and PR-ASR-MULTILINGUAL-01 and PR-CALENDAR-ICS-01

91) PR-CALENDAR-MATCHING-ICS-01: Reliable ICS calendar matching: attendee parsing, sync boundary fixes, runtime calendar_matches population, manual override UI, and downstream summary/glossary context
- Status: DONE
- Tasks file: tasks/PR-CALENDAR-MATCHING-ICS-01.md
- Depends on: PR-GLOSSARY-CORRECTIONS-01 and PR-CALENDAR-ICS-01

92) PR-GPU-SCHEDULER-01: GPU scheduling foundation: lazy diarization, sequential single-GPU execution, warm ASR cache, GPU OOM review reason, and HH:MM:SS duration formatting
- Status: DONE
- Tasks file: tasks/PR-GPU-SCHEDULER-01.md
- Depends on: PR-CALENDAR-MATCHING-ICS-01 and PR-DIARIZATION-GPU-01 and PR-UI-PIPELINE-UX-01

93) PR-CAPTURE-TIME-SEMANTICS-01: Correct upload/Plaud capture-time semantics: local source time -> UTC normalization, provenance fields, and safe legacy backfill
- Status: DONE
- Tasks file: tasks/PR-CAPTURE-TIME-SEMANTICS-01.md
- Depends on: PR-GPU-SCHEDULER-01 and PR-CALENDAR-MATCHING-ICS-01

94) PR-PIPELINE-CHECKPOINTS-RESUME-01: Durable pipeline stage checkpoints and resume from first incomplete/invalid stage under the single-job model
- Status: DONE
- Tasks file: tasks/PR-PIPELINE-CHECKPOINTS-RESUME-01.md
- Depends on: PR-CAPTURE-TIME-SEMANTICS-01

95) PR-LLM-CHUNK-COMPACTION-01: Compact long-transcript LLM input: merged speaker blocks, no per-line timestamps, compact speaker labels, and richer chunk plan metadata
- Status: DONE
- Tasks file: tasks/PR-LLM-CHUNK-COMPACTION-01.md
- Depends on: PR-PIPELINE-CHECKPOINTS-RESUME-01 and PR-LLM-CHUNKING-01

96) PR-LLM-CHUNK-RESUME-TIMEOUTS-01: Chunk-level state, resume, adaptive split-on-timeout, and specific chunk failure reasons for long-transcript LLM processing
- Status: DONE
- Tasks file: tasks/PR-LLM-CHUNK-RESUME-TIMEOUTS-01.md
- Depends on: PR-LLM-CHUNK-COMPACTION-01 and PR-PIPELINE-CHECKPOINTS-RESUME-01

97) PR-STOP-CANCEL-01: Soft Stop from UI: stop button, durable cancel requests, queued-job cancel, and worker cooperative checkpoints
- Status: DONE
- Tasks file: tasks/PR-STOP-CANCEL-01.md
- Depends on: PR-LLM-CHUNK-RESUME-TIMEOUTS-01

98) PR-STOP-CANCEL-HARD-01: Hard-stop escalation for long-running heavy stages via child processes and bounded termination grace period
- Status: DONE
- Tasks file: tasks/PR-STOP-CANCEL-HARD-01.md
- Depends on: PR-STOP-CANCEL-01

99) PR-OBSERVABILITY-ROOT-CAUSE-01: Root-cause-first UI/log diagnostics: current stage, chunk N/M, primary error reason, and stop visibility
- Status: DONE
- Tasks file: tasks/PR-OBSERVABILITY-ROOT-CAUSE-01.md
- Depends on: PR-STOP-CANCEL-HARD-01

100) PR-CALENDAR-MATCHING-STABILIZATION-01: Stabilize calendar matching after corrected capture times: safer scoring, ambiguity handling, and clearer operator warnings
- Status: DONE
- Tasks file: tasks/PR-CALENDAR-MATCHING-STABILIZATION-01.md
- Depends on: PR-CAPTURE-TIME-SEMANTICS-01 and PR-OBSERVABILITY-ROOT-CAUSE-01

101) PR-SNIPPETS-STAGE-REORDER-01: Move snippet generation to its own earlier pipeline stage right after speaker_turns so snippets are available during long LLM processing
- Status: DONE
- Tasks file: tasks/PR-SNIPPETS-STAGE-REORDER-01.md
- Depends on: PR-SNIPPETS-PURITY-01 and PR-PIPELINE-CHECKPOINTS-RESUME-01 and PR-OBSERVABILITY-ROOT-CAUSE-01

102) PR-SNIPPETS-UI-STATE-01: Improve snippet UI states during processing (not started/running/failed/ready) and make Speakers tab truthful before terminal completion
- Status: DONE
- Tasks file: tasks/PR-SNIPPETS-UI-STATE-01.md
- Depends on: PR-SNIPPETS-STAGE-REORDER-01 and PR-OBSERVABILITY-ROOT-CAUSE-01 and PR-SPEAKER-UX-01

103) PR-SNIPPETS-REPAIR-BACKFILL-01: Add repair/backfill paths to regenerate missing snippets for completed recordings and support lazy/admin re-generation
- Status: DONE
- Tasks file: tasks/PR-SNIPPETS-REPAIR-BACKFILL-01.md
- Depends on: PR-SNIPPETS-STAGE-REORDER-01 and PR-SNIPPETS-UI-STATE-01

104) PR-UI-CONTROL-CENTER-01: Extract reusable UI partials and HTMX-friendly render paths for the future 1-page Control Center workflow
- Status: DONE
- Tasks file: tasks/PR-UI-CONTROL-CENTER-01.md
- Depends on: PR-SNIPPETS-REPAIR-BACKFILL-01 and PR-GLOSSARY-CORRECTIONS-01 and PR-SPEAKER-UX-01

105) PR-UI-CONTROL-CENTER-02: Replace the passive dashboard landing page with a real Control Center shell on /
- Status: DONE
- Tasks file: tasks/PR-UI-CONTROL-CENTER-02.md
- Depends on: PR-UI-CONTROL-CENTER-01

106) PR-UI-CONTROL-CENTER-03: Unify Upload + status filters + live Recordings list in the Control Center left pane
- Status: DONE
- Tasks file: tasks/PR-UI-CONTROL-CENTER-03.md
- Depends on: PR-UI-CONTROL-CENTER-02 and PR-UI-UPLOAD-02 and PR-UI-PROGRESS-02

107) PR-UI-CONTROL-CENTER-04: Embed the recording inspector into the Control Center right pane while keeping /recordings/{id} as a full-page fallback
- Status: DONE
- Tasks file: tasks/PR-UI-CONTROL-CENTER-04.md
- Depends on: PR-UI-CONTROL-CENTER-03

108) PR-SPEAKER-REVIEW-01: Add explicit speaker decisions: confirm canonical match, keep unknown, local label only, and separate trusted-sample training
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-REVIEW-01.md
- Depends on: PR-UI-CONTROL-CENTER-04 and PR-SPEAKER-UX-01 and PR-SPEAKER-BANK-CANONICAL-01 and PR-SNIPPETS-UI-STATE-01

109) PR-CORRECTIONS-UX-01: Redesign Glossary into a simpler Corrections / ASR Memory operator UX with quick-entry from recordings
- Status: DONE
- Tasks file: tasks/PR-CORRECTIONS-UX-01.md
- Depends on: PR-UI-CONTROL-CENTER-04 and PR-GLOSSARY-CORRECTIONS-01

110) PR-UI-WORKFLOW-01: Final 1-page workflow polish, navigation cleanup, and smoke coverage centered on the Control Center
- Status: DONE
- Tasks file: tasks/PR-UI-WORKFLOW-01.md
- Depends on: PR-SPEAKER-REVIEW-01 and PR-CORRECTIONS-UX-01

111) PR-UI-CONTROL-CENTER-RECOVERY-01: Rebuild the broken Control Center into a minimal 2-column daily workspace and remove duplicate dashboard/fallback clutter
- Status: DONE
- Tasks file: tasks/PR-UI-CONTROL-CENTER-RECOVERY-01.md
- Depends on: PR-UI-WORKFLOW-01 and PR-SPEAKER-REVIEW-01 and PR-CORRECTIONS-UX-01

112) PR-UI-STITCH-CONTROL-CENTER-01: Implement the approved Google Stitch "Control Center" design as a compact daily workspace plus a separate full-page recording inspector
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-CONTROL-CENTER-01.md
- Depends on: PR-UI-CONTROL-CENTER-RECOVERY-01

113) PR-UI-STITCH-CONTROL-CENTER-02: Rebuild the Control Center shell into a compact operator workspace and move DGX/GPU visibility into the bottom system-bar foundation
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-CONTROL-CENTER-02.md
- Depends on: PR-UI-STITCH-CONTROL-CENTER-01

114) PR-OPS-SYSTEM-BAR-01: Add real DGX Spark and GPU runtime status to the persistent bottom system bar
- Status: DONE
- Tasks file: tasks/PR-OPS-SYSTEM-BAR-01.md
- Depends on: PR-UI-STITCH-CONTROL-CENTER-02 and PR-GPU-SCHEDULER-01 and PR-DIARIZATION-GPU-01 and PR-FIX-SENTIMENT-GPU-01

115) PR-UI-STITCH-WORKLIST-01: Refine the Control Center left worklist, replace Confidence with Progress, and remove bulky filter and status clutter
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-WORKLIST-01.md
- Depends on: PR-UI-STITCH-CONTROL-CENTER-02 and PR-OPS-SYSTEM-BAR-01

116) PR-UI-STITCH-INSPECTOR-COMPACT-01: Build a compact daily inspector for Control Center with next-action-focused overview and no deep-detail overload
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-INSPECTOR-COMPACT-01.md
- Depends on: PR-UI-STITCH-WORKLIST-01

117) PR-SPEAKER-DECISIONS-02: Harden speaker review state model for confirm match, keep unknown, local label only, and separate trusted-sample actions
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-DECISIONS-02.md
- Depends on: PR-UI-STITCH-INSPECTOR-COMPACT-01 and PR-SPEAKER-REVIEW-01

118) PR-SPEAKER-SNIPPET-UX-02: Redesign Speakers into a snippet-first review workspace in compact and full-page modes
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-SNIPPET-UX-02.md
- Depends on: PR-SPEAKER-DECISIONS-02 and PR-SNIPPETS-UI-STATE-01 and PR-SNIPPETS-REPAIR-BACKFILL-01

119) PR-INSPECTOR-FULL-PAGE-02: Rebuild the full-page recording inspector around Overview, Speakers, Transcript, Summary, Diagnostics, and Export
- Status: DONE
- Tasks file: tasks/PR-INSPECTOR-FULL-PAGE-02.md
- Depends on: PR-SPEAKER-SNIPPET-UX-02 and PR-UI-STITCH-INSPECTOR-COMPACT-01

120) PR-UI-STITCH-POLISH-02: Final Stitch-aligned cleanup, consistency polish, and workflow test hardening for the operator-centric UI
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-POLISH-02.md
- Depends on: PR-INSPECTOR-FULL-PAGE-02 and PR-OPS-SYSTEM-BAR-01

121) PR-UI-STITCH-REDUCTION-01: Reduce Control Center clutter to a strict operator contract and keep deep detail full-page only
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-REDUCTION-01.md
- Depends on: PR-UI-STITCH-POLISH-02

122) PR-OPS-SYSTEM-BAR-02: Simplify the bottom system bar to a compact Node/GPU/LLM row and make GPU runtime status truthful
- Status: DONE
- Tasks file: tasks/PR-OPS-SYSTEM-BAR-02.md
- Depends on: PR-UI-STITCH-REDUCTION-01 and PR-OPS-SYSTEM-BAR-01

123) PR-UI-STITCH-WORKLIST-02: Flatten the Control Center worklist into a row-click meeting inbox with derived titles, dot status, and no filter chrome
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-WORKLIST-02.md
- Depends on: PR-OPS-SYSTEM-BAR-02

124) PR-UI-STITCH-RECORDING-DETAILS-01: Replace the compact right pane with a single Recording Details panel and remove embedded tabs
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-RECORDING-DETAILS-01.md
- Depends on: PR-UI-STITCH-WORKLIST-02

125) PR-UI-STITCH-UPLOAD-01: Simplify the Control Center upload area into a compact UPLOAD block with queue cards
- Status: DONE
- Tasks file: tasks/PR-UI-STITCH-UPLOAD-01.md
- Depends on: PR-UI-STITCH-WORKLIST-02

126) PR-TAILWIND-BASE-01: Add Tailwind CSS to base.html and migrate navbar, modal, and global elements
- Status: DONE
- Tasks file: tasks/PR-TAILWIND-BASE-01.md
- Depends on: PR-UI-STITCH-UPLOAD-01

127) PR-TAILWIND-CONTROL-CENTER-01: Migrate Control Center page and all its partials from legacy CSS to Stitch Tailwind design
- Status: DONE
- Tasks file: tasks/PR-TAILWIND-CONTROL-CENTER-01.md
- Depends on: PR-TAILWIND-BASE-01

128) PR-TAILWIND-RECORDING-DETAIL-01: Migrate Recording Detail full-page inspector from legacy CSS to Stitch Tailwind design
- Status: DONE
- Tasks file: tasks/PR-TAILWIND-RECORDING-DETAIL-01.md
- Depends on: PR-TAILWIND-BASE-01

129) PR-TAILWIND-REMAINING-01: Migrate remaining pages to Tailwind and remove all legacy CSS from base.html
- Status: DONE
- Tasks file: tasks/PR-TAILWIND-REMAINING-01.md
- Depends on: PR-TAILWIND-CONTROL-CENTER-01 and PR-TAILWIND-RECORDING-DETAIL-01

130) PR-UI-POLISH-POST-TAILWIND-01: Fix layout bugs, remove duplicate sections, and clean up post-Tailwind migration issues
- Status: DONE
- Tasks file: tasks/PR-UI-POLISH-POST-TAILWIND-01.md
- Depends on: PR-TAILWIND-REMAINING-01

131) PR-UI-POLISH-02: Fix Cyrillic font, compact inspector avatar, navbar flicker, and inspector scroll reset
- Status: DONE
- Tasks file: tasks/PR-UI-POLISH-02.md
- Depends on: PR-UI-POLISH-POST-TAILWIND-01

132) PR-TRANSCRIPT-MERGE-01: Merge short speaker turns and add configurable merge gap to reduce transcript fragmentation
- Status: DONE
- Tasks file: tasks/PR-TRANSCRIPT-MERGE-01.md
- Depends on: PR-UI-POLISH-02

133) PR-DIARIZATION-FLICKER-01: Filter out flickering diarization speakers that appear briefly surrounded by the same dominant speaker
- Status: DONE
- Tasks file: tasks/PR-DIARIZATION-FLICKER-01.md
- Depends on: PR-TRANSCRIPT-MERGE-01

134) PR-TRANSCRIPT-TIMESTAMPS-01: Add timestamps to transcript export and UI as default behavior
- Status: DONE
- Tasks file: tasks/PR-TRANSCRIPT-TIMESTAMPS-01.md
- Depends on: PR-DIARIZATION-FLICKER-01

135) PR-SPEAKER-MERGE-EMBEDDINGS-01: Auto-merge diarization speakers with similar voice embeddings and no temporal overlap
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-MERGE-EMBEDDINGS-01.md
- Depends on: PR-DIARIZATION-FLICKER-01

136) PR-MERGE-DECAPITALIZE-01: Decapitalize false sentence starts left over from segment merging
- Status: DONE
- Tasks file: tasks/PR-MERGE-DECAPITALIZE-01.md
- Depends on: PR-TRANSCRIPT-MERGE-01

137) PR-FIX-CYRILLIC-02: Fix Cyrillic font rendering (diagnose and fix Inter woff2 + Tailwind fontFamily mismatch)
- Status: DONE
- Tasks file: tasks/PR-FIX-CYRILLIC-02.md
- Depends on: PR-MERGE-DECAPITALIZE-01

138) PR-TITLE-EDIT-01: Add inline recording title editing (display_title column + PATCH endpoint + UI pencil icon)
- Status: DONE
- Tasks file: tasks/PR-TITLE-EDIT-01.md
- Depends on: PR-FIX-CYRILLIC-02

139) PR-RECORDING-DETAIL-REDESIGN-01: Redesign full-page Speakers tab to compact table + add transcript preview to Overview + 2-column layout + collapsible pipeline status
- Status: DONE
- Tasks file: tasks/PR-RECORDING-DETAIL-REDESIGN-01.md
- Depends on: PR-TITLE-EDIT-01

140) PR-SPEAKER-MERGE-DIAGNOSTICS-01: Add diagnostics to speaker merge step for debugging merge failures
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-MERGE-DIAGNOSTICS-01.md
- Depends on: PR-SPEAKER-MERGE-EMBEDDINGS-01

141) PR-SPEAKER-MERGE-DIAGNOSTICS-HOTFIX-01: Fix speaker_merge fields missing from diarization_metadata.json due to worker_tasks overwrite
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-MERGE-DIAGNOSTICS-HOTFIX-01.md
- Depends on: PR-SPEAKER-MERGE-DIAGNOSTICS-01

142) PR-SPEAKER-MERGE-TORCH-LOAD-FIX-01: Fix speaker merge embedding model: torch.load weights_only failure and CPU-only device
- Status: DONE
- Tasks file: tasks/PR-SPEAKER-MERGE-TORCH-LOAD-FIX-01.md
- Depends on: PR-SPEAKER-MERGE-DIAGNOSTICS-HOTFIX-01

143) PR-ARTIFACT-SINGLE-WRITER-01: Eliminate dual artifact writes: make orchestrator the single source of truth for all derived JSON artifacts
- Status: DONE
- Tasks file: tasks/PR-ARTIFACT-SINGLE-WRITER-01.md
- Depends on: PR-SPEAKER-MERGE-TORCH-LOAD-FIX-01

144) PR-FORCE-REPROCESS-01: Add Force Full Reprocess button that clears derived artifacts and re-runs pipeline from scratch
- Status: TODO
- Tasks file: tasks/PR-FORCE-REPROCESS-01.md
- Depends on: PR-ARTIFACT-SINGLE-WRITER-01
