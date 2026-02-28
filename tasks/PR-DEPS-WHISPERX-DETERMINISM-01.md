PR-DEPS-WHISPERX-DETERMINISM-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/deps-whisperx-determinism-01
PR title: PR-DEPS-WHISPERX-DETERMINISM-01 Make WhisperX stack deterministic (single resolver pass, pinned versions, pip check) to reduce API drift
Base branch: main

Problem
API drift keeps breaking runtime because dependencies are resolved non-deterministically (unpinned requirements, installing whisperx separately after requirements).

Goals
1) Ensure whisperx, faster-whisper, ctranslate2, pyannote are installed in one resolver pass from a pinned set.
2) Remove separate Dockerfile step "pip install whisperx==..." (move whisperx pin into pinned requirements).
3) Add pip check during docker build (and CI venv) to catch incompatible sets early.

Implementation

A) One source of truth for pins
- Use existing pinned file (requirements.in plus compiled lock) or introduce a pinned lock file.
- Ensure whisperx and related pins live in that file.

B) Dockerfile
- runtime-full and runtime-lite must install only from the pinned file in one pip command.
- Add:
  python -m pip check
- Add lightweight import probe:
  python -c "import whisperx; import faster_whisper; import ctranslate2; print('deps ok')"

C) CI
- Add python -m pip check to scripts/ci.sh after deps install (or equivalent).

Verification
- docker build passes
- scripts/ci.sh passes

Success criteria
- Dockerfile installs WhisperX stack deterministically.
- pip check passes in CI and docker build.
- Reduced probability of future API drift regressions.
```
