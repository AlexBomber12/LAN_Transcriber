PR-SMOKE-DOCKER-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/smoke-docker-01
PR title: PR-SMOKE-DOCKER-01 Strengthen Docker smoke test to catch native dependency loader issues (ctranslate2)
Base branch: main

Goal
Make docker smoke test detect failures that only appear in the real runtime image, such as native loader errors from ctranslate2.

Scope
- Update tests/test_docker_smoke.py only.
- Keep runtime smoke fast (no real transcription).

Implementation

A) Update tests/test_docker_smoke.py
- Run container in detached mode so we can docker exec into it:
  - docker run -d --rm --name lan-smoke -p 17860:7860 "$IMAGE"
- Wait up to 90 seconds for a readiness signal:
  - GET http://127.0.0.1:17860/healthz/app or /openapi.json
- Once ready, execute a dependency import check inside the container:
  - docker exec lan-smoke python -c "import ctranslate2; import faster_whisper; import whisperx"
- Ensure container is stopped in finally:
  - docker stop lan-smoke

B) Keep output useful
- If readiness fails, print container logs:
  - docker logs lan-smoke

Local verification
- Run the test locally by setting SMOKE_IMAGE to a local image tag.

Success criteria
- Smoke test fails when ctranslate2/whisperx imports fail in the runtime image.
- Smoke test still passes quickly when the image is healthy.
- scripts/ci.sh is green.
```
