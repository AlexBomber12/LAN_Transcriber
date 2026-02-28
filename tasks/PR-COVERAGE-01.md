PR-COVERAGE-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/coverage-01
PR title: PR-COVERAGE-01 Expand coverage to include lan_app and enforce coverage thresholds
Base branch: main

Goal
Move toward maximum coverage by including lan_app in coverage reporting and gating.

Constraints
- Keep CI runtime reasonable.
- Avoid sudden large drops by making thresholds configurable.

Implementation

A) Update scripts/ci.sh to measure coverage for lan_app
- Add env vars:
  - COVERAGE_THRESHOLD_TRANSCRIBER (default 90)
  - COVERAGE_THRESHOLD_APP (default 75)
- Run pytest twice to gate each package separately:
  1) Transcriber gate:
     - pytest --cov=lan_transcriber --cov-fail-under="$COVERAGE_THRESHOLD_TRANSCRIBER" -q
  2) App gate:
     - pytest --cov=lan_app --cov-fail-under="$COVERAGE_THRESHOLD_APP" -q

- Ensure the script still respects INSTALL_DEPS and USE_VENV.

B) Update CI workflow to set explicit thresholds
- In .github/workflows/ci.yml set env for the test step:
  - COVERAGE_THRESHOLD_TRANSCRIBER=90
  - COVERAGE_THRESHOLD_APP=75

C) Optional: update .coveragerc
- Ensure coverage excludes:
  - tests
  - scripts
  - migrations if needed

Local verification
- scripts/ci.sh

Success criteria
- Coverage is gated for both lan_transcriber and lan_app.
- Thresholds are configurable via env.
- CI is green with the new tests added earlier.
```
