PR-COVERAGE-INFRA-02

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/coverage-infra-02
PR title: PR-COVERAGE-INFRA-02 Coverage foundation: branch coverage, unified command, htmlcov artifact
Base branch: main

Goals
1) Unify local and CI coverage invocation via scripts/ci.sh.
2) Enable branch coverage.
3) Upload htmlcov artifact in GH Actions.
4) Do not enforce 100% in this PR yet.

Changes
- Update scripts/ci.sh:
  python -m pytest -q --cov=lan_transcriber --cov=lan_app --cov-branch --cov-report=term-missing:skip-covered --cov-report=html
- Add/update .coveragerc with minimal omit/exclude lines.
- Update CI workflow to upload htmlcov/ artifact.

Success criteria
- Local and CI reports match.
- htmlcov artifact available.
```
