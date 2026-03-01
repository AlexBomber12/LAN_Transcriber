PR-FIX-NONETYPE-CALLABLE-01

```text
Role: You are Codex Agent working inside the LAN_Transcriber repository.

Branch: pr/fix-nonetype-callable-01
PR title: PR-FIX-NONETYPE-CALLABLE-01 Fix runtime crash: 'NoneType' object is not callable + regression test
Base branch: main

Problem
Observed crash:
  TypeError: 'NoneType' object is not callable

Goals
1) Identify exact call site via traceback.
2) Fix deterministically (callable guard, safe default, or fail-fast validation).
3) Add regression test reproducing prior crash.

Implementation
- Capture traceback from docker compose logs.
- Apply fix pattern at call site:
  - if value is None, handle safely
  - if not callable, raise clear TypeError
- Add tests/test_nonetype_callable_regression.py hitting the failing function/route.

Success criteria
- Crash eliminated.
- Regression test added.
- scripts/ci.sh is green.
```
