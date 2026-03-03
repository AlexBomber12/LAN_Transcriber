from __future__ import annotations

import os

_DEFAULT_TEST_LLM_MODEL = os.environ.setdefault("LLM_MODEL", "test-llm-model")
os.environ.setdefault("LAN_LLM_MODEL", _DEFAULT_TEST_LLM_MODEL)
