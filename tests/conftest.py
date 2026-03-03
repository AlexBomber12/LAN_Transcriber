from __future__ import annotations

import os

_DEFAULT_TEST_LLM_MODEL = os.environ.get("LLM_MODEL", "").strip() or "test-llm-model"
os.environ["LLM_MODEL"] = _DEFAULT_TEST_LLM_MODEL
os.environ.setdefault("LAN_LLM_MODEL", _DEFAULT_TEST_LLM_MODEL)
