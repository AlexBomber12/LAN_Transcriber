#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COVERAGE_THRESHOLD_TRANSCRIBER="${COVERAGE_THRESHOLD_TRANSCRIBER:-87}"
COVERAGE_THRESHOLD_APP="${COVERAGE_THRESHOLD_APP:-75}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
USE_VENV="${USE_VENV:-1}"
CI_VENV_DIR="${CI_VENV_DIR:-.venv-ci}"
export CI="${CI:-true}"

if [ "$USE_VENV" = "1" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ ! -x "${CI_VENV_DIR}/bin/python" ]; then
    python3 -m venv "${CI_VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  . "${CI_VENV_DIR}/bin/activate"
fi

if [ "$INSTALL_DEPS" = "1" ]; then
  python -m pip install -U pip
  python -m pip install -r ci-requirements.txt
  python -m pip install -e .[test]
  python -m pip check
fi

python -m ruff check .
python -m pytest -q --cov=lan_transcriber --cov=lan_app --cov-branch --cov-report=term-missing:skip-covered --cov-report=html "$@"
python -m coverage report --include='lan_transcriber/*' --fail-under="${COVERAGE_THRESHOLD_TRANSCRIBER}"
python -m coverage report --include='lan_app/*' --fail-under="${COVERAGE_THRESHOLD_APP}"
