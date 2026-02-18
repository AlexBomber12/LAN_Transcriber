#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts

./scripts/ci.sh 2>&1 | tee artifacts/ci.log

{
  git diff --binary HEAD -- . ':(exclude)artifacts/ci.log' ':(exclude)artifacts/pr.patch'

  git ls-files --others --exclude-standard | while IFS= read -r path; do
    case "$path" in
      artifacts/ci.log|artifacts/pr.patch)
        continue
        ;;
    esac
    git diff --binary --no-index -- /dev/null "$path" || true
  done
} > artifacts/pr.patch
