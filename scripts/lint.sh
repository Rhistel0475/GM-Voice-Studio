#!/usr/bin/env bash
# Lint Python code. Uses ruff if available, else exits with instructions.
set -e
cd "$(dirname "$0")/.."
if command -v ruff >/dev/null 2>&1; then
  ruff check app server.py tests
  echo "ruff check OK"
else
  echo "Install ruff: pip install ruff" >&2
  echo "Then run: ruff check app server.py tests" >&2
  exit 1
fi
