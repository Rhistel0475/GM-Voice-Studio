#!/usr/bin/env bash
# Run the FastAPI server (development). Requires venv and deps.
set -e
cd "$(dirname "$0")/.."
export PORT="${PORT:-7862}"
if [[ -n "$VIRTUAL_ENV" ]] && [[ -x "$VIRTUAL_ENV/bin/python" ]]; then
  exec "$VIRTUAL_ENV/bin/python" -m uvicorn server:app --host 0.0.0.0 --port "$PORT" --reload
else
  echo "Activate your venv first, e.g. source .venv/bin/activate" >&2
  exit 1
fi
