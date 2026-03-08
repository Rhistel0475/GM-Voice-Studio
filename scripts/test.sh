#!/usr/bin/env bash
# Run tests. By default skips slow (TTS) and integration markers.
# Usage:
#   ./scripts/test.sh           # fast tests only
#   ./scripts/test.sh --slow     # include slow tests
#   ./scripts/test.sh -m integration  # run integration-marked tests
#   ./scripts/test.sh -m "not slow and not integration"  # unit only
set -e
cd "$(dirname "$0")/.."
if [[ -n "$VIRTUAL_ENV" ]] && [[ -x "$VIRTUAL_ENV/bin/python" ]]; then
  exec "$VIRTUAL_ENV/bin/python" -m pytest tests/ "$@"
else
  exec python3 -m pytest tests/ "$@"
fi
