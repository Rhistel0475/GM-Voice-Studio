#!/usr/bin/env bash
# Build the React frontend into static/frontend (served at /preview).
# Run from repo root.
set -e
cd "$(dirname "$0")/.."
if [[ ! -d frontend ]]; then
  echo "frontend/ not found" >&2
  exit 1
fi
cd frontend
if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies..."
  npm install
fi
npm run build
echo "Frontend built → static/frontend"
