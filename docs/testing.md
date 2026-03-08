# Testing strategy

## Test layout

| Path | Purpose |
|------|---------|
| **tests/** | Root test package. |
| **tests/test_server.py** | API/health tests (no TTS load): health payload, ready 503/200. |
| **tests/integration/** | Integration-style tests: gated download, voice clone, Kani load, etc. Some are runnable as `python -m tests.integration.<module>` or via pytest. |

Pytest discovers tests under `tests/` (see `pytest.ini` and `pyproject.toml`).

## Markers

Configured in `pytest.ini`:

| Marker | Meaning | Default |
|--------|---------|---------|
| **slow** | Loads TTS model or other heavy setup. | **Excluded** — run with `pytest -m slow` to include. |
| **integration** | Hits external services or full stack. | **Included** unless excluded with `-m "not integration"`. |

**Examples:**

```bash
# Fast run (default: exclude slow)
pytest tests/ -v

# Include slow (TTS) tests
pytest tests/ -v -m slow

# Unit-style only (exclude slow and integration)
pytest tests/ -v -m "not slow and not integration"

# Only integration-marked tests
pytest tests/ -v -m integration
```

## Running tests

From repo root with venv activated:

```bash
# Recommended: use the test script (passes extra args to pytest)
./scripts/test.sh
./scripts/test.sh --slow
./scripts/test.sh -m integration
./scripts/test.sh -m "not slow and not integration"
```

Or directly:

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v -m slow
```

## Smoke validation

After refactors or before release, run the full smoke checklist to confirm the app and preview UI work end-to-end:

1. Start the server: `python server.py` or `./scripts/run-server.sh`.
2. Follow [docs/smoke-test-checklist.md](smoke-test-checklist.md): health, docs, preview, narrate, voice clone, optional RAG and WebSocket.

Smoke tests are **manual** (curl + browser). The automated test suite focuses on health/ready and unit behavior; integration tests cover specific flows when run with the right markers or scripts.

## CI / local quick check

A minimal automated check that doesn’t load the TTS model:

```bash
./scripts/test.sh
# Optional: lint
./scripts/lint.sh   # requires: pip install ruff
```
