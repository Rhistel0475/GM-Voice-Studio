# Contributing

Where to add or change code.

## Adding a new API route

- **Routers:** Add a new router module under `app/api/routers/` or add endpoints to `app/api/routers/routes_legacy.py`. Register the router in `app/main.py` with `app.include_router(..., prefix=...)`.
- **Dependencies:** Auth, rate limiting, and DB session are in `app/api/dependencies/`. Use `get_db`, `verify_api_key`, `get_owner_id` as needed.

## Adding or changing a service

- **Services:** Put business logic in `app/services/` (e.g. `app/services/tts_service.py`, `app/services/llm/`). Call repositories and infrastructure adapters from here.
- **Domain:** If the feature belongs to a domain (voice, campaign, live, ai), add or update exports in the corresponding `app/domain/<name>/__init__.py` and document ownership in [current-architecture.md](current-architecture.md).

## Adding an external adapter

- **Protocol and implementation:** Add a new protocol (and default implementation) under `app/infrastructure/adapters/`. Follow the pattern of `LLMAdapter`, `VoiceStorageAdapter`, etc. Register a default getter (e.g. `get_default_llm_adapter()`) if callers should use a single instance.
- **Config:** Use `app/core/settings.py` (and optionally `app/core/config.py`) for any new env vars.

## Adding a migration

- **Campaign DB:** Create a new revision under `migrations/`: `alembic revision -m "description"`, edit the upgrade/downgrade functions, then run `alembic upgrade head`.
- **Voice metadata DB:** Create a new revision under `migrations_voice/` using `alembic -c alembic_voice.ini revision -m "description"`, then `alembic -c alembic_voice.ini upgrade head`.

## Adding tests

- **Unit / API tests:** Add under `tests/`, e.g. `tests/test_server.py` or `tests/test_<module>.py`. Use pytest markers: `@pytest.mark.slow` for tests that load the TTS model, `@pytest.mark.integration` for tests that hit external services.
- **Smoke:** Update [smoke-test-checklist.md](smoke-test-checklist.md) if you add user-facing flows that should be verified manually.

## Lint and test

From repo root with venv activated:

```bash
./scripts/lint.sh
./scripts/test.sh
./scripts/test.sh --slow   # include slow tests
```

See [testing.md](testing.md) for markers and smoke validation.
