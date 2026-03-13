# Deployment and production

## Checklist

- Set `PORT`, `VOICE_STORAGE_PATH` (or use a volume), and optionally `API_KEYS`, `REQUIRE_API_KEY`.
- Run **migrations** before first start (see below).
- Use **readiness** (`GET /ready`) for load balancer health checks so traffic is not sent until the TTS model is loaded.
- Back up voice storage and the database (SQLite file or PostgreSQL).

## Migrations

- **Campaign DB** (`codm.db`): Migrations in `migrations/`. The app runs `alembic upgrade head` on startup. To run manually: `alembic upgrade head`.
- **Voice metadata DB** (when `DATABASE_URL` is set): Migrations in `migrations_voice/`. Run before using voice DB: `alembic -c alembic_voice.ini upgrade head`.

## Environment

See [.env.example](../.env.example) and the Config table in [README](../README.md). Important for production:

| Variable | Notes |
|----------|--------|
| `DATABASE_URL` | SQLite or PostgreSQL for voice metadata. |
| `VOICE_STORAGE_BACKEND` | `local` or `s3`; use S3 for multi-instance or durability. |
| `CELERY_BROKER_URL` | Redis URL for async clone/narrate; then run Celery worker. |
| `CORS_ORIGINS` | Comma-separated origins if the frontend is on another origin. |
| `ADMIN_API_KEY` | Enables `DELETE /admin/voices/{voice_id}` with `X-Admin-Key`. |
| `LOG_JSON` | Set to `1` for JSON log lines (e.g. for log aggregation). |
| `LOG_LEVEL` | e.g. `INFO` or `WARNING`. |

## Rate limiting and abuse

- **Rate limits** (SlowAPI): Configured via `RATE_LIMIT_GLOBAL`, `RATE_LIMIT_TTS`, `RATE_LIMIT_CLONE`. Default implementation is **in-memory** (per process).
- **Abuse** (clone per IP per hour): `ABUSE_CLONE_PER_IP_PER_HOUR`; tracked in-memory in the auth dependency.
- For **multi-instance** deployments, in-memory limits are per instance. To share limits across instances, you would need a shared store (e.g. Redis); that is not implemented by default. Document your limits and consider a single instance or a reverse proxy with rate limiting if needed.

## Health vs readiness

- **GET /health** (liveness): Returns 200 if the process is up. Use for container/orchestrator liveness.
- **GET /ready** (readiness): Returns 503 until the TTS model has been loaded, then 200. Use for load balancer readiness so traffic is not sent before the app can serve TTS.

## Docker

Build and run:

```bash
# Optional: rebuild the React preview before docker build
cd frontend
npm install
npm run build
cd ..

docker build -t kani-tts .
docker run -p 7862:7862 -v kani-voice_storage:/app/voice_storage --env-file .env kani-tts
```

With Docker Compose:

```bash
docker compose up -d app
# Optional: uncomment env_file: .env in docker-compose.yml so the container receives your repo-root env vars.
# Optional Redis + Celery worker:
# docker compose --profile celery up -d redis
# docker compose run --rm -e CELERY_BROKER_URL=redis://redis:6379/0 app celery -A app.infrastructure.tasks.celery_app worker --loglevel=info
```

Use `--env-file .env` for direct `docker run` so the container receives API keys and runtime settings from the repo-root `.env`.
