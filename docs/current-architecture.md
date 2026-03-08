# Current Architecture (Pre-Refactor Baseline)

This document captures the GM-Voice-Studio application state before the refactor so behavior can be verified after each phase.

## Startup Flow

1. **Environment**: `server.py` loads `.env` via `dotenv` at the top (so `HF_TOKEN` and other vars are available before any imports that use them).
2. **Config**: All configuration is read from `config.py`, which uses `os.environ.get()` for each variable (no typed settings).
3. **Database**: On import, `server.py` calls `from database import init_db` then `init_db()`. That:
   - Imports `models.py` (Campaign, NPC, Scene, Location) to register SQLAlchemy models with `Base`.
   - Calls `Base.metadata.create_all(bind=engine)` for the **campaign** DB (`codm.db`).
   - Runs `_ensure_runtime_migrations()` to add any missing columns (e.g. `data_json` on `campaigns`).
4. **Voice metadata DB**: Separate from campaign DB. When `DATABASE_URL` is set, `db_voice.py` connects (SQLite or PostgreSQL) and runs `_init_schema_sqlite` / `_init_schema_pg`, which do `CREATE TABLE IF NOT EXISTS voices` and `ALTER TABLE voices ADD COLUMN` for `owner_id`, `faction`.
5. **Static mounts**: After `init_db()`, `server.py` mounts:
   - `/campaign-assets` → `static/campaign_assets`
   - `/static/img` → `static/img`
6. **Lifespan**: FastAPI `@app.on_event("startup")` runs `startup()` which calls `configure_logging()` from `logging_config.py` and logs HF_TOKEN presence.

## Role of Each Root File

| File | Role |
|------|------|
| `server.py` | FastAPI app: all routes, middleware (CORS, SlowAPI), exception handlers, static mounts, `init_db()`, startup. Single large module (~1750 lines). |
| `config.py` | Environment-based configuration: server, storage, DB, auth, rate limits, TTS, RAG, Deepgram, Anthropic. Flat `os.getenv` reads. |
| `database.py` | Campaign DB: SQLAlchemy engine/session for `codm.db`, `get_db()` dependency, `init_db()` with `create_all` + runtime migrations. |
| `models.py` | SQLAlchemy ORM models for campaigns: Campaign, NPC, Scene, Location. Imports `Base` from `database`. |
| `db_voice.py` | Voice metadata DB: connection to `DATABASE_URL` (SQLite or PostgreSQL), schema init with CREATE/ALTER at runtime, CRUD for voices table. |
| `tts_service.py` | TTS: Kani TTS model load, `generate()`, preset voices, language tags. |
| `voice_clone.py` | Voice cloning: validate audio, create speaker embedding, persist via voice_store and optional DB. |
| `voice_store.py` | Voice storage abstraction: local filesystem or S3, metadata (name, consent_scope), list/delete/load path. Uses config for paths and backend. |
| `live_board.py` | Live session / Co-DM: WebSocket helpers, session state, integration with brain and retrieval. |
| `npc_generator.py` | NPC generation from campaign context; uses AI service. |
| `ai_service.py` | Adventure parsing, full campaign structure extraction (AI), dialogue. |
| `llm_brain.py` | Co-DM “brain”: intent routing, RAG query, NPC answers, narration generation, rules explanation. Single orchestration module. |
| `retrieval.py` | RAG: Pinecone + OpenAI embeddings, query interface. |
| `ingest.py` | Document ingest: chunk and index PDFs/text into Pinecone. |
| `anthropic_client.py` | Anthropic API client for Claude. |
| `celery_app.py` | Celery app for async clone and narrate tasks (when `CELERY_BROKER_URL` is set). |
| `logging_config.py` | Logging configuration. |
| `metrics.py` | Prometheus-style metrics: increment, request duration, text export. |
| `text_utils.py` | TTS text splitting: `split_for_tts`, `MAX_CHUNKS`, `MAX_TOTAL_CHARS`. |

## Endpoints (from server.py)

- **Health / ops**: `GET /health`, `GET /ready`, `GET /config`, `GET /metrics`, `GET /limits`
- **Voices**: `GET /voices`, `POST /voices/clone`, `GET /voices/list`, `GET /voices/{voice_id}`, `PATCH /voices/{voice_id}`, `DELETE /voices/{voice_id}`, `DELETE /admin/voices/{voice_id}`
- **Jobs**: `GET /jobs/{job_id}`, `GET /jobs/{job_id}/result`
- **TTS**: `POST /tts`, `POST /tts/narrate`
- **Campaigns**: `GET /api/campaigns`, `GET /api/campaigns/{campaign_id}`, `DELETE /api/campaigns/{campaign_id}`
- **Adventure**: `POST /adventure/parse`, `POST /adventure/ai-parse`, `POST /adventure/images`
- **RAG / AI**: `POST /rag/query`, `POST /brain/query`, `POST /npc/generate`, `POST /ai/dialogue`
- **Live**: `WebSocket /ws/audio`
- **Static / HTML**: `GET /`, `GET /preview`, `GET /preview/{subpath:path}`, `GET /favicon.ico`

## Features to Smoke-Test

After each refactor phase, verify:

1. **Health**: `GET /health` returns 200 and `{"status":"ok","service":"kani-tts"}`; `GET /ready` returns 503 until model loaded, then 200.
2. **Docs**: `GET /docs` (OpenAPI UI) loads.
3. **Preview frontend**: `GET /preview` serves the React app (or legacy fallback).
4. **Narrate flow**: `POST /tts/narrate` with JSON `{ "text": "...", "voice_id": "alba" }` returns WAV.
5. **Voice clone flow**: `POST /voices/clone` with form `audio` file (and optional name) returns `voice_id` or `job_id`; `GET /voices/list` shows the new voice when ready.
6. **Retrieval flow**: With RAG configured, `POST /rag/query` with appropriate body returns results.
7. **Live board flow**: WebSocket `ws://localhost:PORT/ws/audio` connects and accepts messages (transcription + optional Co-DM query).

## Requirements Files

- **Local dev (minimal)**: `requirements-core.txt` (TTS stack) + `requirements-server.txt` (FastAPI, uvicorn, slowapi, etc.). Often installed as `pip install -r requirements-core.txt && pip install -r requirements-server.txt` or via `requirements.txt`.
- **Full dev**: Add `requirements-rag.txt` for RAG (OpenAI, Pinecone, ingest); `requirements-dev.txt` for pytest and dev tools.
- **Production**: Same as local dev; add `requirements-optional.txt` if using S3, Celery, or PostgreSQL.
- **Single file**: `requirements.txt` lists main deps and points to optional/rag in comments.

See [Smoke Test Checklist](smoke-test-checklist.md) for step-by-step verification.
