# Current Architecture (Pre-Refactor Baseline)

This document captures the GM-Voice-Studio application state before the refactor so behavior can be verified after each phase.

## Startup Flow

1. **Environment**: `server.py` loads `.env` via `dotenv` at the top (so `HF_TOKEN` and other vars are available before any imports that use them).
2. **Config**: All configuration is read from `config.py`, which uses `os.environ.get()` for each variable (no typed settings).
3. **Database**: On startup, `app.main` runs `init_db()` which:
   - Imports campaign models to register with `Base`.
   - Runs **Alembic** migrations for the **campaign** DB (`codm.db`) via `run_alembic_upgrade()` — no `create_all` or runtime column migrations.
4. **Voice metadata DB**: Separate from campaign DB. When `DATABASE_URL` is set, `app.repositories.voice_repository` connects (SQLite or PostgreSQL). Schema is applied via **Alembic**: run `alembic -c alembic_voice.ini upgrade head` before using voice DB (no runtime CREATE/ALTER).
5. **Static mounts**: After `init_db()`, `server.py` mounts:
   - `/campaign-assets` → `static/campaign_assets`
   - `/static/img` → `static/img`
6. **Lifespan**: FastAPI `@app.on_event("startup")` runs `startup()` which calls `configure_logging()` from `logging_config.py` and logs HF_TOKEN presence.

## Role of Each Root File

| File | Role |
|------|------|
| `server.py` | FastAPI app: all routes, middleware (CORS, SlowAPI), exception handlers, static mounts, `init_db()`, startup. Single large module (~1750 lines). |
| `config.py` | Environment-based configuration: server, storage, DB, auth, rate limits, TTS, RAG, Deepgram, Anthropic. Flat `os.getenv` reads. |
| `database.py` | Campaign DB: SQLAlchemy engine/session for `codm.db`, `get_db()` dependency, `init_db()` runs **Alembic** `upgrade head` (migrations in `migrations/`). |
| `db_models.py` | SQLAlchemy ORM models for campaigns: Campaign, NPC, Scene, Location. Imports `Base` from `database`. |
| `voice_repository.py` | Voice metadata DB: connection to `DATABASE_URL` (SQLite or PostgreSQL), CRUD for voices table. Schema via `alembic -c alembic_voice.ini upgrade head` (no runtime CREATE/ALTER). |
| `campaign_repository.py` | Campaign persistence: list_all, get_by_id, delete, create_from_parse_result; owns session handling. |
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

## Domain boundaries (Phase 5)

Code is grouped into four domains under `app/domain/`; each subpackage documents ownership and re-exports main entry points.

| Domain   | Package            | Owns |
|----------|--------------------|------|
| **voice**   | `app.domain.voice`   | TTS, voice clone/store, voice metadata; routes: /voices, /tts, /jobs (clone/narrate). |
| **campaign**| `app.domain.campaign` | Campaign CRUD, adventure parse persist; routes: /api/campaigns, /adventure/* (persist). |
| **live**    | `app.domain.live`   | WebSocket /ws/audio, live board service. |
| **ai**      | `app.domain.ai`     | LLM orchestrator, RAG, NPC generation, dialogue; routes: /rag/query, /brain/query, /npc/generate, /ai/dialogue. |

Callers can use `from app.domain.voice import clone_voice` (etc.) or continue using `app.services` / `app.repositories` directly.

## External adapters (Phase 6)

External services are accessed through adapter interfaces under `app/infrastructure/adapters/`. Default implementations wrap existing infra.

| Adapter | Protocol | Default implementation |
|---------|----------|------------------------|
| **LLM** | `LLMAdapter.complete(system, messages, model, max_tokens)` | `AnthropicLLMAdapter` (Claude via anthropic_client) |
| **Retriever** | `RetrieverAdapter.retrieve(query, top_k, doc_type)` | `PineconeRetrieverAdapter` (Pinecone + OpenAI embeddings) |
| **Indexer** | `IndexerAdapter.ingest(paths, doc_type)` | `PineconeIndexerAdapter` (indexer.ingest) |
| **Transcription** | `TranscriptionAdapter.transcribe(audio_bytes, mime_type)` | `DeepgramTranscriptionAdapter` (Deepgram REST) |
| **Storage** | `VoiceStorageAdapter` (create_voice_id, save_embedding, load_embedding_path, get_metadata, list_voices, update_metadata, delete_voice) | `DefaultVoiceStorageAdapter` (voice_store_service) |

Use `get_default_llm_adapter()`, `get_default_retriever()`, etc. for the default instance. Routes and services can be refactored to accept adapters for testing or alternate backends.

## LLM orchestrator split (Phase 7)

Co-DM brain logic lives under `app/services/llm/`:

| Module | Role |
|--------|------|
| **intent** | `classify_intent(query)` → `'rule_lookup'` \| `'npc_request'` \| `'general_chat'` (keyword-based). |
| **tool_router** | `get_route_result(query, intent)` → `(user_message, sources)`; fetches RAG chunks for rule_lookup, returns `None` user_message for npc_request. |
| **response_planner** | `CO_DM_SYSTEM_PROMPT`, `build_rag_context(chunks)`, `call_claude(user_message)` → `(response_type, content)`; uses LLM adapter, parses `[STAT_BLOCK]` / `[LORE]` / `[CHAT]`. |
| **orchestrator** | `handle_query(query)` → `{type, intent, content, sources}`; composes intent → route → plan. |

`app/services/llm_orchestrator` is a thin shim re-exporting `handle_query` and `classify_intent` from `app.services.llm`.

## React primary frontend (Phase 8)

The React app in `frontend/` is the primary preview UI. It is built into `static/frontend/` and served at `/preview`.

| Item | Description |
|------|-------------|
| **Build** | `cd frontend && npm run build`, or from root `./scripts/build-frontend.sh`. Output: `static/frontend/`. |
| **Dev server** | `cd frontend && npm run dev` — Vite with proxy to backend (port 7862). |
| **API client** | `frontend/src/api.js` — `getBaseUrl()`, `getConfig()`, `createClient(apiKey)` with methods for voices, tts, narrate, brain, campaigns, jobs. |
| **Docs** | [docs/frontend.md](frontend.md) — build, dev server, proxy, API client, env. |

## Packaging (Phase 9)

- **pyproject.toml** — Project metadata, dependencies, optional-dependencies `dev`, `rag`, `optional`. Tool config: pytest (testpaths, pythonpath), ruff (lint).
- **Scripts** (run from repo root):
  - `./scripts/run-server.sh` — Start FastAPI with uvicorn reload (requires venv).
  - `./scripts/lint.sh` — Run ruff on `app`, `server.py`, `tests` (requires `pip install ruff`).
  - `./scripts/test.sh` — Run pytest; pass args (e.g. `--slow`, `-m integration`).

## Testing strategy (Phase 10)

- **Layout:** `tests/test_server.py` (health/ready), `tests/integration/` (scripts and optional pytest integration tests).
- **Markers:** `slow` (excluded by default), `integration` (see [docs/testing.md](testing.md)).
- **Smoke validation:** Manual checklist in [docs/smoke-test-checklist.md](smoke-test-checklist.md); run after refactors or before release.

## Production hardening (Phase 11)

- **Request ID:** Middleware in `app/main.py` generates or forwards `X-Request-ID`, sets `request.state.request_id`, and adds it to response headers. When `LOG_JSON=1`, `app/core/logging.py` includes `request_id` in each request log line for tracing.
- **Structured logging:** Request logs include `request_path`, `status_code`, `duration_seconds`, and when set `voice_id`, `job_id` (see `app/core/logging.py`).
- **Health vs readiness:** `GET /health` is liveness (process up); `GET /ready` returns 503 until TTS model is loaded (for load balancer readiness). Descriptions are in `app/api/routers/health.py` and [docs/deployment.md](deployment.md).
- **Rate limiting / abuse:** SlowAPI (in-memory) and clone-per-IP abuse tracking in `app/api/dependencies/auth.py`. For multi-instance, document that limits are per process; shared Redis is not implemented (see [docs/deployment.md](deployment.md)).
- **Auth / admin:** `verify_api_key`, `get_owner_id`, `check_abuse_clone` in dependencies; admin take-down via `X-Admin-Key` when `ADMIN_API_KEY` is set.
- **Jobs:** Celery when `CELERY_BROKER_URL` is set; status via `GET /jobs/{job_id}` and `GET /jobs/{job_id}/result`. Documented in [docs/api.md](api.md) and README.
- **Storage:** `VoiceStorageAdapter` in `app/infrastructure/adapters/`; local vs S3 via config (see [docs/deployment.md](deployment.md)).

## Documentation (Phase 12)

- **README:** Overview, architecture summary, run server, React preview, config table, API overview, testing, deploy, TTRPG usage, voice cloning. Links to [docs/architecture.md](architecture.md), [docs/current-architecture.md](current-architecture.md), [docs/api.md](api.md), [docs/deployment.md](deployment.md), [docs/testing.md](testing.md), [docs/contributing.md](contributing.md).
- **docs/architecture.md:** High-level flow diagram and pointer to current-architecture and other docs.
- **docs/api.md:** Endpoint groups and link to OpenAPI `/docs`; request ID note.
- **docs/deployment.md:** Production checklist, migrations, env, rate limiting/abuse (in-memory vs multi-instance), health/readiness, Docker.
- **docs/contributing.md:** Where to add routers, services, adapters, migrations, tests; lint/test commands.
- **docs/frontend.md**, **docs/testing.md:** Already present; referenced from README and architecture.
