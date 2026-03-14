# GM Voice Studio (Pocket TTS)

AI voice engine: use built-in voices or clone a voice from a short recording, then generate speech with [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (Kyutai). English only; CPU-optimized, no GPU required.

## Architecture

FastAPI backend with React preview UI at `/preview`. Domains: voice (TTS, clone), campaign (adventure/campaigns), live (WebSocket Co-DM), ai (RAG, LLM). See [docs/architecture.md](docs/architecture.md) for overview and [docs/current-architecture.md](docs/current-architecture.md) for detailed layout. API summary: [docs/api.md](docs/api.md). Deployment: [docs/deployment.md](docs/deployment.md).

## How to start all the servers

**Option A - Docker (one service, recommended)**
Backend serves the main UI and the built React app at `/preview`. No separate frontend process.

```bash
# Optional: rebuild the React preview before building the image
cd frontend
npm install
npm run build
cd ..

docker build -t kani-tts .
docker run -p 7862:7862 -v kani-voice_storage:/app/voice_storage --env-file .env kani-tts
```

Use `--env-file .env` so the container gets `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `PINECONE_API_KEY`, `HF_TOKEN`, and any other runtime config from the repo-root `.env`.

- Main UI: **http://localhost:7862**
- React preview: **http://localhost:7862/preview**

**Option B - Docker Compose (API + optional Redis)**
Same API container, with optional Redis/Celery for async clone and narrate.

```bash
docker compose up -d app
# Optional: uncomment env_file: .env in docker-compose.yml so the container receives your repo-root env vars.
# Optional: start Redis + worker after setting CELERY_BROKER_URL=redis://redis:6379/0:
# docker compose --profile celery up -d redis
# docker compose run --rm -e CELERY_BROKER_URL=redis://redis:6379/0 app celery -A app.infrastructure.tasks.celery_app worker --loglevel=info
```

**Option C - Local backend only (single process)**
One Python process serves everything, including `/preview` when the React build exists in `static/frontend`.

```bash
# From project root, with venv activated
pip install -r requirements-core.txt && pip install -r requirements-server.txt
# Co-DM / PDF / RAG features:
pip install -r requirements-rag.txt
python server.py
```

**Option D - Local backend + React dev server (two processes)**
Use this when actively iterating on the frontend.

1. Terminal 1 - backend
   ```bash
   pip install -r requirements-core.txt && pip install -r requirements-server.txt
   pip install -r requirements-rag.txt
   python server.py
   ```
2. Terminal 2 - React
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

- Main UI: **http://localhost:7862**
- React preview (dev): **http://localhost:5173**

If the frontend is on a different origin, set `CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173` in `.env`.

## Run the server

```bash
# From project root, with venv activated
pip install -r requirements-core.txt && pip install -r requirements-server.txt
# Co-DM / PDF / RAG features:
# pip install -r requirements-rag.txt
python server.py
```

**Python 3.10–3.14**, **PyTorch 2.5+**. Pocket TTS runs on CPU by default and does not require a GPU.

**If pip reports "resolution-too-deep"**: Install in order one package at a time:

```bash
pip install torch>=2.5.0
pip install "soundfile>=0.13.0"
pip install pocket-tts
pip install fastapi uvicorn slowapi "gradio>=6.6.0"
python server.py
```

- Web UI: **http://localhost:7862** (default port; override with `PORT` env var).
- Interactive API docs: **http://localhost:7862/docs**.

### React Preview UI (`/preview`)

`/preview` serves the React app build when available (fallback: legacy `static/index.preview.html`). The React app in `frontend/` is the **primary preview UI**.

```bash
# One-time install
cd frontend
npm install

# Build into static/frontend (served by FastAPI at /preview)
npm run build

# Or from repo root (installs deps if needed)
./scripts/build-frontend.sh
```

**Optional:** Run `npm run dev` in `frontend/` for hot reload; Vite proxies API and WebSocket to the backend (see [docs/frontend.md](docs/frontend.md)).

After `npm run build`, open **http://localhost:7862/preview**.

### Config (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_NAME` | `0.0.0.0` | Bind address |
| `PORT` | `7862` | Server port (override with env var) |
| `VOICE_STORAGE_PATH` | `./voice_storage` | Directory for cloned voice files (.safetensors) and metadata |
| `API_KEYS` | (empty) | Comma-separated API keys; header `X-API-Key` |
| `REQUIRE_API_KEY` | (unset) | Set to `1`/`true`/`yes` to require key for TTS/clone |
| `TTS_PROVIDER` | `kani` | TTS backend: `kani` for local Pocket/Kani TTS, or `hume` for Hume Octave |

Optional features (see [app/core/config.py](app/core/config.py) for full list):

| Variable | Description |
|----------|-------------|
| `VOICE_STORAGE_BACKEND` | `local` or `s3`; use S3 for multi-instance or durability |
| `VOICE_STORAGE_BUCKET` | S3 bucket name when backend is `s3` |
| `DATABASE_URL` | SQLite or PostgreSQL URL for voice metadata (e.g. `sqlite:///voice_metadata.db`) |
| `CELERY_BROKER_URL` | Redis URL to enable async clone (returns `job_id`; poll `GET /jobs/{job_id}`) |
| `CORS_ORIGINS` | Comma-separated origins for CORS (empty = same-origin only) |
| `ADMIN_API_KEY` | When set, `DELETE /admin/voices/{voice_id}` with header `X-Admin-Key` for take-down |
| `ABUSE_CLONE_PER_IP_PER_HOUR` | Max clones per IP per hour (0 = disable) |
| `RATE_LIMIT_GLOBAL`, `RATE_LIMIT_TTS`, `RATE_LIMIT_CLONE` | e.g. `60/minute`; empty = no limit |
| `HF_TOKEN` | Hugging Face token for **voice cloning** (gated model). Optional if you run `hf auth login` first — then the cached token is used. Otherwise create at [hf.co/settings/tokens](https://huggingface.co/settings/tokens), request access at [hf.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), and set `HF_TOKEN=hf_...` in `.env` (no spaces/quotes). |
| `HUME_API_KEY` | Hume API key used when `TTS_PROVIDER=hume` |
| `HUME_SECRET_KEY` | Optional Hume secret key for future integrations; not required by the current TTS path |
| `HUME_BASE_URL` | Base URL for Hume API requests; defaults to `https://api.hume.ai` |
| `HUME_TTS_VERSION` | Hume TTS version string sent to `/v0/tts/file`; defaults to `2` |

## API overview

- **GET /** – Web UI (TTS, voice clone from upload or mic, script narration, Export WAV).
- **GET /health** – Liveness: `{"status":"ok","service":"kani-tts"}`.
- **GET /ready** – Readiness: 503 until TTS model has been loaded (use for load balancer probe).
- **GET /config** – Client config, e.g. `{"require_api_key": true}`.
- **GET /voices** – Returns `language_tags` (e.g. `["en"]`) and `preset_voices` (e.g. `["alba", "marius", ...]`).
- **GET /limits** – Narrate limits: `max_narrate_chars`, `max_narrate_chunks`.
- **POST /tts** – Generate speech: form fields `text`, `language_tag` (ignored; English only), `voice_id` (preset name or cloned voice ID), optional `temperature`, `top_p`, `repetition_penalty`; optional file `reference_audio` for one-off clone. Returns WAV.
- **POST /voices/clone** – Create persistent voice: form fields `audio` (file), optional `name`, `consent_scope`, `faction`; returns `voice_id` or (when Celery enabled) `job_id`.
- **GET /jobs/{job_id}** – When Celery enabled: poll clone (or async) job status; when completed, includes `voice_id`.
- **POST /tts/narrate** – Long-form: JSON `text`, `voice_id` (preset or cloned), optional `language_tag`, `chunk_by`, `max_chars`; returns WAV.
- **GET /voices/list**, **GET /voices/{id}**, **PATCH /voices/{id}**, **DELETE /voices/{id}** – List and manage cloned voices.
- **DELETE /admin/voices/{voice_id}** – Take-down (requires `X-Admin-Key` when `ADMIN_API_KEY` is set).

When `TTS_PROVIDER=hume`, `/voices/list` returns Hume voice IDs in the form `hume:<provider>:<id>`, `/tts` and `/tts/narrate` synthesize through Hume, and local clone endpoints are intentionally unavailable from this server.

Full request/response schemas: **http://localhost:7862/docs** (or your host/port).

## Built-in and cloned voices

- **Preset voices:** Pocket TTS includes built-in voices (alba, marius, javert, jean, fantine, cosette, eponine, azelma). Use `voice_id` set to the preset name (e.g. `alba`) for TTS or narrate without cloning.
- **Cloned voices:** Upload a short clean sample (WAV/MP3) to create a persistent voice; it is stored as a `.safetensors` file and appears in the voice list.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

By default, slow tests (POST /tts, which loads the model) are skipped. To run them: `pytest tests/ -v -m slow`.

**Scripts:** From repo root, `./scripts/test.sh` runs the same; `./scripts/test.sh --slow` includes slow tests. See [docs/testing.md](docs/testing.md) for markers and smoke validation. Where to add routes, services, adapters, and tests: [docs/contributing.md](docs/contributing.md).

## Deploy

See [docs/deployment.md](docs/deployment.md) for production checklist, migrations, env, rate limiting, and health/readiness.

**Docker:** Build and run the API (default port 7862):

```bash
docker build -t kani-tts .
docker run -p 7862:7862 -v kani-voice_storage:/app/voice_storage --env-file .env kani-tts
```

**Docker Compose:** API + Redis (for optional async clone/narrate):

```bash
docker compose up -d app
# Optional: uncomment env_file: .env in docker-compose.yml so the container receives your repo-root env vars.
# With Redis and Celery worker (set CELERY_BROKER_URL=redis://redis:6379/0 in app env):
# docker compose --profile celery up -d redis
# docker compose run --rm -e CELERY_BROKER_URL=redis://redis:6379/0 app celery -A app.infrastructure.tasks.celery_app worker --loglevel=info
```

**Env:** Set `PORT`, `VOICE_STORAGE_PATH` (or use a volume), and optionally `API_KEYS`, `REQUIRE_API_KEY`, `DATABASE_URL`, `CELERY_BROKER_URL`, `CORS_ORIGINS` (see Config table). For production, back up `voice_storage` and your database (SQLite file or PostgreSQL). See [docs/deployment.md](docs/deployment.md).

## Troubleshooting

**PDF parsing or image extraction fails**
Install the RAG/PDF dependencies locally with `pip install -r requirements-rag.txt`. In Docker, rebuild the image after dependency changes: `docker build -t kani-tts .`.

**Container starts but LLM or live mic features fail**
Pass your repo-root `.env` into the container with `--env-file .env` and confirm the relevant keys are set (`ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `PINECONE_API_KEY`, `HF_TOKEN`).

**`Bind for 0.0.0.0:7862 failed: port is already allocated`**
Stop the process already using port `7862`, or run the container on another host port such as `docker run -p 7863:7862 -v kani-voice_storage:/app/voice_storage --env-file .env kani-tts`.

## Use from a TTRPG app (GM Voice Studio API)

The web UI is a GM-focused voice studio; the same API can be called from your TTRPG app (VTT, companion app, or bot).

**Auth:** If the server has `REQUIRE_API_KEY=1`, send `X-API-Key: <your-key>` (or `Authorization: Bearer <key>`) on every request. Voices are scoped per key when `API_KEYS` and a DB are configured.

**Endpoints and example payloads:**

- **Create a character voice:** `POST /voices/clone` — form: `audio` (file), optional `name`, `consent_scope` (e.g. `tts` or `commercial`), `faction`. Returns `voice_id` or (with Celery) `job_id`; poll `GET /jobs/{job_id}` until done.
- **List voices:** `GET /voices/list` — returns cloned voices. Use `GET /voices` for preset voice names.
- **Speak a line:** `POST /tts` — form: `text`, optional `voice_id` (preset name or cloned ID), `temperature`, `top_p`, `repetition_penalty`. Returns WAV bytes.
- **Narrate a scene:** `POST /tts/narrate` — JSON: `{ "text": "...", "voice_id": "alba", "language_tag": "en", "chunk_by": "sentence", "max_chars": 500 }`. Returns WAV. For long scripts, use `"async": true` when Celery is configured; then poll `GET /jobs/{job_id}` and fetch WAV from `GET /jobs/{job_id}/result`.

**Example (preset voice then clone):**

```bash
# Speak with built-in voice
curl -X POST http://localhost:7862/tts -F "text=You approach the gates." -F "voice_id=alba" --output out.wav
# Clone a voice (after uploading audio)
curl -X POST http://localhost:7862/voices/clone -F "audio=@sample.wav" -F "name=Dragon Queen" -H "X-API-Key: YOUR_KEY"
# Then speak as that voice
curl -X POST http://localhost:7862/tts -F "text=You approach the gates." -F "voice_id=VOICE_ID" -H "X-API-Key: YOUR_KEY" --output out.wav
```

## Voice cloning

- **Upload:** WAV/MP3, 3–120 s (Pocket TTS). Clean speech works best.
- **Mic:** Record in the UI; recording can be played back before creating the voice. Use "Re-record" to clear and try again.
- Cloned voices are stored as `.safetensors` and appear in the voice dropdown with built-in presets.
- When Celery is configured, clone returns a `job_id`; the UI polls until the job completes and then shows the new `voice_id`.
