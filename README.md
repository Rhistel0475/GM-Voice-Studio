# Kani TTS

AI voice engine: generate speech with regional accents or clone a voice from a short recording. Uses [Kani-TTS-2](https://huggingface.co/nineninesix/kani-tts-2-en) and a WavLM-based speaker embedder.

## Run the server

```bash
# From project root, with venv activated
pip install -r requirements-core.txt && pip install -r requirements-server.txt
python server.py
```

**If pip reports "resolution-too-deep"**: Your venv may already have the packages (check for "Requirement already satisfied"). Try running `python server.py` first. If you need a clean install, install in order one package at a time so pip never resolves the full tree at once:

```bash
pip install torch>=2.10.0
pip install "transformers==4.56.0"
pip install "soundfile>=0.13.0"
pip install "kani-tts-2==0.0.5"
pip install fastapi uvicorn slowapi "gradio>=6.6.0"
python server.py
```

- Web UI: **http://localhost:7862** (default port; override with `PORT` env var).
- Interactive API docs: **http://localhost:7862/docs**.

### Config (env)

| Variable | Default | Description |
|----------|---------|-------------|
| `KANI_MODEL_NAME` | `nineninesix/kani-tts-2-en` | Hugging Face model ID |
| `SERVER_NAME` | `0.0.0.0` | Bind address |
| `PORT` | `7862` | Server port (override with env var) |
| `VOICE_STORAGE_PATH` | `./voice_storage` | Directory for cloned voice embeddings (local) |
| `API_KEYS` | (empty) | Comma-separated API keys; header `X-API-Key` |
| `REQUIRE_API_KEY` | (unset) | Set to `1`/`true`/`yes` to require key for TTS/clone |

Optional features (see [config.py](config.py) for full list):

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

## API overview

- **GET /** – Web UI (TTS, voice clone from upload or mic, script narration, Export WAV).
- **GET /health** – Liveness: `{"status":"ok","service":"kani-tts"}`.
- **GET /ready** – Readiness: 503 until TTS model has been loaded (use for load balancer probe).
- **GET /config** – Client config, e.g. `{"require_api_key": true}`.
- **GET /limits** – Narrate limits: `max_narrate_chars`, `max_narrate_chunks`.
- **POST /tts** – Generate speech: form fields `text`, `language_tag`, optional `voice_id`, `temperature`, `top_p`, `repetition_penalty`; optional file `reference_audio` for one-off clone. Returns WAV.
- **POST /voices/clone** – Create persistent voice: form fields `audio` (file), optional `name`, `consent_scope`; returns `voice_id` or (when Celery enabled) `job_id`.
- **GET /jobs/{job_id}** – When Celery enabled: poll clone (or async) job status; when completed, includes `voice_id`.
- **POST /tts/narrate** – Long-form: JSON `text`, optional `voice_id`, `language_tag`, `chunk_by`, `max_chars`; returns WAV.
- **GET /voices/list**, **GET /voices/{id}**, **PATCH /voices/{id}**, **DELETE /voices/{id}** – List and manage cloned voices.
- **DELETE /admin/voices/{voice_id}** – Take-down (requires `X-Admin-Key` when `ADMIN_API_KEY` is set).

Full request/response schemas: **http://localhost:7862/docs** (or your host/port).

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

By default, slow tests (POST /tts, which loads the model) are skipped. To run them: `pytest tests/ -v -m slow`.

## Deploy

**Docker:** Build and run the API (default port 7862):

```bash
docker build -t kani-tts .
docker run -p 7862:7862 -v kani-voice_storage:/app/voice_storage kani-tts
```

**Docker Compose:** API + Redis (for optional async clone/narrate):

```bash
docker compose up -d app
# With Redis and Celery worker (set CELERY_BROKER_URL=redis://redis:6379/0 in app env):
# docker compose --profile celery up -d redis && docker compose run -e CELERY_BROKER_URL=redis://redis:6379/0 app celery -A celery_app worker --loglevel=info
```

**Env:** Set `PORT`, `VOICE_STORAGE_PATH` (or use a volume), and optionally `API_KEYS`, `REQUIRE_API_KEY`, `DATABASE_URL`, `CELERY_BROKER_URL`, `CORS_ORIGINS` (see Config table). For production, back up `voice_storage` and your database (SQLite file or PostgreSQL).

## Use from a TTRPG app (GM Voice Studio API)

The web UI is a GM-focused voice studio; the same API can be called from your TTRPG app (VTT, companion app, or bot).

**Auth:** If the server has `REQUIRE_API_KEY=1`, send `X-API-Key: <your-key>` (or `Authorization: Bearer <key>`) on every request. Voices are scoped per key when `API_KEYS` and a DB are configured.

**Endpoints and example payloads:**

- **Create a character voice:** `POST /voices/clone` — form: `audio` (file), optional `name`, `consent_scope` (e.g. `tts` or `commercial`). Returns `voice_id` or (with Celery) `job_id`; poll `GET /jobs/{job_id}` until done.
- **List voices:** `GET /voices/list` — returns `[{ "voice_id", "name", "consent_scope", "created_at" }, ...]`. When using API keys, only voices for that key are returned.
- **Speak a line:** `POST /tts` — form: `text`, `language_tag` (e.g. `en_us`), optional `voice_id`, `temperature`, `top_p`, `repetition_penalty`. Returns WAV bytes.
- **Narrate a scene:** `POST /tts/narrate` — JSON: `{ "text": "...", "voice_id": null, "language_tag": "en_us", "chunk_by": "sentence", "max_chars": 500 }`. Returns WAV. For long scripts, use `"async": true` when Celery is configured; then poll `GET /jobs/{job_id}` and fetch WAV from `GET /jobs/{job_id}/result`.

**Example (create voice then TTS):**

```bash
# Clone a voice (after uploading audio)
curl -X POST http://localhost:7862/voices/clone -F "audio=@sample.wav" -F "name=Dragon Queen" -H "X-API-Key: YOUR_KEY"
# Then speak as that voice
curl -X POST http://localhost:7862/tts -F "text=You approach the gates." -F "voice_id=VOICE_ID" -H "X-API-Key: YOUR_KEY" --output out.wav
```

## Voice cloning

- **Upload:** WAV/MP3, 5–30 s, 16 kHz preferred.
- **Mic:** Record in the UI; recording is resampled to 16 kHz and can be played back before creating the voice. Use "Re-record" to clear and try again.
- Cloned voices appear in the "Or use a cloned voice" dropdown for TTS and in Script to narration.
- When Celery is configured, clone returns a `job_id`; the UI polls until the job completes and then shows the new `voice_id`.
