# API reference

Interactive OpenAPI docs are available at **/docs** (Swagger UI) and **/openapi.json** when the server is running.

## Endpoint groups

- **Health / ops:** `GET /health` (liveness), `GET /ready` (readiness), `GET /config`, `GET /metrics`, `GET /limits`
- **Voices:** `GET /voices`, `POST /voices/clone`, `GET /voices/list`, `GET /voices/{id}`, `PATCH /voices/{id}`, `DELETE /voices/{id}`, `DELETE /admin/voices/{voice_id}` (admin)
- **Jobs:** `GET /jobs/{job_id}`, `GET /jobs/{job_id}/result` (when Celery is configured)
- **TTS:** `POST /tts`, `POST /tts/narrate`
- **Campaigns:** `GET /api/campaigns`, `GET /api/campaigns/{id}`, `DELETE /api/campaigns/{id}`
- **Adventure:** `POST /adventure/parse`, `POST /adventure/ai-parse`, `POST /adventure/images`
- **RAG / AI:** `POST /rag/query`, `POST /brain/query`, `POST /npc/generate`, `POST /ai/dialogue`
- **Live:** `WebSocket /ws/audio`

Request/response schemas, auth (`X-API-Key`, `X-Admin-Key`), and rate limits are described in the OpenAPI schema at **http://localhost:7862/docs** (or your host/port).

## Request IDs

The server generates or forwards `X-Request-ID` and returns it on responses. When `LOG_JSON=1`, logs include `request_id` for tracing.
