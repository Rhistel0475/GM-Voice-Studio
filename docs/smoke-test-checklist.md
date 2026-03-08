# Smoke Test Checklist

Run after each refactor phase to confirm the app still works. Execute from project root with the server running (e.g. `python server.py` in another terminal).

## Prerequisites

- Backend: `pip install -r requirements-core.txt && pip install -r requirements-server.txt` (and optionally `requirements-rag.txt` if testing RAG).
- Frontend (for preview): `cd frontend && npm install && npm run build` (or from root: `./scripts/build-frontend.sh`).
- Default port: 7862 (override with `PORT` env var).

## 1. Health

- [ ] `curl -s http://localhost:7862/health` returns 200 and body contains `"status":"ok"` and `"service":"kani-tts"`.
- [ ] `curl -s -o /dev/null -w "%{http_code}" http://localhost:7862/ready` — expect 503 before first TTS load, 200 after (or run a TTS request first).

## 2. Docs

- [ ] Open http://localhost:7862/docs in a browser; OpenAPI UI loads.

## 3. Preview frontend

- [ ] Open http://localhost:7862/preview; React app or legacy preview page loads without errors.

## 4. Narrate flow

- [ ] `curl -s -X POST http://localhost:7862/tts/narrate -H "Content-Type: application/json" -d '{"text":"Hello world.","voice_id":"alba"}' --output /tmp/narrate.wav`
- [ ] File `/tmp/narrate.wav` exists and is valid WAV (e.g. `file /tmp/narrate.wav`).

## 5. Voice clone flow

- [ ] Create a short WAV (or use a test fixture). Then:  
  `curl -s -X POST http://localhost:7862/voices/clone -F "audio=@/path/to/short.wav" -F "name=SmokeTest"`
- [ ] Response is JSON with `voice_id` or `job_id`. If `job_id`, poll `GET /jobs/{job_id}` until completed.
- [ ] `curl -s http://localhost:7862/voices/list` returns a list that includes the new voice when ready.

## 6. Retrieval flow (optional, requires RAG config)

- [ ] With `OPENAI_API_KEY`, `PINECONE_API_KEY`, and index configured:  
  `curl -s -X POST http://localhost:7862/rag/query -H "Content-Type: application/json" -d '{"query":"test"}'`  
  returns 200 and a result structure (or expected error if index empty).

## 7. Live board / WebSocket (optional)

- [ ] Connect to `ws://localhost:7862/ws/audio` with a WebSocket client; connection is accepted and server responds to expected message format.

## Quick one-liner (health + docs + config)

```bash
curl -s http://localhost:7862/health | grep -q '"status":"ok"' && \
curl -s -o /dev/null -w "%{http_code}" http://localhost:7862/docs | grep -q 200 && \
curl -s http://localhost:7862/config | grep -q 'require_api_key' && echo "Smoke OK" || echo "Smoke FAIL"
```
