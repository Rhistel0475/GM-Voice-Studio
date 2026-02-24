"""
FastAPI app: TTS and voice cloning API with health check and voice_id persistence.
Uses tts_service (thin interface) and voice_store.
"""
import io
import logging
import os
import time
import tempfile
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from pathlib import Path
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config import (
    ABUSE_CLONE_PER_IP_PER_HOUR,
    ADMIN_API_KEY,
    API_KEYS,
    CELERY_BROKER_URL,
    CORS_ORIGINS,
    NARRATE_RESULT_PATH,
    PENDING_CLONE_PATH,
    PORT,
    RATE_LIMIT_CLONE,
    RATE_LIMIT_GLOBAL,
    RATE_LIMIT_TTS,
    REQUIRE_API_KEY,
    SERVER_NAME,
)
from logging_config import configure_logging
from metrics import increment, prometheus_text, record_request_duration
from text_utils import MAX_CHUNKS, MAX_TOTAL_CHARS, split_for_tts
from tts_service import generate as tts_generate, get_supported_language_tags
from voice_clone import clone_voice
from voice_store import delete_voice, get_metadata, list_voices, load_embedding_path, update_metadata

def _lang_tags():
    """Preset accents from the loaded model (or default list)."""
    return get_supported_language_tags()


# Optional API key verification (when REQUIRE_API_KEY and API_KEYS are set)
async def verify_api_key(request: Request) -> None:
    if not REQUIRE_API_KEY or not API_KEYS:
        return
    key = request.headers.get("X-API-Key") or (request.headers.get("Authorization") or "").replace("Bearer ", "").strip()
    if not key or key not in API_KEYS:
        raise HTTPException(401, "Invalid or missing API key")


def get_owner_id(request: Request) -> Optional[str]:
    """Resolve owner from request: valid API key or None. Used for per-user voice scoping when DB is set."""
    if not API_KEYS:
        return None
    key = request.headers.get("X-API-Key") or (request.headers.get("Authorization") or "").replace("Bearer ", "").strip()
    return key if key in API_KEYS else None


# Abuse: clone count per IP (in-memory, last hour)
_clone_times_by_ip: dict[str, list[float]] = {}
def _check_abuse_clone(ip: str) -> None:
    if ABUSE_CLONE_PER_IP_PER_HOUR <= 0:
        return
    now = time.time()
    cutoff = now - 3600
    if ip not in _clone_times_by_ip:
        _clone_times_by_ip[ip] = []
    times = _clone_times_by_ip[ip]
    times.append(now)
    times[:] = [t for t in times if t > cutoff]
    if len(times) > ABUSE_CLONE_PER_IP_PER_HOUR:
        raise HTTPException(429, "Too many voice clones from this IP; try again later")


limiter = Limiter(
    key_func=get_remote_address,
    application_limits=[RATE_LIMIT_GLOBAL] if RATE_LIMIT_GLOBAL else [],
)
app = FastAPI(title="Kani TTS API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
if CORS_ORIGINS:
    origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_logging_and_metrics(request: Request, call_next):
    """Log request path/status/duration and record latency for /metrics."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.scope.get("path") or request.url.path
    status = response.status_code
    record_request_duration(path, duration)
    extra = {
        "request_path": path,
        "status_code": status,
        "duration_seconds": round(duration, 4),
    }
    voice_id_attr = getattr(request.state, "voice_id", None)
    job_id_attr = getattr(request.state, "job_id", None)
    if voice_id_attr is not None:
        extra["voice_id"] = voice_id_attr
    if job_id_attr is not None:
        extra["job_id"] = job_id_attr
    logging.getLogger("request").info("request_finished", extra=extra)
    return response


@app.on_event("startup")
def startup():
    configure_logging()
    logging.info("Kani TTS API starting; models load on first request.")

# --- Client config (e.g. whether API key is required) ---
@app.get("/config")
def get_config():
    """Return client config so the frontend can show API key input when required."""
    return {"require_api_key": REQUIRE_API_KEY}

# --- Health and readiness ---
@app.get("/health")
def health():
    return {"status": "ok", "service": "kani-tts"}


@app.get("/ready")
def ready():
    """Readiness: 503 until TTS model has been loaded (e.g. after first request). Use for load balancer readiness probe."""
    from tts_service import is_model_loaded
    if not is_model_loaded():
        raise HTTPException(503, "Model not yet loaded")
    return {"status": "ready"}

# --- Metrics (Prometheus-style) ---
@app.get("/metrics")
def metrics():
    return PlainTextResponse(prometheus_text(), media_type="text/plain; charset=utf-8")

# --- Limits (for frontend) ---
@app.get("/limits")
def limits():
    """Return narrate limits so the frontend can show counters and disable submit without duplicating constants."""
    return {"max_narrate_chars": MAX_TOTAL_CHARS, "max_narrate_chunks": MAX_CHUNKS}

# --- Voices (preset list) ---
@app.get("/voices")
def voices():
    return {"language_tags": _lang_tags()}

def _use_clone_queue() -> bool:
    return bool(CELERY_BROKER_URL and not CELERY_BROKER_URL.startswith("memory"))


# --- Voice cloning: create persistent voice from upload ---
@app.post("/voices/clone")
@limiter.limit(RATE_LIMIT_CLONE or "1000/minute")
async def create_voice(
    request: Request,
    audio: UploadFile = File(...),
    consent_scope: str = Form("tts"),
    name: str = Form(""),
    _auth: None = Depends(verify_api_key),
):
    """Upload a short audio sample; validate and store speaker embedding. Returns voice_id or job_id when queue is enabled."""
    _check_abuse_clone(get_remote_address(request))
    if not audio.filename:
        raise HTTPException(400, "No file")
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    body = await audio.read()

    if _use_clone_queue():
        os.makedirs(PENDING_CLONE_PATH, exist_ok=True)
        import uuid
        upload_id = str(uuid.uuid4())
        upload_path = os.path.join(PENDING_CLONE_PATH, f"{upload_id}{suffix}")
        with open(upload_path, "wb") as f:
            f.write(body)
        try:
            from celery_app import clone_voice_task
            owner_id = get_owner_id(request)
            task = clone_voice_task.delay(
                upload_path,
                consent_scope=consent_scope,
                name=name or "",
                owner_id=owner_id,
            )
            increment("clone_requests_total")
            request.state.job_id = task.id
            return JSONResponse({"job_id": task.id})
        except Exception as e:
            try:
                os.unlink(upload_path)
            except OSError:
                pass
            increment("errors_total")
            raise HTTPException(503, f"Queue unavailable: {e!s}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(body)
        tmp_path = tmp.name
    try:
        owner_id = get_owner_id(request)
        voice_id = clone_voice(
            tmp_path,
            consent_scope=consent_scope,
            name=name or None,
            owner_id=owner_id,
        )
        increment("clone_requests_total")
        request.state.voice_id = voice_id
        return JSONResponse({"voice_id": voice_id})
    except ValueError as e:
        increment("errors_total")
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        increment("errors_total")
        raise HTTPException(500, str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# --- Job status (when clone or narrate is enqueued) ---
@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Return status and result for an async clone or narrate job. When completed, includes voice_id (clone) or result_url (narrate)."""
    if not _use_clone_queue():
        raise HTTPException(404, "Job not found")
    from celery.result import AsyncResult
    from celery_app import app as celery_app
    result = AsyncResult(job_id, app=celery_app)
    if result.state == "PENDING":
        return {"job_id": job_id, "status": "pending"}
    if result.state == "SUCCESS":
        res = result.result
        if isinstance(res, dict) and res.get("job_type") == "narrate":
            if res.get("status") == "failed":
                return {"job_id": job_id, "status": "failed", "error": res.get("error", "Unknown error")}
            return {"job_id": job_id, "status": "completed", "result_url": f"/jobs/{job_id}/result"}
        voice_id = res.get("voice_id") if isinstance(res, dict) else res
        return {"job_id": job_id, "status": "completed", "voice_id": voice_id}
    if result.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(result.result) if result.result else "Unknown error"}
    return {"job_id": job_id, "status": result.state.lower(), "result": str(result.result)}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    """Return the WAV file for a completed async narrate job. 404 if not found or not a narrate job."""
    if not _use_clone_queue():
        raise HTTPException(404, "Not found")
    from celery.result import AsyncResult
    from celery_app import app as celery_app
    result = AsyncResult(job_id, app=celery_app)
    if result.state != "SUCCESS":
        raise HTTPException(404, "Job not completed")
    res = result.result
    if not isinstance(res, dict) or res.get("job_type") != "narrate" or res.get("status") != "completed":
        raise HTTPException(404, "Not a completed narrate job")
    wav_path = os.path.join(NARRATE_RESULT_PATH, f"{job_id}.wav")
    if not os.path.isfile(wav_path):
        raise HTTPException(404, "Result file not found")
    return FileResponse(wav_path, media_type="audio/wav", filename="narration.wav")

# --- List all voices (for UI dropdown and My voices panel) ---
@app.get("/voices/list")
def voices_list(request: Request, owner_id: Optional[str] = Depends(get_owner_id)):
    return list_voices(owner_id=owner_id)

# --- GDPR: get voice metadata / delete voice ---
@app.get("/voices/{voice_id}")
def get_voice(voice_id: str, request: Request, owner_id: Optional[str] = Depends(get_owner_id)):
    meta = get_metadata(voice_id, owner_id=owner_id)
    if not meta:
        raise HTTPException(404, "Voice not found")
    return meta

@app.delete("/voices/{voice_id}")
def remove_voice(voice_id: str, request: Request, _auth: None = Depends(verify_api_key), owner_id: Optional[str] = Depends(get_owner_id)):
    """Delete voice embedding and metadata (GDPR right to erasure)."""
    if delete_voice(voice_id, owner_id=owner_id):
        return {"deleted": voice_id}
    raise HTTPException(404, "Voice not found")


# --- Admin: take-down (report abuse) ---
@app.delete("/admin/voices/{voice_id}")
def admin_remove_voice(voice_id: str, x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Remove a voice by ID. Requires X-Admin-Key header (ADMIN_API_KEY). Use for take-down of reported content."""
    if not ADMIN_API_KEY or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(403, "Forbidden")
    if delete_voice(voice_id):
        return {"deleted": voice_id}
    raise HTTPException(404, "Voice not found")


class PatchVoiceBody(BaseModel):
    name: Optional[str] = None


@app.patch("/voices/{voice_id}")
def patch_voice(voice_id: str, body: PatchVoiceBody, request: Request, _auth: None = Depends(verify_api_key), owner_id: Optional[str] = Depends(get_owner_id)):
    """Update voice metadata (e.g. name). Body: {"name": "optional new name"}."""
    if not update_metadata(voice_id, name=body.name, owner_id=owner_id):
        raise HTTPException(404, "Voice not found")
    meta = get_metadata(voice_id, owner_id=owner_id)
    return meta if meta else {"voice_id": voice_id}

# --- TTS: preset or custom voice ---
@app.post("/tts")
@limiter.limit(RATE_LIMIT_TTS or "1000/minute")
async def tts_endpoint(
    request: Request,
    text: str = Form(...),
    _auth: None = Depends(verify_api_key),
    language_tag: str = Form("en_us"),
    voice_id: Optional[str] = Form(None),
    temperature: float = Form(0.95),
    top_p: float = Form(0.9),
    repetition_penalty: float = Form(1.15),
    reference_audio: Optional[UploadFile] = File(None),
):
    """
    Generate speech. Use either:
    - language_tag only (preset accent),
    - voice_id (persistent cloned voice),
    - or reference_audio (one-off clone for this request).
    """
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "No text")
    if voice_id:
        request.state.voice_id = voice_id

    # Ensure we always pass a supported preset accent to the engine
    supported = _lang_tags()
    lang_tag = (language_tag or "").strip() or "en_us"
    if lang_tag not in supported and supported:
        lang_tag = supported[0]
    language_tag = lang_tag

    speaker_emb_path: Optional[str] = None

    if voice_id:
        speaker_emb_path = load_embedding_path(voice_id)
        if not speaker_emb_path:
            raise HTTPException(404, "Voice not found")
    elif reference_audio and reference_audio.filename:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await reference_audio.read())
            tmp_path = tmp.name
        pt_path = None
        try:
            from kani_tts import compute_speaker_embedding
            import torch
            emb = compute_speaker_embedding(tmp_path)
            fd, pt_path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            torch.save(emb.cpu() if emb.ndim == 2 else emb.unsqueeze(0).cpu(), pt_path)
            audio, sr = tts_generate(
                text,
                language_tag=language_tag,
                speaker_emb_path=pt_path,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            increment("tts_requests_total")
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV")
            buf.seek(0)
            return StreamingResponse(buf, media_type="audio/wav")
        except Exception:
            increment("errors_total")
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if pt_path and os.path.exists(pt_path):
                try:
                    os.unlink(pt_path)
                except OSError:
                    pass

    try:
        audio, sr = tts_generate(
            text,
            language_tag=language_tag,
            speaker_emb_path=speaker_emb_path,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    except ValueError as e:
        increment("errors_total")
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        increment("errors_total")
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        increment("errors_total")
        logging.exception("TTS failed")
        raise HTTPException(500, str(e))

    increment("tts_requests_total")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


class NarrateBody(BaseModel):
    text: str
    language_tag: Optional[str] = "en_us"
    voice_id: Optional[str] = None
    chunk_by: str = "sentence"
    max_chars: int = 500
    async_: bool = Field(False, alias="async")  # when True and Celery enabled, enqueue and return job_id


@app.post("/tts/narrate")
@limiter.limit("5/minute")
async def tts_narrate(request: Request, body: NarrateBody, _auth: None = Depends(verify_api_key)):
    """
    Long-form narration: split text into chunks, TTS each, concatenate, return one WAV.
    Limits: 5000 chars, 15 chunks (enforced in split_for_tts).
    When async=true and Celery is configured, enqueues and returns job_id; poll GET /jobs/{job_id} then GET /jobs/{job_id}/result for WAV.
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "No text")
    if len(text) > MAX_TOTAL_CHARS:
        raise HTTPException(400, f"Text exceeds {MAX_TOTAL_CHARS} characters")
    chunk_by = body.chunk_by if body.chunk_by in ("sentence", "paragraph", "fixed") else "sentence"
    chunks = split_for_tts(text, chunk_by=chunk_by, max_chars=max(50, min(body.max_chars, 1500)))
    if not chunks:
        raise HTTPException(400, "No chunks produced from text")
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    if body.voice_id:
        request.state.voice_id = body.voice_id

    if body.async_ and _use_clone_queue():
        from celery_app import narrate_task
        job_id = str(uuid.uuid4())
        supported = _lang_tags()
        lang_tag = (body.language_tag or "").strip() or "en_us"
        if lang_tag not in supported and supported:
            lang_tag = supported[0]
        narrate_task.delay(
            job_id,
            text=text,
            language_tag=lang_tag,
            voice_id=body.voice_id,
            chunk_by=chunk_by,
            max_chars=max(50, min(body.max_chars, 1500)),
        )
        increment("tts_requests_total")
        return JSONResponse({"job_id": job_id})

    supported = _lang_tags()
    lang_tag = (body.language_tag or "").strip() or "en_us"
    if lang_tag not in supported and supported:
        lang_tag = supported[0]
    language_tag = lang_tag

    speaker_emb_path: Optional[str] = None
    if body.voice_id:
        speaker_emb_path = load_embedding_path(body.voice_id)
        if not speaker_emb_path:
            raise HTTPException(404, "Voice not found")

    audio_list: list = []
    sr_out: Optional[int] = None
    try:
        for chunk in chunks:
            audio, sr = tts_generate(
                chunk,
                language_tag=language_tag,
                speaker_emb_path=speaker_emb_path,
                temperature=0.95,
                top_p=0.9,
                repetition_penalty=1.15,
            )
            if sr_out is None:
                sr_out = sr
            audio_list.append(audio)
    except ValueError as e:
        increment("errors_total")
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        increment("errors_total")
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        increment("errors_total")
        logging.exception("Narrate TTS failed")
        raise HTTPException(500, str(e))

    concatenated = np.concatenate(audio_list)
    increment("tts_requests_total")
    buf = io.BytesIO()
    sf.write(buf, concatenated, sr_out, format="WAV")
    buf.seek(0)
    response = StreamingResponse(buf, media_type="audio/wav")
    response.headers["Content-Disposition"] = 'attachment; filename="narration.wav"'
    return response


# --- Favicon (browsers request this automatically; 204 avoids 404 in logs) ---
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# --- Web UI: served from static file ---
_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_STATIC_INDEX, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_NAME, port=PORT)
