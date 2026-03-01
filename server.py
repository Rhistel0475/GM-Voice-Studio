"""
FastAPI app: TTS and voice cloning API with health check and voice_id persistence.
Uses tts_service (thin interface) and voice_store.
"""
# Load .env first so HF_TOKEN is available for Pocket TTS voice-cloning model download
import os as _os
try:
    from dotenv import load_dotenv
    load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

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
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config import (
    ABUSE_CLONE_PER_IP_PER_HOUR,
    ADMIN_API_KEY,
    AI_MODEL,
    ANTHROPIC_API_KEY,
    API_KEYS,
    CELERY_BROKER_URL,
    CORS_ORIGINS,
    HF_TOKEN,
    NARRATE_RESULT_PATH,
    PENDING_CLONE_PATH,
    PORT,
    MAX_ADVENTURE_CHARS,
    RATE_LIMIT_AI,
    RATE_LIMIT_CLONE,
    RATE_LIMIT_GLOBAL,
    RATE_LIMIT_PARSE,
    RATE_LIMIT_TTS,
    REQUIRE_API_KEY,
    SERVER_NAME,
)
from logging_config import configure_logging
from metrics import increment, prometheus_text, record_request_duration
from text_utils import MAX_CHUNKS, MAX_TOTAL_CHARS, split_for_tts
from tts_service import generate as tts_generate, get_preset_voices, get_supported_language_tags, _is_preset_voice
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
    if not HF_TOKEN:
        logging.warning("HF_TOKEN is not set. Voice cloning may fail; set HF_TOKEN in .env or the environment.")
    else:
        logging.info("HF_TOKEN is set; voice cloning (gated model) should be available.")
    if not ANTHROPIC_API_KEY:
        logging.warning("ANTHROPIC_API_KEY is not set. POST /ai/dialogue will return 500; add it to .env for Co-GM features.")

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

# --- Voices (preset list + language) ---
@app.get("/voices")
def voices():
    return {"language_tags": _lang_tags(), "preset_voices": get_preset_voices()}

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
    faction: str = Form(""),
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
                faction=faction or "",
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
            faction=faction or None,
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
    language_tag: str = Form("en"),
    voice_id: Optional[str] = Form(None),
    temperature: float = Form(0.65),       # Lowered from 0.75 for stability
    top_p: float = Form(0.80),             # Lowered from 0.85
    repetition_penalty: float = Form(1.15), # Lowered from 2.0 to stop slurring
    reference_audio: Optional[UploadFile] = File(None),
):
    """
    Generate speech. Use either:
    - voice_id (persistent cloned voice),
    - or reference_audio (one-off clone for this request).
    """
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "No text")
    if voice_id:
        request.state.voice_id = voice_id

    # Ensure we always pass a supported language tag to the engine
    supported = _lang_tags()
    lang_tag = (language_tag or "").strip() or "en"
    if lang_tag not in supported and supported:
        lang_tag = supported[0]
    language_tag = lang_tag

    speaker_emb_path: Optional[str] = None

    # Option A: Use a saved voice or preset
    if voice_id:
        if _is_preset_voice(voice_id):
            speaker_emb_path = voice_id.strip()
        else:
            speaker_emb_path = load_embedding_path(voice_id)
        if not speaker_emb_path:
            raise HTTPException(404, "Voice not found")

    # Option B: One-off reference audio (Pocket loads voice from WAV path)
    elif reference_audio and reference_audio.filename:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await reference_audio.read())
            tmp_path = tmp.name
        try:
            audio, sr = tts_generate(
                text,
                language_tag=language_tag,
                speaker_emb_path=tmp_path,
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
    language_tag: Optional[str] = "en"
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
        if not body.voice_id:
            increment("errors_total")
            raise HTTPException(400, "Narrate requires a voice_id.")
        job_id = str(uuid.uuid4())
        supported = _lang_tags()
        lang_tag = (body.language_tag or "").strip() or "en"
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

    if not body.voice_id:
        increment("errors_total")
        raise HTTPException(400, "Narrate requires a voice_id. Select a character voice.")
    if _is_preset_voice(body.voice_id):
        speaker_emb_path = body.voice_id.strip()
    else:
        speaker_emb_path = load_embedding_path(body.voice_id)
    if not speaker_emb_path:
        raise HTTPException(404, "Voice not found")

    supported = _lang_tags()
    lang_tag = (body.language_tag or "").strip() or "en"
    if lang_tag not in supported and supported:
        lang_tag = supported[0]
    language_tag = lang_tag

    audio_list: list = []
    sr_out: Optional[int] = None
    try:
        for chunk in chunks:
            audio, sr = tts_generate(
                chunk,
                language_tag=language_tag,
                speaker_emb_path=speaker_emb_path,
                temperature=0.65,
                top_p=0.80,
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


# --- AI: Co-GM NPC Dialogue Generation ---

class DialogueMessage(BaseModel):
    role: str    # "user" or "assistant"
    content: str


class DialogueRequest(BaseModel):
    npc_name: str
    personality: str
    situation: str
    conversation_history: list[DialogueMessage] = []
    voice_id: Optional[str] = None
    faction: str = ""


class DialogueResponse(BaseModel):
    dialogue: str
    voice_id: Optional[str]


@app.post("/ai/dialogue", response_model=DialogueResponse)
@limiter.limit(RATE_LIMIT_AI or "1000/minute")
async def ai_dialogue(
    request: Request,
    body: DialogueRequest,
    _auth: None = Depends(verify_api_key),
):
    """
    Generate a short in-character NPC line using Claude (Anthropic).
    Returns dialogue text only; call /tts separately to speak it aloud.
    Requires ANTHROPIC_API_KEY in .env.
    """
    if not body.npc_name.strip():
        raise HTTPException(400, "npc_name is required")
    if not body.personality.strip():
        raise HTTPException(400, "personality is required")

    from ai_service import generate_dialogue
    history = [{"role": m.role, "content": m.content} for m in body.conversation_history]

    try:
        dialogue = generate_dialogue(
            npc_name=body.npc_name,
            personality=body.personality,
            situation=body.situation,
            conversation_history=history,
            faction=body.faction,
        )
    except RuntimeError as e:
        increment("errors_total")
        raise HTTPException(500, str(e))

    increment("ai_dialogue_requests_total")
    return DialogueResponse(dialogue=dialogue, voice_id=body.voice_id or None)


# --- AI: Adventure Import (parse read-alouds and NPCs from uploaded adventure) ---

class ReadAloud(BaseModel):
    title: str
    text: str
    scene: str = ""

class ParsedNPC(BaseModel):
    name: str
    personality: str
    faction: str = ""
    description: str = ""
    scene: str = ""

class ParseAdventureResponse(BaseModel):
    read_alouds: list[ReadAloud]
    npcs: list[ParsedNPC]
    char_count: int

@app.post("/ai/parse-adventure", response_model=ParseAdventureResponse)
@limiter.limit(RATE_LIMIT_PARSE or "1000/minute")
async def parse_adventure_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: str = Form(""),
    _auth: None = Depends(verify_api_key),
):
    """
    Upload a PDF, DOCX, or TXT adventure module (or paste text) and extract
    read-aloud passages and NPC profiles using Claude.
    Requires ANTHROPIC_API_KEY in .env and pdfplumber/python-docx for PDF/DOCX files.
    """
    raw_text = ""
    tmp_path = None

    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".txt"
        body_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(body_bytes)
            tmp_path = tmp.name
        try:
            from ai_service import extract_text_from_file
            raw_text = extract_text_from_file(tmp_path, suffix)
        except RuntimeError as e:
            increment("errors_total")
            raise HTTPException(500, str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    elif text.strip():
        raw_text = text.strip()
    else:
        raise HTTPException(400, "Provide a file upload or paste text in the 'text' field.")

    if not raw_text.strip():
        raise HTTPException(400, "No text could be extracted from the provided file.")

    char_count = len(raw_text)
    raw_text = raw_text[:MAX_ADVENTURE_CHARS]

    from ai_service import parse_adventure
    try:
        result = parse_adventure(raw_text)
    except RuntimeError as e:
        increment("errors_total")
        raise HTTPException(500, str(e))

    increment("ai_dialogue_requests_total")
    return ParseAdventureResponse(
        read_alouds=[ReadAloud(**r) for r in result.get("read_alouds", [])],
        npcs=[ParsedNPC(**n) for n in result.get("npcs", [])],
        char_count=char_count,
    )


# --- Favicon (browsers request this automatically; 204 avoids 404 in logs) ---
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# --- Web UI: served from static file ---
_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
_STATIC_TEST  = Path(__file__).resolve().parent / "static" / "test_ui.html"

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_STATIC_INDEX, media_type="text/html")

@app.get("/test", response_class=HTMLResponse)
def test_ui():
    return FileResponse(_STATIC_TEST, media_type="text/html")


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static_files")

# ═══════════════════════════════════════════════════════════════════════════
# GM Voice Studio – Live Board  (Gradio UI, mounted at /live)
# ═══════════════════════════════════════════════════════════════════════════
import gradio as gr

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Inter:wght@400;500;600&display=swap');

:root {
  --wood:      #1b1410;
  --parchment: #f3e2c5;
  --gold:      #d4af37;
  --charcoal:  #25242a;
  --ink:       #2c1a0e;
  --red:       #7a2020;
  --green:     #2a5020;
  --amber:     #b07820;
}

/* Force entire Gradio container background */
body, .gradio-container, .gradio-container .main, .gradio-container .contain {
  background: linear-gradient(135deg, #1b1410 0%, #25242a 50%, #1b1410 100%) !important;
  min-height: 100vh !important;
}

/* Override Gradio's default dark theme */
.dark {
  background: linear-gradient(135deg, #1b1410 0%, #25242a 50%, #1b1410 100%) !important;
}

.header-banner {
  background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
              url('/static/img/forest_campfire.jpg') center/cover no-repeat;
  padding: 40px 24px;
  text-align: center;
  border-radius: 12px;
  border: 4px solid var(--gold);
  margin-bottom: 20px;
  min-height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}

/* Force Group components to use parchment card styling */
.gradio-container .gradio-group {
  background: rgba(243,226,197,0.95) !important;
  border: 4px solid #d4af37 !important;
  border-radius: 12px !important;
  box-shadow: 0 6px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.3) !important;
  padding: 20px !important;
  margin-bottom: 16px !important;
}

/* Dark studio cards - override for specific groups */
.studio-card .gradio-group,
#component-1048 .gradio-group /* Voice Studio */,
#component-1076 .gradio-group /* Co-GM */ {
  background: rgba(37,36,42,0.95) !important;
  border: 4px solid #d4af37 !important;
}

/* Title styling */
.section-title, .studio-title {
  font-family: 'Cinzel', serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 2px !important;
  border-bottom: 3px solid #d4af37 !important;
  padding-bottom: 8px !important;
  margin: 0 0 16px 0 !important;
}

.section-title {
  color: #2c1a0e !important;
}

.studio-title {
  color: #d4af37 !important;
}

/* Quick Tool buttons */
.gradio-container button.quicktool-btn {
  background: #f3e2c5 !important;
  border: 3px solid #d4af37 !important;
  border-radius: 8px !important;
  color: #2c1a0e !important;
  font-family: 'Cinzel', serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
  min-height: 72px !important;
  height: 72px !important;
  transition: transform 0.2s, box-shadow 0.2s !important;
  cursor: pointer !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

.gradio-container button.quicktool-btn:hover {
  transform: scale(1.08) !important;
  box-shadow: 0 0 28px rgba(212,175,55,0.8), 0 4px 12px rgba(0,0,0,0.4) !important;
  background: #faecd6 !important;
}

/* Labels inside parchment cards */
.gradio-group label span {
  color: #2c1a0e !important;
  font-weight: 600 !important;
  font-family: 'Cinzel', serif !important;
}

/* Labels inside studio cards */
#component-1048 label span,
#component-1076 label span {
  color: #d4af37 !important;
}

/* Input fields in parchment cards */
.gradio-group textarea,
.gradio-group input[type=text],
.gradio-group input[type=number],
.gradio-group .svelte-1ed2p3z {
  background: rgba(255,255,255,0.7) !important;
  border: 2px solid #d4af37 !important;
  color: #2c1a0e !important;
  border-radius: 6px !important;
}

/* Dropdowns */
.gradio-group .svelte-1gfkn6j {
  background: rgba(255,255,255,0.7) !important;
  border: 2px solid #d4af37 !important;
  color: #2c1a0e !important;
}

/* Primary action buttons */
.gradio-container button.primary {
  background: #d4af37 !important;
  color: #1b1410 !important;
  border: 3px solid #d4af37 !important;
  font-family: 'Cinzel', serif !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
}

.gradio-container button.primary:hover {
  background: #b8951e !important;
  box-shadow: 0 0 20px rgba(212,175,55,0.6) !important;
}

/* Co-GM chatbot - override Gradio's chat styling */
.chatbot-parchment .message-wrap {
  background: transparent !important;
}

.chatbot-parchment .user-row {
  background: #f3e2c5 !important;
  border: 2px solid #d4af37 !important;
  color: #2c1a0e !important;
  border-radius: 8px !important;
  padding: 12px !important;
}

.chatbot-parchment .bot-row {
  background: #25242a !important;
  border: 2px solid #d4af37 !important;
  color: #f3e2c5 !important;
  border-radius: 8px !important;
  padding: 12px !important;
}

/* Sliders */
.gradio-group input[type=range] {
  accent-color: #d4af37 !important;
}

/* Audio player */
.gradio-group audio {
  filter: sepia(0.3) hue-rotate(5deg) !important;
}
"""


# ── Static HTML panels ────────────────────────────────────────────────────────

_ENCOUNTER_HTML = """
<style>
.enc-row  { display:flex; align-items:center; gap:12px; margin-bottom:10px; 
            padding: 8px; background: rgba(255,255,255,0.4); border-radius: 8px;
            border-left: 4px solid var(--gold); }
.enc-init { font-family:'Cinzel',serif; font-size:14px; font-weight:900;
            color:#d4af37; min-width:32px; text-align:center;
            background: #2c1a0e; border-radius: 50%; width: 32px; height: 32px;
            display: flex; align-items: center; justify-content: center; }
.enc-name { font-family:'Cinzel',serif; font-size:12px; font-weight:700;
            color:#2c1a0e; min-width:120px; }
.enc-hp   { font-size:11px; min-width:42px;
            text-align:right; font-weight:700; }
.enc-bg   { flex:1; height:10px; background:#d4aa70; border-radius:5px; 
            border: 1px solid rgba(44,26,14,0.3); }
.enc-bar  { height:100%; border-radius:5px; transition: width 0.3s; }
</style>
<div>
  <div class="enc-row">
    <div class="enc-init">14</div>
    <span class="enc-name">Goblin Scout</span>
    <div class="enc-bg"><div class="enc-bar" style="width:67%;background:#2a5020"></div></div>
    <span class="enc-hp" style="color:#2a5020">8 / 12</span>
  </div>
  <div class="enc-row">
    <div class="enc-init">12</div>
    <span class="enc-name">Goblin Archer</span>
    <div class="enc-bg"><div class="enc-bar" style="width:60%;background:#b07820"></div></div>
    <span class="enc-hp" style="color:#b07820">6 / 10</span>
  </div>
  <div class="enc-row">
    <div class="enc-init">9</div>
    <span class="enc-name">Goblin Shaman</span>
    <div class="enc-bg"><div class="enc-bar" style="width:60%;background:#b07820"></div></div>
    <span class="enc-hp" style="color:#b07820">9 / 15</span>
  </div>
  <div class="enc-row">
    <div class="enc-init">7</div>
    <span class="enc-name">Captive Wolf</span>
    <div class="enc-bg"><div class="enc-bar" style="width:25%;background:#7a2020"></div></div>
    <span class="enc-hp" style="color:#7a2020">5 / 20</span>
  </div>
</div>
"""

_PARTY_HTML = """
<style>
.pc-row  { display:flex; align-items:center; gap:10px; margin-bottom:10px;
           padding: 8px; background: rgba(255,255,255,0.4); border-radius: 8px;
           border-left: 4px solid var(--gold); }
.pc-name { font-family:'Cinzel',serif; font-size:13px; font-weight:700;
           color:#2c1a0e; min-width:100px; text-transform: uppercase;
           letter-spacing: 0.5px; }
.pc-hp   { font-size:11px; min-width:50px; text-align:right; font-weight:700; }
.pc-bg   { flex:1; height:8px; background:#d4aa70; border-radius:4px;
           border: 1px solid rgba(44,26,14,0.3); }
.pc-bar  { height:100%; border-radius:4px; transition: width 0.3s; }
</style>
<div>
  <div class="pc-row">
    <span class="pc-name">Aethelred</span>
    <div class="pc-bg"><div class="pc-bar" style="width:90%;background:#2a5020"></div></div>
    <span class="pc-hp" style="color:#2a5020">72 / 80</span>
  </div>
  <div class="pc-row">
    <span class="pc-name">Lira</span>
    <div class="pc-bg"><div class="pc-bar" style="width:48%;background:#b07820"></div></div>
    <span class="pc-hp" style="color:#b07820">28 / 58</span>
  </div>
  <div class="pc-row">
    <span class="pc-name">Torin</span>
    <div class="pc-bg"><div class="pc-bar" style="width:91%;background:#2a5020"></div></div>
    <span class="pc-hp" style="color:#2a5020">40 / 44</span>
  </div>
  <div class="pc-row">
    <span class="pc-name">Zephyr</span>
    <div class="pc-bg"><div class="pc-bar" style="width:16%;background:#7a2020"></div></div>
    <span class="pc-hp" style="color:#7a2020">9 / 56</span>
  </div>
  <div class="pc-row">
    <span class="pc-name">Mira</span>
    <div class="pc-bg"><div class="pc-bar" style="width:95%;background:#2a5020"></div></div>
    <span class="pc-hp" style="color:#2a5020">52 / 55</span>
  </div>
</div>
"""

_REVEALS_HTML = """
<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px;">
  <span style="background:#2a5020;color:#fff;padding:6px 14px;border-radius:16px;
               font-size:11px;font-family:'Cinzel',serif;font-weight:700;
               display:flex;align-items:center;gap:6px;">
    <span style="width:8px;height:8px;background:#4ade80;border-radius:50%;"></span>
    Hook line
  </span>
  <span style="background:#b07820;color:#fff;padding:6px 14px;border-radius:16px;
               font-size:11px;font-family:'Cinzel',serif;font-weight:700;
               display:flex;align-items:center;gap:6px;">
    <span style="width:8px;height:8px;background:#fbbf24;border-radius:50%;"></span>
    Temple history
  </span>
  <span style="background:#7a2020;color:#fff;padding:6px 14px;border-radius:16px;
               font-size:11px;font-family:'Cinzel',serif;font-weight:700;
               display:flex;align-items:center;gap:6px;">
    <span style="width:8px;height:8px;background:#ef4444;border-radius:50%;"></span>
    🔒 Secret passage
  </span>
</div>
"""

_NPC_AVATAR = """
<div style="width:80px;height:80px;margin:0 auto 16px;border-radius:50%;
            background:linear-gradient(135deg,#2c1a0e 0%,#1b1410 100%);
            border:4px solid #d4af37;display:flex;align-items:center;justify-content:center;
            font-size:40px;box-shadow:0 4px 12px rgba(0,0,0,0.4);">
  👹
</div>
"""

_WAVEFORM_BAR = """
<div style="width:100%;height:4px;background:linear-gradient(90deg,#3ea18c 0%,#2dd4bf 50%,#3ea18c 100%);
            border-radius:2px;margin:12px 0;box-shadow:0 0 12px rgba(62,161,140,0.6);
            animation:pulse 2s infinite;">
</div>
<style>
@keyframes pulse { 0%,100% { opacity:0.6; } 50% { opacity:1; } }
</style>
"""

# ── Gradio helper functions ───────────────────────────────────────────────────

def _lb_parse_voice(choice: Optional[str]) -> Optional[str]:
    """'Alba (preset)' → 'alba'  |  'Dragon Queen (cloned) [id]' → 'id'."""
    if not choice:
        return None
    if "[" in choice and choice.endswith("]"):
        return choice.split("[")[-1].rstrip("]").strip()
    return choice.split("(")[0].strip().lower() or None


def _lb_get_voices() -> list:
    presets = [f"{v.title()} (preset)"
               for v in ["alba", "marius", "javert", "jean",
                          "fantine", "cosette", "eponine", "azelma"]]
    try:
        from voice_store import list_voices as _lv
        cloned = [f"{v['name']} (cloned) [{v['voice_id']}]" for v in _lv()]
    except Exception:
        cloned = []
    return presets + cloned


def _lb_speak(text: str, voice_choice: str):
    """TTS — returns (sample_rate, audio_ndarray) for gr.Audio."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter some text to speak.")
    voice_id = _lb_parse_voice(voice_choice)
    if not voice_id:
        raise gr.Error("Select a voice first.")
    try:
        from tts_service import generate as _tts
        arr, sr = _tts(text, speaker_emb_path=voice_id)
        return (sr, arr)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc


def _lb_cogm_respond(message: str, history: list,
                     npc_name: str, personality: str, _npc_voice: str):
    """Add GM message, call Claude for Co-GM reply, return updated history."""
    message = (message or "").strip()
    if not message:
        return history, ""
    conv_hist = []
    for pair in (history or []):
        if pair[0]:
            conv_hist.append({"role": "user",      "content": pair[0]})
        if pair[1]:
            conv_hist.append({"role": "assistant", "content": pair[1]})
    try:
        from ai_service import generate_dialogue
        reply = generate_dialogue(
            npc_name=npc_name.strip() or "The NPC",
            personality=personality.strip() or "Neutral",
            situation=message,
            conversation_history=conv_hist,
        )
    except Exception as exc:
        reply = f"[Co-GM error: {exc}]"
    updated = list(history or []) + [[message, reply]]
    return updated, ""


def _lb_speak_last(history: list, npc_voice: str):
    """Speak the most recent Co-GM (bot) reply via TTS."""
    for pair in reversed(history or []):
        if pair[1]:
            return _lb_speak(pair[1], npc_voice)
    raise gr.Error("No Co-GM reply to speak yet.")


# ── Build Gradio Blocks demo ──────────────────────────────────────────────────

def _build_live_demo():
    _voices = _lb_get_voices()
    _default_voice = _voices[0] if _voices else None

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="GM Voice Studio – Live Board",
        analytics_enabled=False,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
<div class="header-banner">
  <h1 style="font-family:'Cinzel',serif;color:#d4af37;font-size:36px;margin:0 0 8px;
             text-shadow:0 2px 12px rgba(0,0,0,0.9);letter-spacing:3px;font-weight:900;">
    GM VOICE STUDIO – LIVE BOARD
  </h1>
  <p style="color:#f3e2c5;font-size:16px;margin:0;letter-spacing:2px;font-weight:600;">
    Campaign: The Shattered Crown
  </p>
</div>""")

        with gr.Row(equal_height=False):

            # ══════════════ LEFT COLUMN (GM controls) ═══════════════
            with gr.Column(scale=2):

                # Quick Tools
                with gr.Group(elem_classes="section-card"):
                    gr.HTML('<h3 class="section-title">⚔️ Quick Tools</h3>')
                    with gr.Row():
                        gr.Button("🎲 Roll Dice",        elem_classes="quicktool-btn")
                        gr.Button("📖 Monster Bestiary", elem_classes="quicktool-btn")
                        gr.Button("✨ Spell Ref",         elem_classes="quicktool-btn")
                    with gr.Row():
                        gr.Button("💰 Loot Table",   elem_classes="quicktool-btn")
                        gr.Button("🎭 Gen NPC",      elem_classes="quicktool-btn")
                        gr.Button("⚔️ Apply Damage", elem_classes="quicktool-btn")

                # Session Notes
                with gr.Group(elem_classes="section-card"):
                    gr.HTML('<h3 class="section-title">📜 Session Notes</h3>')
                    gr.Textbox(
                        label="",
                        placeholder="Scribe your notes here...",
                        lines=4,
                        show_label=False,
                    )

                # Encounter Tracker
                with gr.Group(elem_classes="section-card"):
                    gr.HTML('<h3 class="section-title">⚔️ Encounter Tracker</h3>')
                    gr.HTML(_ENCOUNTER_HTML)

                # Party Roster + World Map
                with gr.Group(elem_classes="section-card"):
                    gr.HTML('<h3 class="section-title">🛡️ Party Roster</h3>')
                    gr.HTML(_PARTY_HTML)
                    gr.HTML("""
<div style="margin-top:20px;border-radius:8px;overflow:hidden;border:4px solid #d4af37;">
  <img src="/static/img/world_map.jpg"
       onerror="this.style.display='none'"
       style="width:100%;display:block;" alt="World Map" />
  <div style="background:rgba(27,20,16,0.85);color:#d4af37;font-family:'Cinzel',serif;
              font-size:11px;text-align:center;padding:6px;letter-spacing:2px;font-weight:700;">
    🗺️ WORLD MAP
  </div>
</div>""")

            # ══════════════ RIGHT COLUMN (Voice + Co-GM) ═════════════
            with gr.Column(scale=1):

                # Voice Studio
                with gr.Group(elem_classes="studio-card"):
                    gr.HTML('<h3 class="studio-title">🎙️ Voice Studio</h3>')
                    gr.HTML(_WAVEFORM_BAR)
                    tts_voice = gr.Dropdown(
                        choices=_voices, value=_default_voice,
                        label="Choose Voice", interactive=True,
                    )
                    with gr.Row():
                        refresh_v_btn = gr.Button("↻ Refresh",    size="sm")
                        gr.Button("🔴 Record",  size="sm", variant="stop")
                        gr.Button("📁 Upload",  size="sm")
                    tts_text = gr.Textbox(
                        label="Speak a line", lines=2,
                        placeholder="Enter NPC dialogue here…",
                    )
                    with gr.Row():
                        tts_temp = gr.Slider(0.0, 1.0, value=0.65, step=0.05,
                                             label="Temperature")
                        gr.Slider(-5, 5, value=0, step=0.5, label="Pitch")
                    speak_btn = gr.Button("▶ Speak", variant="primary")
                    tts_audio = gr.Audio(type="numpy", label="Output",
                                         autoplay=True, show_label=False)

                # NPC Profile
                with gr.Group(elem_classes="section-card"):
                    gr.HTML('<h3 class="section-title">🎭 NPC Profile</h3>')
                    gr.HTML(_NPC_AVATAR)
                    with gr.Row():
                        npc_name_box = gr.Textbox(label="Name",
                                                  placeholder="Temple Guardian", scale=1)
                        gr.Textbox(label="Role", placeholder="Undead Sentinel", scale=1)
                    npc_voice_dd = gr.Dropdown(choices=_voices, label="Voice", value=_default_voice)
                    npc_persona  = gr.Textbox(
                        label="Personality Notes", lines=3,
                        placeholder="Stoic, ancient, speaks in riddles...",
                    )
                    gr.HTML('<p style="font-size:12px;color:#2c1a0e;font-family:\'Cinzel\',' 
                            'serif;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px;font-weight:700;">🎯 Reveals</p>')
                    gr.HTML(_REVEALS_HTML)

                # Co-GM Assistant
                with gr.Group(elem_classes="studio-card"):
                    gr.HTML('<h3 class="studio-title">🤖 Co-GM Assistant</h3>')
                    chatbot = gr.Chatbot(
                        label="", height=300,
                        elem_classes="chatbot-parchment",
                    )
                    with gr.Row():
                        msg_box  = gr.Textbox(
                            show_label=False, placeholder="Describe the situation…",
                            lines=1, scale=4,
                        )
                        send_btn = gr.Button("Send", scale=1, variant="primary")
                    speak_npc_btn = gr.Button("🔊 Speak Last Reply as NPC", size="sm", variant="secondary")
                    cogm_audio = gr.Audio(type="numpy", label="",
                                          autoplay=True, show_label=False)

        # ── Event wiring ──────────────────────────────────────────────────────
        speak_btn.click(_lb_speak, [tts_text, tts_voice], tts_audio)

        refresh_v_btn.click(
            lambda: gr.update(choices=_lb_get_voices()),
            outputs=tts_voice,
        )

        def _send(msg, hist, name, persona, voice):
            return _lb_cogm_respond(msg, hist, name, persona, voice)

        send_btn.click(
            _send,
            [msg_box, chatbot, npc_name_box, npc_persona, npc_voice_dd],
            [chatbot, msg_box],
        )
        msg_box.submit(
            _send,
            [msg_box, chatbot, npc_name_box, npc_persona, npc_voice_dd],
            [chatbot, msg_box],
        )
        speak_npc_btn.click(_lb_speak_last, [chatbot, npc_voice_dd], cogm_audio)

    return demo


_live_demo = _build_live_demo()
app = gr.mount_gradio_app(app, _live_demo, path="/live")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_NAME, port=PORT)
