"""
FastAPI app: TTS and voice cloning API with health check and voice_id persistence.
Uses tts_service (thin interface) and voice_store.
"""
# Load .env first so HF_TOKEN is available for Kani TTS voice-cloning model download
import os as _os
try:
    from dotenv import load_dotenv
    load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

import io
import logging
import os
import re
import time
import tempfile
import uuid
from collections import Counter
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
    HF_TOKEN,
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

# --- Co-DM Adventure document intake ---

_MAX_ADVENTURE_FILES = 6
_MAX_ADVENTURE_FILE_BYTES = 8 * 1024 * 1024
_MAX_ADVENTURE_TOTAL_CHARS = 160_000
_ACT_HEADING_RE = re.compile(r"^\s*act\s+([ivx0-9]+)\s*[-:]\s*(.+)$", re.IGNORECASE)
_SCENE_HEADING_RE = re.compile(r"^\s*(scene|encounter|chapter)\s*([0-9ivx]*)\s*[-:]\s*(.+)$", re.IGNORECASE)
_LABEL_SPLIT_RE = re.compile(r"^\s*([A-Za-z ]{2,24})\s*[:\-]\s*(.+)$")
_TITLE_PHRASE_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")
_SKIP_TITLE_PHRASES = {
    "Act",
    "Scene",
    "Chapter",
    "Campaign",
    "Session",
    "Dungeon Master",
    "Game Master",
    "Read Aloud",
    "Important Npcs",
    "Secrets Clues",
}


def _dedupe_keep_order(items: list[str], limit: int = 10) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean = re.sub(r"\s+", " ", (item or "").strip())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= limit:
            break
    return output


def _extract_pdf_text(data: bytes) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("PDF parsing requires 'pypdf'. Install requirements-rag.txt.") from e
    reader = PdfReader(io.BytesIO(data))
    chunks: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks), len(reader.pages)


async def _read_adventure_upload(upload: UploadFile) -> tuple[str, dict]:
    if not upload.filename:
        raise HTTPException(400, "One of the uploaded files has no filename.")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, f"Unsupported file type: {upload.filename}. Use .txt, .md, or .pdf.")

    raw = await upload.read()
    if not raw:
        raise HTTPException(400, f"{upload.filename} is empty.")
    if len(raw) > _MAX_ADVENTURE_FILE_BYTES:
        raise HTTPException(413, f"{upload.filename} is too large. Max size is {_MAX_ADVENTURE_FILE_BYTES // (1024 * 1024)}MB.")

    page_count: Optional[int] = None
    if suffix == ".pdf":
        try:
            text, page_count = _extract_pdf_text(raw)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    else:
        text = raw.decode("utf-8", errors="ignore")

    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        raise HTTPException(400, f"{upload.filename} has no extractable text.")

    return text, {
        "name": upload.filename,
        "characters": len(text),
        "page_count": page_count,
    }


def _scene_like(line: str) -> bool:
    words = line.split()
    if len(words) < 2 or len(words) > 10:
        return False
    if len(line) > 80:
        return False
    if line.endswith("."):
        return False
    if not line[0].isalpha() or not line[0].isupper():
        return False
    return True


def _extract_acts(lines: list[str]) -> list[dict]:
    acts: list[dict] = []
    current: Optional[dict] = None

    for line in lines:
        act_match = _ACT_HEADING_RE.match(line)
        if act_match:
            act_id = act_match.group(1).upper()
            act_title = act_match.group(2).strip()
            current = {"title": f"Act {act_id} - {act_title}", "scenes": []}
            acts.append(current)
            continue

        scene_match = _SCENE_HEADING_RE.match(line)
        if scene_match:
            scene_title = scene_match.group(3).strip()
            if not current:
                current = {"title": "Act I - Imported Adventure", "scenes": []}
                acts.append(current)
            if scene_title:
                current["scenes"].append(scene_title)
            continue

        if current and _scene_like(line):
            current["scenes"].append(line)

    if not acts:
        fallback_scenes = [line for line in lines if _scene_like(line)]
        if fallback_scenes:
            acts = [{"title": "Act I - Imported Adventure", "scenes": fallback_scenes[:6]}]

    normalized: list[dict] = []
    for act in acts[:6]:
        scenes = _dedupe_keep_order(act.get("scenes", []), limit=8)
        normalized.append({"title": act.get("title", "Act - Imported"), "scenes": scenes})
    return normalized


def _extract_labeled_values(lines: list[str], labels: set[str], limit: int = 10) -> list[str]:
    values: list[str] = []
    for line in lines:
        match = _LABEL_SPLIT_RE.match(line)
        if not match:
            continue
        label = match.group(1).strip().lower()
        value = match.group(2).strip()
        if label in labels and value:
            values.append(value)
    return _dedupe_keep_order(values, limit=limit)


def _extract_title_phrases(text: str, limit: int = 30) -> list[str]:
    counts: Counter[str] = Counter()
    for match in _TITLE_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip()
        if phrase in _SKIP_TITLE_PHRASES:
            continue
        if len(phrase) > 42:
            continue
        counts[phrase] += 1
    return [item for item, _count in counts.most_common(limit)]


def _extract_context_items(text: str, phrases: list[str], keywords: tuple[str, ...], limit: int = 10) -> list[str]:
    out: list[str] = []
    low = text.lower()
    for phrase in phrases:
        needle = phrase.lower()
        idx = low.find(needle)
        if idx == -1:
            continue
        start = max(0, idx - 80)
        end = min(len(low), idx + len(needle) + 80)
        window = low[start:end]
        if any(k in window for k in keywords):
            out.append(phrase)
    return _dedupe_keep_order(out, limit=limit)


def _extract_reveals(lines: list[str]) -> list[str]:
    reveals: list[str] = []
    for line in lines:
        low = line.lower()
        if any(k in low for k in ("clue", "secret", "hook", "reveal", "rumor", "twist")):
            match = _LABEL_SPLIT_RE.match(line)
            reveals.append(match.group(2).strip() if match else line)
    return _dedupe_keep_order(reveals, limit=10)


def _summarize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]
    summary = " ".join(sentences[:2]).strip() if sentences else compact
    if len(summary) < 80:
        summary = compact[:420].rsplit(" ", 1)[0]
    if len(summary) > 420:
        summary = summary[:420].rsplit(" ", 1)[0] + "..."
    return summary


def _parse_adventure_text(text: str) -> dict:
    clipped = text[:_MAX_ADVENTURE_TOTAL_CHARS]
    lines = [
        re.sub(r"\s+", " ", re.sub(r"^[\-\*\+]+\s*", "", raw.strip()))
        for raw in clipped.splitlines()
    ]
    lines = [line for line in lines if line]

    acts = _extract_acts(lines)
    phrases = _extract_title_phrases(clipped)

    labeled_npcs = _extract_labeled_values(lines, {"npc", "name", "character", "villain", "ally"}, limit=10)
    context_npcs = _extract_context_items(
        clipped,
        phrases,
        ("npc", "character", "villain", "ally", "guardian", "priest", "captain", "mage", "merchant"),
        limit=10,
    )
    npcs = _dedupe_keep_order(labeled_npcs + context_npcs, limit=10)

    labeled_locations = _extract_labeled_values(lines, {"location", "region", "site", "area", "setting"}, limit=10)
    context_locations = _extract_context_items(
        clipped,
        phrases,
        ("location", "temple", "ruin", "cave", "forest", "city", "village", "chamber", "lair", "hall"),
        limit=10,
    )
    locations = _dedupe_keep_order(labeled_locations + context_locations, limit=10)

    reveals = _extract_reveals(lines)
    summary = _summarize_text(clipped)

    return {
        "summary": summary,
        "acts": acts,
        "npcs": npcs,
        "locations": locations,
        "reveals": reveals,
        "total_characters": len(clipped),
    }


@app.post("/adventure/parse")
@limiter.limit("15/minute")
async def parse_adventure_docs(
    request: Request,
    files: list[UploadFile] = File(...),
    _auth: None = Depends(verify_api_key),
):
    """Upload adventure docs (.txt/.md/.pdf) and return a parsed prep summary payload."""
    if not files:
        raise HTTPException(400, "Upload at least one document.")
    if len(files) > _MAX_ADVENTURE_FILES:
        raise HTTPException(400, f"Too many files. Max {_MAX_ADVENTURE_FILES} files per parse.")

    all_text_parts: list[str] = []
    uploaded_files: list[dict] = []
    for upload in files:
        text, meta = await _read_adventure_upload(upload)
        all_text_parts.append(text)
        uploaded_files.append(meta)

    merged = "\n\n".join(all_text_parts)
    parsed = _parse_adventure_text(merged)
    return {"files": uploaded_files, **parsed}


# --- Co-DM RAG query ---

class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: Optional[str] = None


@app.post("/rag/query")
@limiter.limit("60/minute")
async def rag_query(req: RagQueryRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """Semantic search over ingested campaign documents. Returns top_k relevant chunks."""
    from retrieval import retrieve
    try:
        results = retrieve(req.query, top_k=req.top_k, doc_type=req.doc_type)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"results": results}


# --- Co-DM LLM brain (Sprint 3) ---

class BrainQueryRequest(BaseModel):
    query: str


@app.post("/brain/query")
@limiter.limit("30/minute")
async def brain_query(req: BrainQueryRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """
    Classify intent, optionally fetch RAG context, call Claude, return structured response.
    Returns: {type, intent, content, sources}
    """
    from llm_brain import handle_query
    try:
        result = handle_query(req.query)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return result


# --- Co-DM NPC Generator (Sprint 4) ---

class NpcGenerateRequest(BaseModel):
    genre: str
    location: str
    name: str
    role: str


@app.post("/npc/generate")
@limiter.limit("10/minute")
async def npc_generate(req: NpcGenerateRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """
    Stream a full NPC profile as Server-Sent Events.
    Each event: data: {"token": "<text>"}\\n\\n
    Final event: data: {"done": true}\\n\\n
    Error event: data: {"error": "<message>"}\\n\\n
    """
    import json as _json
    from npc_generator import generate_npc_stream

    async def sse():
        try:
            for token in generate_npc_stream(req.genre, req.location, req.name, req.role):
                yield f"data: {_json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"
        yield f"data: {_json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


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


# --- Favicon (browsers request this automatically; 204 avoids 404 in logs) ---
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# --- Web UI: served from static file ---
_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
_STATIC_PREVIEW_INDEX = Path(__file__).resolve().parent / "static" / "index.preview.html"
_STATIC_PREVIEW_REACT_DIR = Path(__file__).resolve().parent / "static" / "preview-react"
_STATIC_PREVIEW_REACT_INDEX = _STATIC_PREVIEW_REACT_DIR / "index.html"


def _with_no_cache(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _serve_preview(subpath: Optional[str] = None) -> Response:
    """Serve React preview build when available; fallback to legacy preview HTML."""
    if _STATIC_PREVIEW_REACT_INDEX.is_file():
        if subpath:
            base = _STATIC_PREVIEW_REACT_DIR.resolve()
            candidate = (base / subpath).resolve()
            try:
                candidate.relative_to(base)
            except ValueError:
                raise HTTPException(404, "Not found")
            if candidate.is_file():
                return FileResponse(candidate)
        return _with_no_cache(FileResponse(_STATIC_PREVIEW_REACT_INDEX, media_type="text/html"))
    return _with_no_cache(FileResponse(_STATIC_PREVIEW_INDEX, media_type="text/html"))

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(_STATIC_INDEX, media_type="text/html")


@app.get("/preview", response_class=HTMLResponse)
def preview_index():
    return _serve_preview()


@app.get("/preview/{subpath:path}", response_class=HTMLResponse)
def preview_subpath(subpath: str):
    return _serve_preview(subpath)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_NAME, port=PORT)
