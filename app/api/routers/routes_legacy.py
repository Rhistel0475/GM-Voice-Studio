"""
Legacy route bundle: voices, clone, jobs, tts, campaigns, adventure, rag, brain, npc, ai, websocket.
TODO: Split into app/api/routers/voices.py, clone.py, tts.py, etc.
"""
# Load .env first
import os as _os
try:
    from dotenv import load_dotenv
    load_dotenv(
        _os.path.join(
            _os.path.dirname(
                _os.path.dirname(
                    _os.path.dirname(
                        _os.path.dirname(_os.path.abspath(__file__))
                    )
                )
            ),
            ".env",
        )
    )
except ImportError:
    pass

import base64
import io
import json
import logging
import os
import re
import time
import tempfile
import uuid
import asyncio
import contextlib
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import hashlib
import fitz
import numpy as np
import soundfile as sf
from fastapi import Depends, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from slowapi.util import get_remote_address

from app.api.dependencies.auth import (
    check_abuse_clone,
    get_owner_id,
    limiter,
    verify_api_key,
)
from app.core.config import (
    ADMIN_API_KEY,
    CELERY_BROKER_URL,
    DEEPGRAM_API_KEY,
    DEEPGRAM_LANGUAGE,
    DEEPGRAM_MODEL,
    DEFAULT_VOICE_ID,
    NARRATE_RESULT_PATH,
    PENDING_CLONE_PATH,
    RATE_LIMIT_CLONE,
    RATE_LIMIT_TTS,
)
from app.core.metrics import increment
from app.core.text_utils import MAX_CHUNKS, MAX_TOTAL_CHARS, split_for_tts
from app.services.tts_service import (
    DEFAULT_TTS_REPETITION_PENALTY,
    DEFAULT_TTS_TEMPERATURE,
    DEFAULT_TTS_TOP_P,
    delete_hume_voice,
    generate as tts_generate,
    get_hume_voice,
    get_preset_voices,
    get_supported_language_tags,
    _is_preset_voice,
    is_hume_provider,
    list_hume_voices,
    normalize_stored_voice,
)
from app.services.voice_assignment_service import suggest_voice_for_npc
from app.services.voice_clone_service import clone_voice
from app.services.voice_store_service import (
    delete_voice,
    get_metadata,
    list_voices,
    load_embedding_path,
    update_metadata,
)
from app.repositories import campaign_repository
from app.domain.campaign.systems import (
    DEFAULT_CAMPAIGN_SYSTEM_ID,
    get_campaign_system_preset,
    list_campaign_system_presets,
    normalize_campaign_system,
)
from app.domain.live.session_control import start_session as start_live_session

from fastapi import APIRouter
router = APIRouter()

try:
    import websockets
except ImportError:
    websockets = None

# Static assets dir (used by adventure/campaign routes)
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static" / "campaign_assets"


def _ensure_dir(path: Path) -> None:
    """Create directory with parents. If path exists as a file, remove it first (avoids Errno 17)."""
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _list_campaign_embedded_image_urls(campaign_id: int) -> list[str]:
    """List image files under static/campaign_assets/{id}/embedded/ as /campaign-assets URLs."""
    embedded = _ASSETS_DIR / str(int(campaign_id)) / "embedded"
    if not embedded.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    urls: list[str] = []
    for p in sorted(embedded.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            urls.append(f"/campaign-assets/{int(campaign_id)}/embedded/{p.name}")
    return urls


def _lang_tags():
    return get_supported_language_tags()


def _resolve_voice_target(voice_id: str) -> str:
    raw_voice_id = (voice_id or "").strip()
    if not raw_voice_id:
        raise HTTPException(400, "Voice not found")
    if _is_preset_voice(raw_voice_id):
        return raw_voice_id
    if is_hume_provider():
        raise HTTPException(404, "Voice not found")
    speaker_emb_path = load_embedding_path(raw_voice_id)
    if not speaker_emb_path:
        raise HTTPException(404, "Voice not found")
    return speaker_emb_path


def _audio_to_wav_response(audio, sample_rate: int, filename: str = "narration.wav") -> StreamingResponse:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    response = StreamingResponse(buf, media_type="audio/wav")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def _audio_to_wav_base64(audio, sample_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cleanup_old_sessions(assets_dir: Path, max_age_seconds: int = 3600) -> None:
    """Remove session dirs older than max_age_seconds to prevent disk bloat.

    Skips numeric folder names (DB campaign ids) so embedded PDF images are not purged.
    """
    now = time.time()
    for child in assets_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.isdigit():
            continue
        if (now - child.stat().st_mtime) > max_age_seconds:
            import shutil as _shutil
            _shutil.rmtree(child, ignore_errors=True)


def _transcribe_with_deepgram(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """Send recorded audio bytes to Deepgram and return transcript text."""
    from app.infrastructure.adapters.transcription import get_default_transcription_adapter
    return get_default_transcription_adapter().transcribe(audio_bytes, mime_type)


@router.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """
    Lightweight WebSocket channel for live Co-DM interactions.
    Client messages:
      - {"type":"query","text":"..."} -> routes to llm_brain.handle_query()
      - {"type":"audio_start","mime_type":"audio/webm"} -> begin buffered mic capture
      - binary frames -> appended to buffered audio
      - {"type":"audio_end"} -> Deepgram transcription and transcript push
      - {"type":"transcript","text":"..."} -> echoes as chat payload (UI smoke-test path)
    """
    await websocket.accept()
    send_lock = asyncio.Lock()

    async def safe_send_json(payload: dict) -> None:
        async with send_lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_json(payload)

    audio_chunks = bytearray()
    is_audio_active = False
    audio_mime_type = "audio/webm"
    max_audio_bytes = 10 * 1024 * 1024
    deepgram_ws = None
    deepgram_listener_task = None
    deepgram_streaming = False
    deepgram_last_partial = ""
    deepgram_final_emitted = False

    async def close_deepgram_stream(send_finalize: bool = False) -> None:
        nonlocal deepgram_ws, deepgram_listener_task, deepgram_streaming, deepgram_last_partial
        ws_conn = deepgram_ws
        listener_task = deepgram_listener_task
        deepgram_ws = None
        deepgram_listener_task = None
        deepgram_streaming = False
        deepgram_last_partial = ""

        if ws_conn is not None:
            if send_finalize:
                with contextlib.suppress(Exception):
                    await ws_conn.send(json.dumps({"type": "Finalize"}))
                    await asyncio.sleep(0.35)
            with contextlib.suppress(Exception):
                await ws_conn.close()

        if listener_task is not None:
            listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener_task

    async def start_deepgram_stream() -> bool:
        nonlocal deepgram_ws, deepgram_listener_task, deepgram_streaming, deepgram_last_partial, deepgram_final_emitted

        if not DEEPGRAM_API_KEY or websockets is None:
            return False

        params = {
            "model": DEEPGRAM_MODEL or "nova-3",
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
        }
        if DEEPGRAM_LANGUAGE:
            params["language"] = DEEPGRAM_LANGUAGE
        deepgram_url = f"wss://api.deepgram.com/v1/listen?{urllib.parse.urlencode(params)}"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        try:
            try:
                deepgram_ws = await websockets.connect(
                    deepgram_url,
                    additional_headers=headers,
                    ping_interval=10,
                    ping_timeout=20,
                    max_size=8 * 1024 * 1024,
                )
            except TypeError:
                # Back-compat with older websockets versions.
                deepgram_ws = await websockets.connect(
                    deepgram_url,
                    extra_headers=headers,
                    ping_interval=10,
                    ping_timeout=20,
                    max_size=8 * 1024 * 1024,
                )
        except Exception as e:
            await safe_send_json({"type": "error", "content": f"Deepgram live stream unavailable: {e}"})
            deepgram_ws = None
            return False

        deepgram_streaming = True
        deepgram_last_partial = ""
        deepgram_final_emitted = False

        async def _deepgram_listener():
            nonlocal deepgram_last_partial, deepgram_final_emitted
            try:
                async for deepgram_message in deepgram_ws:
                    if not isinstance(deepgram_message, str):
                        continue
                    try:
                        event = json.loads(deepgram_message)
                    except json.JSONDecodeError:
                        continue

                    event_type = str(event.get("type") or "").lower()
                    if event_type == "error":
                        await safe_send_json({
                            "type": "error",
                            "content": event.get("description") or "Deepgram streaming error.",
                        })
                        continue

                    channel = event.get("channel") or {}
                    alternatives = channel.get("alternatives") or []
                    transcript = ((alternatives[0].get("transcript") if alternatives else "") or "").strip()
                    if not transcript:
                        continue

                    is_final = bool(event.get("is_final"))
                    if is_final:
                        deepgram_final_emitted = True
                        deepgram_last_partial = ""
                    else:
                        if transcript == deepgram_last_partial:
                            continue
                        deepgram_last_partial = transcript

                    await safe_send_json({
                        "type": "transcript",
                        "intent": "general_chat",
                        "content": transcript,
                        "sources": [],
                        "final": is_final,
                    })
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await safe_send_json({"type": "error", "content": f"Deepgram stream interrupted: {e}"})

        deepgram_listener_task = asyncio.create_task(_deepgram_listener())
        return True

    try:
        while True:
            message = await websocket.receive()

            raw_text = message.get("text")
            raw_bytes = message.get("bytes")
            if raw_text is None and raw_bytes is not None:
                if not is_audio_active:
                    await safe_send_json({"type": "error", "content": "Send {\"type\":\"audio_start\"} before streaming audio bytes."})
                    continue
                audio_chunks.extend(raw_bytes)
                if len(audio_chunks) > max_audio_bytes:
                    audio_chunks.clear()
                    is_audio_active = False
                    await close_deepgram_stream()
                    await safe_send_json({"type": "error", "content": "Audio capture exceeded 10MB limit. Please record a shorter sample."})
                    continue

                if deepgram_streaming and deepgram_ws is not None:
                    try:
                        await deepgram_ws.send(raw_bytes)
                    except Exception as e:
                        await safe_send_json({"type": "error", "content": f"Deepgram live stream failed, falling back to final-only mode: {e}"})
                        await close_deepgram_stream()
                continue
            if raw_text is None:
                continue

            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                payload = {"type": "query", "text": raw_text}

            msg_type = (payload.get("type") or "").strip()
            text = (payload.get("text") or "").strip()

            if msg_type == "audio_start":
                audio_chunks.clear()
                is_audio_active = True
                audio_mime_type = (payload.get("mime_type") or "audio/webm").strip() or "audio/webm"
                deepgram_final_emitted = False
                await close_deepgram_stream()
                live_stream_started = await start_deepgram_stream()
                if DEEPGRAM_API_KEY and websockets is None:
                    await safe_send_json({
                        "type": "error",
                        "content": "Install Python package 'websockets' to enable Deepgram live streaming.",
                    })
                await safe_send_json({"type": "status", "content": "listening-live" if live_stream_started else "listening"})
                continue

            if msg_type == "audio_end":
                if not is_audio_active:
                    await safe_send_json({"type": "error", "content": "No active audio stream to finalize."})
                    continue
                is_audio_active = False
                if not audio_chunks:
                    await close_deepgram_stream()
                    await safe_send_json({"type": "error", "content": "No audio received from microphone stream."})
                    continue
                if deepgram_streaming:
                    await close_deepgram_stream(send_finalize=True)
                    if deepgram_final_emitted:
                        audio_chunks.clear()
                        deepgram_final_emitted = False
                        continue

                try:
                    transcript = await run_in_threadpool(_transcribe_with_deepgram, bytes(audio_chunks), audio_mime_type)
                except RuntimeError as e:
                    await safe_send_json({"type": "error", "content": str(e)})
                else:
                    await safe_send_json({
                        "type": "transcript",
                        "intent": "general_chat",
                        "content": transcript,
                        "sources": [],
                        "final": True,
                    })
                finally:
                    audio_chunks.clear()
                    deepgram_final_emitted = False
                continue

            if not text:
                await safe_send_json({"type": "error", "content": "Empty message."})
                continue

            if msg_type == "transcript":
                await safe_send_json({"type": "chat", "intent": "general_chat", "content": text, "sources": []})
                continue

            if msg_type == "query":
                from app.services.llm_orchestrator import handle_query
                try:
                    result = await run_in_threadpool(handle_query, text)
                except RuntimeError as e:
                    await safe_send_json({"type": "error", "content": str(e)})
                    continue
                await safe_send_json(result)
                continue

            await safe_send_json({"type": "error", "content": f"Unsupported message type: {msg_type or '(missing)'}"})
    except WebSocketDisconnect:
        return
    except RuntimeError as e:
        # Starlette may raise RuntimeError on receive() after disconnect instead of WebSocketDisconnect.
        if "disconnect message" in str(e).lower():
            return
        raise
    finally:
        await close_deepgram_stream()


def _use_clone_queue() -> bool:
    return bool(CELERY_BROKER_URL and not CELERY_BROKER_URL.startswith("memory"))


# --- Voice cloning: create persistent voice from upload ---
@router.post("/voices/clone")
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
    if is_hume_provider():
        raise HTTPException(
            501,
            "Voice cloning is not managed by this server when TTS_PROVIDER=hume. Hume custom voices must be created in Hume first, then used here by voice ID.",
        )
    check_abuse_clone(get_remote_address(request))
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
            from app.infrastructure.tasks.celery_app import clone_voice_task
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
        voice_id = await run_in_threadpool(
            clone_voice,
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
@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Return status and result for an async clone or narrate job. When completed, includes voice_id (clone) or result_url (narrate)."""
    if not _use_clone_queue():
        raise HTTPException(404, "Job not found")
    from celery.result import AsyncResult
    from app.infrastructure.tasks.celery_app import app as celery_app
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


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    """Return the WAV file for a completed async narrate job. 404 if not found or not a narrate job."""
    if not _use_clone_queue():
        raise HTTPException(404, "Not found")
    from celery.result import AsyncResult
    from app.infrastructure.tasks.celery_app import app as celery_app
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
@router.get("/voices/list")
def voices_list(request: Request, owner_id: Optional[str] = Depends(get_owner_id)):
    if is_hume_provider():
        return list_hume_voices()
    voices = list_voices(owner_id=owner_id)
    usable = []
    for voice in voices:
        voice_id = (voice or {}).get("voice_id")
        if not voice_id:
            continue
        if load_embedding_path(voice_id):
            usable.append(normalize_stored_voice(voice))
    return usable

# --- GDPR: get voice metadata / delete voice ---
@router.get("/voices/{voice_id}")
def get_voice(voice_id: str, request: Request, owner_id: Optional[str] = Depends(get_owner_id)):
    if is_hume_provider():
        meta = get_hume_voice(voice_id)
        if not meta:
            raise HTTPException(404, "Voice not found")
        return meta
    meta = get_metadata(voice_id, owner_id=owner_id)
    if not meta:
        raise HTTPException(404, "Voice not found")
    return normalize_stored_voice(meta)

@router.delete("/voices/{voice_id}")
def remove_voice(voice_id: str, request: Request, _auth: None = Depends(verify_api_key), owner_id: Optional[str] = Depends(get_owner_id)):
    """Delete voice embedding and metadata (GDPR right to erasure)."""
    if is_hume_provider():
        try:
            if delete_hume_voice(voice_id):
                return {"deleted": voice_id}
        except ValueError as e:
            raise HTTPException(400, str(e))
        raise HTTPException(404, "Voice not found")
    if delete_voice(voice_id, owner_id=owner_id):
        return {"deleted": voice_id}
    raise HTTPException(404, "Voice not found")


# --- Admin: take-down (report abuse) ---
@router.delete("/admin/voices/{voice_id}")
def admin_remove_voice(voice_id: str, x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Remove a voice by ID. Requires X-Admin-Key header (ADMIN_API_KEY). Use for take-down of reported content."""
    if not ADMIN_API_KEY or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(403, "Forbidden")
    if delete_voice(voice_id):
        return {"deleted": voice_id}
    raise HTTPException(404, "Voice not found")


class PatchVoiceBody(BaseModel):
    name: Optional[str] = None


@router.patch("/voices/{voice_id}")
def patch_voice(voice_id: str, body: PatchVoiceBody, request: Request, _auth: None = Depends(verify_api_key), owner_id: Optional[str] = Depends(get_owner_id)):
    """Update voice metadata (e.g. name). Body: {"name": "optional new name"}."""
    if is_hume_provider():
        raise HTTPException(501, "Voice metadata edits are not supported by this server when TTS_PROVIDER=hume.")
    if not update_metadata(voice_id, name=body.name, owner_id=owner_id):
        raise HTTPException(404, "Voice not found")
    meta = get_metadata(voice_id, owner_id=owner_id)
    return meta if meta else {"voice_id": voice_id}

# --- Co-DM Adventure document intake ---

_MAX_ADVENTURE_FILES = 6
_MAX_ADVENTURE_FILE_BYTES = 200 * 1024 * 1024  # 200 MB — text is truncated to MAX_ADVENTURE_CHARS after extraction
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
        import pymupdf4llm
        import pymupdf
    except Exception as e:
        raise RuntimeError("PDF parsing requires 'pymupdf4llm'. Install requirements-rag.txt.") from e
    doc = pymupdf.open(stream=data, filetype="pdf")
    page_count = doc.page_count
    use_ocr = _has_tesseract_language_data("eng")

    # Some pymupdf4llm versions run in a "legacy mode" and ignore OCR kwargs.
    # Keep extraction resilient across versions and fall back to raw page text
    # if markdown extraction is unexpectedly sparse.
    try:
        if use_ocr:
            try:
                md_text = pymupdf4llm.to_markdown(doc, use_ocr=True, ocr_language="eng")
            except TypeError:
                logging.debug("pymupdf4llm OCR kwargs unsupported; retrying without kwargs.")
                md_text = pymupdf4llm.to_markdown(doc)
        else:
            logging.debug("Tesseract OCR unavailable; using embedded PDF text (sufficient for most published PDFs).")
            md_text = pymupdf4llm.to_markdown(doc)
    except Exception as e:
        logging.warning("pymupdf4llm extraction failed; falling back to plain page text: %s", e)
        md_text = ""

    plain_text_parts: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text:
            plain_text_parts.append(text)
    plain_text = "\n\n".join(plain_text_parts)

    md_len = len((md_text or "").strip())
    plain_len = len((plain_text or "").strip())
    if md_len < 1200 and plain_len > md_len:
        logging.info("PDF extraction fallback used: markdown=%d chars, plain=%d chars", md_len, plain_len)
        return plain_text, page_count

    return (md_text if md_len >= plain_len else plain_text), page_count


def _has_tesseract_language_data(language: str = "eng") -> bool:
    try:
        import pymupdf
    except Exception:
        return False

    try:
        tessdata = pymupdf.get_tessdata()
    except Exception:
        return False

    if not tessdata:
        return False

    tessdata_path = Path(str(tessdata))
    if not tessdata_path.exists():
        return False

    wanted = [token.strip() for token in re.split(r"[+,]", language or "") if token.strip()]
    if not wanted:
        return True

    return all((tessdata_path / f"{token}.traineddata").exists() for token in wanted)


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
            text, page_count = await run_in_threadpool(_extract_pdf_text, raw)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    else:
        text = raw.decode("utf-8", errors="ignore")

    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        if suffix == ".pdf" and not _has_tesseract_language_data("eng"):
            raise HTTPException(
                400,
                (
                    f"{upload.filename} has no extractable text. "
                    "This PDF may be image-only, and OCR is unavailable because Tesseract English language data is missing."
                ),
            )
        raise HTTPException(400, f"{upload.filename} has no extractable text.")

    return text, {
        "name": upload.filename,
        "characters": len(text),
        "page_count": page_count,
    }


def _extract_pdf_text_pdfplumber_or_pypdf(data: bytes) -> str:
    """
    Plain PDF text extraction for /adventure/extract-text.
    Prefer pdfplumber when installed; otherwise use pypdf. Pages joined with \\n\\n.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        pdfplumber = None

    if pdfplumber is not None:
        try:
            parts: list[str] = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t and str(t).strip():
                        parts.append(str(t).strip())
            if parts:
                return "\n\n".join(parts)
        except Exception as e:
            logging.warning("pdfplumber text extraction failed, trying pypdf: %s", e)

    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "PDF text extraction requires pypdf. Install with: pip install pypdf"
        ) from e

    reader = PdfReader(io.BytesIO(data))
    parts2: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t and str(t).strip():
            parts2.append(str(t).strip())
    return "\n\n".join(parts2)


def _extract_raw_document_text(filename: str, raw: bytes) -> str:
    """UTF-8 text from .txt/.md, or pdfplumber/pypdf for .pdf. No truncation."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, f"Unsupported file type: {filename}. Use .txt, .md, or .pdf.")
    if not raw:
        raise HTTPException(400, f"{filename} is empty.")
    if len(raw) > _MAX_ADVENTURE_FILE_BYTES:
        raise HTTPException(
            413, f"{filename} is too large. Max size is {_MAX_ADVENTURE_FILE_BYTES // (1024 * 1024)}MB."
        )

    if suffix in (".txt", ".md"):
        text = raw.decode("utf-8", errors="ignore")
    else:
        try:
            text = _extract_pdf_text_pdfplumber_or_pypdf(raw)
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e

    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        raise HTTPException(400, f"{filename} has no extractable text.")
    return text


@router.post("/adventure/extract-text")
@limiter.limit("30/minute")
async def extract_adventure_text(
    request: Request,
    files: list[UploadFile] = File(...),
    _auth: None = Depends(verify_api_key),
):
    """Upload one adventure document; return raw extracted text (no AI / no parse)."""
    if not files:
        raise HTTPException(400, "Upload at least one document.")
    upload = files[0]
    if not upload.filename:
        raise HTTPException(400, "Uploaded file has no filename.")

    raw = await upload.read()
    text = await run_in_threadpool(_extract_raw_document_text, upload.filename, raw)
    return {"text": text}


_UPLOADS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "uploads"


@router.post("/adventure/upload-doc")
@limiter.limit("30/minute")
async def upload_adventure_document(
    request: Request,
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
):
    """Save one uploaded document under uploads/ and return a URL for static serving."""
    if not file.filename:
        raise HTTPException(400, "Uploaded file has no filename.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, "Unsupported file type. Use .txt, .md, or .pdf.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, f"{file.filename} is empty.")
    if len(raw) > _MAX_ADVENTURE_FILE_BYTES:
        raise HTTPException(
            413, f"File too large. Max size is {_MAX_ADVENTURE_FILE_BYTES // (1024 * 1024)}MB.",
        )
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest = _UPLOADS_DIR / safe_name
    dest.write_bytes(raw)
    return {"file_url": f"/uploads/{safe_name}"}


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
        "items": [],
        "quests": [],
        "factions": [],
        "lore": [],
        "total_characters": len(clipped),
    }


@router.post("/adventure/parse")
@limiter.limit("15/minute")
async def parse_adventure_docs(
    request: Request,
    files: list[UploadFile] = File(...),
    campaign_system: str = Form(DEFAULT_CAMPAIGN_SYSTEM_ID),
    _auth: None = Depends(verify_api_key),
):
    """Upload adventure docs (.txt/.md/.pdf) and return a parsed prep summary payload."""
    if not files:
        raise HTTPException(400, "Upload at least one document.")
    if len(files) > _MAX_ADVENTURE_FILES:
        raise HTTPException(400, f"Too many files. Max {_MAX_ADVENTURE_FILES} files per parse.")

    try:
        all_text_parts: list[str] = []
        uploaded_files: list[dict] = []
        pdf_raws: list[bytes] = []
        for upload in files:
            raw_peek = await upload.read()
            await upload.seek(0)
            text, meta = await _read_adventure_upload(upload)
            all_text_parts.append(text)
            uploaded_files.append(meta)
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix == ".pdf":
                pdf_raws.append(raw_peek)

        merged = "\n\n".join(all_text_parts)
        parsed = _parse_adventure_text(merged)
        system_id = normalize_campaign_system(campaign_system)
        parsed["system_id"] = system_id
        parsed["systemId"] = system_id
        parsed["system"] = get_campaign_system_preset(system_id)

        # normalize_campaign_entities expects NPCs/locations as dicts, not plain strings
        if isinstance(parsed.get("npcs"), list):
            coerced_npcs: list[dict] = []
            for n in parsed["npcs"]:
                if isinstance(n, str) and n.strip():
                    coerced_npcs.append({"name": n.strip(), "summary": "", "description": ""})
                elif isinstance(n, dict):
                    coerced_npcs.append(n)
            parsed["npcs"] = coerced_npcs
        if isinstance(parsed.get("locations"), list):
            coerced_locs: list[dict] = []
            for loc in parsed["locations"]:
                if isinstance(loc, str) and loc.strip():
                    coerced_locs.append({"name": loc.strip(), "description": ""})
                elif isinstance(loc, dict):
                    coerced_locs.append(loc)
            parsed["locations"] = coerced_locs

        try:
            cid = campaign_repository.create_from_parse_result(parsed)
            parsed["campaign_id"] = cid
        except Exception as e:
            logging.warning("Failed to persist fast-parse campaign to DB: %s", e)

        if pdf_raws and parsed.get("campaign_id") is not None:
            cid = int(parsed["campaign_id"])
            embedded_dir = _ASSETS_DIR / str(cid) / "embedded"
            _ensure_dir(embedded_dir)
            img_counter = 0
            for raw_pdf in pdf_raws:
                try:
                    _new_imgs, _pages = await run_in_threadpool(
                        _extract_embedded_images, raw_pdf, embedded_dir, img_counter, str(cid)
                    )
                    img_counter += len(_new_imgs)
                except Exception as ex:
                    logging.warning("PDF image extraction (fast parse) failed: %s", ex)

        return {"files": uploaded_files, **parsed}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Parse adventure failed: %s", e)
        msg = str(e) if str(e) else type(e).__name__
        raise HTTPException(500, f"Parse failed: {msg}") from e


def _extract_embedded_images(raw_pdf: bytes, embedded_dir: Path, start_counter: int, session_id: str) -> tuple[list[dict], int]:
    import fitz
    import hashlib
    raw_images: list[dict] = []
    total_pages = 0
    img_counter = start_counter
    seen_hashes = set()
    try:
        doc = fitz.open(stream=raw_pdf, filetype="pdf")
        total_pages = doc.page_count
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            for img in page.get_images(full=True):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    data = base_image["image"]

                    # 1. Deduplication Filter
                    img_hash = hashlib.md5(data).hexdigest()
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    # 2. Dimensions & Aspect Ratio Filter
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    if width < 150 or height < 150:
                        continue
                    aspect_ratio = width / height if height > 0 else 0
                    if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                        continue

                    ext = base_image["ext"]
                    img_counter += 1
                    fname = f"img_{img_counter:04d}.{ext}"
                    (embedded_dir / fname).write_bytes(data)
                    raw_images.append({
                        "idx": img_counter,
                        "page": page_num + 1,
                        "url": f"/campaign-assets/{session_id}/embedded/{fname}",
                    })
                except Exception:
                    continue
    except Exception as e:
        logging.warning("Image extraction during ai-parse failed: %s", e)
    return raw_images, total_pages


@router.post("/adventure/ai-parse")
@limiter.limit("10/minute")
async def ai_parse_adventure_docs(
    request: Request,
    files: list[UploadFile] = File(...),
    campaign_system: str = Form(DEFAULT_CAMPAIGN_SYSTEM_ID),
    _auth: None = Depends(verify_api_key),
):
    """Upload adventure docs and use Claude to extract a full structured campaign object."""
    if not files:
        raise HTTPException(400, "Upload at least one document.")
    if len(files) > _MAX_ADVENTURE_FILES:
        raise HTTPException(400, f"Too many files. Max {_MAX_ADVENTURE_FILES} files per parse.")

    all_text_parts: list[str] = []
    uploaded_files: list[dict] = []
    pdf_raws: list[bytes] = []  # keep raw bytes for image extraction

    for upload in files:
        # Peek raw bytes before _read_adventure_upload consumes the stream
        raw_peek = await upload.read()
        await upload.seek(0)
        text, meta = await _read_adventure_upload(upload)
        all_text_parts.append(text)
        uploaded_files.append(meta)
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix == ".pdf":
            pdf_raws.append(raw_peek)

    merged = "\n\n".join(all_text_parts)
    from app.services.ai_service import ai_full_parse, assign_images_to_entities
    system_id = normalize_campaign_system(campaign_system)
    try:
        result = await run_in_threadpool(ai_full_parse, merged)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    result["system_id"] = system_id
    result["systemId"] = system_id
    result["system"] = get_campaign_system_preset(system_id)

    # --- Persist first so embedded images land under campaign_id ---
    campaign_id: int | None = None
    try:
        campaign_id = campaign_repository.create_from_parse_result(result)
        result["campaign_id"] = campaign_id
    except Exception as e:
        logging.warning("Failed to persist campaign to DB: %s", e)

    asset_key = str(campaign_id) if campaign_id is not None else str(uuid.uuid4())
    _cleanup_old_sessions(_ASSETS_DIR)
    session_dir = _ASSETS_DIR / asset_key
    embedded_dir = session_dir / "embedded"
    _ensure_dir(embedded_dir)

    raw_images: list[dict] = []
    total_pages = 0
    img_counter = 0

    for raw_pdf in pdf_raws:
        new_images, pages = await run_in_threadpool(
            _extract_embedded_images, raw_pdf, embedded_dir, img_counter, asset_key
        )
        raw_images.extend(new_images)
        total_pages += pages
        img_counter += len(new_images)

    if raw_images:
        assigned = await run_in_threadpool(assign_images_to_entities, raw_images, result, total_pages or 1)
        result["images"] = assigned
    else:
        result["images"] = []

    return {"files": uploaded_files, **result}


class AnalyzePageBody(BaseModel):
    page_text: str = ""
    page_number: int = Field(1, ge=1)
    campaign_system: str = DEFAULT_CAMPAIGN_SYSTEM_ID
    image_url: str = ""


_ANALYZE_PAGE_PROMPT = """You are analyzing a single page from a Pathfinder tabletop RPG adventure module (Kingmaker style).
Extract ALL relevant content from this page. Do not skip anything.

This module uses these specific formats — learn them:

READ-ALOUD TEXT:
- Opening descriptive paragraphs at the start of a location section
- Text after a location header like "Oleg's Trading Post" or "Q. Rickety Bridge (Landmark)"
- Any text describing what players see, hear, or experience
- Example: "Oleg's trading post is surrounded by a wooden palisade that stands 10 feet high. At each corner of the palisade are 20-foot-square watchtowers..."

GM NOTES:
- Lettered/numbered location keys like "A5. Middens: Three 3-foot-deep composting pits"
- Mechanical info: DCs, trap stats, locked doors, hidden items
- Tactical notes the GM needs but players don't hear
- Example: "A8. Office: This room is where Oleg keeps his ledgers. DC 15 Perception to notice the hidden compartment."

NPCS:
- Named characters with descriptions and personality
- Format: Name (alignment, race, class, level) — any details given
- Example: "Oleg Leveton (CG male human expert 2) — stern and unimaginative, owns the trading post"
- Example: "Svetlana (NG female human expert 2) — Oleg's wife, pleaded with him to abandon the post"
- Example: "Jhod Kavken — traveling priest, has a recurring dream about a temple"

MONSTERS / VILLAINS:
- Stat block headers like "HAPPS BYDON CR 1/2" or "BEAR TRAP CR 1"
- Always include: name, CR, XP, AC, HP, type
- Villains are NPCs with combat stat blocks (LE/CE/NE alignment usually)
- Traps count as monsters — extract them with their trigger and effect
- Example villain: "Happs Bydon, CR 1/2, XP 200, Male human ranger 1, LE, AC 14, touch 12"
- Example trap: "Bear Trap, CR 1, XP 400, mechanical, Perception DC 15, Disable Device DC 20, Atk +10 melee"

SCENE TITLE:
- Major headings like "OLEG'S TRADING POST", "ARRIVAL AT OLEG'S", "C. TRAP-FILLED GLADE"
- Landmark names like "Q. RICKETY BRIDGE (LANDMARK)"

Return ONLY valid JSON — no markdown, no preamble:
{
  "scene_title": "the main heading or location name on this page, empty string if none",
  "read_aloud": "the opening descriptive paragraph for the location, written as something the GM reads to players. If multiple locations on the page, use the first/main one.",
  "gm_notes": "all lettered/numbered location keys and their descriptions, DCs, trap mechanics, tactical info. Preserve the A1/A2/Q format.",
  "npcs": [
    {
      "name": "full name",
      "role": "ally|villain|neutral|quest-giver",
      "description": "physical description and stat info",
      "personality": "how they act, what they want",
      "hp": "hp value as string",
      "ac": 0,
      "cr": "CR value as string e.g. CR 1/2"
    }
  ],
  "monsters": [
    {
      "name": "name",
      "hp": "hp value",
      "ac": 0,
      "cr": "CR value",
      "notes": "type, trigger if trap, key abilities"
    }
  ],
  "is_new_scene": true,
  "scene_type": "combat|social|exploration|trap|travel"
}

EXTRACTION RULES:
- A page with "ARRIVAL AT OLEG'S" is is_new_scene: true, scene_type: social
- A page with "TRAP-FILLED GLADE" is is_new_scene: true, scene_type: trap
- A page with only a map and location keys is is_new_scene: true, scene_type: exploration
- Happs Bydon and similar bandit leaders are monsters (villains with stat blocks)
- Bear Trap and similar are monsters (traps with stat blocks)
- Oleg, Svetlana, Jhod are npcs (named characters without combat stat blocks)
- If a page has ONLY a map image with no text, return empty strings and empty arrays
- Never return empty read_aloud if there is any descriptive text on the page"""


@router.post("/adventure/analyze-page")
@limiter.limit("30/minute")
async def analyze_single_page(
    request: Request,
    body: AnalyzePageBody,
    _auth: None = Depends(verify_api_key),
):
    """Send a single page of text (or image) to Claude for structured extraction."""
    from app.core.config import ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set. Add it to .env.")

    text = (body.page_text or "").strip()
    has_image = bool((body.image_url or "").strip())
    logging.info(
        "analyze-page: page=%d, text_len=%d, has_image=%s",
        body.page_number, len(text), has_image,
    )

    if not text and not has_image:
        return {"error": "No content to analyze — this page has no text and no image was provided."}

    raw_text = ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        if text:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"{_ANALYZE_PAGE_PROMPT}\n\n--- PAGE {body.page_number} ---\n{text[:12000]}"}
                ],
            )
        else:
            image_raw = body.image_url.strip()
            if "," in image_raw:
                base64_data = image_raw.split(",", 1)[1]
            else:
                base64_data = image_raw
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{_ANALYZE_PAGE_PROMPT}\n\nPage {body.page_number}: Analyze the image above.",
                        },
                    ],
                }],
            )

        raw_text = message.content[0].text if message.content else ""
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        return {"error": "Could not parse Claude response as JSON", "raw": raw_text[:2000]}
    except anthropic.AuthenticationError as e:
        raise HTTPException(401, f"Invalid ANTHROPIC_API_KEY: {e}")
    except Exception as e:
        logging.exception("analyze-page failed for page %d: %s", body.page_number, e)
        return {"error": str(e), "raw": ""}


# ── Chapter detection (regex only, no AI) ────────────────────────────────────

_HEADING_ALL_CAPS = re.compile(r"^([A-Z][A-Z\s'''\-:,]{5,})$")
_HEADING_PART_CHAPTER = re.compile(
    r"^(?:Part\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+)"
    r"|Chapter\s+\d+"
    r"|Section\s+\d+)\b",
    re.IGNORECASE,
)
_HEADING_LOCATION_KEY = re.compile(
    r"^[A-Z]\d*\.\s+.+",
)
_HEADING_LANDMARK = re.compile(
    r"^[A-Z]\.\s+.+\((?:Landmark|landmark)\)",
)


class DetectChaptersBody(BaseModel):
    pages_text: list[dict]


@router.post("/adventure/detect-chapters")
@limiter.limit("60/minute")
async def detect_chapters(
    request: Request,
    body: DetectChaptersBody,
    _auth: None = Depends(verify_api_key),
):
    """Scan page texts for chapter/section headings using regex heuristics."""
    chapters: list[dict] = []
    current: dict | None = None

    for entry in body.pages_text:
        page_num = int(entry.get("page", 0))
        text = str(entry.get("text", ""))
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        found_heading = None
        confidence = "low"

        for i, line in enumerate(lines):
            if len(line) > 120:
                continue

            if _HEADING_ALL_CAPS.match(line) and len(line.split()) >= 3:
                found_heading = line.title()
                has_prose_after = i + 1 < len(lines) and len(lines[i + 1]) > 60
                confidence = "high" if has_prose_after else "medium"
                break

            if _HEADING_PART_CHAPTER.match(line):
                found_heading = line.strip().rstrip(":")
                confidence = "high"
                break

            if _HEADING_LANDMARK.match(line):
                found_heading = line.strip()
                confidence = "high"
                break

            if (
                len(line) < 60
                and (line == line.upper() or line.istitle())
                and i + 1 < len(lines)
                and len(lines[i + 1]) > 60
            ):
                found_heading = line.title() if line == line.upper() else line
                confidence = "medium"
                break

        if found_heading:
            if current:
                chapters.append(current)
            current = {
                "title": found_heading,
                "start_page": page_num,
                "end_page": page_num,
                "confidence": confidence,
            }
        elif current:
            current["end_page"] = page_num

    if current:
        chapters.append(current)

    logging.info("detect-chapters: found %d chapters across %d pages", len(chapters), len(body.pages_text))
    return {"chapters": chapters}


# ── Analyze full chapter (multi-page Claude call) ────────────────────────────

class AnalyzeChapterBody(BaseModel):
    chapter_title: str = ""
    pages_text: list[str]
    page_numbers: list[int]
    campaign_system: str = "pathfinder1e"


@router.post("/adventure/analyze-chapter")
@limiter.limit("20/minute")
async def analyze_chapter(
    request: Request,
    body: AnalyzeChapterBody,
    _auth: None = Depends(verify_api_key),
):
    """Send combined chapter pages to Claude for structured extraction."""
    from app.core.config import ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set. Add it to .env.")

    if not body.pages_text or not body.page_numbers:
        return {"error": "No pages provided."}

    combined_parts = []
    for i, page_text in enumerate(body.pages_text):
        pg = body.page_numbers[i] if i < len(body.page_numbers) else i + 1
        combined_parts.append(f"--- PAGE {pg} ---\n{page_text}")
    combined_text = "\n\n".join(combined_parts)

    chapter_prompt = (
        f"You are analyzing pages {body.page_numbers[0]}–{body.page_numbers[-1]} "
        f"of a tabletop RPG module. These pages all belong to the chapter or location: "
        f'"{body.chapter_title}".\n\n'
        f"Treat all these pages as ONE unified scene. The map and room descriptions "
        f"belong together. Sub-sections like \"Arrival at Oleg's\" or \"Ambush at Oleg's\" "
        f"should be captured in gm_notes but the whole chapter is one scene.\n\n"
        f"{_ANALYZE_PAGE_PROMPT}\n\n"
        f"Combined chapter text:\n{combined_text[:15000]}"
    )

    logging.info(
        "analyze-chapter: title=%r, pages=%s, combined_len=%d",
        body.chapter_title, body.page_numbers, len(combined_text),
    )

    raw_text = ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": chapter_prompt}
            ],
        )
        raw_text = message.content[0].text if message.content else ""
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        return {"error": "Could not parse Claude response as JSON", "raw": raw_text[:2000]}
    except anthropic.AuthenticationError as e:
        raise HTTPException(401, f"Invalid ANTHROPIC_API_KEY: {e}")
    except Exception as e:
        logging.exception("analyze-chapter failed for %r: %s", body.chapter_title, e)
        return {"error": str(e), "raw": ""}


# ── Parse PDF chunk via native document upload to Claude ─────────────────────

_MAX_CLAUDE_PDF_CHUNK_PAGES = 100


def _extract_pdf_page_range_for_claude(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
) -> tuple[bytes, str, bool]:
    """
    Build a smaller PDF containing only the requested 1-based page range.
    Returns (bytes, human_note, used_slice). On failure returns (original, reason, False).
    """
    if not pdf_bytes:
        return pdf_bytes, "empty", False
    src = None
    dst = None
    try:
        src = fitz.open(stream=pdf_bytes, filetype="pdf")
        if getattr(src, "needs_pass", False):
            if not src.authenticate(""):
                return pdf_bytes, "encrypted_needs_password", False
        n = src.page_count
        if n < 1:
            return pdf_bytes, "no_pages", False
        start_idx = max(0, min(int(start_page) - 1, n - 1))
        if end_page and int(end_page) > 0:
            end_idx = min(int(end_page) - 1, n - 1)
        else:
            end_idx = n - 1
        if start_idx > end_idx:
            return pdf_bytes, "bad_range", False
        span = end_idx - start_idx + 1
        truncated = False
        if span > _MAX_CLAUDE_PDF_CHUNK_PAGES:
            end_idx = start_idx + _MAX_CLAUDE_PDF_CHUNK_PAGES - 1
            truncated = True
        dst = fitz.open()
        dst.insert_pdf(src, from_page=start_idx, to_page=end_idx)
        try:
            out = dst.tobytes(deflate=True, garbage=3, clean=True)
        except TypeError:
            out = dst.tobytes()
        note = f"sliced_pages_{start_idx + 1}-{end_idx + 1}"
        if truncated:
            note += f"_truncated_max_{_MAX_CLAUDE_PDF_CHUNK_PAGES}"
        return out, note, True
    except Exception as e:
        logging.warning("PDF page extract for Claude failed, using full file: %s", e)
        return pdf_bytes, f"slice_failed:{e}", False
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()


@router.post("/adventure/parse-pdf-chunk")
@limiter.limit("10/minute")
async def parse_pdf_chunk(
    request: Request,
    pdf_file: UploadFile = File(...),
    chunk_title: str = Form(""),
    start_page: int = Form(1),
    end_page: int = Form(0),
    campaign_system: str = Form("pathfinder1e"),
    _auth: None = Depends(verify_api_key),
):
    """Send a PDF directly to Claude as a native document for structured extraction."""
    from app.core.config import ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set. Add it to .env.")

    pdf_bytes = await pdf_file.read()
    logging.info(
        "parse-pdf-chunk: title=%r, start=%d, end=%d, pdf_size=%d bytes",
        chunk_title, start_page, end_page, len(pdf_bytes),
    )

    if not pdf_bytes:
        return {"error": "Uploaded PDF is empty."}

    payload_bytes, slice_note, used_slice = _extract_pdf_page_range_for_claude(
        pdf_bytes, start_page, end_page
    )
    logging.info(
        "parse-pdf-chunk: slice_note=%s, used_slice=%s, payload_size=%d",
        slice_note, used_slice, len(payload_bytes),
    )

    if slice_note == "encrypted_needs_password":
        return {
            "error": "This PDF is password-protected. Remove security in a PDF app and upload again.",
            "raw": "",
        }

    pdf_b64 = base64.b64encode(payload_bytes).decode("utf-8")

    if used_slice:
        orig_hi = end_page if end_page and int(end_page) > 0 else None
        range_desc = f"original module pages {start_page}" + (
            f"–{orig_hi}" if orig_hi else " through end of document"
        )
        page_instruction = (
            f"The attached PDF is a server-side extract ({range_desc}). "
            "It contains only those pages — analyze the entire attachment."
        )
    elif end_page > 0:
        page_instruction = f"Focus ONLY on pages {start_page} to {end_page} of this PDF. Ignore all other pages."
    else:
        page_instruction = f"Start from page {start_page} of this PDF."

    prompt = f"""{page_instruction}
Chapter/Section being extracted: "{chunk_title or 'Full document'}"

{_ANALYZE_PAGE_PROMPT}"""

    raw_text = ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }],
        )
        raw_text = message.content[0].text if message.content else ""
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        return {"error": "Could not parse Claude response as JSON", "raw": raw_text[:2000]}
    except anthropic.AuthenticationError as e:
        raise HTTPException(401, f"Invalid ANTHROPIC_API_KEY: {e}")
    except anthropic.BadRequestError as e:
        logging.warning("parse-pdf-chunk Anthropic 400: %s", e)
        hint = (
            "Claude could not read this PDF. Try: (1) set an explicit End page so only a smaller range is sent, "
            "(2) export or print-to-PDF from your viewer to flatten the file, "
            "(3) remove password protection. "
            f"Details: {e}"
        )
        return {"error": hint, "raw": str(e)}
    except Exception as e:
        logging.exception("parse-pdf-chunk failed for %r: %s", chunk_title, e)
        return {"error": str(e), "raw": ""}


@router.get("/api/campaigns")
@limiter.limit("60/minute")
async def list_campaigns(request: Request, _auth: None = Depends(verify_api_key)):
    """Return all saved campaigns (id, title, summary) ordered newest first."""
    return campaign_repository.list_all()


@router.get("/api/campaign-systems")
@limiter.limit("60/minute")
async def list_campaign_systems(request: Request, _auth: None = Depends(verify_api_key)):
    """Return supported campaign-system presets for campaign creation and later flavor layers."""
    return {
        "default_system_id": DEFAULT_CAMPAIGN_SYSTEM_ID,
        "systems": list_campaign_system_presets(),
    }


@router.get("/api/campaigns/{campaign_id}")
@limiter.limit("60/minute")
async def get_campaign(campaign_id: int, request: Request, _auth: None = Depends(verify_api_key)):
    """Return a single campaign payload (full JSON when available, relational fallback otherwise)."""
    data = campaign_repository.get_by_id(campaign_id)
    if data is None:
        raise HTTPException(404, "Campaign not found")
    return data


@router.get("/api/campaigns/{campaign_id}/images")
@limiter.limit("60/minute")
async def list_campaign_images(
    campaign_id: int,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """List embedded image URLs under static/campaign_assets/{campaign_id}/embedded/."""
    if campaign_repository.get_by_id(campaign_id) is None:
        raise HTTPException(404, "Campaign not found")
    return {"images": _list_campaign_embedded_image_urls(campaign_id)}


@router.delete("/api/campaigns/{campaign_id}")
@limiter.limit("30/minute")
async def delete_campaign(campaign_id: int, request: Request, _auth: None = Depends(verify_api_key)):
    """Delete a campaign and all related NPCs, scenes, and locations."""
    if not campaign_repository.delete(campaign_id):
        raise HTTPException(404, "Campaign not found")
    return {"deleted": campaign_id}


class AssignVoiceBody(BaseModel):
    voice_id: str


class SuggestNpcVoiceBody(BaseModel):
    npc_id: str = Field(..., min_length=1)


@router.post("/api/campaigns/{campaign_id}/documents")
@limiter.limit("20/minute")
async def upload_campaign_documents(
    campaign_id: int,
    request: Request,
    files: list[UploadFile] = File(...),
    _auth: None = Depends(verify_api_key),
):
    """Upload and index campaign source documents for Campaign Brain retrieval."""
    if not files:
        raise HTTPException(400, "Upload at least one campaign document.")

    uploads: list[dict[str, object]] = []
    for upload in files:
        content = await upload.read()
        uploads.append(
            {
                "filename": upload.filename or "document",
                "content_type": upload.content_type or "",
                "content": content,
            }
        )

    from app.services.campaign_brain_service import ingest_campaign_documents

    try:
        return await run_in_threadpool(ingest_campaign_documents, campaign_id, uploads)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))


@router.get("/api/campaigns/{campaign_id}/documents")
@limiter.limit("60/minute")
async def list_campaign_documents(
    campaign_id: int,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """List indexed campaign documents for Campaign Brain."""
    from app.services.campaign_brain_service import list_campaign_documents as list_campaign_documents_service

    data = campaign_repository.get_by_id(campaign_id)
    if data is None:
        raise HTTPException(404, "Campaign not found")
    return {"campaign_id": campaign_id, "documents": await run_in_threadpool(list_campaign_documents_service, campaign_id)}


class SceneRecordPatchBody(BaseModel):
    """PATCH scene: read_aloud, GM notes (notes or gm_notes), and npc name list."""

    read_aloud: Optional[str] = None
    notes: Optional[str] = None
    gm_notes: Optional[str] = None
    npcs: Optional[list[str]] = None


class NpcGlobalVoiceBody(BaseModel):
    voice_id: str = ""


class SessionActiveSceneBody(BaseModel):
    active_scene_index: int = Field(..., ge=0)
    campaign_id: Optional[int] = None


@router.patch("/api/campaigns/{campaign_id}/npcs/{npc_name}/voice")
@limiter.limit("120/minute")
async def assign_npc_voice(
    campaign_id: int,
    npc_name: str,
    body: AssignVoiceBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """Persist voice assignment for an NPC within a campaign. Updates relational row + data_json."""
    updated = campaign_repository.assign_npc_voice(campaign_id, npc_name, body.voice_id)
    if not updated:
        # Campaign may not be in DB yet; return 200 so frontend doesn't treat as hard error
        return {"ok": False, "reason": "campaign or npc not found in db"}
    return {"ok": True, "campaign_id": campaign_id, "npc_name": npc_name, "voice_id": body.voice_id}


@router.patch("/api/campaigns/{campaign_id}/scenes/{scene_ref}")
@router.patch("/campaigns/{campaign_id}/scenes/{scene_ref}")
@limiter.limit("120/minute")
async def patch_campaign_scene_record(
    campaign_id: int,
    scene_ref: str,
    body: SceneRecordPatchBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """
    Patch a scene by numeric DB id or title / JSON id (single path segment).
    Updates read_aloud, gm_notes (via notes or gm_notes), and npcs list in SQLite + data_json.
    Returns the updated scene record.
    """
    notes_eff = body.gm_notes if body.gm_notes is not None else body.notes
    if body.read_aloud is None and notes_eff is None and body.npcs is None:
        raise HTTPException(400, detail="no fields to update")
    rec = await run_in_threadpool(
        campaign_repository.patch_campaign_scene,
        campaign_id,
        scene_ref,
        read_aloud=body.read_aloud,
        gm_notes=notes_eff,
        npcs=body.npcs,
    )
    if rec is None:
        raise HTTPException(404, detail="campaign or scene not found")
    return rec


@router.patch("/api/npcs/{npc_id}/voice")
@router.patch("/npcs/{npc_id}/voice")
@limiter.limit("120/minute")
async def patch_npc_voice_global(
    npc_id: str,
    body: NpcGlobalVoiceBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """Assign voice_id to an NPC by integer id or name. Returns updated NPC record."""
    rec = await run_in_threadpool(campaign_repository.patch_npc_voice_global, npc_id, body.voice_id)
    if rec is None:
        raise HTTPException(404, detail="NPC not found")
    return rec


@router.post("/api/sessions/{session_id}/scene")
@router.post("/sessions/{session_id}/scene")
@limiter.limit("120/minute")
async def post_session_active_scene(
    session_id: str,
    body: SessionActiveSceneBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """Persist active scene index for a client session (game_sessions table)."""
    try:
        return await run_in_threadpool(
            campaign_repository.upsert_session_active_scene,
            session_id,
            body.active_scene_index,
            body.campaign_id,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc


@router.post("/npc/suggest-voice")
@limiter.limit("120/minute")
async def suggest_npc_voice(
    body: SuggestNpcVoiceBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
    owner_id: Optional[str] = Depends(get_owner_id),
):
    npc = campaign_repository.get_npc_record(body.npc_id)
    if npc is None:
        raise HTTPException(404, "NPC not found")

    suggested_voice = await run_in_threadpool(suggest_voice_for_npc, npc, owner_id)
    return {"suggested_voice": suggested_voice}


class SessionEventBody(BaseModel):
    type: str = "assistant"
    text: str
    scene_id: Optional[str] = None
    session_id: Optional[str] = None
    id: Optional[str] = None
    created_at: Optional[str] = None


class SessionMemoryEventBody(BaseModel):
    event_type: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    npc_id: Optional[str] = None
    tags: Optional[list[str]] = None
    campaign_id: Optional[int] = None
    scene_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class StartSessionBody(BaseModel):
    campaign_id: int
    scene_id: str
    narrator_voice: str = Field(..., min_length=1)


@router.post("/api/campaigns/{campaign_id}/events")
@limiter.limit("300/minute")
async def append_campaign_event(
    campaign_id: int,
    body: SessionEventBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """Append a session event (action log entry) to a campaign. Returns event id."""
    try:
        event_id = campaign_repository.append_session_event(
            campaign_id=campaign_id,
            event_type=body.type,
            text=body.text,
            scene_id=body.scene_id,
            session_id=body.session_id,
            event_id=body.id,
            created_at=body.created_at,
        )
        return {"ok": True, "event_id": event_id, "campaign_id": campaign_id}
    except Exception as exc:
        logging.warning("Failed to append session event for campaign %s: %s", campaign_id, exc)
        raise HTTPException(500, "Failed to store session event")


@router.get("/api/campaigns/{campaign_id}/events")
@limiter.limit("60/minute")
async def get_campaign_events(
    campaign_id: int,
    request: Request,
    scene_id: Optional[str] = None,
    limit: int = 100,
    _auth: None = Depends(verify_api_key),
):
    """Return session events for a campaign, optionally filtered by scene_id."""
    events = campaign_repository.get_session_events(campaign_id, scene_id=scene_id, limit=min(limit, 500))
    return {"campaign_id": campaign_id, "events": events}


@router.post("/session/event")
@limiter.limit("180/minute")
async def record_session_memory_event(
    body: SessionMemoryEventBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """Record an important session-memory event against the active live session."""
    from app.services.session_memory_service import record_event

    try:
        memory_event = await run_in_threadpool(
            record_event,
            event_type=body.event_type,
            description=body.description,
            npc_id=body.npc_id,
            tags=body.tags,
            campaign_id=body.campaign_id,
            scene_id=body.scene_id,
            session_id=body.session_id,
            timestamp=body.timestamp,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logging.warning("Failed to record session memory event: %s", exc)
        raise HTTPException(500, "Failed to store session memory event")

    return {"ok": True, "session_memory": memory_event}


@router.post("/session/start")
@limiter.limit("60/minute")
async def start_session_route(
    body: StartSessionBody,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """Start a guided live session and return the refreshed campaign payload."""
    try:
        return await run_in_threadpool(
            start_live_session,
            int(body.campaign_id),
            str(body.scene_id),
            str(body.narrator_voice),
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _extract_images_from_pdf(
    raw: bytes,
    embedded_dir: Path,
    pages_dir: Path,
    img_counter_start: int,
    session_id: str,
) -> tuple[list[str], list[str]]:
    import fitz
    import hashlib
    from pdf2image import convert_from_bytes
    embedded_urls: list[str] = []
    page_urls: list[str] = []
    img_counter = img_counter_start
    seen_hashes = set()

    # --- Embedded images via PyMuPDF (fitz) ---
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            for img in page.get_images(full=True):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    data = base_image["image"]

                    img_hash = hashlib.md5(data).hexdigest()
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    if width < 150 or height < 150:
                        continue
                    aspect_ratio = width / height if height > 0 else 0
                    if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                        continue
                    ext = base_image["ext"]
                    img_counter += 1
                    fname = f"img_{img_counter:04d}.{ext}"
                    (embedded_dir / fname).write_bytes(data)
                    embedded_urls.append(f"/campaign-assets/{session_id}/embedded/{fname}")
                except Exception:
                    continue
    except Exception as e:
        logging.warning("Embedded image extraction failed: %s", e)
    # --- Page thumbnails via pdf2image ---
    try:
        pages = convert_from_bytes(raw, dpi=96, fmt="jpeg", thread_count=2)
        for i, page_img in enumerate(pages):
            fname = f"page_{i + 1:04d}.jpg"
            page_img.save(str(pages_dir / fname), "JPEG", quality=75)
            page_urls.append(f"/campaign-assets/{session_id}/pages/{fname}")
    except Exception as e:
        logging.warning("Page thumbnail extraction failed: %s", e)
    return embedded_urls, page_urls


@router.post("/adventure/images")
@limiter.limit("10/minute")
async def extract_adventure_images(
    request: Request,
    files: list[UploadFile] = File(...),
    _auth: None = Depends(verify_api_key),
):
    """
    Extract embedded images and page thumbnails from uploaded PDFs.
    Saves to static/campaign_assets/<session_id>/ and returns URLs.
    Returns: {"embedded": [...urls], "pages": [...urls]}
    """
    if not files:
        raise HTTPException(400, "Upload at least one document.")

    _cleanup_old_sessions(_ASSETS_DIR)
    session_id = str(uuid.uuid4())
    session_dir = _ASSETS_DIR / session_id
    embedded_dir = session_dir / "embedded"
    pages_dir = session_dir / "pages"
    _ensure_dir(embedded_dir)
    _ensure_dir(pages_dir)

    embedded_urls: list[str] = []
    page_urls: list[str] = []
    img_counter = 0

    for upload in files:
        if not upload.filename:
            continue
        if Path(upload.filename).suffix.lower() != ".pdf":
            continue
        raw = await upload.read()
        if not raw:
            continue
        new_embedded, new_pages = await run_in_threadpool(
            _extract_images_from_pdf, raw, embedded_dir, pages_dir, img_counter, session_id
        )
        img_counter += len(new_embedded)
        embedded_urls.extend(new_embedded)
        page_urls.extend(new_pages)

    return {
        "embedded": embedded_urls,
        "pages": page_urls,
        "total_embedded": len(embedded_urls),
        "total_pages": len(page_urls),
    }


# --- Co-DM RAG query ---

class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: Optional[str] = None


class CampaignQueryRequest(BaseModel):
    campaign_id: int
    question: str
    top_k: int = 5


@router.post("/rag/query")
@limiter.limit("60/minute")
async def rag_query(req: RagQueryRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """Semantic search over ingested campaign documents. Returns top_k relevant chunks."""
    from app.infrastructure.retrieval.pinecone_retriever import retrieve
    try:
        results = await run_in_threadpool(retrieve, req.query, top_k=req.top_k, doc_type=req.doc_type)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"results": results}


@router.post("/campaign/query")
@limiter.limit("60/minute")
async def campaign_query(req: CampaignQueryRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """Query campaign-ingested documents and return matched chunks plus a concise answer."""
    from app.services.campaign_brain_service import query_campaign_documents

    try:
        return await run_in_threadpool(query_campaign_documents, req.campaign_id, req.question, req.top_k)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))


# --- Co-DM LLM brain (Sprint 3) ---

class BrainQueryRequest(BaseModel):
    query: str


class SessionAssistantAnalyzeRequest(BaseModel):
    transcript_entries: list[str]
    scene_title: str = ""
    scene_summary: str = ""
    location_name: str = ""
    active_quests: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    npcs: list[dict] = Field(default_factory=list)


@router.post("/brain/query")
@limiter.limit("30/minute")
async def brain_query(req: BrainQueryRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """
    Classify intent, optionally fetch RAG context, call Claude, return structured response.
    Returns: {type, intent, content, sources}
    """
    from app.services.llm_orchestrator import handle_query
    try:
        result = await run_in_threadpool(handle_query, req.query)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return result


@router.post("/session-assistant/analyze")
@limiter.limit("20/minute")
async def session_assistant_analyze(
    req: SessionAssistantAnalyzeRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
):
    """
    Analyze recent transcript lines and return actionable live-play suggestions.
    Returns: {"suggestions": [{type, title, text, npc_name, spoken_text, action_prompt}, ...]}
    """
    from app.services.ai_service import analyze_session_context

    if not req.transcript_entries:
        return {"suggestions": []}

    try:
        suggestions = await run_in_threadpool(
            analyze_session_context,
            transcript_entries=req.transcript_entries,
            scene_title=req.scene_title,
            scene_summary=req.scene_summary,
            location_name=req.location_name,
            active_quests=req.active_quests,
            recent_events=req.recent_events,
            npcs=req.npcs,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"suggestions": suggestions}


# --- Co-DM NPC Generator (Sprint 4) ---

class NpcGenerateRequest(BaseModel):
    genre: str
    location: str
    name: str
    role: str


@router.post("/npc/generate")
@limiter.limit("10/minute")
async def npc_generate(req: NpcGenerateRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """
    Stream a full NPC profile as Server-Sent Events.
    Each event: data: {"token": "<text>"}\\n\\n
    Final event: data: {"done": true}\\n\\n
    Error event: data: {"error": "<message>"}\\n\\n
    """
    import json as _json
    from app.services.npc_generator_service import generate_npc_stream

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


# --- Co-GM NPC Dialogue ---

class DialogueRequest(BaseModel):
    npc_name: str
    personality: str
    situation: str
    conversation_history: list[dict]
    faction: str = ""
    scene_id: Optional[str] = None
    scene_summary: str = ""
    location_name: str = ""
    recent_events: list[str] = Field(default_factory=list)
    scene_npcs: list[str] = Field(default_factory=list)
    related_quests: list[str] = Field(default_factory=list)
    codex_titles: list[str] = Field(default_factory=list)


class SceneTriggerBody(BaseModel):
    scene_id: str
    trigger_name: str


class SceneActivateBody(BaseModel):
    scene_id: str
    reset_atmosphere_override: bool = False


class SceneSuggestionBody(BaseModel):
    current_scene_id: str
    player_action: str = ""


class EncounterLaunchBody(BaseModel):
    encounter_id: str


@router.post("/ai/dialogue")
@limiter.limit("20/minute")
async def ai_dialogue(req: DialogueRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """
    Generate a short in-character NPC line via Claude.
    Returns: {"dialogue": "<spoken line>"}
    """
    from app.services.ai_service import generate_dialogue
    from app.services.live_context_service import build_scene_live_context
    live_context_lines: list[str] = []
    if req.scene_id:
        try:
            live_context = await run_in_threadpool(build_scene_live_context, scene_id=req.scene_id)
        except Exception:
            live_context = {}
        summary = str((live_context or {}).get("summary") or "").strip()
        if summary:
            live_context_lines.append(summary)
    if req.scene_summary.strip():
        live_context_lines.append(f"Scene summary: {req.scene_summary.strip()}")
    if req.location_name.strip():
        live_context_lines.append(f"Location: {req.location_name.strip()}")
    if req.scene_npcs:
        live_context_lines.append("NPCs in scene: " + ", ".join(item for item in req.scene_npcs if str(item).strip()))
    if req.related_quests:
        live_context_lines.append("Related quests: " + ", ".join(item for item in req.related_quests if str(item).strip()))
    if req.codex_titles:
        live_context_lines.append("Relevant codex: " + ", ".join(item for item in req.codex_titles if str(item).strip()))
    if req.recent_events:
        live_context_lines.append("Recent events: " + " | ".join(item for item in req.recent_events if str(item).strip()))
    try:
        line = await run_in_threadpool(
            generate_dialogue,
            npc_name=req.npc_name,
            personality=req.personality,
            situation=req.situation,
            conversation_history=req.conversation_history,
            faction=req.faction,
            live_context_summary="\n".join(line for line in live_context_lines if line),
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"dialogue": line}


@router.post("/scene/trigger")
@limiter.limit("30/minute")
async def scene_trigger(body: SceneTriggerBody, request: Request, _auth: None = Depends(verify_api_key)):
    """
    Execute a scene trigger through the live domain and return text plus optional WAV audio.
    """
    from app.domain.live.scene_control import execute_scene_trigger

    try:
        payload = await run_in_threadpool(execute_scene_trigger, body.scene_id, body.trigger_name)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        increment("errors_total")
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        increment("errors_total")
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        increment("errors_total")
        logging.exception("Scene trigger execution failed")
        raise HTTPException(503, str(exc))

    voice_id = str(payload.get("voice_id") or "").strip()
    if voice_id:
        request.state.voice_id = voice_id

    audio = payload.pop("audio", None)
    sample_rate = payload.pop("sample_rate", None)
    if audio is not None and sample_rate:
        increment("tts_requests_total")
        payload["audio_base64"] = _audio_to_wav_base64(audio, int(sample_rate))
        payload["mime_type"] = payload.get("mime_type") or "audio/wav"
    elif payload.get("audio_base64"):
        increment("tts_requests_total")

    return payload


@router.post("/scene/activate")
@limiter.limit("60/minute")
async def activate_scene_route(body: SceneActivateBody, request: Request, _auth: None = Depends(verify_api_key)):
    from app.domain.live.scene_control import activate_scene

    try:
        return await run_in_threadpool(activate_scene, body.scene_id, bool(body.reset_atmosphere_override))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/scene/combat-start")
@limiter.limit("60/minute")
async def scene_combat_start(body: SceneActivateBody, request: Request, _auth: None = Depends(verify_api_key)):
    from app.domain.live.scene_control import start_scene_combat

    try:
        return await run_in_threadpool(start_scene_combat, body.scene_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/scene/suggestions")
@limiter.limit("120/minute")
async def scene_suggestions(body: SceneSuggestionBody, request: Request, _auth: None = Depends(verify_api_key)):
    from app.domain.live.scene_control import suggest_next_scenes

    try:
        payload = await run_in_threadpool(suggest_next_scenes, body.current_scene_id, body.player_action)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return {"scenes": payload.get("suggested_scenes", [])}


@router.post("/encounter/launch")
@limiter.limit("30/minute")
async def encounter_launch(body: EncounterLaunchBody, request: Request, _auth: None = Depends(verify_api_key)):
    from app.domain.live.encounter_control import launch_encounter

    try:
        payload = await run_in_threadpool(launch_encounter, body.encounter_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        logging.exception("Encounter launch failed")
        raise HTTPException(503, str(exc))

    enemy_voice_id = str(((payload.get("enemy_dialogue_audio") or {}).get("voice_id")) or "").strip()
    narration_voice_id = str(((payload.get("narration_audio") or {}).get("voice_id")) or "").strip()
    if enemy_voice_id:
        request.state.voice_id = enemy_voice_id
    elif narration_voice_id:
        request.state.voice_id = narration_voice_id

    return payload


# --- TTS: preset or custom voice ---
@router.post("/tts")
@limiter.limit(RATE_LIMIT_TTS or "1000/minute")
async def tts_endpoint(
    request: Request,
    text: str = Form(...),
    _auth: None = Depends(verify_api_key),
    language_tag: str = Form("en"),
    voice_id: Optional[str] = Form(None),
    temperature: float = Form(DEFAULT_TTS_TEMPERATURE),
    top_p: float = Form(DEFAULT_TTS_TOP_P),
    repetition_penalty: float = Form(DEFAULT_TTS_REPETITION_PENALTY),
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
            if is_hume_provider():
                raise HTTPException(404, "Voice not found")
            speaker_emb_path = load_embedding_path(voice_id)
        if not speaker_emb_path:
            raise HTTPException(404, "Voice not found")

    # Option B: One-off reference audio (Pocket loads voice from WAV path)
    elif reference_audio and reference_audio.filename:
        if is_hume_provider():
            raise HTTPException(501, "Reference-audio cloning is not supported when TTS_PROVIDER=hume.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await reference_audio.read())
            tmp_path = tmp.name
        try:
            audio, sr = await run_in_threadpool(
                tts_generate,
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
        audio, sr = await run_in_threadpool(
            tts_generate,
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


class NpcDialogueBody(BaseModel):
    npc_id: str
    text: str


class NarrateSceneBody(BaseModel):
    scene_id: str


class GenerateNpcDialogueBody(BaseModel):
    npc_id: str
    player_input: str
    scene_id: Optional[str] = None


class NarrateAnswerBody(BaseModel):
    campaign_id: int
    answer: str
    voice_id: Optional[str] = None


def _synthesize_text(
    *,
    text: str,
    voice_id: str,
    language_tag: str = "en",
):
    supported = _lang_tags()
    lang_tag = (language_tag or "").strip() or "en"
    if lang_tag not in supported and supported:
        lang_tag = supported[0]
    speaker_emb_path = _resolve_voice_target(voice_id)
    return tts_generate(
        text.strip(),
        language_tag=lang_tag,
        speaker_emb_path=speaker_emb_path,
        temperature=DEFAULT_TTS_TEMPERATURE,
        top_p=DEFAULT_TTS_TOP_P,
        repetition_penalty=DEFAULT_TTS_REPETITION_PENALTY,
    )


@router.post("/tts/narrate")
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
        from app.infrastructure.tasks.celery_app import narrate_task
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
        if is_hume_provider():
            raise HTTPException(404, "Voice not found")
        speaker_emb_path = load_embedding_path(body.voice_id)
    if not speaker_emb_path:
        raise HTTPException(404, "Voice not found")

    supported = _lang_tags()
    lang_tag = (body.language_tag or "").strip() or "en"
    if lang_tag not in supported and supported:
        lang_tag = supported[0]
    language_tag = lang_tag

    def _run_narrate_chunks() -> tuple[list, int]:
        audio_list: list = []
        sr_out: Optional[int] = None
        for chunk in chunks:
            audio, sr = tts_generate(
                chunk,
                language_tag=language_tag,
                speaker_emb_path=speaker_emb_path,
                temperature=DEFAULT_TTS_TEMPERATURE,
                top_p=DEFAULT_TTS_TOP_P,
                repetition_penalty=DEFAULT_TTS_REPETITION_PENALTY,
            )
            if sr_out is None:
                sr_out = sr
            audio_list.append(audio)
        return audio_list, sr_out

    try:
        audio_list, sr_out = await run_in_threadpool(_run_narrate_chunks)
    except ValueError as e:
        increment("errors_total")
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        increment("errors_total")
        raise HTTPException(404, str(e))
    except ImportError as e:
        increment("errors_total")
        logging.exception("TTS model or dependency failed to load")
        raise HTTPException(503, f"TTS model unavailable: {e!s}")
    except RuntimeError as e:
        increment("errors_total")
        logging.exception("Narrate TTS failed")
        raise HTTPException(500, str(e))
    except Exception as e:
        increment("errors_total")
        logging.exception("Narrate failed with unexpected error")
        raise HTTPException(503, f"TTS error: {type(e).__name__}: {e!s}")

    concatenated = np.concatenate(audio_list)
    increment("tts_requests_total")
    buf = io.BytesIO()
    sf.write(buf, concatenated, sr_out, format="WAV")
    buf.seek(0)
    response = StreamingResponse(buf, media_type="audio/wav")
    response.headers["Content-Disposition"] = 'attachment; filename="narration.wav"'
    return response


@router.post("/tts/npc-dialogue")
@limiter.limit("20/minute")
async def tts_npc_dialogue(request: Request, body: NpcDialogueBody, _auth: None = Depends(verify_api_key)):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "No text")

    npc = await run_in_threadpool(campaign_repository.get_npc_record, body.npc_id)
    if not npc:
        raise HTTPException(404, "NPC not found")

    voice_id = str(npc.get("voice_id") or "").strip()
    if not voice_id:
        raise HTTPException(400, "NPC has no assigned voice.")
    request.state.voice_id = voice_id

    try:
        audio, sr = await run_in_threadpool(
            _synthesize_text,
            text=text,
            voice_id=voice_id,
        )
    except ValueError as e:
        increment("errors_total")
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        increment("errors_total")
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        increment("errors_total")
        logging.exception("NPC dialogue TTS failed")
        raise HTTPException(500, str(e))

    increment("tts_requests_total")
    filename = f"npc-{npc.get('id') or body.npc_id}-dialogue.wav"
    return _audio_to_wav_response(audio, sr, filename=filename)


@router.post("/tts/narrate-scene")
@limiter.limit("10/minute")
async def tts_narrate_scene(request: Request, body: NarrateSceneBody, _auth: None = Depends(verify_api_key)):
    scene = await run_in_threadpool(campaign_repository.get_scene_record, body.scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")

    text = str(scene.get("read_aloud") or scene.get("notes") or "").strip()
    if not text:
        raise HTTPException(400, "Scene has no narration text.")

    voice_id = str(scene.get("narrator_voice_id") or DEFAULT_VOICE_ID or "").strip()
    if not voice_id:
        raise HTTPException(400, "No narrator voice configured for this scene or campaign.")
    request.state.voice_id = voice_id

    try:
        audio, sr = await run_in_threadpool(
            _synthesize_text,
            text=text,
            voice_id=voice_id,
        )
    except ValueError as e:
        increment("errors_total")
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        increment("errors_total")
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        increment("errors_total")
        logging.exception("Scene narration TTS failed")
        raise HTTPException(500, str(e))

    increment("tts_requests_total")
    filename = f"scene-{scene.get('id') or body.scene_id}-narration.wav"
    return _audio_to_wav_response(audio, sr, filename=filename)


@router.post("/tts/narrate-answer")
@limiter.limit("20/minute")
async def tts_narrate_answer(request: Request, body: NarrateAnswerBody, _auth: None = Depends(verify_api_key)):
    text = (body.answer or "").strip()
    if not text:
        raise HTTPException(400, "Answer text is required.")
    if len(text) > MAX_TOTAL_CHARS:
        raise HTTPException(400, f"Answer exceeds {MAX_TOTAL_CHARS} characters")

    voice_id = (
        str(body.voice_id or "").strip()
        or await run_in_threadpool(campaign_repository.get_narrator_voice_id, body.campaign_id)
        or str(DEFAULT_VOICE_ID or "").strip()
    )
    if not voice_id:
        raise HTTPException(400, "No narrator voice configured for this campaign.")
    request.state.voice_id = voice_id

    try:
        audio, sr = await run_in_threadpool(
            _synthesize_text,
            text=text,
            voice_id=voice_id,
        )
    except ValueError as exc:
        increment("errors_total")
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        increment("errors_total")
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        increment("errors_total")
        logging.exception("Campaign answer narration failed")
        raise HTTPException(500, str(exc))

    increment("tts_requests_total")
    filename = f"campaign-{body.campaign_id}-brain-answer.wav"
    return _audio_to_wav_response(audio, sr, filename=filename)


@router.post("/npc/generate-dialogue")
@limiter.limit("20/minute")
async def npc_generate_dialogue(request: Request, body: GenerateNpcDialogueBody, _auth: None = Depends(verify_api_key)):
    player_input = (body.player_input or "").strip()
    if not player_input:
        raise HTTPException(400, "Player input is required.")

    npc = await run_in_threadpool(campaign_repository.get_npc_record, body.npc_id)
    if not npc:
        raise HTTPException(404, "NPC not found")

    voice_id = str(npc.get("voice_id") or "").strip()
    if not voice_id:
        raise HTTPException(400, "NPC has no assigned voice.")
    request.state.voice_id = voice_id

    personality = (
        str(npc.get("description") or "").strip()
        or str(npc.get("personality") or "").strip()
        or str(npc.get("role") or "").strip()
        or "An NPC in a tabletop RPG campaign."
    )

    from app.services.ai_service import generate_dialogue
    from app.services.live_context_service import build_scene_live_context
    from app.services.session_memory_service import get_session_context, record_event

    session_context = await run_in_threadpool(
        get_session_context,
        campaign_id=int(npc.get("campaign_id")) if npc.get("campaign_id") is not None else None,
        npc_id=str(npc.get("id") or "").strip() or None,
    )
    live_context_summary = ""
    if body.scene_id:
        try:
            live_context = await run_in_threadpool(build_scene_live_context, scene_id=body.scene_id)
            live_context_summary = str((live_context or {}).get("summary") or "").strip()
        except Exception:
            live_context_summary = ""

    try:
        generated_text = await run_in_threadpool(
            generate_dialogue,
            npc_name=str(npc.get("name") or "NPC"),
            personality=personality,
            situation=player_input,
            conversation_history=[{"role": "user", "content": player_input}],
            faction=str(npc.get("faction") or "").strip(),
            live_context_summary=live_context_summary,
            session_context=str(session_context.get("summary") or "").strip(),
            npc_memory_summary=str(session_context.get("npc_memory_summary") or "").strip(),
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    try:
        await run_in_threadpool(
            record_event,
            event_type="npc_interaction",
            description=f"Players addressed {str(npc.get('name') or 'the NPC').strip()}: {player_input}",
            npc_id=str(npc.get("id") or "").strip() or None,
            tags=["npc_interaction", "player_input"],
            campaign_id=int(npc.get("campaign_id")) if npc.get("campaign_id") is not None else None,
        )
        await run_in_threadpool(
            record_event,
            event_type="important_dialogue",
            description=f"{str(npc.get('name') or 'NPC').strip()} replied: {generated_text}",
            npc_id=str(npc.get("id") or "").strip() or None,
            tags=["important_dialogue", "npc_response"],
            campaign_id=int(npc.get("campaign_id")) if npc.get("campaign_id") is not None else None,
        )
    except ValueError:
        pass
    except Exception as exc:
        logging.warning("Failed to record NPC dialogue memory for %s: %s", body.npc_id, exc)

    try:
        audio, sr = await run_in_threadpool(
            _synthesize_text,
            text=generated_text,
            voice_id=voice_id,
        )
    except ValueError as e:
        increment("errors_total")
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        increment("errors_total")
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        increment("errors_total")
        logging.exception("Generated NPC dialogue TTS failed")
        raise HTTPException(500, str(e))

    increment("tts_requests_total")
    return {
        "npc_id": str(npc.get("id") or body.npc_id),
        "npc_name": npc.get("name") or "NPC",
        "voice_id": voice_id,
        "generated_text": generated_text,
        "audio_base64": _audio_to_wav_base64(audio, sr),
        "mime_type": "audio/wav",
    }
