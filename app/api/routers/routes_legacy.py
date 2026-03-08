"""
Legacy route bundle: voices, clone, jobs, tts, campaigns, adventure, rag, brain, npc, ai, websocket.
TODO: Split into app/api/routers/voices.py, clone.py, tts.py, etc.
"""
# Load .env first
import os as _os
try:
    from dotenv import load_dotenv
    load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

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
    NARRATE_RESULT_PATH,
    PENDING_CLONE_PATH,
    RATE_LIMIT_CLONE,
    RATE_LIMIT_TTS,
)
from app.core.metrics import increment
from app.core.text_utils import MAX_CHUNKS, MAX_TOTAL_CHARS, split_for_tts
from app.services.tts_service import (
    generate as tts_generate,
    get_preset_voices,
    get_supported_language_tags,
    _is_preset_voice,
)
from app.services.voice_clone_service import clone_voice
from app.services.voice_store_service import (
    delete_voice,
    get_metadata,
    list_voices,
    load_embedding_path,
    update_metadata,
)
from app.repositories import campaign_repository

from fastapi import APIRouter
router = APIRouter()

try:
    import websockets
except ImportError:
    websockets = None

# Static assets dir (used by adventure/campaign routes)
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static" / "campaign_assets"


def _lang_tags():
    return get_supported_language_tags()


def _cleanup_old_sessions(assets_dir: Path, max_age_seconds: int = 3600) -> None:
    """Remove session dirs older than max_age_seconds to prevent disk bloat."""
    now = time.time()
    for child in assets_dir.iterdir():
        if child.is_dir() and (now - child.stat().st_mtime) > max_age_seconds:
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
    voices = list_voices(owner_id=owner_id)
    usable = []
    for voice in voices:
        voice_id = (voice or {}).get("voice_id")
        if not voice_id:
            continue
        if load_embedding_path(voice_id):
            usable.append(voice)
    return usable

# --- GDPR: get voice metadata / delete voice ---
@router.get("/voices/{voice_id}")
def get_voice(voice_id: str, request: Request, owner_id: Optional[str] = Depends(get_owner_id)):
    meta = get_metadata(voice_id, owner_id=owner_id)
    if not meta:
        raise HTTPException(404, "Voice not found")
    return meta

@router.delete("/voices/{voice_id}")
def remove_voice(voice_id: str, request: Request, _auth: None = Depends(verify_api_key), owner_id: Optional[str] = Depends(get_owner_id)):
    """Delete voice embedding and metadata (GDPR right to erasure)."""
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
    md_text = pymupdf4llm.to_markdown(doc)
    return md_text, page_count


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


@router.post("/adventure/parse")
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
    try:
        result = await run_in_threadpool(ai_full_parse, merged)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    # --- Auto-extract meaningful images and assign them to campaign entities ---
    _cleanup_old_sessions(_ASSETS_DIR)
    session_id = str(uuid.uuid4())
    session_dir = _ASSETS_DIR / session_id
    embedded_dir = session_dir / "embedded"
    embedded_dir.mkdir(parents=True)

    raw_images: list[dict] = []
    total_pages = 0
    img_counter = 0

    for raw_pdf in pdf_raws:
        new_images, pages = await run_in_threadpool(
            _extract_embedded_images, raw_pdf, embedded_dir, img_counter, session_id
        )
        raw_images.extend(new_images)
        total_pages += pages
        img_counter += len(new_images)

    if raw_images:
        assigned = await run_in_threadpool(assign_images_to_entities, raw_images, result, total_pages or 1)
        result["images"] = assigned
    else:
        result["images"] = []

    # --- Persist to database ---
    try:
        result["campaign_id"] = campaign_repository.create_from_parse_result(result)
    except Exception as e:
        logging.warning("Failed to persist campaign to DB: %s", e)

    return {"files": uploaded_files, **result}


@router.get("/api/campaigns")
@limiter.limit("60/minute")
async def list_campaigns(request: Request, _auth: None = Depends(verify_api_key)):
    """Return all saved campaigns (id, title, summary) ordered newest first."""
    return campaign_repository.list_all()


@router.get("/api/campaigns/{campaign_id}")
@limiter.limit("60/minute")
async def get_campaign(campaign_id: int, request: Request, _auth: None = Depends(verify_api_key)):
    """Return a single campaign payload (full JSON when available, relational fallback otherwise)."""
    data = campaign_repository.get_by_id(campaign_id)
    if data is None:
        raise HTTPException(404, "Campaign not found")
    return data


@router.delete("/api/campaigns/{campaign_id}")
@limiter.limit("30/minute")
async def delete_campaign(campaign_id: int, request: Request, _auth: None = Depends(verify_api_key)):
    """Delete a campaign and all related NPCs, scenes, and locations."""
    if not campaign_repository.delete(campaign_id):
        raise HTTPException(404, "Campaign not found")
    return {"deleted": campaign_id}


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
    embedded_dir.mkdir(parents=True)
    pages_dir.mkdir(parents=True)

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


# --- Co-DM LLM brain (Sprint 3) ---

class BrainQueryRequest(BaseModel):
    query: str


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


@router.post("/ai/dialogue")
@limiter.limit("20/minute")
async def ai_dialogue(req: DialogueRequest, request: Request, _auth: None = Depends(verify_api_key)):
    """
    Generate a short in-character NPC line via Claude.
    Returns: {"dialogue": "<spoken line>"}
    """
    from app.services.ai_service import generate_dialogue
    try:
        line = await run_in_threadpool(
            generate_dialogue,
            npc_name=req.npc_name,
            personality=req.personality,
            situation=req.situation,
            conversation_history=req.conversation_history,
            faction=req.faction,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"dialogue": line}


# --- TTS: preset or custom voice ---
@router.post("/tts")
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
                temperature=0.65,
                top_p=0.80,
                repetition_penalty=1.15,
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
