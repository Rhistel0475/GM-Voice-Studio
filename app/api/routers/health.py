"""Health, readiness, metrics, and limits endpoints."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.core.metrics import prometheus_text
from app.core.text_utils import MAX_CHUNKS, MAX_TOTAL_CHARS

router = APIRouter()


@router.get(
    "/health",
    summary="Liveness",
    description="Liveness probe: returns 200 if the process is running. Use for container/orchestrator liveness checks.",
)
def health():
    """Liveness: process is up."""
    return {"status": "ok", "service": "kani-tts"}


@router.get(
    "/ready",
    summary="Readiness",
    description="Readiness probe: returns 503 until the TTS model has been loaded. Use for load balancer readiness so traffic is not sent before the app can serve TTS.",
)
def ready():
    """Readiness: 503 until TTS model has been loaded. Use for load balancer readiness probe."""
    from app.services.tts_service import is_model_loaded
    if not is_model_loaded():
        raise HTTPException(503, "Model not yet loaded")
    return {"status": "ready"}


@router.get("/metrics")
def metrics():
    return PlainTextResponse(prometheus_text(), media_type="text/plain; charset=utf-8")


@router.get("/limits")
def limits():
    """Return narrate limits for the frontend."""
    return {"max_narrate_chars": MAX_TOTAL_CHARS, "max_narrate_chunks": MAX_CHUNKS}
