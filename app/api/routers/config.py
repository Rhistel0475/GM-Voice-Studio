"""Client config endpoint."""
from fastapi import APIRouter

from app.core.config import AUTO_QUERY_ON_VOICE, DEFAULT_VOICE_ID, REQUIRE_API_KEY, TTS_PROVIDER

router = APIRouter()


@router.get("/config")
def get_config():
    """Return client config so the frontend can show API key input when required."""
    return {
        "require_api_key": REQUIRE_API_KEY,
        "auto_query_on_voice": AUTO_QUERY_ON_VOICE,
        "default_voice_id": DEFAULT_VOICE_ID,
        "tts_provider": TTS_PROVIDER,
    }
