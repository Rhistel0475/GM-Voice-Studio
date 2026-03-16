"""
Shared lazy Anthropic client — imported by llm_brain, npc_generator, and ai_service.
"""
from typing import Any, Optional

try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None

from app.core.config import ANTHROPIC_API_KEY

_client: Optional[Any] = None


def get_client() -> Any:
    global _client
    if _client is None:
        if anthropic is None:
            raise RuntimeError(
                "Anthropic SDK is not installed. Install the 'anthropic' package in the active environment."
            )
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client
