"""
Shared lazy Anthropic client — imported by llm_brain, npc_generator, and ai_service.
"""
from typing import Optional

import anthropic

from app.core.config import ANTHROPIC_API_KEY

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client
