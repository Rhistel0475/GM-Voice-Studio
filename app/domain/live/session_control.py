"""Live domain entrypoints for guided session startup."""
from __future__ import annotations

from typing import Any

from app.services.session_service import start_session as _start_session


def start_session(campaign_id: int, scene_id: str, narrator_voice: str) -> dict[str, Any]:
    """Start a guided live session and return the updated campaign payload."""
    return _start_session(
        campaign_id=campaign_id,
        scene_id=scene_id,
        narrator_voice=narrator_voice,
    )
