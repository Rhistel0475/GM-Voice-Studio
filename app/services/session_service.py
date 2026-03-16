"""Guided session startup service for LiveBoard."""
from __future__ import annotations

from typing import Any

from app.repositories import campaign_repository


def start_session(campaign_id: int, scene_id: str, narrator_voice: str) -> dict[str, Any]:
    """
    Create an active session for the requested campaign/scene and return
    the refreshed campaign payload for frontend hydration.
    """
    return campaign_repository.start_session(
        campaign_id=campaign_id,
        scene_id=scene_id,
        narrator_voice=narrator_voice,
    )
