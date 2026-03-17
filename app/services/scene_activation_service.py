from __future__ import annotations

from typing import Any

from app.repositories import campaign_repository
from app.services.atmosphere_service import get_scene_atmosphere
from app.services.live_context_service import build_scene_live_context


def activate_scene(scene_id: str, reset_atmosphere_override: bool = False) -> dict[str, Any]:
    scene = campaign_repository.activate_scene(
        scene_id,
        reset_atmosphere_override=reset_atmosphere_override,
    )
    if scene is None:
        raise FileNotFoundError("Scene not found")

    atmosphere = get_scene_atmosphere(scene)
    ambience_audio = atmosphere.get("ambience_audio")
    return {
        "scene": {**scene, "ambience_track": ambience_audio},
        "ambience_audio": ambience_audio,
        "live_context": build_scene_live_context(scene_id=scene_id),
    }


def start_scene_combat(scene_id: str) -> dict[str, Any]:
    scene = campaign_repository.activate_scene(scene_id, atmosphere_override_type="combat")
    if scene is None:
        raise FileNotFoundError("Scene not found")

    if "atmosphere_override_type" not in scene:
        scene = {**scene, "atmosphere_override_type": "combat"}
    atmosphere = get_scene_atmosphere(scene)
    ambience_audio = atmosphere.get("ambience_audio")
    return {
        "scene": {**scene, "ambience_track": ambience_audio},
        "ambience_audio": ambience_audio,
        "live_context": build_scene_live_context(scene_id=scene_id),
    }
