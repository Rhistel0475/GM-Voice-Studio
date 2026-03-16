from __future__ import annotations

from typing import Any

from app.repositories import campaign_repository
from app.services.atmosphere_service import get_atmosphere_audio


def activate_scene(scene_id: str, reset_atmosphere_override: bool = False) -> dict[str, Any]:
    scene = campaign_repository.activate_scene(
        scene_id,
        reset_atmosphere_override=reset_atmosphere_override,
    )
    if scene is None:
        raise FileNotFoundError("Scene not found")

    atmosphere = get_atmosphere_audio(scene)
    return {
        "scene": scene,
        "ambience_audio": atmosphere.get("ambience_track"),
    }


def start_scene_combat(scene_id: str) -> dict[str, Any]:
    scene = campaign_repository.activate_scene(scene_id, atmosphere_override_type="combat")
    if scene is None:
        raise FileNotFoundError("Scene not found")

    if "atmosphere_override_type" not in scene:
        scene = {**scene, "atmosphere_override_type": "combat"}
    atmosphere = get_atmosphere_audio(scene)
    return {
        "scene": scene,
        "ambience_audio": atmosphere.get("ambience_track"),
    }
