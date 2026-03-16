"""Live domain entrypoints for scene control."""
from __future__ import annotations

from typing import Any

from app.services.scene_trigger_service import (
    execute_scene_trigger as _execute_scene_trigger,
    get_scene_triggers as _get_scene_triggers,
)
from app.services.scene_activation_service import (
    activate_scene as _activate_scene,
    start_scene_combat as _start_scene_combat,
)
from app.services.scene_suggestion_service import (
    suggest_next_scenes_for_scene as _suggest_next_scenes_for_scene,
)


def get_scene_triggers(scene_id: str) -> list[dict[str, Any]]:
    """Return normalized scene-control triggers for a scene."""
    return _get_scene_triggers(scene_id)


def execute_scene_trigger(scene_id: str, trigger_name: str) -> dict[str, Any]:
    """Execute a named trigger for a scene and return the result payload."""
    return _execute_scene_trigger(scene_id, trigger_name)


def activate_scene(scene_id: str, reset_atmosphere_override: bool = False) -> dict[str, Any]:
    """Activate a scene and resolve its ambient audio track."""
    return _activate_scene(scene_id, reset_atmosphere_override=reset_atmosphere_override)


def start_scene_combat(scene_id: str) -> dict[str, Any]:
    """Switch the active scene ambience into combat mode."""
    return _start_scene_combat(scene_id)


def suggest_next_scenes(scene_id: str, player_action: str) -> dict[str, Any]:
    """Suggest likely next scenes for the active campaign flow."""
    return _suggest_next_scenes_for_scene(scene_id, player_action)
