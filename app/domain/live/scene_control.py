"""Live domain entrypoints for scene control."""
from __future__ import annotations

from typing import Any

from app.services.scene_trigger_service import (
    execute_scene_trigger as _execute_scene_trigger,
    get_scene_triggers as _get_scene_triggers,
)


def get_scene_triggers(scene_id: str) -> list[dict[str, Any]]:
    """Return normalized scene-control triggers for a scene."""
    return _get_scene_triggers(scene_id)


def execute_scene_trigger(scene_id: str, trigger_name: str) -> dict[str, Any]:
    """Execute a named trigger for a scene and return the result payload."""
    return _execute_scene_trigger(scene_id, trigger_name)
