"""Live domain entrypoints for encounter launch."""
from __future__ import annotations

from typing import Any

from app.services.encounter_service import launch_encounter as _launch_encounter


def launch_encounter(encounter_id: str) -> dict[str, Any]:
    """Launch an encounter with intro narration and combat ambience."""
    return _launch_encounter(encounter_id)
