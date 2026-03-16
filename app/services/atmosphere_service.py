from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ATMOSPHERE_DIR = _REPO_ROOT / "static" / "audio" / "atmosphere"

_ATMOSPHERE_TRACKS: dict[str, dict[str, Any]] = {
    "forest": {
        "filename": "forest.wav",
        "label": "Forest Ambience",
        "volume": 0.34,
    },
    "tavern": {
        "filename": "tavern.wav",
        "label": "Tavern Murmur",
        "volume": 0.38,
    },
    "town": {
        "filename": "town.wav",
        "label": "Town Square",
        "volume": 0.34,
    },
    "dungeon": {
        "filename": "dungeon.wav",
        "label": "Dungeon Echoes",
        "volume": 0.32,
    },
    "combat": {
        "filename": "combat.wav",
        "label": "Combat Drums",
        "volume": 0.4,
    },
}


def _normalize_atmosphere_type(scene: dict[str, Any]) -> str:
    return str(
        scene.get("atmosphere_override_type")
        or scene.get("atmosphere_type")
        or "town"
    ).strip().lower() or "town"


def get_atmosphere_audio(scene: dict[str, Any]) -> dict[str, Any]:
    atmosphere_type = _normalize_atmosphere_type(scene)
    track = _ATMOSPHERE_TRACKS.get(atmosphere_type) or _ATMOSPHERE_TRACKS["town"]
    filepath = _ATMOSPHERE_DIR / track["filename"]

    ambience_track = None
    if filepath.is_file():
        ambience_track = {
            "atmosphere_type": atmosphere_type if atmosphere_type in _ATMOSPHERE_TRACKS else "town",
            "label": track["label"],
            "url": f"/static/audio/atmosphere/{track['filename']}",
            "loop": True,
            "volume": track["volume"],
        }

    return {"ambience_track": ambience_track}
