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
    "mystery": {
        "filename": "dungeon.wav",
        "label": "Mystery Underscore",
        "volume": 0.3,
    },
}


def _normalize_atmosphere_type(scene: dict[str, Any]) -> str:
    return str(
        scene.get("atmosphere_override_type")
        or scene.get("atmosphere_type")
        or "town"
    ).strip().lower() or "town"


def _normalize_ambience_track_key(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw in _ATMOSPHERE_TRACKS:
        return raw
    if raw.endswith(".wav"):
        for atmosphere_type, track in _ATMOSPHERE_TRACKS.items():
            if track["filename"].lower() == raw:
                return atmosphere_type
    return None


def get_scene_atmosphere(scene: dict[str, Any]) -> dict[str, Any]:
    atmosphere_type = _normalize_atmosphere_type(scene)
    track_key = (
        _normalize_ambience_track_key(scene.get("ambience_track") or scene.get("ambienceTrack"))
        or atmosphere_type
    )
    track = _ATMOSPHERE_TRACKS.get(track_key) or _ATMOSPHERE_TRACKS["town"]
    filepath = _ATMOSPHERE_DIR / track["filename"]

    ambience_audio = None
    if filepath.is_file():
        ambience_audio = {
            "track_id": track_key if track_key in _ATMOSPHERE_TRACKS else "town",
            "atmosphere_type": atmosphere_type if atmosphere_type in _ATMOSPHERE_TRACKS else "town",
            "label": track["label"],
            "filename": track["filename"],
            "url": f"/static/audio/atmosphere/{track['filename']}",
            "loop": True,
            "volume": track["volume"],
        }

    return {
        "ambience_audio": ambience_audio,
        "loop": bool(ambience_audio and ambience_audio.get("loop", True)),
    }


def get_atmosphere_audio(scene: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible wrapper for existing callers."""
    payload = get_scene_atmosphere(scene)
    return {"ambience_track": payload.get("ambience_audio")}
