"""Scene trigger normalization helpers for LiveBoard scene control."""
from __future__ import annotations

from typing import Any

TRIGGER_TYPE_NARRATION = "narration"
TRIGGER_TYPE_DIALOGUE = "dialogue"
TRIGGER_TYPE_AI_ACTION = "ai_action"

_TRIGGER_TYPE_ALIASES = {
    "narration": TRIGGER_TYPE_NARRATION,
    "narrate": TRIGGER_TYPE_NARRATION,
    "read_aloud": TRIGGER_TYPE_NARRATION,
    "readaloud": TRIGGER_TYPE_NARRATION,
    "dialogue": TRIGGER_TYPE_DIALOGUE,
    "speak": TRIGGER_TYPE_DIALOGUE,
    "npc_dialogue": TRIGGER_TYPE_DIALOGUE,
    "npcdialogue": TRIGGER_TYPE_DIALOGUE,
    "ai_action": TRIGGER_TYPE_AI_ACTION,
    "ai": TRIGGER_TYPE_AI_ACTION,
    "action": TRIGGER_TYPE_AI_ACTION,
}


def normalize_trigger_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _TRIGGER_TYPE_ALIASES.get(raw, raw or TRIGGER_TYPE_AI_ACTION)


def _coerce_action(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {"prompt": value.strip()}
    return None


def _scene_text(scene: dict[str, Any]) -> str:
    return (
        str(scene.get("read_aloud") or "").strip()
        or str(scene.get("notes") or "").strip()
        or str(scene.get("summary") or "").strip()
        or str(scene.get("title") or "").strip()
    )


def resolve_scene_npcs(scene: dict[str, Any], npcs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    npc_name_lookup = {
        str((npc or {}).get("name") or "").strip(): npc
        for npc in (npcs or [])
        if str((npc or {}).get("name") or "").strip()
    }
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_name in scene.get("npcs") or []:
        name = str(raw_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        resolved.append(npc_name_lookup.get(name, {"name": name}))
    return resolved


def _default_generate_dialogue_prompt(scene: dict[str, Any], npc_name: str, *, greeting: bool) -> str:
    context = _scene_text(scene) or f"the scene titled {str(scene.get('title') or 'Unknown Scene').strip()}"
    if greeting:
        return (
            f"Offer a brief in-character greeting or opening reaction as {npc_name} "
            f"for this scene: {context}"
        )
    return (
        f"Respond in character as {npc_name} to the current scene situation. "
        f"Keep it short and table-ready. Context: {context}"
    )


def build_fallback_scene_triggers(
    scene: dict[str, Any],
    npcs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    read_aloud = str(scene.get("read_aloud") or "").strip()
    notes = str(scene.get("notes") or "").strip()

    if read_aloud:
        triggers.append({
            "name": "Narrate Scene",
            "type": TRIGGER_TYPE_NARRATION,
            "text": read_aloud,
            "action": None,
        })
    elif notes:
        triggers.append({
            "name": "Narrate Scene",
            "type": TRIGGER_TYPE_NARRATION,
            "text": notes,
            "action": None,
        })

    if notes and notes != read_aloud:
        triggers.append({
            "name": "Reveal Lore",
            "type": TRIGGER_TYPE_NARRATION,
            "text": notes,
            "action": None,
        })

    scene_npcs = resolve_scene_npcs(scene, npcs=npcs)
    primary_npc = next((npc for npc in scene_npcs if str((npc or {}).get("name") or "").strip()), None)
    if primary_npc is not None:
        npc_name = str(primary_npc.get("name") or "").strip()
        triggers.append({
            "name": f"Speak as {npc_name}",
            "type": TRIGGER_TYPE_DIALOGUE,
            "text": "",
            "action": {
                "kind": "generate_dialogue",
                "npc_name": npc_name,
                "prompt": _default_generate_dialogue_prompt(scene, npc_name, greeting=True),
            },
        })
        triggers.append({
            "name": "Generate Dialogue",
            "type": TRIGGER_TYPE_DIALOGUE,
            "text": "",
            "action": {
                "kind": "generate_dialogue",
                "npc_name": npc_name,
                "prompt": _default_generate_dialogue_prompt(scene, npc_name, greeting=False),
            },
        })

    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for trigger in triggers:
        name = str(trigger.get("name") or "").strip()
        if not name:
            continue
        normalized_name = name.lower()
        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        deduped.append(trigger)
    return deduped


def normalize_scene_triggers(
    scene: dict[str, Any],
    npcs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    explicit = scene.get("triggers")
    normalized: list[dict[str, Any]] = []

    if isinstance(explicit, list):
        scene_npcs = resolve_scene_npcs(scene, npcs=npcs)
        primary_npc = next((npc for npc in scene_npcs if str((npc or {}).get("name") or "").strip()), None)

        for item in explicit:
            if not isinstance(item, dict):
                continue
            trigger_type = normalize_trigger_type(item.get("type"))
            name = str(item.get("name") or "").strip()
            text = str(item.get("text") or "").strip()
            action = _coerce_action(item.get("action"))

            if not name:
                if trigger_type == TRIGGER_TYPE_NARRATION:
                    name = "Narrate Scene"
                elif trigger_type == TRIGGER_TYPE_DIALOGUE:
                    name = "Speak Dialogue"
                else:
                    name = "Run Scene Action"

            if not text and action is None:
                if trigger_type == TRIGGER_TYPE_NARRATION:
                    text = _scene_text(scene)
                elif trigger_type == TRIGGER_TYPE_DIALOGUE and primary_npc is not None:
                    npc_name = str(primary_npc.get("name") or "").strip()
                    if npc_name:
                        action = {
                            "kind": "generate_dialogue",
                            "npc_name": npc_name,
                            "prompt": _default_generate_dialogue_prompt(scene, npc_name, greeting=True),
                        }

            if not text and action is None:
                continue

            normalized.append({
                "name": name,
                "type": trigger_type,
                "text": text,
                "action": action,
            })

    if not normalized:
        normalized = build_fallback_scene_triggers(scene, npcs=npcs)

    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for trigger in normalized:
        name = str(trigger.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        deduped.append(trigger)
    return deduped

