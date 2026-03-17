"""
Active-scene context assembly for LiveBoard dialogue and narration flows.
"""
from __future__ import annotations

from typing import Any

from app.repositories import campaign_repository


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _entity_name(entity: dict[str, Any]) -> str:
    for key in ("name", "title", "id"):
        value = str(entity.get(key) or "").strip()
        if value:
            return value
    return ""


def _match_name(candidate: Any, *values: Any) -> bool:
    wanted = {_normalize(value) for value in values if _normalize(value)}
    return bool(wanted) and _normalize(candidate) in wanted


def _scene_location(scene: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any] | None:
    location_ref = str(scene.get("location") or scene.get("location_id") or "").strip()
    if not location_ref:
        related_locations = scene.get("related_locations")
        if isinstance(related_locations, list) and related_locations:
            location_ref = str(related_locations[0]).strip()
    if not location_ref:
        return None

    for location in campaign.get("locations", []) or []:
        if not isinstance(location, dict):
            continue
        if _match_name(location.get("id"), location_ref) or _match_name(location.get("name"), location_ref):
            return location

    for entry in campaign.get("codex_entries", []) or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type") or "").strip().lower() != "location":
            continue
        if _match_name(entry.get("id"), location_ref) or _match_name(entry.get("title"), location_ref):
            return {
                "id": str(entry.get("id") or location_ref).strip(),
                "name": str(entry.get("title") or location_ref).strip(),
                "description": str(entry.get("summary") or entry.get("content") or "").strip(),
                "tags": list(entry.get("tags") or []),
            }
    return None


def _related_scene_quests(scene: dict[str, Any], scene_npcs: list[dict[str, Any]], campaign: dict[str, Any]) -> list[dict[str, Any]]:
    scene_name = str(scene.get("title") or scene.get("id") or "").strip()
    scene_location = str(scene.get("location") or "").strip()
    scene_npc_names = {str(npc.get("name") or "").strip() for npc in scene_npcs if str(npc.get("name") or "").strip()}
    relationships = campaign.get("relationships") if isinstance(campaign.get("relationships"), list) else []

    explicit_quest_names: set[str] = set()
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        if str(relationship.get("from_type") or "").strip() == "scene" and _match_name(relationship.get("from_id"), scene_name):
            if str(relationship.get("to_type") or "").strip() == "quest":
                explicit_quest_names.add(str(relationship.get("to_id") or "").strip())
        if str(relationship.get("to_type") or "").strip() == "scene" and _match_name(relationship.get("to_id"), scene_name):
            if str(relationship.get("from_type") or "").strip() == "quest":
                explicit_quest_names.add(str(relationship.get("from_id") or "").strip())

    related: list[dict[str, Any]] = []
    seen: set[str] = set()
    for quest in campaign.get("quests", []) or []:
        if not isinstance(quest, dict):
            continue
        quest_name = str(quest.get("name") or quest.get("title") or "").strip()
        if not quest_name or quest_name in seen:
            continue
        related_npcs = {str(item).strip() for item in quest.get("related_npcs", []) or [] if str(item).strip()}
        related_locations = {str(item).strip() for item in quest.get("related_locations", []) or [] if str(item).strip()}
        explicit_match = any(_match_name(quest_name, candidate) for candidate in explicit_quest_names)
        npc_match = bool(scene_npc_names.intersection(related_npcs))
        location_match = bool(scene_location and any(_match_name(scene_location, ref) for ref in related_locations))
        scene_match = any(_match_name(scene_name, ref) for ref in quest.get("related_scenes", []) or [])
        if explicit_match or npc_match or location_match or scene_match:
            related.append(quest)
            seen.add(quest_name)
    return related


def _related_codex_entries(scene: dict[str, Any], scene_npcs: list[dict[str, Any]], location: dict[str, Any] | None, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    scene_name = str(scene.get("title") or scene.get("id") or "").strip()
    npc_names = {str(npc.get("name") or "").strip() for npc in scene_npcs if str(npc.get("name") or "").strip()}
    location_name = str((location or {}).get("name") or "").strip()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in campaign.get("codex_entries", []) or []:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id") or entry.get("title") or "").strip()
        if not entry_id or entry_id in seen:
            continue
        related_scenes = entry.get("related_scenes") or []
        related_npcs = entry.get("related_npcs") or []
        related_locations = entry.get("related_locations") or []
        if any(_match_name(scene_name, value) for value in related_scenes) or any(
            any(_match_name(npc_name, value) for value in related_npcs) for npc_name in npc_names
        ) or any(_match_name(location_name, value) for value in related_locations if location_name):
            entries.append(entry)
            seen.add(entry_id)
    return entries


def build_scene_live_context(*, scene_id: str | None = None, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    active_bundle = bundle
    if active_bundle is None and scene_id:
        active_bundle = campaign_repository.get_scene_bundle(scene_id)
    if not active_bundle:
        return {}

    scene = active_bundle.get("scene") or {}
    campaign_id = active_bundle.get("campaign_id")
    campaign = campaign_repository.get_by_id(int(campaign_id)) if campaign_id is not None else None
    if not isinstance(campaign, dict):
        return {}

    scene_npcs = active_bundle.get("scene_npcs")
    if not isinstance(scene_npcs, list):
        scene_npcs = []
    location = _scene_location(scene, campaign)
    related_quests = _related_scene_quests(scene, scene_npcs, campaign)
    related_codex_entries = _related_codex_entries(scene, scene_npcs, location, campaign)

    quest_names = [str(item.get("name") or item.get("title") or "").strip() for item in related_quests]
    context_lines = [
        f"Scene: {str(scene.get('title') or scene.get('id') or 'Unknown scene').strip()}",
    ]
    if location is not None:
        context_lines.append(f"Location: {str(location.get('name') or '').strip()}")
        location_desc = str(location.get("description") or location.get("summary") or "").strip()
        if location_desc:
            context_lines.append(f"Location details: {location_desc}")
    if scene_npcs:
        context_lines.append(
            "NPCs present: " + ", ".join(
                str(npc.get("name") or "").strip() for npc in scene_npcs if str(npc.get("name") or "").strip()
            )
        )
    if quest_names:
        context_lines.append("Related quests: " + ", ".join(name for name in quest_names if name))

    return {
        "scene": scene,
        "scene_npcs": scene_npcs,
        "location": location,
        "related_quests": related_quests,
        "related_codex_entries": related_codex_entries,
        "summary": "\n".join(line for line in context_lines if line).strip(),
    }
