"""
Cross-entity normalization for parser output before persistence/UI consumption.

This service keeps the parser output system-agnostic and payload-compatible while
making downstream product features more reliable:
- merge conservative duplicate entities
- canonicalize references after merging
- preserve relationships as entity-local related_* fields for LiveBoard/Codex use
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from app.services.parsing.dedupe import (
    canonicalize_location_reference,
    canonicalize_location_references,
    canonicalize_npc_reference,
    canonicalize_npc_references,
    canonicalize_scene_reference,
    dedupe_codex_entries,
    dedupe_locations,
    dedupe_npcs,
    dedupe_scenes,
)
from app.services.parsing.extractors.quest import canonicalize_quests


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _canonical_name(value: Any) -> str:
    text = _normalize_name(value)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    return " ".join(text.split())


def _source(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry.get("source")
    return source if isinstance(source, dict) else {}


def _shared_context(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_source = _source(left)
    right_source = _source(right)
    if _normalize_name(left_source.get("document_id")) != _normalize_name(right_source.get("document_id")):
        return False

    left_heading = _normalize_name(left_source.get("heading"))
    right_heading = _normalize_name(right_source.get("heading"))
    if left_heading and right_heading and left_heading == right_heading:
        return True

    left_page = left_source.get("page_number")
    right_page = right_source.get("page_number")
    try:
        if left_page is not None and right_page is not None and abs(int(left_page) - int(right_page)) <= 1:
            return True
    except (TypeError, ValueError):
        return False
    return False


def _text_similarity(left: Any, right: Any) -> float:
    return SequenceMatcher(None, _canonical_name(left), _canonical_name(right)).ratio()


def _preserve_order_unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        output.append(clean)
    return output


def _dedupe_named_records(
    entries: list[dict[str, Any]],
    *,
    name_keys: tuple[str, ...],
    list_fields: tuple[str, ...] = ("tags",),
    text_fields: tuple[str, ...] = ("description", "summary", "content"),
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        current = dict(entry)
        current_name = ""
        for key in name_keys:
            current_name = str(current.get(key) or "").strip()
            if current_name:
                break
        if not current_name:
            continue

        match_index = -1
        for index, existing in enumerate(merged):
            existing_name = ""
            for key in name_keys:
                existing_name = str(existing.get(key) or "").strip()
                if existing_name:
                    break
            if not existing_name:
                continue
            same_name = _canonical_name(existing_name) == _canonical_name(current_name)
            near_match = _shared_context(existing, current) and _text_similarity(existing_name, current_name) >= 0.9
            if same_name or near_match:
                match_index = index
                break

        if match_index == -1:
            merged.append(current)
            continue

        existing = dict(merged[match_index])
        for field in text_fields:
            left = str(existing.get(field) or "").strip()
            right = str(current.get(field) or "").strip()
            if not left and right:
                existing[field] = right
            elif right and len(right) > len(left):
                existing[field] = right
        for field in list_fields:
            combined = [*list(existing.get(field) or []), *list(current.get(field) or [])]
            existing[field] = _preserve_order_unique([str(item) for item in combined])
        for field in ("related_npcs", "related_locations", "related_scenes", "goals"):
            if field in existing or field in current:
                combined = [*list(existing.get(field) or []), *list(current.get(field) or [])]
                existing[field] = _preserve_order_unique([str(item) for item in combined])
        existing["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), float(current.get("confidence", 0.0) or 0.0))
        merged[match_index] = existing

    return merged


def _dedupe_named_entries(entries: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in keys:
            raw = str(entry.get(key) or "").strip().casefold()
            if not raw:
                continue
            if raw not in seen or float(entry.get("confidence", 0.0) or 0.0) >= float(seen[raw].get("confidence", 0.0) or 0.0):
                seen[raw] = dict(entry)
            break
    return list(seen.values())


def _canonicalize_scene_refs(
    scenes: list[dict[str, Any]],
    npcs: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scene in scenes:
        item = dict(scene)
        item["npcs"] = canonicalize_npc_references(item.get("npcs"), npcs)
        item["location"] = canonicalize_location_reference(item.get("location"), locations)
        output.append(item)
    return output


def _canonicalize_quest_refs(
    quests: list[dict[str, Any]],
    npcs: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for quest in quests:
        item = dict(quest)
        item["related_npcs"] = canonicalize_npc_references(item.get("related_npcs"), npcs)
        item["related_locations"] = canonicalize_location_references(item.get("related_locations"), locations)
        output.append(item)
    return output


def _canonicalize_item_refs(
    items: list[dict[str, Any]],
    npcs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        updated = dict(item)
        updated["owner"] = canonicalize_npc_reference(updated.get("owner"), npcs)
        scene_ref = str(updated.get("scene") or "").strip()
        if scene_ref:
            updated["scene"] = canonicalize_scene_reference(scene_ref, scenes)
            if updated["scene"] == scene_ref:
                updated["scene"] = canonicalize_location_reference(scene_ref, locations)
        output.append(updated)
    return output


def _canonicalize_relationships(payload: dict[str, Any]) -> list[dict[str, Any]]:
    npcs = payload.get("npcs") or []
    scenes = payload.get("scenes") or []
    locations = payload.get("locations") or []
    relationships = payload.get("relationships")
    if not isinstance(relationships, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        item = dict(relationship)
        if item.get("from_type") == "npc":
            item["from_id"] = canonicalize_npc_reference(item.get("from_id"), npcs)
        elif item.get("from_type") == "scene":
            item["from_id"] = canonicalize_scene_reference(item.get("from_id"), scenes)
        elif item.get("from_type") == "location":
            item["from_id"] = canonicalize_location_reference(item.get("from_id"), locations)

        if item.get("to_type") == "npc":
            item["to_id"] = canonicalize_npc_reference(item.get("to_id"), npcs)
        elif item.get("to_type") == "scene":
            item["to_id"] = canonicalize_scene_reference(item.get("to_id"), scenes)
        elif item.get("to_type") == "location":
            item["to_id"] = canonicalize_location_reference(item.get("to_id"), locations)

        key = (
            str(item.get("from_type") or ""),
            str(item.get("from_id") or ""),
            str(item.get("relation") or ""),
            str(item.get("to_type") or ""),
            str(item.get("to_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _entity_lookup(entries: list[dict[str, Any]], *keys: str) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in keys:
            raw = str(entry.get(key) or "").strip()
            if raw:
                lookup[raw] = entry
                lookup[_canonical_name(raw)] = entry
    return lookup


def _append_related(entity: dict[str, Any], field: str, value: str) -> None:
    existing = entity.get(field)
    values = list(existing) if isinstance(existing, list) else []
    if value not in values:
        values.append(value)
    entity[field] = values


def _attach_relationship_sets(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    npcs = [dict(item) for item in result.get("npcs", []) if isinstance(item, dict)]
    scenes = [dict(item) for item in result.get("scenes", []) if isinstance(item, dict)]
    locations = [dict(item) for item in result.get("locations", []) if isinstance(item, dict)]
    quests = [dict(item) for item in result.get("quests", []) if isinstance(item, dict)]
    factions = [dict(item) for item in result.get("factions", []) if isinstance(item, dict)]
    codex_entries = [dict(item) for item in result.get("codex_entries", []) if isinstance(item, dict)]
    relationships = _canonicalize_relationships(result)

    by_type = {
        "npc": _entity_lookup(npcs, "name"),
        "scene": _entity_lookup(scenes, "title", "id"),
        "location": _entity_lookup(locations, "name", "id"),
        "quest": _entity_lookup(quests, "name", "title", "id"),
        "faction": _entity_lookup(factions, "name", "title", "id"),
        "codex": _entity_lookup(codex_entries, "title", "id"),
    }

    related_fields = {
        "npc": "related_npcs",
        "scene": "related_scenes",
        "location": "related_locations",
        "quest": "related_quests",
        "faction": "related_factions",
        "codex": "related_codex_entries",
    }

    for relationship in relationships:
        from_type = str(relationship.get("from_type") or "").strip()
        to_type = str(relationship.get("to_type") or "").strip()
        from_id = str(relationship.get("from_id") or "").strip()
        to_id = str(relationship.get("to_id") or "").strip()
        if not from_type or not to_type or not from_id or not to_id:
            continue

        from_entity = by_type.get(from_type, {}).get(from_id) or by_type.get(from_type, {}).get(_canonical_name(from_id))
        to_entity = by_type.get(to_type, {}).get(to_id) or by_type.get(to_type, {}).get(_canonical_name(to_id))
        if from_entity is not None and to_type in related_fields:
            _append_related(from_entity, related_fields[to_type], to_id)
        if to_entity is not None and from_type in related_fields:
            _append_related(to_entity, related_fields[from_type], from_id)

    result["npcs"] = npcs
    result["scenes"] = scenes
    result["locations"] = locations
    result["quests"] = quests
    result["factions"] = factions
    result["codex_entries"] = codex_entries
    result["relationships"] = relationships
    return result


def normalize_campaign_entities(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)

    npcs = dedupe_npcs(list(result.get("npcs") or []))
    locations = dedupe_locations(list(result.get("locations") or []))
    scenes = dedupe_scenes(list(result.get("scenes") or []))
    codex_entries = dedupe_codex_entries(list(result.get("codex_entries") or []))
    quests = canonicalize_quests(list(result.get("quests") or []))
    items = _dedupe_named_entries(list(result.get("items") or []), "id", "name")
    encounters = _dedupe_named_entries(list(result.get("encounters") or []), "id", "name")
    factions = _dedupe_named_records(
        list(result.get("factions") or []),
        name_keys=("name", "title", "id"),
        text_fields=("description", "summary", "content"),
    )
    lore = _dedupe_named_records(
        list(result.get("lore") or []),
        name_keys=("title", "name", "id"),
        text_fields=("summary", "description", "content"),
    )

    scenes = _canonicalize_scene_refs(scenes, npcs, locations)
    quests = _canonicalize_quest_refs(quests, npcs, locations)
    items = _canonicalize_item_refs(items, npcs, scenes, locations)

    result["npcs"] = npcs
    result["locations"] = locations
    result["scenes"] = scenes
    result["codex_entries"] = codex_entries
    result["quests"] = quests
    result["items"] = items
    result["encounters"] = encounters
    result["factions"] = factions
    result["lore"] = lore

    return _attach_relationship_sets(result)
