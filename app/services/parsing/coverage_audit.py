"""Post-parse coverage audit pass."""
from __future__ import annotations

import re
from typing import Any

from app.services.parsing.models import SectionChunk


_PERSON_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_NON_PERSON_TERMS = {
    "Read Aloud",
    "Quest Hook",
    "Blackfen Marsh",
    "Moonwell Chapel",
    "Old Quarry",
}
_LOW_SIGNAL_AUDIT_TERMS = ("manifest", "inventory", "ledger", "supply list", "stock counts")
_QUEST_LOGISTICS_TERMS = ("quest hook", "objective", "reward", "rewards", "mission", "goals")
_CHARACTER_CONTEXT_TERMS = (
    "npc",
    "scene",
    "encounter",
    "speaks",
    "says",
    "introduces",
    "captain",
    "baron",
    "priest",
    "merchant",
    "guard",
)
_PERSON_TITLE_PREFIXES = (
    "captain",
    "baron",
    "lady",
    "lord",
    "sir",
    "dame",
    "priest",
    "priestess",
    "master",
    "mistress",
    "doctor",
    "dr",
)


def _detect_named_people(chunks: list[SectionChunk]) -> set[str]:
    people: set[str] = set()
    for chunk in chunks:
        for match in _PERSON_NAME_RE.findall(str(chunk.body or "")):
            if match in _NON_PERSON_TERMS:
                continue
            people.add(match)
    return people


def _normalize_person_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z\s]", " ", str(name or "")).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    parts = normalized.split()
    if parts and parts[0] in _PERSON_TITLE_PREFIXES:
        parts = parts[1:]
    return " ".join(parts)


def _is_low_signal_chunk(chunk: SectionChunk) -> bool:
    text = " ".join(
        part for part in (chunk.heading, chunk.subheading, chunk.body) if str(part).strip()
    ).lower()
    return any(term in text for term in _LOW_SIGNAL_AUDIT_TERMS)


def _is_quest_logistics_chunk(chunk: SectionChunk) -> bool:
    text = " ".join(
        part for part in (chunk.heading, chunk.subheading, chunk.body) if str(part).strip()
    ).lower()
    return any(term in text for term in _QUEST_LOGISTICS_TERMS)


def _has_character_context(chunk: SectionChunk) -> bool:
    text = " ".join(
        part for part in (chunk.heading, chunk.subheading, chunk.body) if str(part).strip()
    ).lower()
    return any(term in text for term in _CHARACTER_CONTEXT_TERMS)


def audit_coverage(result: dict[str, Any], chunks: list[SectionChunk]) -> dict[str, Any]:
    headings_without_entities: list[str] = []
    chunk_hits: dict[str, int] = {}
    for collection_key in (
        "npcs",
        "locations",
        "scenes",
        "quests",
        "items",
        "clues",
        "secrets",
        "rumors",
        "hooks",
        "rewards",
        "consequences",
    ):
        for item in result.get(collection_key, []):
            chunk_id = str(item.get("source_chunk_id") or item.get("source", {}).get("source_chunk_id") or "").strip()
            if chunk_id:
                chunk_hits[chunk_id] = chunk_hits.get(chunk_id, 0) + 1

    for chunk in chunks:
        if not chunk.heading:
            continue
        if chunk_hits.get(chunk.chunk_id(), 0) == 0:
            headings_without_entities.append(chunk.heading)
    low_signal_heading_count = sum(1 for chunk in chunks if chunk.heading and _is_low_signal_chunk(chunk))

    scenes_without_npcs = [
        str(scene.get("title") or "")
        for scene in result.get("scenes", [])
        if not (scene.get("npcs") or [])
    ]
    locations_without_hooks = [
        str(location.get("name") or "")
        for location in result.get("locations", [])
        if not str(location.get("description") or "").strip()
    ]
    quests_missing_parts = [
        str(quest.get("name") or quest.get("title") or "")
        for quest in result.get("quests", [])
        if not str(quest.get("objective") or "").strip() or not str(quest.get("stakes") or quest.get("rewards") or "").strip()
    ]
    encounter_without_scene = [
        str(encounter.get("name") or encounter.get("id") or "")
        for encounter in result.get("encounters", [])
        if not str(encounter.get("scene_id") or "").strip()
    ]
    people_chunks = [chunk for chunk in chunks if not _is_quest_logistics_chunk(chunk) or _has_character_context(chunk)]
    detected_people = _detect_named_people(people_chunks)
    extracted_npcs = {
        str(item.get("name") or "").strip()
        for item in result.get("npcs", [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    detected_people_norm = {
        _normalize_person_name(name)
        for name in detected_people
        if _normalize_person_name(name)
    }
    extracted_npcs_norm = {
        _normalize_person_name(name)
        for name in extracted_npcs
        if _normalize_person_name(name)
    }
    likely_unpromoted_people = len(detected_people_norm - extracted_npcs_norm)
    lexical_cue_misses = 0
    if any("secret" in str(chunk.body).lower() for chunk in chunks) and not result.get("secrets"):
        lexical_cue_misses += 1
    if any("clue" in str(chunk.body).lower() for chunk in chunks) and not result.get("clues"):
        lexical_cue_misses += 1

    heading_confidence = "low" if low_signal_heading_count >= max(1, len(headings_without_entities) // 2) else "medium"
    flags = [
        ("heading_gaps", len(headings_without_entities), heading_confidence),
        ("scene_npc_gaps", len(scenes_without_npcs), "medium"),
        ("location_detail_gaps", len(locations_without_hooks), "medium"),
        ("quest_detail_gaps", len(quests_missing_parts), "high"),
        ("encounter_link_gaps", len(encounter_without_scene), "high"),
        ("likely_unpromoted_people", likely_unpromoted_people, "low" if likely_unpromoted_people <= 2 else "medium"),
        ("lexical_cue_gaps", lexical_cue_misses, "medium"),
    ]
    total_gaps = sum(value for _name, value, confidence in flags if confidence in {"high", "medium"})
    return {
        "summary": {
            "heading_gaps": len(headings_without_entities),
            "scene_npc_gaps": len(scenes_without_npcs),
            "location_detail_gaps": len(locations_without_hooks),
            "quest_detail_gaps": len(quests_missing_parts),
            "encounter_link_gaps": len(encounter_without_scene),
            "likely_unpromoted_people": likely_unpromoted_people,
            "lexical_cue_gaps": lexical_cue_misses,
            "total_gaps": total_gaps,
        },
        "gap_confidence": {name: confidence for name, _value, confidence in flags},
        "headings_without_entities": headings_without_entities[:50],
        "scenes_without_npcs": scenes_without_npcs[:50],
        "locations_without_hooks_or_details": locations_without_hooks[:50],
        "quests_missing_objective_or_reward": quests_missing_parts[:50],
        "encounters_without_scene_link": encounter_without_scene[:50],
    }
