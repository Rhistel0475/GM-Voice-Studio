"""Post-parse coverage audit pass."""
from __future__ import annotations

from typing import Any

from app.services.parsing.models import SectionChunk


def _count_named_mentions(chunks: list[SectionChunk]) -> int:
    count = 0
    for chunk in chunks:
        words = [token for token in str(chunk.body).split() if token.istitle()]
        count += len(words)
    return count


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
    named_mentions = _count_named_mentions(chunks)
    npc_count = len(result.get("npcs", []))
    likely_unpromoted_people = max(0, named_mentions - npc_count * 4)
    lexical_cue_misses = 0
    if any("secret" in str(chunk.body).lower() for chunk in chunks) and not result.get("secrets"):
        lexical_cue_misses += 1
    if any("clue" in str(chunk.body).lower() for chunk in chunks) and not result.get("clues"):
        lexical_cue_misses += 1

    total_gaps = (
        len(headings_without_entities)
        + len(scenes_without_npcs)
        + len(locations_without_hooks)
        + len(quests_missing_parts)
        + len(encounter_without_scene)
        + likely_unpromoted_people
        + lexical_cue_misses
    )
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
        "headings_without_entities": headings_without_entities[:50],
        "scenes_without_npcs": scenes_without_npcs[:50],
        "locations_without_hooks_or_details": locations_without_hooks[:50],
        "quests_missing_objective_or_reward": quests_missing_parts[:50],
        "encounters_without_scene_link": encounter_without_scene[:50],
    }
