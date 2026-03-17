"""
Orchestrator: run the full parsing pipeline and return a campaign-shaped dict.
"""
import json
import logging
from typing import Any

from app.core.config import MAX_ADVENTURE_CHARS

from app.services.parsing.normalize import normalize_text
from app.services.parsing.sections import split_into_sections
from app.services.parsing.classify import classify_chunks
from app.services.parsing.confidence import annotate_campaign_confidence
from app.services.parsing.extraction import extract_typed_entities
from app.services.parsing.extractors.quest import canonicalize_quests
from app.services.parsing.relationships import extract_relationships
from app.services.entity_normalization_service import normalize_campaign_entities
from app.services.parsing.dedupe import (
    canonicalize_location_reference,
    canonicalize_location_references,
    canonicalize_npc_reference,
    canonicalize_npc_references,
    canonicalize_scene_reference,
    dedupe_npcs,
    dedupe_locations,
    dedupe_scenes,
    dedupe_codex_entries,
)


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def _extract_title_summary(text: str, model: str | None = None) -> tuple[str, str]:
    """One short LLM call to get adventure title and 2-sentence summary."""
    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL
    preview = (text or "")[:3000].strip()
    if not preview:
        return "", ""

    prompt = (
        "From this adventure document excerpt, extract:\n"
        "1. title: the adventure or module title (one short line)\n"
        "2. summary: a 2-sentence premise or overview\n\n"
        "Return ONLY a JSON object: {\"title\": \"...\", \"summary\": \"...\"}\n\n"
        f"Excerpt:\n---\n{preview}\n---"
    )
    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        return (
            (data.get("title") or "").strip(),
            (data.get("summary") or "").strip(),
        )
    except Exception as e:
        logging.warning("Title/summary extraction failed: %s", e)
        return ("", "")


def run_parsing_pipeline(
    text: str,
    max_chars: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Run the full pipeline:
    normalize → section-aware chunking → pass 1 classification → pass 2 type-specific extraction
    → dedupe → relationship linking.

    Returns a campaign dict with title, summary, npcs, scenes, locations, quests, factions,
    lore, reveals, items, encounters, party, codex_entries, relationships.
    """
    cap = max_chars or min(MAX_ADVENTURE_CHARS, 60_000)
    normalized = normalize_text(text, max_chars=cap)
    if not normalized:
        return _empty_result()

    sections = split_into_sections(normalized)
    if not sections:
        return _empty_result()

    # Classify
    classify_chunks(sections, model=model)

    # Title/summary from start of document
    title, summary = _extract_title_summary(normalized, model=model)

    extracted = extract_typed_entities(sections, model=model)
    npcs = dedupe_npcs(extracted["npcs"])
    locations = dedupe_locations(extracted["locations"])
    scenes = dedupe_scenes(extracted["scenes"])
    codex_entries = dedupe_codex_entries(extracted["codex_entries"])
    quests = canonicalize_quests(extracted["quests"])
    items = _dedupe_named_entries(extracted["items"], ("id", "name"))
    encounters = _dedupe_named_entries(extracted["encounters"], ("id", "name"))
    scenes = _canonicalize_scene_references(scenes, npcs, locations)
    quests = _canonicalize_quest_references(quests, npcs, locations)
    items = _canonicalize_item_references(items, npcs, scenes, locations)
    factions = _build_factions_from_codex_entries(codex_entries)
    lore_entries = _build_lore_from_codex_entries(codex_entries)

    # Relationships
    relationships = extract_relationships(
        npcs,
        locations,
        scenes,
        codex_entries,
        model=model,
        quests=quests,
        items=items,
        factions=factions,
        encounters=encounters,
    )

    result = {
        "title": title,
        "summary": summary,
        "npcs": npcs,
        "party": [],
        "scenes": scenes,
        "locations": locations,
        "encounters": encounters,
        "reveals": _build_reveals_from_quests(quests),
        "items": items,
        "quests": quests,
        "factions": factions,
        "lore": lore_entries,
        "codex_entries": codex_entries,
        "relationships": relationships,
    }
    result = normalize_campaign_entities(result)
    result = annotate_campaign_confidence(result, sections)
    return result


def _empty_result() -> dict[str, Any]:
    return {
        "title": "",
        "summary": "",
        "npcs": [],
        "party": [],
        "scenes": [],
        "locations": [],
        "encounters": [],
        "reveals": [],
        "items": [],
        "quests": [],
        "factions": [],
        "lore": [],
        "codex_entries": [],
        "relationships": [],
    }


def _dedupe_named_entries(
    entries: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for entry in entries:
        for key in keys:
            raw = str(entry.get(key) or "").strip().casefold()
            if not raw:
                continue
            if raw not in seen or (entry.get("confidence") or 0) >= (seen[raw].get("confidence") or 0):
                seen[raw] = dict(entry)
            break
    return list(seen.values())


def _build_factions_from_codex_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    factions: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get("type") or "").strip().lower() != "faction":
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        factions.append(
            {
                "id": str(entry.get("id") or "").strip() or title,
                "name": title,
                "description": str(entry.get("summary") or entry.get("content") or "").strip(),
                "tags": list(entry.get("tags") or []),
                "confidence": float(entry.get("confidence", 0.8)),
            }
        )
    return _dedupe_named_entries(factions, ("id", "name"))


def _build_lore_from_codex_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lore_entries: list[dict[str, Any]] = []
    for entry in entries:
        entry_type = str(entry.get("type") or "").strip().lower()
        if entry_type not in {"lore", "rule"}:
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        lore_entries.append(
            {
                "id": str(entry.get("id") or "").strip() or title,
                "title": title,
                "summary": str(entry.get("summary") or "").strip(),
                "content": str(entry.get("content") or "").strip(),
                "type": entry_type,
                "tags": list(entry.get("tags") or []),
                "confidence": float(entry.get("confidence", 0.8)),
            }
        )
    return _dedupe_named_entries(lore_entries, ("id", "title"))


def _build_reveals_from_quests(quests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reveals: list[dict[str, Any]] = []
    for quest in quests:
        name = str(quest.get("name") or "").strip()
        if not name:
            continue
        reveals.append(
            {
                "name": name,
                "when": "",
                "type": "hook",
            }
        )
    return reveals


def _canonicalize_scene_references(
    scenes: list[dict[str, Any]],
    npcs: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonicalized: list[dict[str, Any]] = []
    for scene in scenes:
        item = dict(scene)
        item["npcs"] = canonicalize_npc_references(item.get("npcs"), npcs)
        location_name = canonicalize_location_reference(str(item.get("location") or "").strip(), locations)
        item["location"] = location_name
        canonicalized.append(item)
    return canonicalized


def _canonicalize_quest_references(
    quests: list[dict[str, Any]],
    npcs: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonicalized: list[dict[str, Any]] = []
    for quest in quests:
        item = dict(quest)
        item["related_npcs"] = canonicalize_npc_references(item.get("related_npcs"), npcs)
        item["related_locations"] = canonicalize_location_references(item.get("related_locations"), locations)
        canonicalized.append(item)
    return canonicalized


def _canonicalize_item_references(
    items: list[dict[str, Any]],
    npcs: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonicalized: list[dict[str, Any]] = []
    for item in items:
        normalized = dict(item)
        owner = canonicalize_npc_reference(str(normalized.get("owner") or "").strip(), npcs)
        scene_name = canonicalize_scene_reference(str(normalized.get("scene") or "").strip(), scenes)
        location_name = canonicalize_location_reference(str(normalized.get("location") or "").strip(), locations)
        normalized["owner"] = owner
        normalized["scene"] = scene_name
        if location_name:
            normalized["location"] = location_name
        canonicalized.append(normalized)
    return canonicalized
