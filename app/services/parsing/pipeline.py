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
from app.services.parsing.candidates import build_candidates
from app.services.parsing.aggregation import fuse_candidates
from app.services.parsing.importance import score_importance
from app.services.parsing.coverage_audit import audit_coverage
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


def _source_chunk_id(entity: dict[str, Any]) -> str:
    src = entity.get("source") or {}
    return str(
        entity.get("source_chunk_id")
        or src.get("source_chunk_id")
        or src.get("chunk_id")
        or ""
    ).strip()


def _backfill_scene_read_aloud_from_read_alouds(
    scenes: list[dict[str, Any]],
    read_alouds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """When scene seeds lack read_aloud but the same chunk produced a read_aloud entry, copy full content."""
    if not read_alouds:
        return scenes
    by_chunk: dict[str, str] = {}
    for ra in read_alouds:
        if not isinstance(ra, dict):
            continue
        cid = _source_chunk_id(ra)
        text = str(ra.get("content") or ra.get("summary") or "").strip()
        if not cid or not text:
            continue
        prev = by_chunk.get(cid, "")
        if len(text) > len(prev):
            by_chunk[cid] = text

    out: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        s = dict(scene)
        if not str(s.get("read_aloud") or "").strip():
            cid = _source_chunk_id(s)
            if cid and cid in by_chunk:
                s["read_aloud"] = by_chunk[cid]
        out.append(s)
    return out


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
    candidates = build_candidates(extracted, sections)
    fused = fuse_candidates(candidates)
    fused = score_importance(fused)
    npcs = dedupe_npcs(fused["npcs"])
    locations = dedupe_locations(fused["locations"])
    scenes = dedupe_scenes(fused["scenes"])
    scenes = _backfill_scene_read_aloud_from_read_alouds(scenes, fused.get("read_alouds", []))
    codex_entries = dedupe_codex_entries(fused["codex_entries"])
    quests = canonicalize_quests(fused["quests"])
    items = _dedupe_named_entries(fused["items"], ("id", "name"))
    encounters = _dedupe_named_entries(fused["encounters"], ("id", "name"))
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
        "clues": fused.get("clues", []),
        "secrets": fused.get("secrets", []),
        "rumors": fused.get("rumors", []),
        "read_alouds": fused.get("read_alouds", []),
        "consequences": fused.get("consequences", []),
        "rewards": fused.get("rewards", []),
        "hooks": fused.get("hooks", []),
        "parse_candidates": candidates,
    }
    result = normalize_campaign_entities(result)
    result = annotate_campaign_confidence(result, sections)
    result["coverage_report"] = audit_coverage(result, sections)
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
        "clues": [],
        "secrets": [],
        "rumors": [],
        "read_alouds": [],
        "consequences": [],
        "rewards": [],
        "hooks": [],
        "parse_candidates": [],
        "coverage_report": {"summary": {"total_gaps": 0}},
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
