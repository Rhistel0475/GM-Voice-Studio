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
from app.services.parsing.extractors import (
    extract_npcs,
    extract_locations,
    extract_scene_seeds,
    extract_codex_entries,
)
from app.services.parsing.relationships import extract_relationships
from app.services.parsing.dedupe import (
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
    Run the full pipeline: normalize → section → classify → extract → relationships → dedupe.
    Returns a campaign dict with title, summary, npcs, scenes, locations, reveals, items,
    party, codex_entries, relationships. Entities include confidence where applicable.
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

    # Extract by type
    npcs: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    codex_entries: list[dict[str, Any]] = []

    for chunk in sections:
        ct = chunk.content_type or "lore"
        if ct == "npc":
            npcs.extend(extract_npcs(chunk, model=model))
        elif ct == "location":
            locations.extend(extract_locations(chunk, model=model))
        elif ct == "encounter":
            scenes.extend(extract_scene_seeds(chunk, model=model))
        elif ct in ("lore", "rule", "faction", "boxed_text"):
            codex_entries.extend(
                extract_codex_entries(chunk, content_type=ct, model=model)
            )
        # quest_hook -> reveals, loot -> items: leave for future or leave empty
        elif ct == "quest_hook":
            pass
        elif ct == "loot":
            pass

    # Relationships
    relationships = extract_relationships(
        npcs, locations, scenes, codex_entries, model=model
    )

    # Dedupe
    npcs = dedupe_npcs(npcs)
    locations = dedupe_locations(locations)
    scenes = dedupe_scenes(scenes)
    codex_entries = dedupe_codex_entries(codex_entries)

    # Strip confidence from npcs/locations/scenes for DB if desired; keep in payload for frontend.
    # Plan says "store confidence levels" so we keep them in the returned dict.

    return {
        "title": title,
        "summary": summary,
        "npcs": npcs,
        "party": [],
        "scenes": scenes,
        "locations": locations,
        "reveals": [],
        "items": [],
        "codex_entries": codex_entries,
        "relationships": relationships,
    }


def _empty_result() -> dict[str, Any]:
    return {
        "title": "",
        "summary": "",
        "npcs": [],
        "party": [],
        "scenes": [],
        "locations": [],
        "reveals": [],
        "items": [],
        "codex_entries": [],
        "relationships": [],
    }
