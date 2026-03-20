"""
Structured extraction for items, clues, treasure, and loot.
"""
import logging
import re
from typing import Any, List

from app.services.llm_json import parse_llm_json_array
from app.services.parsing.models import SectionChunk


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def _slug_id(name: str, prefix: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "item").lower()).strip("_")
    return f"{prefix}_{slug}" if slug else f"{prefix}_entry"


def extract_items(chunk: SectionChunk, model: str | None = None) -> List[dict[str, Any]]:
    """
    Extract notable items or loot from a section chunk.
    """
    from app.core.config import AI_MODEL

    lowered = chunk.full_text().lower()
    inventory_like = any(token in lowered for token in ("inventory", "manifest", "ledger", "supply list", "stock counts"))
    notable_item_cues = any(
        token in lowered
        for token in ("treasure", "reward", "artifact", "relic", "magic item", "key item", "heirloom", "loot")
    )
    if inventory_like and not notable_item_cues:
        return []

    client = _get_client()
    effective_model = model or AI_MODEL

    prompt = (
        "Extract notable loot, items, artifacts, documents, or clues from this RPG chunk. "
        "Return ONLY a JSON array of objects. Each object must have: name, description (<=25 words), "
        "rarity (brief), owner (brief), scene (scene or location name), magical (true|false), "
        "tags (array of strings), confidence (0.0-1.0). If no notable item is present, return [].\n\n"
        f"Chunk:\n---\n{chunk.llm_context()}\n---"
    )

    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        items = parse_llm_json_array(raw)
    except Exception as e:
        logging.warning("extract_items failed: %s", e)
        return []

    if not isinstance(items, list):
        return []

    result: List[dict[str, Any]] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        name = (obj.get("name") or chunk.heading or "Item").strip()
        if not name:
            continue
        tags = obj.get("tags") if isinstance(obj.get("tags"), list) else []
        result.append(
            {
                "id": _slug_id(name),
                "name": name,
                "description": (obj.get("description") or "").strip(),
                "rarity": (obj.get("rarity") or "").strip(),
                "owner": (obj.get("owner") or "").strip(),
                "scene": (obj.get("scene") or "").strip(),
                "magical": bool(obj.get("magical", False)),
                "tags": [str(item).strip() for item in tags if str(item).strip()],
                "confidence": float(obj.get("confidence", 0.8)),
            }
        )
    if inventory_like:
        # Extra guardrail for logistics-heavy chunks: keep only strongly notable items.
        result = [
            item
            for item in result
            if any(
                token in f"{item.get('name', '')} {item.get('description', '')}".lower()
                for token in ("artifact", "relic", "magic", "heirloom", "key")
            )
        ]
    return result
