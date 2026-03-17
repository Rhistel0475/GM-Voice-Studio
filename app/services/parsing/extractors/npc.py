"""
Structured extraction for NPC sections.
Returns list of NPC dicts compatible with db_models.NPC and frontend.
"""
import logging
from typing import Any, List

from app.services.llm_json import parse_llm_json_array
from app.services.parsing.models import SectionChunk


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def extract_npcs(chunk: SectionChunk, model: str | None = None) -> List[dict[str, Any]]:
    """
    Extract one or more NPCs from a section chunk. Returns structured objects with
    name, role, personality, faction, description, motivation, secrets, and optional stat hints.
    """
    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL

    prompt = (
        "Extract NPC/character data from this RPG section. Return ONLY a JSON array of objects. "
        "Each object must have: name, role (villain|ally|quest-giver|neutral), personality (short), "
        "faction, description, motivation, secrets. Optional only when explicitly present in the text: "
        "hp, ac, cr, and notable roleplaying cues. Add \"confidence\" (0.0-1.0) for each. "
        "Use the section heading and subheading as source clues, but do not merge multiple unrelated people. "
        "Do not invent system-specific combat stats for characters when the section does not provide them. "
        "If no NPC is described, return []. Keep all string values brief (≤20 words each).\n\n"
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
        logging.warning("extract_npcs failed: %s", e)
        return []

    if not isinstance(items, list):
        return []

    result: List[dict[str, Any]] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        name = (obj.get("name") or "").strip()
        if not name:
            continue
        result.append({
            "name": name,
            "role": (obj.get("role") or "neutral").strip(),
            "personality": (obj.get("personality") or "").strip(),
            "faction": (obj.get("faction") or "").strip(),
            "description": (obj.get("description") or "").strip(),
            "motivation": (obj.get("motivation") or "").strip(),
            "secrets": (obj.get("secrets") or "").strip(),
            "hp": str(obj.get("hp") or ""),
            "ac": int(obj.get("ac")) if obj.get("ac") is not None else 0,
            "cr": (obj.get("cr") or "").strip(),
            "confidence": float(obj.get("confidence", 0.8)),
        })
    return result
