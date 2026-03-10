"""
Structured extraction for NPC sections.
Returns list of NPC dicts compatible with db_models.NPC and frontend.
"""
import json
import logging
from typing import Any, List

from app.services.parsing.models import SectionChunk


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def extract_npcs(chunk: SectionChunk, model: str | None = None) -> List[dict[str, Any]]:
    """
    Extract one or more NPCs from a section chunk. Returns structured objects with
    name, role, personality, faction, description, motivation, secrets, hp, ac, cr, confidence.
    """
    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL

    prompt = (
        "Extract NPC/character data from this RPG section. Return ONLY a JSON array of objects. "
        "Each object must have: name, role (villain|ally|quest-giver|neutral), personality (short), "
        "faction, description, motivation, secrets, hp (e.g. \"45\" or \"3d8\"), ac (int or 0), cr (e.g. \"CR 3\"). "
        "Add \"confidence\" (0.0-1.0) for each. If no NPC is described, return []. "
        "Keep all string values brief (≤20 words each).\n\n"
        f"Section:\n---\n{chunk.full_text()}\n---"
    )

    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        items = json.loads(raw)
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
