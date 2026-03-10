"""
Stage 5: Extract relationships between entities (NPC→location, scene→NPCs, etc.).
Uses one LLM call over aggregated entity names/titles to output relationship pairs.
"""
import json
import logging
from typing import Any, List


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def extract_relationships(
    npcs: List[dict],
    locations: List[dict],
    scenes: List[dict],
    codex_entries: List[dict],
    model: str | None = None,
) -> List[dict[str, Any]]:
    """
    Infer relationships between entities from names/titles. Returns list of:
    { from_type, from_id, relation, to_type, to_id }.
    from_id/to_id are names or titles (or codex id) for linking; no DB ids yet.
    """
    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL

    npc_names = [n.get("name", "").strip() for n in npcs if n.get("name")]
    location_names = [loc.get("name", "").strip() for loc in locations if loc.get("name")]
    scene_titles = [s.get("title", "").strip() for s in scenes if s.get("title")]
    codex_titles = [c.get("title", "").strip() for c in codex_entries if c.get("title")]

    if not npc_names and not location_names and not scene_titles:
        return []

    prompt = (
        "Given these RPG campaign entities, list likely relationships. "
        "NPCs: " + ", ".join(npc_names[:30]) + "\n"
        "Locations: " + ", ".join(location_names[:30]) + "\n"
        "Scenes: " + ", ".join(scene_titles[:20]) + "\n"
        "Codex entries: " + ", ".join(codex_titles[:20]) + "\n\n"
        "Return ONLY a JSON array of objects. Each object: "
        "from_type (npc|location|scene|codex), from_id (exact name/title from lists), "
        "relation (appears_in|located_at|related_to|references), "
        "to_type, to_id. Only include relationships you can infer (e.g. NPC appears in scene, NPC at location). "
        "Use exact names/titles from the lists. If unsure, omit."
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
        logging.warning("extract_relationships failed: %s", e)
        return []

    if not isinstance(items, list):
        return []

    allowed = {"npc", "location", "scene", "codex"}
    allowed_rel = {"appears_in", "located_at", "related_to", "references"}
    result: List[dict[str, Any]] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        from_type = (obj.get("from_type") or "").strip().lower()
        to_type = (obj.get("to_type") or "").strip().lower()
        if from_type not in allowed or to_type not in allowed:
            continue
        rel = (obj.get("relation") or "").strip().lower()
        if rel not in allowed_rel:
            rel = "related_to"
        result.append({
            "from_type": from_type,
            "from_id": (obj.get("from_id") or "").strip(),
            "relation": rel,
            "to_type": to_type,
            "to_id": (obj.get("to_id") or "").strip(),
        })
    return result
