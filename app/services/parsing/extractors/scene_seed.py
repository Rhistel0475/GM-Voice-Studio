"""
Structured extraction for encounter/scene sections (scene seeds).
Returns list of scene dicts compatible with db_models.Scene and frontend.
"""
import json
import logging
from typing import Any, List

from app.services.parsing.models import SectionChunk


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def extract_scene_seeds(chunk: SectionChunk, model: str | None = None) -> List[dict[str, Any]]:
    """
    Extract one or more scene/encounter seeds from a section chunk. Returns structured objects with
    title, act, type, read_aloud, npcs (list of names), location, difficulty, rewards, notes, confidence.
    """
    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL

    prompt = (
        "Extract encounter/scene data from this RPG section. Return ONLY a JSON array of objects. "
        "Each object must have: title, act (optional), type (combat|social|exploration|mystery), "
        "read_aloud (≤40 words, boxed text to read to players), npcs (array of NPC names), "
        "location (place name), difficulty (easy|medium|hard|deadly|none), rewards (brief), notes (brief), "
        "confidence (0.0-1.0). If no encounter is described, return []. Keep strings short.\n\n"
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
        logging.warning("extract_scene_seeds failed: %s", e)
        return []

    if not isinstance(items, list):
        return []

    result: List[dict[str, Any]] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        title = (obj.get("title") or chunk.heading or "Scene").strip()
        npcs = obj.get("npcs")
        if isinstance(npcs, list):
            npcs = [str(n).strip() for n in npcs if n]
        else:
            npcs = []
        result.append({
            "title": title,
            "act": (obj.get("act") or "").strip(),
            "type": (obj.get("type") or "exploration").strip(),
            "read_aloud": (obj.get("read_aloud") or "").strip(),
            "npcs": npcs,
            "location": (obj.get("location") or "").strip(),
            "difficulty": (obj.get("difficulty") or "").strip(),
            "rewards": (obj.get("rewards") or "").strip(),
            "notes": (obj.get("notes") or "").strip(),
            "confidence": float(obj.get("confidence", 0.8)),
        })
    return result
