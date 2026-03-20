"""
Structured extraction for encounter/scene sections (scene seeds).
Returns list of scene dicts compatible with db_models.Scene and frontend.
"""
import logging
from typing import Any, List

from app.services.llm_json import parse_llm_json_array
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
        "Extract scene or encounter data from this RPG chunk. Return ONLY a JSON array of objects.\n\n"
        "Each object must have:\n"
        "- title: scene name\n"
        "- act: optional act/chapter label\n"
        "- type: combat|social|exploration|mystery|travel|investigation\n"
        "- read_aloud: the full boxed/atmospheric text to read aloud to players verbatim (up to 150 words). "
        "Copy it exactly from the source if present; do not summarize or truncate it. Empty string if none.\n"
        "- gm_notes: brief GM-facing instruction (what happens, triggers, cues). Not read to players. (≤40 words)\n"
        "- npcs: array of ALL NPC names mentioned or present in this scene — include incidental characters too\n"
        "- location: place name where the scene occurs\n"
        "- difficulty: short label or empty string\n"
        "- rewards: brief description or empty string\n"
        "- confidence: 0.0–1.0\n\n"
        "Keep scene framing system-agnostic. Do not invent combat math. "
        "The chunk may describe a location, a set-piece, an NPC introduction, or boxed read-aloud text. "
        "If no scene beat is described, return [].\n\n"
        f"Chunk:\n---\n{chunk.llm_context()}\n---"
    )

    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        items = parse_llm_json_array(raw)
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
            "gm_notes": (obj.get("gm_notes") or obj.get("notes") or "").strip(),
            "npcs": npcs,
            "location": (obj.get("location") or "").strip(),
            "difficulty": (obj.get("difficulty") or "").strip(),
            "rewards": (obj.get("rewards") or "").strip(),
            "confidence": float(obj.get("confidence", 0.8)),
        })
    return result
