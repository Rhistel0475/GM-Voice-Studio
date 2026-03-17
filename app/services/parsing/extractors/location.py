"""
Structured extraction for location sections.
Returns list of location dicts compatible with db_models.Location.
"""
import logging
from typing import Any, List

from app.services.llm_json import parse_llm_json_array
from app.services.parsing.models import SectionChunk


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def extract_locations(chunk: SectionChunk, model: str | None = None) -> List[dict[str, Any]]:
    """
    Extract one or more locations from a section chunk. Returns structured objects with
    name, description, confidence.
    """
    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL

    prompt = (
        "Extract location/place data from this RPG chunk. Return ONLY a JSON array of objects. "
        "Each object must have: name, description (brief, ≤30 words), confidence (0.0-1.0). "
        "Use heading and subheading context to separate the named place from nearby scene action. "
        "If no distinct location is described, return [].\n\n"
        f"Chunk:\n---\n{chunk.llm_context()}\n---"
    )

    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        items = parse_llm_json_array(raw)
    except Exception as e:
        logging.warning("extract_locations failed: %s", e)
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
            "description": (obj.get("description") or "").strip(),
            "confidence": float(obj.get("confidence", 0.8)),
        })
    return result
