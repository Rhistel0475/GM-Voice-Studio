"""
Structured extraction for location sections.
Returns list of location dicts compatible with db_models.Location.
"""
import json
import logging
from typing import Any, List

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
        "Extract location/place data from this RPG section. Return ONLY a JSON array of objects. "
        "Each object must have: name, description (brief, ≤30 words), confidence (0.0-1.0). "
        "If no distinct location is described, return [].\n\n"
        f"Section:\n---\n{chunk.full_text()}\n---"
    )

    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=512,
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
