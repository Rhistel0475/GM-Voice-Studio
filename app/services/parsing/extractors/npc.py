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
        "Extract NPC/character data from this RPG section. Return ONLY a JSON array of objects.\n\n"
        "Each object must have:\n"
        "- name: full name as written\n"
        "- role: villain|ally|quest-giver|neutral\n"
        "- personality: how this character acts and feels (≤25 words)\n"
        "- description: physical appearance and notable features (≤50 words)\n"
        "- motivation: what this character wants and why (≤50 words)\n"
        "- secrets: hidden information about this character (≤50 words, empty string if none)\n"
        "- faction: organization or group affiliation (short label or empty string)\n"
        "- voice_notes: how to voice this character — accent, speech patterns, tone, mannerisms "
        "(≤30 words). Examples: 'gruff gravelly voice, speaks in short sentences', "
        "'formal archaic speech, condescending tone', 'nervous stutter, whispers'.\n"
        "- read_aloud: the exact boxed/atmospheric text to read when players first meet this NPC, "
        "copied verbatim if present in the chunk (up to 150 words). Empty string if none.\n"
        "- confidence: 0.0–1.0\n\n"
        "Optional fields (include only when explicitly in the text): hp, ac, cr.\n\n"
        "Do not merge multiple unrelated characters into one entry. "
        "Do not invent combat stats if the text does not provide them. "
        "If no character is described, return [].\n\n"
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
            "voice_notes": (obj.get("voice_notes") or "").strip(),
            "read_aloud": (obj.get("read_aloud") or "").strip(),
            "hp": str(obj.get("hp") or ""),
            "ac": int(obj.get("ac")) if obj.get("ac") is not None else 0,
            "cr": (obj.get("cr") or "").strip(),
            "confidence": float(obj.get("confidence", 0.8)),
        })
    return result
