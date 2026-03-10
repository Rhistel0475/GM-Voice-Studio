"""
Stage 3: Classify each section chunk into content types.
Uses one batched LLM call to assign content_type (and optional secondary_type).
"""
import json
import logging
from typing import List

from app.services.parsing.models import CONTENT_TYPES, SectionChunk


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def classify_chunks(chunks: List[SectionChunk], model: str | None = None) -> List[SectionChunk]:
    """
    Classify each chunk into one primary content_type. Optionally set secondary_type.
    Mutates each chunk in place and returns the same list.

    Content types: npc, location, encounter, quest_hook, rule, boxed_text, loot, faction, lore.
    """
    if not chunks:
        return chunks

    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL

    # Build payload: index + heading + short body preview (to stay under context limits)
    max_body_len = 800
    lines: List[str] = []
    for i, c in enumerate(chunks):
        preview = (c.body[:max_body_len] + "..." if len(c.body) > max_body_len else c.body).strip()
        lines.append(f"[{i}] Heading: {c.heading or '(none)'}\n{preview}")

    prompt = (
        "Classify each of the following document sections into exactly one content type.\n\n"
        "Content types: npc, location, encounter, quest_hook, rule, boxed_text, loot, faction, lore.\n"
        "- npc: character stat block, named NPC description, villain/ally write-up\n"
        "- location: place description, room, area, settlement\n"
        "- encounter: combat or social encounter, scene with read-aloud\n"
        "- quest_hook: hook, rumor, secret, clue, twist\n"
        "- rule: game mechanic, rule, DC, check\n"
        "- boxed_text: read-aloud text, flavor text in a box\n"
        "- loot: treasure, items, rewards\n"
        "- faction: organization, group, faction description\n"
        "- lore: background, history, world-building\n\n"
        "Return ONLY a JSON array of objects with keys \"content_type\" and optional \"secondary_type\". "
        "One object per section, in the same order as the sections (indices 0 to N-1). "
        "Use only the content types listed above.\n\n"
        "Sections:\n---\n" + "\n---\n".join(lines) + "\n---"
    )

    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        classifications = json.loads(raw)
    except Exception as e:
        logging.warning("Classify LLM call failed, defaulting all to lore: %s", e)
        for c in chunks:
            c.content_type = "lore"
            c.secondary_type = None
        return chunks

    if not isinstance(classifications, list) or len(classifications) != len(chunks):
        logging.warning("Classify returned wrong length or format, defaulting to lore")
        for c in chunks:
            c.content_type = "lore"
            c.secondary_type = None
        return chunks

    for i, item in enumerate(classifications):
        if i >= len(chunks):
            break
        ct = (item.get("content_type") or "").strip().lower()
        if ct not in CONTENT_TYPES:
            ct = "lore"
        chunks[i].content_type = ct
        st = (item.get("secondary_type") or "").strip().lower()
        chunks[i].secondary_type = st if st in CONTENT_TYPES else None

    return chunks
