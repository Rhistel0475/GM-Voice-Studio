"""
Structured extraction for lore/rule/faction sections into codex entries.
Returns list of codex entry dicts compatible with frontend CodexEntry type.
"""
import json
import logging
import re
from typing import Any, List

from app.services.llm_json import parse_llm_json_array
from app.services.parsing.models import SectionChunk


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def _slug_id(title: str, prefix: str = "codex") -> str:
    """Generate a stable id from title for codex entry."""
    slug = re.sub(r"[^a-z0-9]+", "_", (title or "entry").lower()).strip("_")
    return f"{prefix}_{slug}" if slug else f"{prefix}_entry"


def _coerce_short_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_coerce_short_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value).strip()
    return str(value).strip()


def extract_codex_entries(
    chunk: SectionChunk,
    content_type: str | None = None,
    model: str | None = None,
) -> List[dict[str, Any]]:
    """
    Extract one or more codex entries from a section (lore, rule, faction, or boxed_text).
    content_type is the classified type (lore, rule, faction); maps to CodexEntryType.
    Returns list of dicts with id, type, title, summary, content, tags, confidence.
    relatedNpcIds/relatedSceneIds can be filled later by relationship step.
    """
    from app.core.config import AI_MODEL
    client = _get_client()
    effective_model = model or AI_MODEL

    # Map pipeline content_type to frontend CodexEntryType
    type_map = {"lore": "lore", "rule": "rule", "faction": "faction", "boxed_text": "lore", "mixed": "lore"}
    entry_type = type_map.get((content_type or chunk.content_type or "lore"), "lore")

    prompt = (
        "Extract codex-style entries (lore, rules, faction info) from this RPG chunk. "
        "Return ONLY a JSON array of objects. Each object: title, summary (≤25 words), "
        "content (key facts or full text, concise), tags (array of strings), confidence (0.0-1.0). "
        "If the section is a single coherent entry, return one object. If multiple distinct topics, return multiple. "
        "Keep prose minimal; prefer structured bullets or short phrases.\n\n"
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
        logging.warning("extract_codex_entries failed: %s", e)
        return []

    if not isinstance(items, list):
        return []

    result: List[dict[str, Any]] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        title = (obj.get("title") or chunk.heading or "Entry").strip()
        if not title:
            continue
        tags = obj.get("tags")
        if isinstance(tags, list):
            tags = [str(t).strip() for t in tags if t]
        else:
            tags = []
        result.append({
            "id": _slug_id(title),
            "type": entry_type,
            "title": title,
            "summary": _coerce_short_text(obj.get("summary")),
            "content": _coerce_short_text(obj.get("content")),
            "tags": tags,
            "confidence": float(obj.get("confidence", 0.8)),
        })
    return result
