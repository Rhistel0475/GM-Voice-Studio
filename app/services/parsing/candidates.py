"""Recall-first candidate collection layer."""
from __future__ import annotations

import re
from typing import Any

from app.services.parsing.models import SectionChunk

ENTITY_KEYS = (
    "npcs",
    "locations",
    "scenes",
    "quests",
    "items",
    "codex_entries",
    "encounters",
    "clues",
    "secrets",
    "rumors",
    "read_alouds",
    "consequences",
    "rewards",
    "hooks",
)

_NAME_KEYS: dict[str, tuple[str, ...]] = {
    "npcs": ("name",),
    "locations": ("name",),
    "scenes": ("title", "name"),
    "quests": ("name", "title"),
    "items": ("name",),
    "codex_entries": ("title", "name"),
    "encounters": ("name", "title"),
    "clues": ("title", "name"),
    "secrets": ("title", "name"),
    "rumors": ("title", "name"),
    "read_alouds": ("title",),
    "consequences": ("title",),
    "rewards": ("title", "name"),
    "hooks": ("title", "name"),
}


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _build_hint(item: dict[str, Any], entity_type: str) -> str:
    for key in _NAME_KEYS.get(entity_type, ("name", "title")):
        value = str(item.get(key) or "").strip()
        if value:
            return _normalize_name(value)
    return _normalize_name(item.get("id") or entity_type)


def build_candidates(
    extracted: dict[str, list[dict[str, Any]]],
    chunks: list[SectionChunk],
) -> list[dict[str, Any]]:
    chunk_lookup = {chunk.chunk_id(): chunk for chunk in chunks}
    candidates: list[dict[str, Any]] = []
    for entity_type in ENTITY_KEYS:
        for item in extracted.get(entity_type, []):
            if not isinstance(item, dict):
                continue
            source_chunk_id = str(item.get("source_chunk_id") or item.get("source", {}).get("source_chunk_id") or "")
            source_chunk = chunk_lookup.get(source_chunk_id)
            confidence = float(item.get("confidence", 0.5) or 0.5)
            candidates.append(
                {
                    "entity_type": entity_type,
                    "canonical_hint": _build_hint(item, entity_type),
                    "raw_payload": dict(item),
                    "extraction_method": "typed_extractor",
                    "confidence": confidence,
                    "source_chunk_id": source_chunk_id,
                    "page_number": item.get("page_number") or (source_chunk.page_number if source_chunk else None),
                    "heading": item.get("heading") or (source_chunk.heading if source_chunk else ""),
                    "subheading": item.get("subheading") or (source_chunk.subheading if source_chunk else ""),
                    "source_document_id": item.get("source_document_id")
                    or item.get("source", {}).get("source_document_id")
                    or item.get("source", {}).get("document_id")
                    or (source_chunk.document_id if source_chunk else "document_1"),
                    "evidence_text": str(item.get("evidence_text") or item.get("evidence", {}).get("evidence_text") or "").strip(),
                }
            )
    return candidates
