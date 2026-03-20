"""Evidence helpers for traceable parser entities."""
from __future__ import annotations

from typing import Any

from app.services.parsing.models import SectionChunk


def bounded_evidence_text(text: str, limit: int = 220) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1].rstrip() + "…"


def build_evidence(chunk: SectionChunk, *, entity_name: str = "", text: str = "", confidence: float = 0.5) -> dict[str, Any]:
    source = chunk.source_metadata()
    excerpt_seed = text or entity_name or chunk.subheading or chunk.body[:280]
    return {
        "source_document_id": source.get("source_document_id") or source.get("document_id"),
        "page_number": source.get("page_number"),
        "page_range": source.get("page_range") or [source.get("page_number"), source.get("page_number")],
        "heading": source.get("heading", ""),
        "subheading": source.get("subheading", ""),
        "source_chunk_id": source.get("source_chunk_id") or source.get("chunk_id"),
        "evidence_text": bounded_evidence_text(excerpt_seed),
        "confidence": float(confidence or 0.0),
        "importance_score": 0.0,
    }


def attach_entity_evidence(item: dict[str, Any], chunk: SectionChunk) -> dict[str, Any]:
    enriched = dict(item)
    confidence = float(enriched.get("confidence", 0.5) or 0.5)
    name = str(enriched.get("name") or enriched.get("title") or "")
    evidence = build_evidence(chunk, entity_name=name, text=str(enriched.get("description") or enriched.get("summary") or ""), confidence=confidence)
    source = chunk.source_metadata()
    enriched["source"] = dict(source)
    enriched["evidence"] = dict(evidence)
    enriched["source_document_id"] = evidence["source_document_id"]
    enriched["page_number"] = evidence["page_number"]
    enriched["page_range"] = evidence["page_range"]
    enriched["heading"] = evidence["heading"]
    enriched["subheading"] = evidence["subheading"]
    enriched["source_chunk_id"] = evidence["source_chunk_id"]
    enriched["evidence_text"] = evidence["evidence_text"]
    enriched["importance_score"] = float(enriched.get("importance_score", 0.0) or 0.0)
    return enriched
