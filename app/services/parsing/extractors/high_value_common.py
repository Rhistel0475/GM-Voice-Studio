"""Heuristic extraction helpers for high-value GM content."""
from __future__ import annotations

import re
from typing import Any

from app.services.parsing.models import SectionChunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(str(text or "").strip()) if part.strip()]


def _label_from_heading(chunk: SectionChunk, fallback: str) -> str:
    value = str(chunk.subheading or chunk.heading or fallback).strip()
    return value[:120] if value else fallback


def extract_keyword_entries(
    chunk: SectionChunk,
    *,
    entry_type: str,
    keywords: tuple[str, ...],
    confidence: float,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    lowered = chunk.full_text().lower()
    if not any(keyword in lowered for keyword in keywords):
        return []

    entries: list[dict[str, Any]] = []
    for sentence in _sentences(chunk.full_text()):
        lowered_sentence = sentence.lower()
        if not any(keyword in lowered_sentence for keyword in keywords):
            continue
        entries.append(
            {
                "id": f"{entry_type}_{chunk.chunk_id()}_{len(entries) + 1}",
                "type": entry_type,
                "title": _label_from_heading(chunk, entry_type.replace("_", " ").title()),
                "summary": sentence[:180],
                "content": sentence,
                "tags": [entry_type.replace("_", "-")],
                "confidence": confidence,
            }
        )
        if len(entries) >= max_items:
            break
    return entries
