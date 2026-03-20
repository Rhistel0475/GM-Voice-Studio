"""Heuristic clue extraction for missed investigation details."""
from __future__ import annotations

from typing import Any

from app.services.parsing.extractors.high_value_common import extract_keyword_entries
from app.services.parsing.models import SectionChunk


def extract_clues(chunk: SectionChunk, model: str | None = None) -> list[dict[str, Any]]:
    _ = model
    return extract_keyword_entries(
        chunk,
        entry_type="clue",
        keywords=("clue", "evidence", "hint", "trail", "proof", "lead"),
        confidence=0.72,
    )
