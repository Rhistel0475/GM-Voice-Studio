"""Heuristic rumor extraction."""
from __future__ import annotations

from app.services.parsing.extractors.high_value_common import extract_keyword_entries
from app.services.parsing.models import SectionChunk


def extract_rumors(chunk: SectionChunk, model: str | None = None) -> list[dict]:
    _ = model
    return extract_keyword_entries(
        chunk,
        entry_type="rumor",
        keywords=("rumor", "rumour", "whisper", "legend says", "tale says"),
        confidence=0.65,
    )
