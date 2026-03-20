"""Reward extraction."""
from __future__ import annotations

from app.services.parsing.extractors.high_value_common import extract_keyword_entries
from app.services.parsing.models import SectionChunk


def extract_rewards(chunk: SectionChunk, model: str | None = None) -> list[dict]:
    _ = model
    return extract_keyword_entries(
        chunk,
        entry_type="reward",
        keywords=("reward", "treasure", "gain", "payment", "xp", "experience"),
        confidence=0.69,
    )
