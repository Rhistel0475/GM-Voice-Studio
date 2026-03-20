"""Consequence/outcome extraction."""
from __future__ import annotations

from app.services.parsing.extractors.high_value_common import extract_keyword_entries
from app.services.parsing.models import SectionChunk


def extract_consequences(chunk: SectionChunk, model: str | None = None) -> list[dict]:
    _ = model
    return extract_keyword_entries(
        chunk,
        entry_type="consequence",
        keywords=("if they fail", "if they succeed", "consequence", "outcome", "aftermath", "resulting"),
        confidence=0.7,
    )
