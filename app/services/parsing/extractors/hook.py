"""Quest hook extraction."""
from __future__ import annotations

from app.services.parsing.extractors.high_value_common import extract_keyword_entries
from app.services.parsing.models import SectionChunk


def extract_hooks(chunk: SectionChunk, model: str | None = None) -> list[dict]:
    _ = model
    return extract_keyword_entries(
        chunk,
        entry_type="hook",
        keywords=("hook", "objective", "goal", "mission", "investigate", "must"),
        confidence=0.71,
    )
