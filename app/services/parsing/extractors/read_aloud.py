"""Read-aloud and boxed text extraction."""
from __future__ import annotations

from app.services.parsing.extractors.high_value_common import extract_keyword_entries
from app.services.parsing.models import SectionChunk


def extract_read_aloud(chunk: SectionChunk, model: str | None = None) -> list[dict]:
    _ = model
    entries = extract_keyword_entries(
        chunk,
        entry_type="read_aloud",
        keywords=("read aloud", "boxed text", "paraphrase", "as you enter", "when the players"),
        confidence=0.78,
    )
    if chunk.chunk_type_guess == "boxed_text" and not entries:
        entries = [
            {
                "id": f"read_aloud_{chunk.chunk_id()}",
                "type": "read_aloud",
                "title": str(chunk.subheading or chunk.heading or "Read Aloud").strip(),
                "summary": str(chunk.body).strip()[:180],
                "content": str(chunk.body).strip(),
                "tags": ["boxed-text"],
                "confidence": 0.8,
            }
        ]
    return entries
