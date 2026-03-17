"""Shared types for the parsing pipeline."""
from dataclasses import dataclass, field
from typing import Any, Optional


CONTENT_TYPES = (
    "npc",
    "location",
    "scene",
    "encounter",
    "quest",
    "lore",
    "item",
    "faction",
    "mixed",
)

CHUNK_TYPE_GUESSES = (
    "section",
    "boxed_text",
    "stat_block",
    "encounter_section",
    "location_section",
    "quest_section",
    "lore_section",
    "item_section",
    "faction_section",
    "mixed",
)


@dataclass
class SectionChunk:
    """A semantically meaningful section of document text with preserved source metadata."""

    heading: str
    level: int  # 0 = no heading, 1 = top-level, 2 = subsection, etc.
    body: str
    start_offset: int = 0
    content_type: Optional[str] = None
    secondary_type: Optional[str] = None
    document_id: str = "document_1"
    page_number: int = 1
    subheading: str = ""
    chunk_type_guess: str = "section"
    raw_text: str = ""
    heading_path: list[str] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    classification_confidence: float = 0.0
    classification_method: str = "heuristic"

    def __post_init__(self) -> None:
        if not self.raw_text:
            self.raw_text = self.body.strip()
        if not self.heading_path:
            self.heading_path = [part for part in (self.heading,) if part]
        if not self.content_types:
            self.content_types = [
                item
                for item in (self.content_type, self.secondary_type)
                if item in CONTENT_TYPES
            ]

    def full_text(self) -> str:
        """Heading + subheading + body for LLM context."""
        parts = [self.heading, self.subheading, self.body]
        return "\n\n".join(part.strip() for part in parts if str(part).strip()).strip()

    def source_metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "heading": self.heading,
            "subheading": self.subheading,
            "heading_path": list(self.heading_path),
            "chunk_type_guess": self.chunk_type_guess,
            "start_offset": self.start_offset,
        }

    def llm_context(self) -> str:
        """Render the chunk with explicit structure so prompts preserve source semantics."""
        lines = [
            f"Document: {self.document_id}",
            f"Page: {self.page_number}",
            f"Heading: {self.heading or '(none)'}",
        ]
        if self.subheading:
            lines.append(f"Subheading: {self.subheading}")
        if self.heading_path:
            lines.append("Heading Path: " + " > ".join(self.heading_path))
        lines.append(f"Chunk Type Guess: {self.chunk_type_guess}")
        lines.append("Text:")
        lines.append(self.full_text() or self.raw_text or "")
        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "heading": self.heading,
            "subheading": self.subheading,
            "heading_path": list(self.heading_path),
            "level": self.level,
            "body": self.body,
            "raw_text": self.raw_text,
            "chunk_type_guess": self.chunk_type_guess,
            "start_offset": self.start_offset,
            "content_type": self.content_type,
            "secondary_type": self.secondary_type,
            "content_types": list(self.content_types),
            "classification_confidence": float(self.classification_confidence or 0.0),
            "classification_method": self.classification_method,
        }
