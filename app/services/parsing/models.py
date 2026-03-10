"""
Shared types for the parsing pipeline.
"""
from dataclasses import dataclass
from typing import Any, Optional


CONTENT_TYPES = (
    "npc",
    "location",
    "encounter",
    "quest_hook",
    "rule",
    "boxed_text",
    "loot",
    "faction",
    "lore",
)


@dataclass
class SectionChunk:
    """A single section of document text with optional heading and level."""

    heading: str
    level: int  # 0 = no heading, 1 = top-level, 2 = subsection, etc.
    body: str
    start_offset: int = 0
    content_type: Optional[str] = None
    secondary_type: Optional[str] = None

    def full_text(self) -> str:
        """Heading + body for LLM context."""
        if self.heading:
            return f"{self.heading}\n\n{self.body}".strip()
        return self.body

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "level": self.level,
            "body": self.body,
            "start_offset": self.start_offset,
            "content_type": self.content_type,
            "secondary_type": self.secondary_type,
        }
