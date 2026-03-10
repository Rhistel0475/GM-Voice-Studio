"""
Stage 2: Chunk document by headings/semantic sections.
Splits on #, ##, **Title**, "Chapter N", and numbered section patterns.
"""
import re
from typing import List

from app.services.parsing.models import SectionChunk


# Markdown-style: # Title, ## Subtitle
_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Bold on own line: **Name** or **Name:**
_BOLD_HEADING_RE = re.compile(r"^\*{2}([^*]+)\*{2}\s*:?\s*$", re.MULTILINE)
# Chapter N / Part N / Act N
_CHAPTER_RE = re.compile(
    r"^(Chapter\s+\d+|Part\s+[IVXLCDM\d]+|Act\s+[IVXLCDM\d]+)\s*[.:]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Numbered section: "1. Title" or "1) Title"
_NUMBERED_SECTION_RE = re.compile(r"^(\d+)[.)]\s+(.+)$", re.MULTILINE)


def _split_by_atx(text: str) -> List[SectionChunk]:
    """Split using # / ## / ### headings. Each match gets body until next match."""
    chunks: List[SectionChunk] = []
    matches = list(_ATX_HEADING_RE.finditer(text))
    if not matches:
        return []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        chunks.append(
            SectionChunk(
                heading=heading,
                level=min(level, 6),
                body=body,
                start_offset=m.start(),
            )
        )
    return chunks


def _split_by_pattern(
    text: str, pattern: re.Pattern, default_level: int
) -> List[SectionChunk]:
    """Split at each pattern match; segment before first match is first chunk if non-empty."""
    chunks: List[SectionChunk] = []
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    for i, m in enumerate(matches):
        if pattern == _BOLD_HEADING_RE:
            heading = m.group(1).strip()
            level = 2
        elif pattern == _CHAPTER_RE:
            heading = m.group(1).strip()
            level = 1
        elif pattern == _NUMBERED_SECTION_RE:
            heading = m.group(2).strip()
            level = 2
        else:
            heading = m.group(0).strip()
            level = default_level
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        chunks.append(
            SectionChunk(
                heading=heading,
                level=level,
                body=body,
                start_offset=m.start(),
            )
        )
    return chunks


def split_into_sections(text: str) -> List[SectionChunk]:
    """
    Split document into sections by headings. Tries ATX (#), then bold (**),
    chapter/act, then numbered sections. Preserves order and offsets.

    Args:
        text: Normalized document text.

    Returns:
        List of SectionChunk with heading, level, body, start_offset.
    """
    if not text.strip():
        return []

    # Try ATX first (Markdown)
    chunks = _split_by_atx(text)
    if chunks:
        return chunks

    # Fallback: bold, chapter, or numbered
    for pattern, level in [
        (_BOLD_HEADING_RE, 2),
        (_CHAPTER_RE, 1),
        (_NUMBERED_SECTION_RE, 2),
    ]:
        chunks = _split_by_pattern(text, pattern, level)
        if chunks:
            return chunks

    # Single chunk
    return [
        SectionChunk(
            heading="",
            level=0,
            body=text.strip(),
            start_offset=0,
        )
    ]
