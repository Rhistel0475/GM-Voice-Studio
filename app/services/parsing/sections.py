"""
Stage 2: Section-aware chunking.

Preserves document structure across headings, subheadings, boxed read-aloud text,
stat blocks, encounter sections, location sections, quest sections, and lore sections.
"""
import re
from typing import Iterable, List

from app.services.parsing.models import SectionChunk


_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BOLD_HEADING_RE = re.compile(r"^\*{2}([^*]+?)\*{2}\s*:?\s*$")
_CHAPTER_RE = re.compile(
    r"^(Chapter\s+\d+|Part\s+[IVXLCDM\d]+|Act\s+[IVXLCDM\d]+)\s*[.: -]*\s*(.*)$",
    re.IGNORECASE,
)
_NUMBERED_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(.+?)\s*$")
_DOCUMENT_MARKER_RE = re.compile(r"^\s*<<<\s*document\s*:\s*(.+?)\s*>>>\s*$", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"^\s*(?:\[\s*page\s*(\d+)\s*\]|page\s*(\d+)|p\.\s*(\d+))\s*$", re.IGNORECASE)
_SUBHEADING_RE = re.compile(
    r"^\s*(read aloud|boxed text|narration|flavor text|for the gm|gm reads|"
    r"development|quest hook|quest|objective|treasure|reward|"
    r"lore|background|encounter|location|area|features?|npcs?|clues?|secrets?|rumors?|"
    r"stat block)\s*[:\-]\s*(.*)$",
    re.IGNORECASE,
)
_BLOCKQUOTE_LINE_RE = re.compile(r"^\s*>\s*(.+)$")
_READ_ALOUD_INLINE_RE = re.compile(r"^\s*(?:read aloud|boxed text)\s*[:\-]\s*(.+)$", re.IGNORECASE)
_STAT_SIGNAL_RE = re.compile(
    r"\b(?:ac|armor class|hp|hit points|cr|challenge|str|dex|con|int|wis|cha|san|move|"
    r"initiative|speed|skills?|saves?|melee|ranged)\b",
    re.IGNORECASE,
)
_TITLEISH_RE = re.compile(r"^[A-Z][A-Za-z0-9'(),:&/ -]{2,88}$")
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_TOC_ENTRY_RE = re.compile(r"^(?=.*\d+\s*$).{4,}$")

_ENCOUNTER_WORDS = (
    "encounter",
    "ambush",
    "battle",
    "combat",
    "fight",
    "attack",
    "showdown",
    "skirmish",
)
_LOCATION_WORDS = (
    "location",
    "area",
    "room",
    "tavern",
    "inn",
    "road",
    "forest",
    "town",
    "village",
    "dungeon",
    "temple",
    "crypt",
    "cave",
    "keep",
)
_QUEST_WORDS = (
    "quest",
    "hook",
    "objective",
    "mission",
    "clue",
    "rumor",
    "secret",
    "goal",
    "investigation",
)
_LORE_WORDS = (
    "lore",
    "history",
    "legend",
    "background",
    "origin",
    "culture",
    "myth",
)
_ITEM_WORDS = ("treasure", "loot", "artifact", "item", "relic", "reward", "key")
_FACTION_WORDS = ("faction", "guild", "cult", "order", "house", "tribe", "clan", "cabal")
_CREDIT_KEYWORDS = (
    "credits",
    "design:",
    "editing:",
    "editing and development by",
    "typesetting:",
    "cartography:",
    "web production",
    "web development:",
    "graphic design:",
    "written by",
    "layout by",
    "player testers",
)
_LEGAL_KEYWORDS = (
    "copyright",
    "all rights reserved",
    "trademark",
    "wizards of the coast",
    "chaosium",
    "open game content",
    "open gaming license",
    "written permission",
    "reproduction",
    "unauthorized use",
    "this product is a work of fiction",
    "this scenario is best used with",
    "find more",
    "www.",
)


def _normalize_document_id(value: str, fallback: str = "document_1") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return cleaned or fallback


def _split_document_segments(text: str, default_document_id: str) -> list[tuple[str, str, int]]:
    lines = text.splitlines(keepends=True)
    segments: list[tuple[str, str, int]] = []
    current_lines: list[str] = []
    current_document_id = _normalize_document_id(default_document_id)
    segment_start = 0
    offset = 0

    for line in lines:
        match = _DOCUMENT_MARKER_RE.match(line.strip())
        if match:
            segment_text = "".join(current_lines).strip()
            if segment_text:
                segments.append((current_document_id, segment_text, segment_start))
            current_document_id = _normalize_document_id(match.group(1), current_document_id)
            current_lines = []
            segment_start = offset + len(line)
        else:
            current_lines.append(line)
        offset += len(line)

    segment_text = "".join(current_lines).strip()
    if segment_text:
        segments.append((current_document_id, segment_text, segment_start))
    return segments or [(current_document_id, text.strip(), 0)]


def _split_pages(text: str, base_offset: int) -> list[tuple[int, str, int]]:
    if "\f" in text:
        pages: list[tuple[int, str, int]] = []
        offset = base_offset
        for index, part in enumerate(text.split("\f"), start=1):
            clean = part.strip()
            if clean:
                pages.append((index, clean, offset))
            offset += len(part) + 1
        return pages or [(1, text.strip(), base_offset)]

    lines = text.splitlines(keepends=True)
    pages: list[tuple[int, str, int]] = []
    current_lines: list[str] = []
    page_number = 1
    page_start = base_offset
    offset = base_offset

    for line in lines:
        marker = _PAGE_MARKER_RE.match(line.strip())
        if marker:
            page_text = "".join(current_lines).strip()
            if page_text:
                pages.append((page_number, page_text, page_start))
            number = next((int(group) for group in marker.groups() if group), None)
            page_number = number if number is not None else page_number + 1
            current_lines = []
            page_start = offset + len(line)
        else:
            current_lines.append(line)
        offset += len(line)

    page_text = "".join(current_lines).strip()
    if page_text:
        pages.append((page_number, page_text, page_start))
    return pages or [(1, text.strip(), base_offset)]


def _normalize_boilerplate_line(line: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (line or "").strip().lower()).strip()


def _is_repeated_boilerplate_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    if len(stripped) > 80:
        return False
    if stripped.endswith((".", "!", "?")) and not (len(stripped) <= 40 and len(stripped.split()) <= 6):
        return False
    return len(stripped.split()) <= 8


def _strip_repeated_page_boilerplate(pages: list[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    counts: dict[str, int] = {}
    page_edges: list[tuple[int, set[int], list[str]]] = []
    document_wide_counts: dict[str, int] = {}

    for _page_number, page_text, _page_offset in pages:
        lines = page_text.splitlines()
        positions = [index for index, line in enumerate(lines) if line.strip()]
        edge_positions = set(positions[:4] + positions[-3:])
        page_edges.append((len(lines), edge_positions, lines))
        seen_on_page: set[str] = set()
        document_seen_on_page: set[str] = set()

        for line in lines:
            normalized = _normalize_boilerplate_line(line)
            if not normalized or normalized in document_seen_on_page:
                continue
            if _is_repeated_boilerplate_candidate(line) and len(line.strip().split()) >= 3:
                document_wide_counts[normalized] = document_wide_counts.get(normalized, 0) + 1
                document_seen_on_page.add(normalized)

        for index in edge_positions:
            normalized = _normalize_boilerplate_line(lines[index])
            if not normalized or normalized in seen_on_page:
                continue
            if _is_repeated_boilerplate_candidate(lines[index]):
                counts[normalized] = counts.get(normalized, 0) + 1
                seen_on_page.add(normalized)

    repeated = {value for value, count in counts.items() if count >= 2}
    repeated_anywhere = {value for value, count in document_wide_counts.items() if count >= 2}
    if not repeated and not repeated_anywhere:
        return pages

    cleaned_pages: list[tuple[int, str, int]] = []
    for (page_number, _page_text, page_offset), (_line_count, edge_positions, lines) in zip(pages, page_edges):
        kept_lines: list[str] = []
        for index, line in enumerate(lines):
            normalized = _normalize_boilerplate_line(line)
            if index in edge_positions and normalized in repeated:
                continue
            if normalized in repeated_anywhere:
                continue
            kept_lines.append(line)
        cleaned_text = "\n".join(line.rstrip() for line in kept_lines).strip()
        if cleaned_text:
            cleaned_pages.append((page_number, cleaned_text, page_offset))
    return cleaned_pages


def _looks_like_toc_entry(line: str) -> bool:
    stripped = line.strip()
    if not stripped or not _TOC_ENTRY_RE.match(stripped):
        return False
    if not re.search(r"\d+\s*$", stripped):
        return False
    punctuation_hits = len(re.findall(r"[^\w\s]", stripped))
    return stripped.count(".") >= 3 or punctuation_hits >= 4


def _is_cover_title_paragraph(paragraph: str, *, page_number: int, paragraph_index: int) -> bool:
    if page_number != 1 or paragraph_index > 1:
        return False
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(lines) < 1 or len(lines) > 6:
        return False
    if any(_parse_heading(line) is not None for line in lines):
        return False
    if any(_SUBHEADING_RE.match(line) for line in lines):
        return False
    if any(len(line) > 60 for line in lines):
        return False
    sentence_like = sum(
        1
        for line in lines
        if (
            (line.endswith((".", "!", "?")) and (len(line) > 40 or len(line.split()) >= 7))
            or len(line.split()) >= 10
        )
    )
    return sentence_like == 0


def _is_credit_paragraph(paragraph: str) -> bool:
    normalized = paragraph.strip().lower()
    if not normalized:
        return False
    if any(keyword in normalized for keyword in _CREDIT_KEYWORDS):
        return True
    lines = [line.strip().lower() for line in paragraph.splitlines() if line.strip()]
    colon_lines = sum(1 for line in lines if line.endswith(":"))
    return len(lines) >= 3 and colon_lines >= 2


def _is_name_list_paragraph(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines or len(lines) > 8:
        return False
    pattern = re.compile(r"^[A-Z][A-Za-z.'-]+(?:,?\s+[A-Z][A-Za-z.'-]+){0,4}$")
    return all(pattern.match(line) for line in lines)


def _is_legal_paragraph(paragraph: str) -> bool:
    normalized = paragraph.strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _LEGAL_KEYWORDS)


def _is_table_of_contents_paragraph(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False
    normalized_lines = [line.lower() for line in lines]
    if any(line == "table of contents" for line in normalized_lines):
        return True
    toc_hits = sum(1 for line in lines if _looks_like_toc_entry(line))
    return toc_hits >= 2 or (toc_hits >= 1 and len(lines) >= 3 and "contents" in " ".join(normalized_lines))


def _cleanup_page_text(page_text: str, *, page_number: int) -> str:
    lowered_page = page_text.lower()
    if "table of contents" in lowered_page:
        toc_lines = sum(1 for line in page_text.splitlines() if _looks_like_toc_entry(line))
        if toc_lines >= 2:
            return ""

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", page_text) if part.strip()]
    kept: list[str] = []
    previous_was_credit = False

    for index, paragraph in enumerate(paragraphs):
        if _PAGE_NUMBER_RE.match(paragraph):
            continue
        if _is_table_of_contents_paragraph(paragraph):
            continue
        if _is_credit_paragraph(paragraph):
            previous_was_credit = True
            continue
        if previous_was_credit and _is_name_list_paragraph(paragraph):
            continue
        if _is_legal_paragraph(paragraph):
            previous_was_credit = False
            continue
        if _is_cover_title_paragraph(paragraph, page_number=page_number, paragraph_index=index):
            previous_was_credit = False
            continue
        previous_was_credit = False
        kept.append(paragraph)

    return "\n\n".join(kept).strip()


def _is_standalone_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(char.isdigit() for char in stripped):
        return False
    if stripped.endswith((".", "!", "?")):
        return False
    if ":" in stripped and _SUBHEADING_RE.match(stripped):
        return False
    if stripped.startswith((">", "-", "*")):
        return False
    if len(stripped.split()) > 10:
        return False
    if not _TITLEISH_RE.match(stripped):
        return False
    lowercase_words = sum(1 for token in stripped.split() if token and token[0].islower())
    return lowercase_words <= 2


def _parse_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None

    match = _ATX_HEADING_RE.match(stripped)
    if match:
        return min(len(match.group(1)), 6), match.group(2).strip()

    match = _BOLD_HEADING_RE.match(stripped)
    if match:
        return 2, match.group(1).strip()

    match = _CHAPTER_RE.match(stripped)
    if match:
        title = " - ".join(part for part in (match.group(1).strip(), match.group(2).strip()) if part)
        return 1, title

    match = _NUMBERED_SECTION_RE.match(stripped)
    if match:
        level = min(match.group(1).count(".") + 1, 4)
        return max(level, 2), match.group(2).strip()

    if _is_standalone_heading(stripped):
        return 2, stripped

    return None


def _flush_section(
    chunks: list[SectionChunk],
    *,
    heading: str,
    level: int,
    body_lines: list[str],
    start_offset: int,
    document_id: str,
    page_number: int,
    heading_path: list[str],
) -> None:
    body = "".join(body_lines).strip()
    if not heading and not body:
        return
    chunks.append(
        SectionChunk(
            heading=heading,
            level=level,
            body=body,
            start_offset=start_offset,
            document_id=document_id,
            page_number=page_number,
            heading_path=list(heading_path),
            raw_text=body,
        )
    )


def _build_page_sections(page_text: str, *, document_id: str, page_number: int, start_offset: int) -> list[SectionChunk]:
    lines = page_text.splitlines(keepends=True)
    chunks: list[SectionChunk] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = ""
    current_level = 0
    current_heading_path: list[str] = []
    current_body: list[str] = []
    current_start = start_offset
    offset = start_offset
    started = False

    for line in lines:
        heading_match = _parse_heading(line)
        if heading_match:
            _flush_section(
                chunks,
                heading=current_heading,
                level=current_level,
                body_lines=current_body,
                start_offset=current_start,
                document_id=document_id,
                page_number=page_number,
                heading_path=current_heading_path,
            )

            level, title = heading_match
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_heading = title
            current_level = level
            current_heading_path = [part for _level, part in heading_stack]
            current_body = []
            current_start = offset
            started = True
        else:
            if not started and not line.strip():
                offset += len(line)
                continue
            if not started:
                current_heading = ""
                current_level = 0
                current_heading_path = []
                current_body = []
                current_start = offset
                started = True
            current_body.append(line)
        offset += len(line)

    _flush_section(
        chunks,
        heading=current_heading,
        level=current_level,
        body_lines=current_body,
        start_offset=current_start,
        document_id=document_id,
        page_number=page_number,
        heading_path=current_heading_path,
    )
    return chunks


def _is_boxed_text(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if _READ_ALOUD_INLINE_RE.match(stripped):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and all(_BLOCKQUOTE_LINE_RE.match(line) for line in lines):
        return True
    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) <= 800:
        return True
    return False


def _is_stat_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    hits = len(_STAT_SIGNAL_RE.findall(stripped))
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    compact_lines = sum(1 for line in lines if len(line.split()) <= 12)
    return hits >= 2 and compact_lines >= 1


_READ_ALOUD_SUBHEADING = frozenset(
    {
        "read aloud",
        "boxed text",
        "narration",
        "flavor text",
        "for the gm",
        "gm reads",
    }
)


def _guess_chunk_type(heading: str, subheading: str, text: str) -> str:
    """Infer structural type. Read-aloud subheadings must stay boxed_text so scene/readout extraction runs."""
    sub_norm = (subheading or "").strip().lower()
    if sub_norm in _READ_ALOUD_SUBHEADING or sub_norm.startswith(("read aloud", "boxed text")):
        return "boxed_text"
    context = " ".join(part for part in (heading, subheading, text) if part).lower()
    if _is_boxed_text(text):
        return "boxed_text"
    if _is_stat_block(text):
        return "stat_block"
    if any(word in context for word in _ENCOUNTER_WORDS):
        return "encounter_section"
    if any(word in context for word in _QUEST_WORDS):
        return "quest_section"
    if any(word in context for word in _LOCATION_WORDS):
        return "location_section"
    if any(word in context for word in _FACTION_WORDS):
        return "faction_section"
    if any(word in context for word in _ITEM_WORDS):
        return "item_section"
    if any(word in context for word in _LORE_WORDS):
        return "lore_section"
    return "mixed" if subheading else "section"


def _emit_chunk(
    source: SectionChunk,
    *,
    body: str,
    subheading: str = "",
    start_offset: int | None = None,
) -> SectionChunk | None:
    clean_body = body.strip()
    if not clean_body and not source.heading:
        return None
    chunk = SectionChunk(
        heading=source.heading,
        level=source.level,
        body=clean_body,
        start_offset=source.start_offset if start_offset is None else start_offset,
        document_id=source.document_id,
        page_number=source.page_number,
        subheading=subheading.strip(),
        chunk_type_guess=_guess_chunk_type(source.heading, subheading, clean_body),
        raw_text=clean_body,
        heading_path=list(source.heading_path),
    )
    return chunk


def _split_semantic_subsections(section: SectionChunk) -> Iterable[SectionChunk]:
    body = section.body.strip()
    if not body:
        emitted = _emit_chunk(section, body="", subheading="")
        return [emitted] if emitted is not None else []

    chunks: list[SectionChunk] = []
    narrative_blocks: list[str] = []
    cursor = 0

    def flush_narrative() -> None:
        if not narrative_blocks:
            return
        combined = "\n\n".join(block.strip() for block in narrative_blocks if block.strip()).strip()
        if not combined:
            narrative_blocks.clear()
            return
        offset_in_body = section.body.find(narrative_blocks[0], 0)
        chunk = _emit_chunk(
            section,
            body=combined,
            subheading="",
            start_offset=section.start_offset + max(offset_in_body, 0),
        )
        if chunk is not None:
            chunks.append(chunk)
        narrative_blocks.clear()

    for block in re.split(r"\n{2,}", body):
        raw_block = block.strip()
        if not raw_block:
            continue
        block_index = section.body.find(block, cursor)
        if block_index >= 0:
            cursor = block_index + len(block)
        block_start = section.start_offset + max(block_index, 0)

        subheading_match = _SUBHEADING_RE.match(raw_block)
        if subheading_match:
            flush_narrative()
            label = subheading_match.group(1).strip().title()
            remainder = subheading_match.group(2).strip()
            emitted = _emit_chunk(
                section,
                body=remainder or raw_block,
                subheading=label,
                start_offset=block_start,
            )
            if emitted is not None:
                chunks.append(emitted)
            continue

        if _is_boxed_text(raw_block):
            flush_narrative()
            emitted = _emit_chunk(
                section,
                body=raw_block,
                subheading="Read Aloud",
                start_offset=block_start,
            )
            if emitted is not None:
                chunks.append(emitted)
            continue

        if _is_stat_block(raw_block):
            flush_narrative()
            emitted = _emit_chunk(
                section,
                body=raw_block,
                subheading="Stat Block",
                start_offset=block_start,
            )
            if emitted is not None:
                chunks.append(emitted)
            continue

        narrative_blocks.append(raw_block)

    flush_narrative()
    return chunks or [section]


def split_into_sections(text: str, document_id: str = "document_1") -> List[SectionChunk]:
    """
    Split a normalized document into section-aware chunks with structural metadata.

    Args:
        text: Normalized document text.
        document_id: Fallback document id when the text does not contain explicit markers.

    Returns:
        A list of SectionChunk items with page, heading, subheading, raw text, and chunk guesses.
    """
    if not text.strip():
        return []

    chunks: list[SectionChunk] = []
    for active_document_id, segment_text, segment_offset in _split_document_segments(text, document_id):
        cleaned_pages = _strip_repeated_page_boilerplate(_split_pages(segment_text, segment_offset))
        for page_number, page_text, page_offset in cleaned_pages:
            page_text = _cleanup_page_text(page_text, page_number=page_number)
            if not page_text:
                continue
            for section in _build_page_sections(
                page_text,
                document_id=active_document_id,
                page_number=page_number,
                start_offset=page_offset,
            ):
                chunks.extend(_split_semantic_subsections(section))

    if chunks:
        return chunks

    return [
        SectionChunk(
            heading="",
            level=0,
            body=text.strip(),
            start_offset=0,
            document_id=_normalize_document_id(document_id),
            page_number=1,
            chunk_type_guess=_guess_chunk_type("", "", text.strip()),
            raw_text=text.strip(),
        )
    ]
