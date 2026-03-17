"""
Structured extraction for quest hooks, missions, and open objectives.
"""
from difflib import SequenceMatcher
import logging
import re
from typing import Any, List

from app.services.llm_json import parse_llm_json_array
from app.services.parsing.models import SectionChunk


_BACKGROUND_LIKE_TERMS = {
    "background",
    "overview",
    "synopsis",
    "introduction",
    "setup",
    "premise",
}
_EXPLICIT_QUEST_TERMS = {
    "quest hook",
    "quest",
    "objective",
    "mission",
    "assignment",
    "task",
    "adventure hook",
    "job",
}
_RUMOR_LIKE_TERMS = {
    "rumor",
    "rumors",
    "legend",
    "legends",
    "myth",
    "myths",
    "history",
    "lore",
    "whispers",
    "whispered",
    "tale",
    "tales",
}
_GENERIC_QUEST_NAMES = {
    "background",
    "overview",
    "setup",
    "rumor",
    "rumors",
    "lore",
    "legend",
    "history",
    "development",
    "objective",
    "mission",
    "quest",
    "quest hook",
    "hook",
}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "into",
    "that",
    "this",
    "their",
    "they",
    "them",
    "your",
    "about",
    "have",
    "has",
    "will",
    "must",
    "need",
    "after",
    "before",
    "through",
    "while",
    "during",
    "where",
    "what",
    "when",
    "more",
    "than",
    "such",
    "same",
    "only",
    "major",
    "objective",
    "quest",
    "hook",
    "mission",
    "mystery",
}
_ACTIONABLE_QUEST_RE = re.compile(
    r"\b(?:recover|rescue|investigate|find|stop|protect|escort|reach|enter|escape|"
    r"search|track|clear|solve|discover|learn|deliver|retrieve|convince|negotiate|"
    r"prove|uncover|locate|explore|survive|win|help)\b",
    re.IGNORECASE,
)
_PLAYER_DIRECTIVE_RE = re.compile(
    r"\b(?:must|"
    r"need(?:s)?(?:\s+\w+){0,4}\s+to|"
    r"asked(?:\s+\w+){0,4}\s+to|"
    r"tasked(?:\s+\w+){0,4}\s+to|"
    r"hired(?:\s+\w+){0,4}\s+to|"
    r"sent(?:\s+\w+){0,4}\s+to|"
    r"commissioned(?:\s+\w+){0,4}\s+to|"
    r"charged(?:\s+\w+){0,4}\s+with|"
    r"the party can|the investigators can|the characters can|the players can|"
    r"characters may|party may)\b",
    re.IGNORECASE,
)


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client
    return get_client()


def _slug_id(name: str, prefix: str = "quest") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "quest").lower()).strip("_")
    return f"{prefix}_{slug}" if slug else f"{prefix}_entry"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _source(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry.get("source")
    return source if isinstance(source, dict) else {}


def _page_number(entry: dict[str, Any]) -> int | None:
    value = _source(entry).get("page_number")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _same_document(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_doc = str(_source(left).get("document_id") or "").strip()
    right_doc = str(_source(right).get("document_id") or "").strip()
    return bool(left_doc and right_doc and left_doc == right_doc)


def _heading_context_text(heading: str, subheading: str, heading_path: list[str]) -> str:
    return _normalize_text(" ".join(part for part in [heading, subheading, *list(heading_path or [])] if str(part).strip()))


def _chunk_context_text(chunk: SectionChunk) -> str:
    return _heading_context_text(chunk.heading, chunk.subheading, list(chunk.heading_path or []))


def _source_context_text(entry: dict[str, Any]) -> str:
    source = _source(entry)
    return _heading_context_text(
        str(source.get("heading") or ""),
        str(source.get("subheading") or ""),
        list(source.get("heading_path") or []),
    )


def _token_set(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in re.findall(r"[a-z0-9']+", _normalize_text(value)):
            if len(token) < 3 or token in _STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _normalized_names(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {_normalize_text(item) for item in values if _normalize_text(item)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _is_background_like_chunk(chunk: SectionChunk) -> bool:
    normalized = _chunk_context_text(chunk)
    return any(term in normalized for term in _BACKGROUND_LIKE_TERMS)


def _has_explicit_quest_context(text: str) -> bool:
    return any(term in text for term in _EXPLICIT_QUEST_TERMS)


def _is_rumor_like_chunk(chunk: SectionChunk) -> bool:
    context = _chunk_context_text(chunk)
    return any(term in context for term in _RUMOR_LIKE_TERMS) and not _has_explicit_quest_context(context)


def _chunk_has_player_directive(chunk: SectionChunk) -> bool:
    preview = _normalize_text(chunk.full_text()[:1600])
    return bool(_PLAYER_DIRECTIVE_RE.search(preview))


def _has_actionable_language(*values: Any) -> bool:
    text = " ".join(str(value or "") for value in values if str(value or "").strip())
    return bool(_ACTIONABLE_QUEST_RE.search(text))


def _contentful_token_count(*values: Any) -> int:
    return len(_token_set(*values))


def _is_generic_quest_name(name: Any) -> bool:
    normalized = _normalize_text(name)
    return not normalized or normalized in _GENERIC_QUEST_NAMES


def _quest_record_quality(entry: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
    source_context = _source_context_text(entry)
    objective = str(entry.get("objective") or "").strip()
    stakes = str(entry.get("stakes") or "").strip()
    anchors = len(_normalized_names(entry.get("related_npcs"))) + len(_normalized_names(entry.get("related_locations")))
    content_tokens = _contentful_token_count(
        entry.get("name"),
        entry.get("description"),
        objective,
        stakes,
    )
    return (
        1 if _has_explicit_quest_context(source_context) else 0,
        1 if objective else 0,
        1 if stakes else 0,
        anchors,
        content_tokens,
        float(entry.get("confidence", 0.0)),
    )


def _should_keep_quest(chunk: SectionChunk, quest: dict[str, Any]) -> bool:
    name = str(quest.get("name") or "").strip()
    description = str(quest.get("description") or "").strip()
    objective = str(quest.get("objective") or "").strip()
    stakes = str(quest.get("stakes") or "").strip()
    has_anchors = bool(_normalized_names(quest.get("related_npcs")) or _normalized_names(quest.get("related_locations")))
    actionable = _has_actionable_language(name, description, objective, stakes)
    explicit_context = _has_explicit_quest_context(_chunk_context_text(chunk))
    directive = _chunk_has_player_directive(chunk)
    generic_name = _is_generic_quest_name(name)
    detail_count = sum(
        1
        for value in (description, objective, stakes)
        if str(value).strip()
    ) + (1 if has_anchors else 0)

    if generic_name and detail_count < 2:
        return False

    if _is_rumor_like_chunk(chunk):
        return directive and actionable and bool(objective or stakes)

    if _is_background_like_chunk(chunk):
        return (directive or explicit_context) and (actionable or has_anchors) and bool(objective or stakes or has_anchors)

    if explicit_context:
        return bool(objective or stakes or has_anchors or actionable)

    return directive and actionable and bool(objective or stakes or has_anchors)


def _should_merge_background_quests(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_tokens = _token_set(left.get("name"), left.get("description"), left.get("objective"), left.get("stakes"))
    right_tokens = _token_set(right.get("name"), right.get("description"), right.get("objective"), right.get("stakes"))
    token_overlap = _jaccard(left_tokens, right_tokens)

    left_npcs = _normalized_names(left.get("related_npcs"))
    right_npcs = _normalized_names(right.get("related_npcs"))
    left_locations = _normalized_names(left.get("related_locations"))
    right_locations = _normalized_names(right.get("related_locations"))

    npc_overlap = _jaccard(left_npcs, right_npcs)
    location_overlap = _jaccard(left_locations, right_locations)
    objective_similarity = SequenceMatcher(
        None,
        _normalize_text(left.get("objective")),
        _normalize_text(right.get("objective")),
    ).ratio()
    description_similarity = SequenceMatcher(
        None,
        _normalize_text(left.get("description")),
        _normalize_text(right.get("description")),
    ).ratio()

    if objective_similarity >= 0.58 or description_similarity >= 0.72:
        return True
    if token_overlap >= 0.2 and (npc_overlap >= 0.2 or location_overlap >= 0.2):
        return True
    if npc_overlap >= 0.34 and location_overlap >= 0.2:
        return True
    return False


def _sources_are_nearby(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _same_document(left, right):
        return False

    left_source = _source(left)
    right_source = _source(right)
    left_context = _source_context_text(left)
    right_context = _source_context_text(right)
    if left_context and right_context and left_context == right_context:
        return True

    left_path = {_normalize_text(item) for item in left_source.get("heading_path", []) or [] if str(item).strip()}
    right_path = {_normalize_text(item) for item in right_source.get("heading_path", []) or [] if str(item).strip()}
    if left_path and right_path and left_path.intersection(right_path):
        return True

    left_page = _page_number(left)
    right_page = _page_number(right)
    if left_page is None or right_page is None:
        return False
    return abs(left_page - right_page) <= 1


def _should_merge_quest_records(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name = _normalize_text(left.get("name"))
    right_name = _normalize_text(right.get("name"))
    if not left_name or not right_name:
        return False

    if left_name == right_name:
        return True

    if not _sources_are_nearby(left, right):
        return False

    left_tokens = _token_set(left.get("name"), left.get("description"), left.get("objective"), left.get("stakes"))
    right_tokens = _token_set(right.get("name"), right.get("description"), right.get("objective"), right.get("stakes"))
    token_overlap = _jaccard(left_tokens, right_tokens)

    left_npcs = _normalized_names(left.get("related_npcs"))
    right_npcs = _normalized_names(right.get("related_npcs"))
    left_locations = _normalized_names(left.get("related_locations"))
    right_locations = _normalized_names(right.get("related_locations"))
    npc_overlap = _jaccard(left_npcs, right_npcs)
    location_overlap = _jaccard(left_locations, right_locations)
    shared_anchor = npc_overlap >= 0.2 or location_overlap >= 0.2

    name_similarity = SequenceMatcher(None, left_name, right_name).ratio()
    objective_similarity = SequenceMatcher(
        None,
        _normalize_text(left.get("objective")),
        _normalize_text(right.get("objective")),
    ).ratio()
    description_similarity = SequenceMatcher(
        None,
        _normalize_text(left.get("description")),
        _normalize_text(right.get("description")),
    ).ratio()

    if name_similarity >= 0.88 and (shared_anchor or token_overlap >= 0.2):
        return True
    if objective_similarity >= 0.66 and shared_anchor:
        return True
    if description_similarity >= 0.78 and shared_anchor:
        return True
    if _should_merge_background_quests(left, right) and (shared_anchor or name_similarity >= 0.72):
        return True
    return False


def _should_prefer_secondary_name(primary: dict[str, Any], secondary: dict[str, Any]) -> bool:
    primary_name = str(primary.get("name") or "").strip()
    secondary_name = str(secondary.get("name") or "").strip()
    if not secondary_name:
        return False
    if not primary_name:
        return True

    primary_conf = float(primary.get("confidence", 0.0))
    secondary_conf = float(secondary.get("confidence", 0.0))
    if secondary_conf > primary_conf:
        return True

    primary_generic = _is_generic_quest_name(primary_name)
    secondary_generic = _is_generic_quest_name(secondary_name)
    if primary_generic and not secondary_generic:
        return True

    primary_tokens = _contentful_token_count(primary_name)
    secondary_tokens = _contentful_token_count(secondary_name)
    return secondary_tokens > primary_tokens and secondary_conf >= primary_conf - 0.08


def _merge_quest_pair(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)

    if _should_prefer_secondary_name(primary, secondary):
        merged["name"] = str(secondary.get("name") or merged.get("name") or "").strip()
        merged["id"] = _slug_id(merged["name"])

    for field in ("description", "objective", "stakes"):
        current = str(merged.get(field) or "").strip()
        candidate = str(secondary.get(field) or "").strip()
        if len(candidate) > len(current):
            merged[field] = candidate

    merged["related_npcs"] = sorted(
        {
            str(item).strip()
            for item in list(primary.get("related_npcs") or []) + list(secondary.get("related_npcs") or [])
            if str(item).strip()
        }
    )
    merged["related_locations"] = sorted(
        {
            str(item).strip()
            for item in list(primary.get("related_locations") or []) + list(secondary.get("related_locations") or [])
            if str(item).strip()
        }
    )
    merged["tags"] = sorted(
        {
            str(item).strip()
            for item in list(primary.get("tags") or []) + list(secondary.get("tags") or [])
            if str(item).strip()
        }
    )
    merged["confidence"] = max(
        float(primary.get("confidence", 0.0)),
        float(secondary.get("confidence", 0.0)),
    )
    if _quest_record_quality(secondary) > _quest_record_quality(primary):
        merged["source"] = dict(_source(secondary))
    elif _source(primary):
        merged["source"] = dict(_source(primary))
    elif _source(secondary):
        merged["source"] = dict(_source(secondary))
    return merged


def _canonicalize_background_quests(chunk: SectionChunk, quests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(quests) <= 1 or not _is_background_like_chunk(chunk):
        return quests

    ordered = sorted(
        quests,
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            -len(str(item.get("objective") or "")),
            -len(str(item.get("description") or "")),
        ),
    )
    canonical: list[dict[str, Any]] = []

    for quest in ordered:
        merged = False
        for index, existing in enumerate(canonical):
            if _should_merge_background_quests(existing, quest):
                canonical[index] = _merge_quest_pair(existing, quest)
                merged = True
                break
        if not merged:
            canonical.append(dict(quest))

    return canonical[:2]


def canonicalize_quests(quests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(quests) <= 1:
        return [dict(item) for item in quests if isinstance(item, dict)]

    ordered = sorted(
        (dict(item) for item in quests if isinstance(item, dict)),
        key=lambda item: (
            -_quest_record_quality(item)[0],
            -_quest_record_quality(item)[1],
            -_quest_record_quality(item)[2],
            -_quest_record_quality(item)[3],
            -_quest_record_quality(item)[4],
            -_quest_record_quality(item)[5],
            _page_number(item) if _page_number(item) is not None else 10_000,
        ),
    )
    canonical: list[dict[str, Any]] = []

    for quest in ordered:
        if not str(quest.get("name") or "").strip():
            continue
        merged = False
        for index, existing in enumerate(canonical):
            if _should_merge_quest_records(existing, quest):
                canonical[index] = _merge_quest_pair(existing, quest)
                merged = True
                break
        if not merged:
            canonical.append(dict(quest))

    return canonical


def extract_quests(chunk: SectionChunk, model: str | None = None) -> List[dict[str, Any]]:
    """
    Extract quest hooks or goals from a section chunk.
    """
    from app.core.config import AI_MODEL

    client = _get_client()
    effective_model = model or AI_MODEL
    background_guidance = (
        "This chunk looks like background or overview material. Prefer 1-2 canonical player-facing objectives "
        "instead of multiple overlapping restatements of the same investigation or mission. "
    ) if _is_background_like_chunk(chunk) else ""

    prompt = (
        "Extract quest hooks, missions, mysteries, or major objectives from this RPG chunk. "
        + background_guidance +
        "Only extract player-facing objectives clearly supported by the text. "
        "Do not turn pure lore, flavor, atmosphere, or unsupported rumor into a quest. "
        "When the text restates the same mission in multiple ways, prefer one canonical quest record. "
        "Return ONLY a JSON array of objects. Each object must have: name, description (<=30 words), "
        "objective (brief), stakes (brief), related_npcs (array of names), related_locations (array of names), "
        "tags (array of strings), confidence (0.0-1.0). If no quest or hook is present, return [].\n\n"
        f"Chunk:\n---\n{chunk.llm_context()}\n---"
    )

    try:
        response = client.messages.create(
            model=effective_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        items = parse_llm_json_array(raw)
    except Exception as e:
        logging.warning("extract_quests failed: %s", e)
        return []

    if not isinstance(items, list):
        return []

    result: List[dict[str, Any]] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        name = (obj.get("name") or chunk.heading or "Quest Hook").strip()
        if not name:
            continue
        tags = obj.get("tags") if isinstance(obj.get("tags"), list) else []
        quest = {
            "id": _slug_id(name),
            "name": name,
            "description": (obj.get("description") or "").strip(),
            "objective": (obj.get("objective") or "").strip(),
            "stakes": (obj.get("stakes") or "").strip(),
            "related_npcs": [str(item).strip() for item in obj.get("related_npcs", []) if str(item).strip()],
            "related_locations": [str(item).strip() for item in obj.get("related_locations", []) if str(item).strip()],
            "tags": [str(item).strip() for item in tags if str(item).strip()],
            "confidence": float(obj.get("confidence", 0.8)),
        }
        if _should_keep_quest(chunk, quest):
            result.append(quest)
    return _canonicalize_background_quests(chunk, result)
