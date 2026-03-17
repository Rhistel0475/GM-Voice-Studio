"""
Confidence scoring and review prioritization for parsed campaign entities.

This layer is intentionally conservative:
- keep existing extractor confidence as one signal, not the only signal
- combine chunk classification confidence, extraction completeness, and source consistency
- annotate entities with review metadata without changing existing core fields
"""
from __future__ import annotations

from typing import Any, Iterable

from app.services.parsing.models import SectionChunk


HIGH_CONFIDENCE_THRESHOLD = 0.82
MEDIUM_CONFIDENCE_THRESHOLD = 0.58


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _entity_name(entity: dict[str, Any]) -> str:
    for key in ("name", "title", "id"):
        value = str(entity.get(key) or "").strip()
        if value:
            return value
    return ""


def _list_values(entity: dict[str, Any], key: str) -> list[str]:
    values = entity.get(key)
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _chunk_matches_source(chunk: SectionChunk, source: dict[str, Any]) -> bool:
    if not isinstance(source, dict):
        return False
    source_doc = _normalize_text(source.get("document_id"))
    source_page = source.get("page_number")
    source_heading = _normalize_text(source.get("heading"))
    source_subheading = _normalize_text(source.get("subheading"))

    if source_doc and _normalize_text(chunk.document_id) != source_doc:
        return False
    if source_page is not None and chunk.page_number != source_page:
        return False
    if source_heading and _normalize_text(chunk.heading) != source_heading:
        return False
    if source_subheading and _normalize_text(chunk.subheading) != source_subheading:
        return False
    return True


def _chunk_for_entity(chunks: Iterable[SectionChunk], entity: dict[str, Any]) -> SectionChunk | None:
    source = entity.get("source")
    if not isinstance(source, dict):
        return None

    exact_match = next((chunk for chunk in chunks if _chunk_matches_source(chunk, source)), None)
    if exact_match is not None:
        return exact_match

    entity_name = _normalize_text(_entity_name(entity))
    if not entity_name:
        return None

    source_doc = _normalize_text(source.get("document_id"))
    source_page = source.get("page_number")
    for chunk in chunks:
        if source_doc and _normalize_text(chunk.document_id) != source_doc:
            continue
        if source_page is not None and chunk.page_number != source_page:
            continue
        blob = _normalize_text(" ".join((chunk.heading, chunk.subheading, chunk.raw_text[:500])))
        if entity_name and entity_name in blob:
            return chunk
    return None


def _filled_ratio(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1.0 for value in values if value) / len(values)


def _completeness_score(entity_type: str, entity: dict[str, Any]) -> float:
    def has_text(*keys: str) -> bool:
        return any(str(entity.get(key) or "").strip() for key in keys)

    def has_list(*keys: str) -> bool:
        return any(bool(_list_values(entity, key)) for key in keys)

    checks_by_type: dict[str, list[bool]] = {
        "npc": [
            has_text("name"),
            has_text("description", "personality"),
            has_text("role"),
            has_text("motivation"),
            has_text("faction"),
        ],
        "scene": [
            has_text("title"),
            has_text("read_aloud", "notes"),
            has_text("location"),
            has_list("npcs"),
            has_text("type"),
        ],
        "location": [
            has_text("name"),
            has_text("description"),
            has_list("tags"),
        ],
        "quest": [
            has_text("name"),
            has_text("description", "objective"),
            has_text("stakes"),
            has_list("related_npcs"),
            has_list("related_locations"),
        ],
        "item": [
            has_text("name"),
            has_text("description"),
            has_text("owner"),
            has_text("scene"),
            has_list("tags"),
        ],
        "encounter": [
            has_text("name"),
            has_text("intro_text", "summary", "introText"),
            has_list("enemies"),
            has_text("scene_id", "scene"),
            has_list("treasure_or_rewards"),
        ],
        "faction": [
            has_text("name"),
            has_text("description"),
            has_list("tags"),
            has_list("related_npcs"),
            has_list("related_locations"),
        ],
        "lore": [
            has_text("title"),
            has_text("summary"),
            has_text("content"),
            has_list("tags"),
        ],
        "codex": [
            has_text("title"),
            has_text("summary"),
            has_text("content"),
            has_list("tags"),
        ],
    }
    checks = checks_by_type.get(entity_type, [has_text("name", "title")])
    return max(0.2, _filled_ratio(checks))


def _consistency_score(entity_type: str, entity: dict[str, Any], chunk: SectionChunk | None) -> float:
    name = _normalize_text(_entity_name(entity))
    if chunk is None:
        return 0.45 if name else 0.2

    heading_blob = _normalize_text(" ".join((chunk.heading, chunk.subheading, " ".join(chunk.heading_path))))
    text_blob = _normalize_text(chunk.raw_text[:1600])
    score = 0.35

    if name and (name in heading_blob or name in text_blob):
        score += 0.25
    elif name:
        score += 0.1

    if entity_type in set(chunk.content_types or []):
        score += 0.2
    elif chunk.content_type == "mixed":
        score += 0.08

    if entity_type == "scene" and str(entity.get("location") or "").strip():
        score += 0.08
    if entity_type == "quest" and (_list_values(entity, "related_npcs") or _list_values(entity, "related_locations")):
        score += 0.08
    if entity_type in {"faction", "codex", "lore"} and _list_values(entity, "tags"):
        score += 0.05

    return max(0.2, min(1.0, score))


def _review_label(score: float) -> str:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _review_priority(label: str) -> str:
    if label == "high":
        return "auto_approve"
    if label == "medium":
        return "review_queue"
    return "hidden"


def _annotate_entity(entity_type: str, entity: dict[str, Any], chunks: Iterable[SectionChunk]) -> dict[str, Any]:
    item = dict(entity)
    chunk = _chunk_for_entity(chunks, item)

    base_confidence = item.get("confidence")
    try:
        base_score = float(base_confidence) if base_confidence is not None else 0.68
    except (TypeError, ValueError):
        base_score = 0.68

    classification_score = float(getattr(chunk, "classification_confidence", 0.62) or 0.62) if chunk is not None else 0.62
    completeness_score = _completeness_score(entity_type, item)
    consistency_score = _consistency_score(entity_type, item, chunk)

    score = (
        base_score * 0.35
        + classification_score * 0.3
        + completeness_score * 0.2
        + consistency_score * 0.15
    )
    score = max(0.05, min(0.99, round(score, 3)))
    label = _review_label(score)

    item["confidence"] = score
    item["confidence_score"] = score
    item["confidence_label"] = label
    item["needs_review"] = label != "high"
    item["review_priority"] = _review_priority(label)
    if chunk is not None:
        item["classification_confidence"] = round(float(chunk.classification_confidence or 0.0), 3)
        item["classification_method"] = str(chunk.classification_method or "heuristic")
    return item


def annotate_campaign_confidence(
    payload: dict[str, Any],
    chunks: Iterable[SectionChunk] | None = None,
) -> dict[str, Any]:
    result = dict(payload)
    chunk_list = list(chunks or [])

    list_entity_types = {
        "npcs": "npc",
        "scenes": "scene",
        "locations": "location",
        "encounters": "encounter",
        "items": "item",
        "quests": "quest",
        "factions": "faction",
        "lore": "lore",
        "codex_entries": "codex",
    }

    review_counts = {"auto_approve": 0, "review_queue": 0, "hidden": 0}
    for key, entity_type in list_entity_types.items():
        items = result.get(key)
        if not isinstance(items, list):
            continue
        annotated_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            annotated = _annotate_entity(entity_type, item, chunk_list)
            review_counts[str(annotated.get("review_priority") or "review_queue")] += 1
            annotated_items.append(annotated)
        result[key] = annotated_items

    result["review_summary"] = {
        "auto_approve_count": review_counts["auto_approve"],
        "review_queue_count": review_counts["review_queue"],
        "hidden_count": review_counts["hidden"],
    }
    return result
