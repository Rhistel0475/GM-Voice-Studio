"""
Post-extraction relationship linking.

Builds a lightweight connected campaign graph using:
- exact and fuzzy name matches
- heading inheritance and source overlap
- section proximity and page proximity
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable, List


_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)")


def _normalize_name(value: str) -> str:
    text = (value or "").strip().lower()
    text = _PAREN_SUFFIX_RE.sub("", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = _LEADING_ARTICLE_RE.sub("", text)
    return re.sub(r"[^a-z0-9'\s]+", " ", text).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in _normalize_name(value).split() if token}


def _ordered_tokens(value: str) -> list[str]:
    return [token for token in _normalize_name(value).split() if token]


def _contains_ordered_tokens(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if haystack[index : index + width] == needle:
            return True
    return False


def _names_match(a: str, b: str) -> bool:
    left = _normalize_name(a)
    right = _normalize_name(b)
    if not left or not right:
        return False
    if left == right:
        return True

    left_tokens = _ordered_tokens(a)
    right_tokens = _ordered_tokens(b)
    if len(left_tokens) >= 2 and _contains_ordered_tokens(left_tokens, right_tokens):
        return True
    if len(right_tokens) >= 2 and _contains_ordered_tokens(right_tokens, left_tokens):
        return True

    left_token_set = set(left_tokens)
    right_token_set = set(right_tokens)
    if (
        len(left_tokens) >= 2
        and len(right_tokens) >= 2
        and left_token_set
        and right_token_set
        and (left_token_set.issubset(right_token_set) or right_token_set.issubset(left_token_set))
    ):
        return True

    return SequenceMatcher(None, left, right).ratio() >= 0.9


def _text_mentions(name: str, text: str) -> bool:
    needle = _normalize_name(name)
    haystack = _normalize_name(text)
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True

    needle_tokens = _ordered_tokens(name)
    haystack_tokens = _ordered_tokens(text)
    if len(needle_tokens) >= 2 and _contains_ordered_tokens(needle_tokens, haystack_tokens):
        return True
    if len(needle_tokens) >= 2 and set(needle_tokens).issubset(set(haystack_tokens)):
        return True
    if len(needle_tokens) == 1:
        return bool(re.search(rf"\b{re.escape(needle_tokens[0])}\b", haystack))
    return False


def _fuzzy_match(a: str, b: str) -> bool:
    return _names_match(a, b) or _text_mentions(a, b) or _text_mentions(b, a)


def _entity_name(entity: dict[str, Any]) -> str:
    for key in ("name", "title", "id"):
        value = str(entity.get(key) or "").strip()
        if value:
            return value
    return ""


def _entity_source(entity: dict[str, Any]) -> dict[str, Any]:
    source = entity.get("source")
    return source if isinstance(source, dict) else {}


def _source_heading(source: dict[str, Any]) -> str:
    return str(source.get("heading") or "").strip()


def _source_subheading(source: dict[str, Any]) -> str:
    return str(source.get("subheading") or "").strip()


def _source_page(source: dict[str, Any]) -> int | None:
    page_number = source.get("page_number")
    try:
        return int(page_number) if page_number is not None else None
    except (TypeError, ValueError):
        return None


def _same_document(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("document_id") or "").strip() == str(right.get("document_id") or "").strip()


def _page_proximity(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, int | None]:
    left_page = _source_page(left)
    right_page = _source_page(right)
    if left_page is None or right_page is None:
        return False, None
    distance = abs(left_page - right_page)
    return distance <= 1, distance


def _shared_heading(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_path = {
        _normalize_name(item)
        for item in left.get("heading_path", []) or []
        if str(item).strip()
    }
    right_path = {
        _normalize_name(item)
        for item in right.get("heading_path", []) or []
        if str(item).strip()
    }
    if left_path and right_path and left_path.intersection(right_path):
        return True
    left_heading = _normalize_name(_source_heading(left))
    right_heading = _normalize_name(_source_heading(right))
    return bool(left_heading and right_heading and left_heading == right_heading)


def _source_text(source: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            _source_heading(source),
            _source_subheading(source),
            *[str(item).strip() for item in source.get("heading_path", []) or [] if str(item).strip()],
        ]
        if str(part).strip()
    )


def _heading_mentions_name(source: dict[str, Any], name: str) -> bool:
    return _text_mentions(name, _source_text(source))


def _entity_text(entity: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "description",
        "summary",
        "content",
        "notes",
        "read_aloud",
        "objective",
        "stakes",
        "owner",
        "location",
        "rewards",
        "intro_text",
    ):
        value = entity.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _list_values(entity: dict[str, Any], key: str) -> list[str]:
    values = entity.get(key)
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    value = str(values or "").strip()
    return [value] if value else []


def _relation_key(rel: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(rel.get("from_type") or ""),
        str(rel.get("from_id") or ""),
        str(rel.get("relation") or ""),
        str(rel.get("to_type") or ""),
        str(rel.get("to_id") or ""),
    )


def _add_relationship(
    bucket: dict[tuple[str, str, str, str, str], dict[str, Any]],
    *,
    from_type: str,
    from_id: str,
    relation: str,
    to_type: str,
    to_id: str,
    confidence: float,
    method: str,
    left_source: dict[str, Any],
    right_source: dict[str, Any],
) -> None:
    if not from_id or not to_id:
        return
    same_page, page_distance = _page_proximity(left_source, right_source)
    item = {
        "from_type": from_type,
        "from_id": from_id,
        "relation": relation,
        "to_type": to_type,
        "to_id": to_id,
        "confidence": round(confidence, 3),
        "method": method,
        "same_heading": _shared_heading(left_source, right_source),
        "same_document": _same_document(left_source, right_source),
        "page_distance": page_distance,
    }
    if same_page and item["page_distance"] is None:
        item["page_distance"] = 0

    key = _relation_key(item)
    existing = bucket.get(key)
    if existing is None or float(item["confidence"]) > float(existing.get("confidence", 0.0)):
        bucket[key] = item


def extract_relationships(
    npcs: List[dict],
    locations: List[dict],
    scenes: List[dict],
    codex_entries: List[dict],
    model: str | None = None,
    *,
    quests: List[dict] | None = None,
    items: List[dict] | None = None,
    factions: List[dict] | None = None,
    encounters: List[dict] | None = None,
) -> List[dict[str, Any]]:
    """
    Link extracted entities into a connected campaign graph.

    The `model` parameter is retained for compatibility with older callers, but
    this stage now uses deterministic heuristics for traceable linking.
    """
    del model

    quests = quests or []
    items = items or []
    factions = factions or []
    encounters = encounters or []

    relationships: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    scene_location_map: dict[str, set[str]] = {}

    for npc in npcs:
        npc_name = _entity_name(npc)
        npc_source = _entity_source(npc)
        if not npc_name:
            continue
        npc_text = _entity_text(npc)
        npc_location_links: set[str] = set()

        for scene in scenes:
            scene_id = _entity_name(scene)
            scene_source = _entity_source(scene)
            scene_text = " ".join(
                [
                    " ".join(_list_values(scene, "npcs")),
                    _entity_text(scene),
                    _source_heading(scene_source),
                    _source_subheading(scene_source),
                ]
            )
            listed_npcs = _list_values(scene, "npcs")
            if any(_fuzzy_match(npc_name, candidate) for candidate in listed_npcs):
                _add_relationship(
                    relationships,
                    from_type="npc",
                    from_id=npc_name,
                    relation="appears_in",
                    to_type="scene",
                    to_id=scene_id,
                    confidence=0.96,
                    method="exact_name",
                    left_source=npc_source,
                    right_source=scene_source,
                )
            elif _text_mentions(npc_name, scene_text):
                _add_relationship(
                    relationships,
                    from_type="npc",
                    from_id=npc_name,
                    relation="appears_in",
                    to_type="scene",
                    to_id=scene_id,
                    confidence=0.84,
                    method="fuzzy_name",
                    left_source=npc_source,
                    right_source=scene_source,
                )
            elif _shared_heading(npc_source, scene_source):
                same_page, page_distance = _page_proximity(npc_source, scene_source)
                if same_page or (
                    page_distance is not None and page_distance <= 1 and _heading_mentions_name(scene_source, npc_name)
                ):
                    _add_relationship(
                        relationships,
                        from_type="npc",
                        from_id=npc_name,
                        relation="appears_in",
                        to_type="scene",
                        to_id=scene_id,
                        confidence=0.72,
                        method="heading_inheritance",
                        left_source=npc_source,
                        right_source=scene_source,
                    )

        for location in locations:
            location_name = _entity_name(location)
            location_source = _entity_source(location)
            if not location_name:
                continue
            if _text_mentions(location_name, npc_text) or _heading_mentions_name(npc_source, location_name):
                _add_relationship(
                    relationships,
                    from_type="npc",
                    from_id=npc_name,
                    relation="located_at",
                    to_type="location",
                    to_id=location_name,
                    confidence=0.9,
                    method="exact_name",
                    left_source=npc_source,
                    right_source=location_source,
                )
                npc_location_links.add(location_name)
            elif _shared_heading(npc_source, location_source):
                same_page, _distance = _page_proximity(npc_source, location_source)
                if same_page:
                    npc_location_links.add(f"fallback::{location_name}")

        fallback_npc_locations = [value.split("fallback::", 1)[1] for value in npc_location_links if value.startswith("fallback::")]
        direct_npc_locations = {value for value in npc_location_links if not value.startswith("fallback::")}
        if not direct_npc_locations and len(fallback_npc_locations) == 1:
            fallback_name = fallback_npc_locations[0]
            fallback_location = next(
                (location for location in locations if _entity_name(location) == fallback_name),
                None,
            )
            if fallback_location is not None:
                _add_relationship(
                    relationships,
                    from_type="npc",
                    from_id=npc_name,
                    relation="located_at",
                    to_type="location",
                    to_id=fallback_name,
                    confidence=0.7,
                    method="section_proximity",
                    left_source=npc_source,
                    right_source=_entity_source(fallback_location),
                )

    for scene in scenes:
        scene_id = _entity_name(scene)
        scene_source = _entity_source(scene)
        scene_text = " ".join(
            [
                scene_id,
                _entity_text(scene),
                _source_text(scene_source),
            ]
        )
        scene_location = str(scene.get("location") or "").strip()
        scene_npcs = _list_values(scene, "npcs")
        scene_locations = {scene_location} if scene_location else set()
        fallback_scene_locations: list[tuple[str, dict[str, Any]]] = []

        for location in locations:
            location_name = _entity_name(location)
            location_source = _entity_source(location)
            if not location_name:
                continue

            if scene_location and _names_match(scene_location, location_name):
                _add_relationship(
                    relationships,
                    from_type="scene",
                    from_id=scene_id,
                    relation="occurs_at",
                    to_type="location",
                    to_id=location_name,
                    confidence=0.97,
                    method="exact_name",
                    left_source=scene_source,
                    right_source=location_source,
                )
                scene_locations.add(location_name)
            elif _heading_mentions_name(scene_source, location_name) or _text_mentions(location_name, scene_text):
                _add_relationship(
                    relationships,
                    from_type="scene",
                    from_id=scene_id,
                    relation="occurs_at",
                    to_type="location",
                    to_id=location_name,
                    confidence=0.84,
                    method="heading_inheritance" if _heading_mentions_name(scene_source, location_name) else "fuzzy_name",
                    left_source=scene_source,
                    right_source=location_source,
                )
                scene_locations.add(location_name)
            elif _shared_heading(scene_source, location_source):
                same_page, page_distance = _page_proximity(scene_source, location_source)
                if same_page or (page_distance is not None and page_distance <= 1 and scene_location):
                    fallback_scene_locations.append((location_name, location_source))

        if not scene_locations and len(fallback_scene_locations) == 1:
            fallback_name, fallback_source = fallback_scene_locations[0]
            _add_relationship(
                relationships,
                from_type="scene",
                from_id=scene_id,
                relation="occurs_at",
                to_type="location",
                to_id=fallback_name,
                confidence=0.72,
                method="section_proximity",
                left_source=scene_source,
                right_source=fallback_source,
            )
            scene_locations.add(fallback_name)

        scene_location_map[scene_id] = {value for value in scene_locations if str(value).strip()}

        for quest in quests:
            quest_id = _entity_name(quest)
            quest_source = _entity_source(quest)
            related_npcs = _list_values(quest, "related_npcs")
            related_locations = _list_values(quest, "related_locations")
            npc_hits = any(
                any(_names_match(npc_name, ref_name) for ref_name in related_npcs)
                for npc_name in scene_npcs
            )
            location_hits = any(
                any(_names_match(location_name, ref_name) for ref_name in related_locations)
                for location_name in scene_location_map.get(scene_id, set())
            )
            heading_hits = _shared_heading(scene_source, quest_source)
            quest_text = " ".join([quest_id, _entity_text(quest), _source_text(quest_source)])
            if npc_hits or location_hits or _text_mentions(quest_id, scene_text):
                _add_relationship(
                    relationships,
                    from_type="scene",
                    from_id=scene_id,
                    relation="advances",
                    to_type="quest",
                    to_id=quest_id,
                    confidence=0.89 if npc_hits or location_hits else 0.78,
                    method="exact_name" if npc_hits or location_hits else "fuzzy_name",
                    left_source=scene_source,
                    right_source=quest_source,
                )
            elif heading_hits:
                same_page, page_distance = _page_proximity(scene_source, quest_source)
                if same_page or (
                    page_distance is not None
                    and page_distance <= 1
                    and (
                        any(_heading_mentions_name(scene_source, ref_name) for ref_name in related_locations)
                        or any(_heading_mentions_name(scene_source, ref_name) for ref_name in related_npcs)
                        or _text_mentions(scene_id, quest_text)
                    )
                ):
                    _add_relationship(
                        relationships,
                        from_type="scene",
                        from_id=scene_id,
                        relation="advances",
                        to_type="quest",
                        to_id=quest_id,
                        confidence=0.76,
                        method="section_proximity",
                        left_source=scene_source,
                        right_source=quest_source,
                    )

        for encounter in encounters:
            encounter_id = _entity_name(encounter)
            encounter_source = _entity_source(encounter)
            linked_scene = str(encounter.get("scene_id") or encounter.get("scene") or "").strip()
            if linked_scene and (_fuzzy_match(linked_scene, scene_id) or _fuzzy_match(linked_scene, scene.get("id") or "")):
                _add_relationship(
                    relationships,
                    from_type="scene",
                    from_id=scene_id,
                    relation="contains",
                    to_type="encounter",
                    to_id=encounter_id,
                    confidence=0.97,
                    method="exact_name",
                    left_source=scene_source,
                    right_source=encounter_source,
                )
            elif _fuzzy_match(scene_id, encounter_id) or _shared_heading(scene_source, encounter_source):
                _add_relationship(
                    relationships,
                    from_type="scene",
                    from_id=scene_id,
                    relation="contains",
                    to_type="encounter",
                    to_id=encounter_id,
                    confidence=0.78,
                    method="section_proximity",
                    left_source=scene_source,
                    right_source=encounter_source,
                )

    for quest in quests:
        quest_id = _entity_name(quest)
        quest_source = _entity_source(quest)
        for location in locations:
            location_name = _entity_name(location)
            location_source = _entity_source(location)
            if any(
                _names_match(location_name, ref_name)
                for ref_name in _list_values(quest, "related_locations")
            ) or _text_mentions(location_name, _entity_text(quest)) or _heading_mentions_name(quest_source, location_name):
                _add_relationship(
                    relationships,
                    from_type="quest",
                    from_id=quest_id,
                    relation="unfolds_at",
                    to_type="location",
                    to_id=location_name,
                    confidence=0.9,
                    method="exact_name",
                    left_source=quest_source,
                    right_source=location_source,
                )

    for item in items:
        item_id = _entity_name(item)
        item_source = _entity_source(item)
        owner = str(item.get("owner") or "").strip()
        for npc in npcs:
            npc_name = _entity_name(npc)
            npc_source = _entity_source(npc)
            if owner and _fuzzy_match(owner, npc_name):
                _add_relationship(
                    relationships,
                    from_type="item",
                    from_id=item_id,
                    relation="owned_by",
                    to_type="npc",
                    to_id=npc_name,
                    confidence=0.94,
                    method="exact_name",
                    left_source=item_source,
                    right_source=npc_source,
                )
            elif _fuzzy_match(npc_name, _entity_text(item)):
                _add_relationship(
                    relationships,
                    from_type="item",
                    from_id=item_id,
                    relation="owned_by",
                    to_type="npc",
                    to_id=npc_name,
                    confidence=0.76,
                    method="fuzzy_name",
                    left_source=item_source,
                    right_source=npc_source,
                )

    for faction in factions:
        faction_id = _entity_name(faction)
        faction_source = _entity_source(faction)
        faction_text = " ".join([faction_id, _entity_text(faction)])
        explicit_faction_locations: set[str] = set()
        fallback_faction_locations: list[tuple[str, dict[str, Any]]] = []
        for npc in npcs:
            npc_name = _entity_name(npc)
            npc_source = _entity_source(npc)
            npc_faction = str(npc.get("faction") or "").strip()
            if npc_faction and _fuzzy_match(npc_faction, faction_id):
                _add_relationship(
                    relationships,
                    from_type="faction",
                    from_id=faction_id,
                    relation="includes",
                    to_type="npc",
                    to_id=npc_name,
                    confidence=0.95,
                    method="exact_name",
                    left_source=faction_source,
                    right_source=npc_source,
                )
            elif _fuzzy_match(npc_name, faction_text):
                _add_relationship(
                    relationships,
                    from_type="faction",
                    from_id=faction_id,
                    relation="includes",
                    to_type="npc",
                    to_id=npc_name,
                    confidence=0.74,
                    method="fuzzy_name",
                    left_source=faction_source,
                    right_source=npc_source,
                )

        for location in locations:
            location_name = _entity_name(location)
            location_source = _entity_source(location)
            if _fuzzy_match(location_name, faction_text):
                _add_relationship(
                    relationships,
                    from_type="faction",
                    from_id=faction_id,
                    relation="operates_in",
                    to_type="location",
                    to_id=location_name,
                    confidence=0.82 if _fuzzy_match(location_name, faction_text) else 0.7,
                    method="exact_name",
                    left_source=faction_source,
                    right_source=location_source,
                )
                explicit_faction_locations.add(location_name)
            elif _shared_heading(faction_source, location_source):
                same_page, _distance = _page_proximity(faction_source, location_source)
                if same_page:
                    fallback_faction_locations.append((location_name, location_source))

        if not explicit_faction_locations and len(fallback_faction_locations) == 1:
            fallback_name, fallback_source = fallback_faction_locations[0]
            _add_relationship(
                relationships,
                from_type="faction",
                from_id=faction_id,
                relation="operates_in",
                to_type="location",
                to_id=fallback_name,
                confidence=0.7,
                method="page_proximity",
                left_source=faction_source,
                right_source=fallback_source,
            )

    for codex in codex_entries:
        codex_id = _entity_name(codex)
        codex_source = _entity_source(codex)
        codex_text = _entity_text(codex)
        for entity_type, entities in (
            ("npc", npcs),
            ("location", locations),
            ("scene", scenes),
            ("faction", factions),
        ):
            for entity in entities:
                entity_name = _entity_name(entity)
                entity_source = _entity_source(entity)
                if _fuzzy_match(entity_name, codex_text) or _shared_heading(codex_source, entity_source):
                    _add_relationship(
                        relationships,
                        from_type="codex",
                        from_id=codex_id,
                        relation="references",
                        to_type=entity_type,
                        to_id=entity_name,
                        confidence=0.72 if _fuzzy_match(entity_name, codex_text) else 0.64,
                        method="fuzzy_name" if _fuzzy_match(entity_name, codex_text) else "heading_inheritance",
                        left_source=codex_source,
                        right_source=entity_source,
                    )

    return list(relationships.values())
