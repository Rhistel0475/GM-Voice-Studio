"""
Stage 6: Deduplicate similar entities and merge canonical variants.

This layer is intentionally conservative:
- merge exact canonical variants (articles, punctuation, parenthetical suffixes)
- merge source-aware near-duplicates across long PDFs
- avoid collapsing clearly distinct hierarchical entities like
  "Hightower" vs "Hightower Main Entrance"
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable, List


_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)")


def _normalize_name(s: str) -> str:
    """Lowercase, strip, collapse spaces for comparison."""
    return " ".join((s or "").lower().split())


def _canonical_name(s: str) -> str:
    text = _normalize_name(s)
    text = _PAREN_SUFFIX_RE.sub("", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = _LEADING_ARTICLE_RE.sub("", text)
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    return " ".join(text.split())


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", _canonical_name(s))


def _ordered_tokens(s: str) -> list[str]:
    return [token for token in _tokens(s) if token]


def _confidence(entry: dict[str, Any]) -> float:
    try:
        return float(entry.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _canonical_name(left), _canonical_name(right)).ratio()


def _source(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry.get("source")
    return source if isinstance(source, dict) else {}


def _same_document(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(_source(left).get("document_id") or "").strip() == str(_source(right).get("document_id") or "").strip()


def _page_number(entry: dict[str, Any]) -> int | None:
    value = _source(entry).get("page_number")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _shared_source_context(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _same_document(left, right):
        return False

    left_source = _source(left)
    right_source = _source(right)
    left_heading = _normalize_name(left_source.get("heading") or "")
    right_heading = _normalize_name(right_source.get("heading") or "")
    if left_heading and right_heading and left_heading == right_heading:
        return True

    left_path = {_normalize_name(item) for item in left_source.get("heading_path", []) or [] if str(item).strip()}
    right_path = {_normalize_name(item) for item in right_source.get("heading_path", []) or [] if str(item).strip()}
    if left_path and right_path and left_path.intersection(right_path):
        return True

    left_page = _page_number(left)
    right_page = _page_number(right)
    if left_page is not None and right_page is not None:
        return abs(left_page - right_page) <= 1

    return False


def _preserve_order_unique(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _prefix_match(short_tokens: list[str], long_tokens: list[str]) -> bool:
    if not short_tokens or len(short_tokens) > len(long_tokens):
        return False
    return short_tokens == long_tokens[: len(short_tokens)]


def _suffix_match(short_tokens: list[str], long_tokens: list[str]) -> bool:
    if not short_tokens or len(short_tokens) > len(long_tokens):
        return False
    return short_tokens == long_tokens[-len(short_tokens) :]


def _contained_alias(short_name: str, long_name: str) -> bool:
    short_tokens = _tokens(short_name)
    long_tokens = _tokens(long_name)
    if len(short_tokens) < 2:
        return False
    return _prefix_match(short_tokens, long_tokens)


def _contains_ordered_tokens(needle: list[str], haystack: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if haystack[index : index + width] == needle:
            return True
    return False


def _prefer_primary(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if _confidence(right) > _confidence(left):
        return dict(right), dict(left)
    return dict(left), dict(right)


def _merge_scalar_fields(primary: dict[str, Any], secondary: dict[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        left = str(primary.get(field) or "").strip()
        right = str(secondary.get(field) or "").strip()
        if not left and right:
            primary[field] = right
        elif right and len(right) > len(left):
            primary[field] = right


def _merge_union_list(primary: dict[str, Any], secondary: dict[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        left = primary.get(field) or []
        right = secondary.get(field) or []
        if isinstance(left, list) or isinstance(right, list):
            primary[field] = _preserve_order_unique(list(left) + list(right))


def _should_merge_npc(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name = str(left.get("name") or "").strip()
    right_name = str(right.get("name") or "").strip()
    if not left_name or not right_name:
        return False

    if _canonical_name(left_name) == _canonical_name(right_name):
        return True

    if _contained_alias(left_name, right_name) or _contained_alias(right_name, left_name):
        return True

    left_tokens = _tokens(left_name)
    right_tokens = _tokens(right_name)
    if len(left_tokens) >= 2 and _suffix_match(left_tokens, right_tokens):
        return True
    if len(right_tokens) >= 2 and _suffix_match(right_tokens, left_tokens):
        return True

    if _shared_source_context(left, right) and _text_similarity(left_name, right_name) >= 0.86:
        return True

    return False


def _should_merge_location(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name = str(left.get("name") or "").strip()
    right_name = str(right.get("name") or "").strip()
    if not left_name or not right_name:
        return False

    if _canonical_name(left_name) == _canonical_name(right_name):
        return True

    if _shared_source_context(left, right):
        if _contained_alias(left_name, right_name) or _contained_alias(right_name, left_name):
            return True
        if _text_similarity(left_name, right_name) >= 0.88:
            return True

    return False


def _should_merge_scene(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_title = str(left.get("title") or "").strip()
    right_title = str(right.get("title") or "").strip()
    if not left_title or not right_title:
        return False

    if _canonical_name(left_title) == _canonical_name(right_title):
        return True

    if _shared_source_context(left, right):
        if _contained_alias(left_title, right_title) or _contained_alias(right_title, left_title):
            return True
        if _text_similarity(left_title, right_title) >= 0.88:
            return True

    return False


def _merge_npc_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    primary, secondary = _prefer_primary(left, right)

    primary["name"] = str(primary.get("name") or secondary.get("name") or "").strip()
    _merge_scalar_fields(
        primary,
        secondary,
        ("role", "personality", "faction", "description", "motivation", "secrets", "hp", "cr", "image_url", "voice_id"),
    )

    if not primary.get("ac") and secondary.get("ac"):
        primary["ac"] = secondary.get("ac")

    for field in ("secrets", "motivation", "personality"):
        left_text = str(primary.get(field) or "").strip()
        right_text = str(secondary.get(field) or "").strip()
        if right_text and right_text not in left_text:
            primary[field] = f"{left_text}; {right_text}".strip("; ").strip() if left_text else right_text

    if "source" not in primary and secondary.get("source"):
        primary["source"] = secondary["source"]

    primary["confidence"] = max(_confidence(left), _confidence(right))
    return primary


def _merge_location_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    primary, secondary = _prefer_primary(left, right)
    primary["name"] = str(primary.get("name") or secondary.get("name") or "").strip()
    _merge_scalar_fields(primary, secondary, ("description", "image_url"))
    _merge_union_list(primary, secondary, ("tags",))
    if "source" not in primary and secondary.get("source"):
        primary["source"] = secondary["source"]
    primary["confidence"] = max(_confidence(left), _confidence(right))
    return primary


def _merge_scene_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    primary, secondary = _prefer_primary(left, right)
    primary["title"] = str(primary.get("title") or secondary.get("title") or "").strip()
    _merge_scalar_fields(
        primary,
        secondary,
        ("act", "type", "read_aloud", "location", "difficulty", "rewards", "notes", "image_url", "description"),
    )
    _merge_union_list(primary, secondary, ("npcs", "triggers", "connected_scenes", "reveals", "items", "tags"))
    if "source" not in primary and secondary.get("source"):
        primary["source"] = secondary["source"]
    primary["confidence"] = max(_confidence(left), _confidence(right))
    return primary


def _npc_reference_matches(reference: str, candidate: str) -> bool:
    if _canonical_name(reference) == _canonical_name(candidate):
        return True
    if _contained_alias(reference, candidate) or _contained_alias(candidate, reference):
        return True

    reference_tokens = _ordered_tokens(reference)
    candidate_tokens = _ordered_tokens(candidate)
    if len(reference_tokens) >= 2 and _contains_ordered_tokens(reference_tokens, candidate_tokens):
        return True
    if len(candidate_tokens) >= 2 and _contains_ordered_tokens(candidate_tokens, reference_tokens):
        return True
    if len(reference_tokens) >= 2 and _suffix_match(reference_tokens, candidate_tokens):
        return True
    if len(candidate_tokens) >= 2 and _suffix_match(candidate_tokens, reference_tokens):
        return True
    return _text_similarity(reference, candidate) >= 0.9


def _location_reference_matches(reference: str, candidate: str) -> bool:
    if _canonical_name(reference) == _canonical_name(candidate):
        return True

    reference_tokens = _ordered_tokens(reference)
    candidate_tokens = _ordered_tokens(candidate)
    shorter, longer = (reference_tokens, candidate_tokens) if len(reference_tokens) <= len(candidate_tokens) else (candidate_tokens, reference_tokens)
    if len(shorter) >= 3 and _prefix_match(shorter, longer):
        return True
    return _text_similarity(reference, candidate) >= 0.92


def _scene_reference_matches(reference: str, candidate: str) -> bool:
    if _canonical_name(reference) == _canonical_name(candidate):
        return True
    return _text_similarity(reference, candidate) >= 0.9


def _resolve_reference_name(
    value: str,
    entries: list[dict[str, Any]],
    *,
    key: str,
    matcher,
) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    best_name = raw
    best_score = (-1.0, -1, -1)
    for entry in entries:
        candidate = str(entry.get(key) or "").strip()
        if not candidate or not matcher(raw, candidate):
            continue
        score = (
            _confidence(entry),
            len(_ordered_tokens(candidate)),
            len(candidate),
        )
        if score > best_score:
            best_name = candidate
            best_score = score
    return best_name


def canonicalize_npc_reference(value: str, npcs: List[dict[str, Any]]) -> str:
    return _resolve_reference_name(value, npcs, key="name", matcher=_npc_reference_matches)


def canonicalize_npc_references(values: Any, npcs: List[dict[str, Any]]) -> list[str]:
    if not isinstance(values, list):
        return []
    return _preserve_order_unique(
        canonicalize_npc_reference(str(value).strip(), npcs)
        for value in values
        if str(value).strip()
    )


def canonicalize_location_reference(value: str, locations: List[dict[str, Any]]) -> str:
    return _resolve_reference_name(value, locations, key="name", matcher=_location_reference_matches)


def canonicalize_location_references(values: Any, locations: List[dict[str, Any]]) -> list[str]:
    if not isinstance(values, list):
        return []
    return _preserve_order_unique(
        canonicalize_location_reference(str(value).strip(), locations)
        for value in values
        if str(value).strip()
    )


def canonicalize_scene_reference(value: str, scenes: List[dict[str, Any]]) -> str:
    return _resolve_reference_name(value, scenes, key="title", matcher=_scene_reference_matches)


def dedupe_npcs(npcs: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge NPCs with same or canonicalized near-identical names."""
    merged: list[dict[str, Any]] = []
    for npc in npcs:
        name = str(npc.get("name") or "").strip()
        if not name:
            continue
        for index, existing in enumerate(merged):
            if _should_merge_npc(existing, npc):
                merged[index] = _merge_npc_pair(existing, npc)
                break
        else:
            merged.append(dict(npc))
    return merged


def dedupe_locations(locations: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge locations with same or canonicalized near-identical names."""
    merged: list[dict[str, Any]] = []
    for location in locations:
        name = str(location.get("name") or "").strip()
        if not name:
            continue
        for index, existing in enumerate(merged):
            if _should_merge_location(existing, location):
                merged[index] = _merge_location_pair(existing, location)
                break
        else:
            merged.append(dict(location))
    return merged


def dedupe_scenes(scenes: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge scenes with same or canonicalized near-identical titles."""
    merged: list[dict[str, Any]] = []
    for scene in scenes:
        title = str(scene.get("title") or "").strip()
        if not title:
            continue
        for index, existing in enumerate(merged):
            if _should_merge_scene(existing, scene):
                merged[index] = _merge_scene_pair(existing, scene)
                break
        else:
            merged.append(dict(scene))
    return merged


def dedupe_codex_entries(entries: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge codex entries with same id or normalized title; keep higher confidence."""
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry_id = (entry.get("id") or "").strip() or _canonical_name(entry.get("title") or "")
        if not entry_id:
            continue
        if entry_id not in by_id or _confidence(entry) >= _confidence(by_id[entry_id]):
            by_id[entry_id] = dict(entry)
    return list(by_id.values())
