"""
Stage 6: Deduplicate similar entities (by name/title); prefer higher confidence, merge fields.
"""
import logging
from typing import Any, List


def _normalize_name(s: str) -> str:
    """Lowercase, strip, collapse spaces for comparison."""
    return " ".join((s or "").lower().split())


def dedupe_npcs(npcs: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge NPCs with same normalized name; keep higher confidence, merge list-like fields."""
    by_name: dict[str, dict[str, Any]] = {}
    for n in npcs:
        name = (n.get("name") or "").strip()
        if not name:
            continue
        key = _normalize_name(name)
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = dict(n)
            continue
        # Merge: prefer higher confidence; merge secrets/motivation if different
        if (n.get("confidence") or 0) > (existing.get("confidence") or 0):
            by_name[key] = dict(n)
            other = existing
        else:
            other = n
        for field in ("secrets", "motivation", "personality"):
            a = (existing.get(field) or "").strip()
            b = (other.get(field) or "").strip()
            if b and b not in a:
                by_name[key][field] = f"{a}; {b}".strip() if a else b
    return list(by_name.values())


def dedupe_locations(locations: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge locations with same normalized name; keep higher confidence."""
    by_name: dict[str, dict[str, Any]] = {}
    for loc in locations:
        name = (loc.get("name") or "").strip()
        if not name:
            continue
        key = _normalize_name(name)
        if key not in by_name or (loc.get("confidence") or 0) >= (by_name[key].get("confidence") or 0):
            by_name[key] = dict(loc)
    return list(by_name.values())


def dedupe_scenes(scenes: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge scenes with same normalized title; keep higher confidence, merge npcs list."""
    by_title: dict[str, dict[str, Any]] = {}
    for s in scenes:
        title = (s.get("title") or "").strip()
        if not title:
            continue
        key = _normalize_name(title)
        existing = by_title.get(key)
        if existing is None:
            by_title[key] = dict(s)
            continue
        if (s.get("confidence") or 0) > (existing.get("confidence") or 0):
            by_title[key] = dict(s)
            other = existing
        else:
            other = s
        # Merge npcs list
        npcs = list(set((existing.get("npcs") or []) + (other.get("npcs") or [])))
        by_title[key]["npcs"] = npcs
    return list(by_title.values())


def dedupe_codex_entries(entries: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge codex entries with same id or normalized title; keep higher confidence."""
    by_id: dict[str, dict[str, Any]] = {}
    for e in entries:
        eid = (e.get("id") or "").strip() or _normalize_name(e.get("title") or "")
        if not eid:
            continue
        if eid not in by_id or (e.get("confidence") or 0) >= (by_id[eid].get("confidence") or 0):
            by_id[eid] = dict(e)
    return list(by_id.values())
