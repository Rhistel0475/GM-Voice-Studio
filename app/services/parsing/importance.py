"""Importance scoring for extracted entities."""
from __future__ import annotations

from typing import Any


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mentions_score(item: dict[str, Any]) -> float:
    mentions = int(item.get("mention_count", 1) or 1)
    return _bounded(min(mentions, 5) / 5.0)


def _heading_score(item: dict[str, Any]) -> float:
    heading = str(item.get("heading") or item.get("source", {}).get("heading") or "").strip()
    subheading = str(item.get("subheading") or item.get("source", {}).get("subheading") or "").strip()
    if heading and subheading:
        return 1.0
    if heading:
        return 0.7
    return 0.2


def _linkage_score(item: dict[str, Any]) -> float:
    linked = 0
    for key in ("related_npcs", "related_locations", "related_scenes", "related_factions", "npcs"):
        value = item.get(key)
        if isinstance(value, list):
            linked += len([v for v in value if str(v).strip()])
    return _bounded(min(linked, 5) / 5.0)


def _content_signal_score(item: dict[str, Any]) -> float:
    text = " ".join(
        str(item.get(key) or "")
        for key in ("objective", "rewards", "stakes", "read_aloud", "description", "summary", "type")
    ).lower()
    weight = 0.0
    for token in ("objective", "clue", "secret", "reward", "consequence", "encounter", "hook", "read aloud"):
        if token in text:
            weight += 0.18
    return _bounded(weight)


def _entity_type_weight(entity_type: str) -> float:
    key = (entity_type or "").strip().lower()
    weights = {
        "quests": 1.0,
        "npcs": 0.95,
        "scenes": 0.9,
        "locations": 0.85,
        "hooks": 0.82,
        "secrets": 0.82,
        "clues": 0.8,
        "consequences": 0.8,
        "factions": 0.78,
        "read_alouds": 0.74,
        "rumors": 0.68,
        "items": 0.62,
        "rewards": 0.58,
    }
    return weights.get(key, 0.7)


def score_importance(entities: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    scored: dict[str, list[dict[str, Any]]] = {}
    for key, items in entities.items():
        scored[key] = []
        mention_index: dict[str, int] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("title") or "").strip().casefold()
            if name:
                mention_index[name] = mention_index.get(name, 0) + 1

        for item in items:
            enriched = dict(item)
            base_conf = float(enriched.get("confidence", 0.5) or 0.5)
            name_key = str(enriched.get("name") or enriched.get("title") or "").strip().casefold()
            recurrence_bonus = _bounded((mention_index.get(name_key, 1) - 1) / 4.0)
            type_weight = _entity_type_weight(key)
            score = (
                0.26 * _mentions_score(enriched)
                + 0.22 * _heading_score(enriched)
                + 0.22 * _linkage_score(enriched)
                + 0.18 * _content_signal_score(enriched)
                + 0.12 * recurrence_bonus
            )
            score = _bounded((0.58 * score + 0.42 * base_conf) * (0.65 + 0.35 * type_weight))
            enriched["importance_score"] = score
            evidence = enriched.get("evidence")
            if isinstance(evidence, dict):
                evidence["importance_score"] = score
                enriched["evidence"] = evidence
            scored[key].append(enriched)
    return scored
