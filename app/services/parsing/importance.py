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


def score_importance(entities: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    scored: dict[str, list[dict[str, Any]]] = {}
    for key, items in entities.items():
        scored[key] = []
        for item in items:
            enriched = dict(item)
            base_conf = float(enriched.get("confidence", 0.5) or 0.5)
            score = (
                0.3 * _mentions_score(enriched)
                + 0.25 * _heading_score(enriched)
                + 0.2 * _linkage_score(enriched)
                + 0.25 * _content_signal_score(enriched)
            )
            score = _bounded(0.55 * score + 0.45 * base_conf)
            enriched["importance_score"] = score
            evidence = enriched.get("evidence")
            if isinstance(evidence, dict):
                evidence["importance_score"] = score
                enriched["evidence"] = evidence
            scored[key].append(enriched)
    return scored
