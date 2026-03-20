"""Cross-chunk candidate fusion."""
from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any

from app.services.parsing.candidates import ENTITY_KEYS


def _norm(value: Any) -> str:
    text = re.sub(r"\s*\([^)]*\)", "", str(value or "").strip().lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _name_from_payload(payload: dict[str, Any], entity_type: str) -> str:
    for key in ("name", "title", "id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return entity_type


def _is_mergeable(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("entity_type") != right.get("entity_type"):
        return False
    if str(left.get("source_document_id") or "") != str(right.get("source_document_id") or ""):
        return False
    left_hint = _norm(left.get("canonical_hint"))
    right_hint = _norm(right.get("canonical_hint"))
    if not left_hint or not right_hint:
        return False
    if left_hint == right_hint:
        return True
    similarity = SequenceMatcher(None, left_hint, right_hint).ratio()
    left_page = left.get("page_number")
    right_page = right.get("page_number")
    try:
        nearby = abs(int(left_page) - int(right_page)) <= 2
    except (TypeError, ValueError):
        nearby = False
    if similarity >= 0.9 and nearby:
        return True
    # Alias-style merge: "captain ilvara" vs "ilvara" on adjacent pages/headings.
    suffix_alias = left_hint.endswith(f" {right_hint}") or right_hint.endswith(f" {left_hint}")
    return suffix_alias and nearby


def _merge_candidate(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged_payload = dict(left.get("raw_payload") or {})
    right_payload = dict(right.get("raw_payload") or {})
    for key, value in right_payload.items():
        left_value = merged_payload.get(key)
        if isinstance(value, list):
            current = list(left_value or [])
            for item in value:
                if item not in current:
                    current.append(item)
            merged_payload[key] = current
            continue
        if (not left_value and value) or (isinstance(value, str) and len(value) > len(str(left_value or ""))):
            merged_payload[key] = value
    merged["raw_payload"] = merged_payload
    merged["confidence"] = max(float(left.get("confidence", 0.0) or 0.0), float(right.get("confidence", 0.0) or 0.0))
    evidence = list(left.get("evidence_list") or [])
    evidence.extend(list(right.get("evidence_list") or []))
    if not evidence:
        evidence = [item for item in (left.get("raw_payload", {}).get("evidence"), right.get("raw_payload", {}).get("evidence")) if item]
    merged["evidence_list"] = evidence
    merged["mention_count"] = max(1, int(left.get("mention_count", 1))) + max(1, int(right.get("mention_count", 1)))
    return merged


def fuse_candidates(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in ENTITY_KEYS}
    merged_groups: dict[str, list[dict[str, Any]]] = {key: [] for key in ENTITY_KEYS}
    for item in candidates:
        entity_type = str(item.get("entity_type") or "")
        if entity_type not in grouped:
            continue
        grouped[entity_type].append(dict(item))

    for entity_type, items in grouped.items():
        for item in items:
            merged = False
            for index, existing in enumerate(merged_groups[entity_type]):
                if _is_mergeable(existing, item):
                    merged_groups[entity_type][index] = _merge_candidate(existing, item)
                    merged = True
                    break
            if not merged:
                seeded = dict(item)
                seeded.setdefault("mention_count", 1)
                evidence = seeded.get("raw_payload", {}).get("evidence")
                seeded["evidence_list"] = [evidence] if evidence else []
                merged_groups[entity_type].append(seeded)

    output: dict[str, list[dict[str, Any]]] = {key: [] for key in ENTITY_KEYS}
    for entity_type, items in merged_groups.items():
        for item in items:
            payload = dict(item.get("raw_payload") or {})
            payload["confidence"] = max(float(payload.get("confidence", 0.0) or 0.0), float(item.get("confidence", 0.0) or 0.0))
            payload["mention_count"] = int(item.get("mention_count", 1) or 1)
            payload["importance_score"] = float(payload.get("importance_score", 0.0) or 0.0)
            payload["canonical_name"] = _name_from_payload(payload, entity_type)
            payload["evidence_list"] = list(item.get("evidence_list") or [])
            output[entity_type].append(payload)
    return output
