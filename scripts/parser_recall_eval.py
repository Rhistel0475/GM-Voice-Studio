#!/usr/bin/env python3
"""Parser recall evaluation harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.parsing.pipeline import run_parsing_pipeline

ENTITY_KEYS = ("npcs", "scenes", "locations", "quests", "factions", "items")
HIGH_VALUE_KEYS = ("clues", "rumors", "secrets", "read_alouds", "consequences", "rewards", "hooks")
WEIGHTS = {
    "entity": 0.30,
    "high_value": 0.25,
    "merge": 0.20,
    "evidence": 0.10,
    "importance": 0.10,
    "coverage": 0.05,
}


def _name_index(items: list[dict[str, Any]]) -> set[str]:
    values = set()
    for item in items:
        name = str(item.get("name") or item.get("title") or "").strip()
        if name:
            values.add(name.casefold())
    return values


def _score_evidence(payload: dict[str, Any]) -> tuple[int, int]:
    total = 0
    complete = 0
    for key in (*ENTITY_KEYS, *HIGH_VALUE_KEYS, "codex_entries"):
        for item in payload.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            total += 1
            ok = all(
                [
                    bool(item.get("source_document_id") or item.get("source", {}).get("document_id")),
                    item.get("page_number") is not None or item.get("source", {}).get("page_number") is not None,
                    bool(item.get("source_chunk_id") or item.get("source", {}).get("source_chunk_id") or item.get("source", {}).get("chunk_id")),
                    bool(item.get("evidence_text") or item.get("evidence", {}).get("evidence_text")),
                    item.get("confidence") is not None,
                    item.get("importance_score") is not None,
                ]
            )
            if ok:
                complete += 1
    return complete, total


def _top_names(payload: dict[str, Any], limit: int = 10) -> list[str]:
    scored: list[tuple[float, str]] = []
    for key in (*ENTITY_KEYS, *HIGH_VALUE_KEYS):
        for item in payload.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("title") or "").strip()
            if not name:
                continue
            scored.append((float(item.get("importance_score", 0.0) or 0.0), name))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [name for _, name in scored[:limit]]


def evaluate_document(doc_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    payload = run_parsing_pipeline(doc_path.read_text(encoding="utf-8", errors="ignore"))
    entity_expected = expected.get("entities", {})
    high_expected = expected.get("high_value", {})
    merge_expected = expected.get("merge_expectations", [])
    importance_expected = expected.get("importance_expectations", [])
    coverage_expected = expected.get("coverage_expectations", {})

    checks: list[dict[str, Any]] = []

    # Entity checks
    for key in ENTITY_KEYS:
        expected_names = entity_expected.get(key)
        expected_min = entity_expected.get(f"{key}_min")
        expected_max = entity_expected.get(f"{key}_max")
        actual_items = payload.get(key, []) or []
        if isinstance(expected_names, list):
            actual_names = _name_index([i for i in actual_items if isinstance(i, dict)])
            missing = [name for name in expected_names if name.casefold() not in actual_names]
            checks.append({"dimension": "entity", "key": key, "pass": len(missing) == 0, "missing": missing})
        elif expected_min is not None:
            checks.append(
                {
                    "dimension": "entity",
                    "key": key,
                    "pass": len(actual_items) >= int(expected_min),
                    "expected_min": int(expected_min),
                    "actual": len(actual_items),
                }
            )
        if expected_max is not None:
            checks.append(
                {
                    "dimension": "entity",
                    "key": key,
                    "pass": len(actual_items) <= int(expected_max),
                    "expected_max": int(expected_max),
                    "actual": len(actual_items),
                }
            )

    # High-value checks
    for key in HIGH_VALUE_KEYS:
        expected_min = high_expected.get(f"{key}_min")
        expected_max = high_expected.get(f"{key}_max")
        if expected_min is None:
            if expected_max is None:
                continue
        actual = len(payload.get(key, []) or [])
        if expected_min is not None:
            checks.append(
                {
                    "dimension": "high_value",
                    "key": key,
                    "pass": actual >= int(expected_min),
                    "expected_min": int(expected_min),
                    "actual": actual,
                }
            )
        if expected_max is not None:
            checks.append(
                {
                    "dimension": "high_value",
                    "key": key,
                    "pass": actual <= int(expected_max),
                    "expected_max": int(expected_max),
                    "actual": actual,
                }
            )

    # Merge checks
    for entry in merge_expected:
        key = str(entry.get("type", "")).strip().lower()
        canonical = str(entry.get("canonical", "")).strip()
        max_dup = int(entry.get("max_duplicates", 1))
        collection = f"{key}s" if key.endswith("t") else f"{key}s"
        actual = payload.get(collection, []) or []
        matches = 0
        for item in actual:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("title") or "").strip()
            if canonical and canonical.casefold() in name.casefold():
                matches += 1
        checks.append(
            {
                "dimension": "merge",
                "key": key,
                "canonical": canonical,
                "pass": matches <= max_dup and matches > 0,
                "actual_matches": matches,
                "max_duplicates": max_dup,
            }
        )

    # Importance checks
    top_names = _top_names(payload, limit=20)
    top_names_cf = [name.casefold() for name in top_names]
    for entry in importance_expected:
        name = str(entry.get("name", "")).strip()
        top_k = int(entry.get("top_k", 10))
        rank_max = entry.get("rank_max")
        top_k_names = top_names_cf[:top_k]
        item = {
            "dimension": "importance",
            "name": name,
            "top_k": top_k,
            "pass": name.casefold() in top_k_names if name else False,
        }
        if rank_max is not None and name:
            rank = next((i + 1 for i, candidate in enumerate(top_names_cf) if candidate == name.casefold()), None)
            item["rank"] = rank
            item["rank_max"] = int(rank_max)
            item["pass"] = bool(item["pass"] and rank is not None and rank <= int(rank_max))
        checks.append(
            {
                **item
            }
        )

    # Evidence completeness
    complete, total = _score_evidence(payload)
    evidence_ratio = 1.0 if total == 0 else complete / total
    checks.append({"dimension": "evidence", "pass": evidence_ratio >= 0.98, "complete": complete, "total": total, "ratio": evidence_ratio})

    # Coverage checks
    total_gaps = int((payload.get("coverage_report") or {}).get("summary", {}).get("total_gaps", 0))
    max_overflag = coverage_expected.get("must_not_overflag_max")
    if max_overflag is not None:
        checks.append(
            {
                "dimension": "coverage",
                "pass": total_gaps <= int(max_overflag),
                "total_gaps": total_gaps,
                "max_allowed": int(max_overflag),
            }
        )

    # Dimension-weighted scoring and hard gates
    by_dimension: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        by_dimension.setdefault(str(check.get("dimension")), []).append(check)
    dimension_scores: dict[str, float] = {}
    for dimension, items in by_dimension.items():
        passed_count = len([item for item in items if item.get("pass")])
        dimension_scores[dimension] = 0.0 if not items else passed_count / len(items)
    weighted_score = 0.0
    for dimension, weight in WEIGHTS.items():
        weighted_score += weight * dimension_scores.get(dimension, 1.0)
    hard_gates = {
        "evidence_ratio": evidence_ratio >= 0.98,
        "merge_quality": dimension_scores.get("merge", 1.0) >= 0.80,
        "high_value_recall": dimension_scores.get("high_value", 1.0) >= 0.85,
    }

    passed = [item for item in checks if item.get("pass")]
    return {
        "document": doc_path.name,
        "checks": checks,
        "score": 0 if not checks else round(100.0 * weighted_score, 2),
        "dimension_scores": {k: round(v, 3) for k, v in dimension_scores.items()},
        "hard_gates": hard_gates,
        "hard_pass": all(hard_gates.values()),
        "coverage_summary": (payload.get("coverage_report") or {}).get("summary", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate parser recall against gold expectations.")
    parser.add_argument("--docs-dir", default="/home/brian/GM-Voice-Studio/tests/fixtures/parsing")
    parser.add_argument("--gold-file", default="/home/brian/GM-Voice-Studio/tests/fixtures/parsing/recall_gold.json")
    parser.add_argument("--output-json", default="/home/brian/GM-Voice-Studio/tests/fixtures/parsing/recall_eval_output.json")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    gold = json.loads(Path(args.gold_file).read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for spec in gold.get("documents", []):
        doc_name = spec.get("doc")
        if not doc_name:
            continue
        doc_path = docs_dir / doc_name
        if not doc_path.exists():
            results.append({"document": doc_name, "score": 0, "checks": [{"dimension": "setup", "pass": False, "reason": "document_missing"}]})
            continue
        results.append(evaluate_document(doc_path, spec.get("expected", {})))

    aggregate_score = round(sum(item.get("score", 0.0) for item in results) / max(len(results), 1), 2)
    aggregate_hard_pass = all(bool(item.get("hard_pass", False)) for item in results) if results else False
    out = {
        "aggregate_score": aggregate_score,
        "aggregate_hard_pass": aggregate_hard_pass,
        "documents": results,
    }
    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
