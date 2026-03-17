"""
Pass 1: classify section-aware chunks into one or more generic campaign content types.

The classifier uses deterministic structural heuristics first, then optionally asks the
LLM to refine ambiguous chunks while preserving the parser's system-agnostic schema.
"""
import json
import logging
import re
from collections import defaultdict
from typing import Any, Iterable, List

from app.services.parsing.models import CONTENT_TYPES, SectionChunk


_TYPE_KEYWORDS = {
    "npc": (
        "npc",
        "villain",
        "ally",
        "merchant",
        "guard",
        "cultist",
        "scholar",
        "priest",
        "captain",
        "stat block",
        "armor class",
        "hit points",
    ),
    "location": (
        "location",
        "room",
        "area",
        "forest",
        "tavern",
        "inn",
        "road",
        "village",
        "town",
        "dungeon",
        "temple",
        "cave",
        "crypt",
        "watchtower",
    ),
    "scene": (
        "scene",
        "read aloud",
        "boxed text",
        "arrive",
        "discover",
        "investigate",
        "conversation",
        "social",
        "exploration",
        "travel",
    ),
    "encounter": (
        "encounter",
        "ambush",
        "combat",
        "battle",
        "fight",
        "attack",
        "initiative",
        "monster",
        "enemy",
        "hostile",
    ),
    "quest": (
        "quest",
        "quest hook",
        "hook",
        "objective",
        "mission",
        "clue",
        "rumor",
        "secret",
        "goal",
        "investigation",
    ),
    "lore": (
        "lore",
        "history",
        "legend",
        "background",
        "origin",
        "myth",
        "culture",
        "truth",
    ),
    "item": (
        "treasure",
        "loot",
        "artifact",
        "item",
        "relic",
        "reward",
        "document",
        "letter",
        "journal",
        "key",
    ),
    "faction": (
        "faction",
        "guild",
        "cult",
        "order",
        "house",
        "clan",
        "tribe",
        "cabal",
        "organization",
    ),
}

_CHUNK_TYPE_PRIORS = {
    "boxed_text": ("scene", "lore"),
    "stat_block": ("npc", "encounter"),
    "encounter_section": ("encounter", "scene"),
    "location_section": ("location",),
    "quest_section": ("quest",),
    "lore_section": ("lore",),
    "item_section": ("item",),
    "faction_section": ("faction", "lore"),
}

_STAT_SIGNAL_RE = re.compile(
    r"\b(?:ac|armor class|hp|hit points|cr|challenge|str|dex|con|int|wis|cha|san|move|initiative|speed)\b",
    re.IGNORECASE,
)
_PERSON_TITLE_RE = re.compile(
    r"\b(?:captain|lord|lady|sir|dame|priest|merchant|guard|chief|baron|duke|professor|doctor)\b",
    re.IGNORECASE,
)
_SCENE_CUE_RE = re.compile(
    r"\b(?:read aloud|boxed text|paraphrase|when the pcs|when the investigators|"
    r"when the players|when the party|when characters|upon entering|as you enter|"
    r"the creature attacks|the enemy attacks|an encounter begins|initiative|"
    r"search check|spot hidden|perception check|survival check|social encounter|"
    r"combat encounter|ambush|attack|conversation|speaks|greets)\b",
    re.IGNORECASE,
)
_LOCATION_CUE_RE = re.compile(
    r"\b(?:located|entrance|chamber|hall|room|cellar|cabin|shop|tower|road|vale|"
    r"forest|crypt|doorway|shelves|stonework|structure|terrain|façade|floor|wall|"
    r"ceiling|parlor)\b",
    re.IGNORECASE,
)


def _get_client():
    from app.infrastructure.llm.anthropic_client import get_client

    return get_client()


def _normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def _dedupe_labels(labels: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for label in labels:
        clean = _normalize_label(label)
        if clean not in CONTENT_TYPES or clean in seen or clean == "mixed":
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _score_keywords(text: str, keywords: Iterable[str]) -> float:
    score = 0.0
    for keyword in keywords:
        if keyword in text:
            score += 1.0
    return score


def _score_to_confidence(top_score: float, second_score: float = 0.0, ambiguous: bool = False) -> float:
    margin = max(0.0, top_score - second_score)
    confidence = 0.48 + min(top_score, 6.0) * 0.06 + min(margin, 3.0) * 0.08
    if ambiguous:
        confidence -= 0.12
    return max(0.2, min(0.97, confidence))


def _has_scene_cues(chunk: SectionChunk) -> bool:
    text = " ".join(
        part for part in (chunk.subheading, chunk.raw_text[:1500]) if str(part).strip()
    )
    if not text.strip():
        return False
    return bool(_SCENE_CUE_RE.search(text))


def _has_location_cues(chunk: SectionChunk) -> bool:
    text = " ".join(
        part for part in (chunk.heading, chunk.subheading, chunk.raw_text[:1500]) if str(part).strip()
    )
    if not text.strip():
        return False
    return bool(_LOCATION_CUE_RE.search(text))


def _heuristic_classification(chunk: SectionChunk) -> dict[str, Any]:
    context = " ".join(
        part
        for part in (
            chunk.heading,
            chunk.subheading,
            " ".join(chunk.heading_path),
            chunk.raw_text[:1600],
        )
        if str(part).strip()
    ).lower()

    scores: dict[str, float] = defaultdict(float)
    for label in _CHUNK_TYPE_PRIORS.get(chunk.chunk_type_guess, ()):
        scores[label] += 2.5

    for label, keywords in _TYPE_KEYWORDS.items():
        scores[label] += _score_keywords(context, keywords)

    if len(_STAT_SIGNAL_RE.findall(context)) >= 2:
        scores["npc"] += 3.5
        scores["encounter"] += 1.5

    if chunk.heading and _PERSON_TITLE_RE.search(chunk.heading):
        scores["npc"] += 1.5

    if chunk.subheading.lower() in {"read aloud", "boxed text"}:
        scores["scene"] += 3.0
    if _has_scene_cues(chunk):
        scores["scene"] += 1.75
    if _has_location_cues(chunk):
        scores["location"] += 1.0

    if not scores:
        scores["lore"] = 1.0

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_label, top_score = ordered[0]
    additional = [
        label
        for label, score in ordered[1:]
        if score >= max(2.0, top_score - 1.25)
    ][:2]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0

    if chunk.chunk_type_guess == "stat_block" and top_label == "npc" and "encounter" not in additional:
        additional.append("encounter")
    if chunk.chunk_type_guess == "encounter_section" and top_label == "encounter" and "scene" not in additional:
        additional.append("scene")
    if (
        chunk.chunk_type_guess == "location_section"
        and top_label == "location"
        and _has_scene_cues(chunk)
        and "scene" not in additional
    ):
        additional.append("scene")

    content_types = _dedupe_labels([top_label, *additional])
    ambiguous = len(content_types) > 1 and top_score < 4.0
    primary = "mixed" if ambiguous else top_label
    secondary = content_types[1] if len(content_types) > 1 else None
    confidence = _score_to_confidence(top_score, second_score, ambiguous=ambiguous)

    return {
        "primary_type": primary,
        "content_types": content_types or ["lore"],
        "secondary_type": secondary,
        "classification_confidence": confidence,
        "classification_method": "heuristic",
    }


def _classify_chunks_with_llm(
    chunks: List[SectionChunk],
    heuristics: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]] | None:
    from app.core.config import AI_MODEL

    if not chunks:
        return []

    client = _get_client()
    effective_model = model or AI_MODEL

    payload_lines: list[str] = []
    for index, chunk in enumerate(chunks):
        heuristic = heuristics[index]
        preview = chunk.raw_text[:900].strip()
        payload_lines.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"Document: {chunk.document_id}",
                    f"Page: {chunk.page_number}",
                    f"Heading: {chunk.heading or '(none)'}",
                    f"Subheading: {chunk.subheading or '(none)'}",
                    f"Chunk type guess: {chunk.chunk_type_guess}",
                    f"Heuristic types: {', '.join(heuristic['content_types']) or 'lore'}",
                    preview,
                ]
            )
        )

    prompt = (
        "Classify each RPG adventure chunk into one primary type and optional additional types.\n\n"
        "Allowed types: npc, location, scene, encounter, quest, lore, item, faction, mixed.\n"
        "- Use only generic campaign semantics. Do not infer game-system-specific mechanics.\n"
        "- Prefer npc for named character writeups or stat blocks.\n"
        "- Prefer scene for read-aloud, social, exploration, or narrative beats.\n"
        "- Prefer encounter for hostile or set-piece conflicts; add scene as an additional type when useful.\n"
        "- Prefer quest for hooks, objectives, mysteries, clues, rumors, or mission structure.\n"
        "- Prefer lore for background and setting context.\n"
        "- Prefer mixed only when a chunk genuinely contains multiple equally important entity types.\n\n"
        "Return ONLY a JSON array. One object per chunk in order.\n"
        "Each object must be: {\"primary_type\": \"...\", \"additional_types\": [\"...\", ...]}.\n\n"
        "Chunks:\n---\n"
        + "\n---\n".join(payload_lines)
        + "\n---"
    )

    response = client.messages.create(
        model=effective_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    items = json.loads(raw)
    if not isinstance(items, list) or len(items) != len(chunks):
        return None
    return items


def _merge_classification(
    heuristic: dict[str, Any],
    llm_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(llm_result, dict):
        return heuristic

    primary = _normalize_label(llm_result.get("primary_type") or "")
    additional = llm_result.get("additional_types")
    if not isinstance(additional, list):
        additional = []

    llm_types = _dedupe_labels([primary, *(str(item) for item in additional)])
    if not llm_types:
        llm_types = list(heuristic.get("content_types") or [])

    merged_types = _dedupe_labels([*llm_types, *(heuristic.get("content_types") or [])])
    if not merged_types:
        merged_types = ["lore"]

    if primary not in CONTENT_TYPES:
        primary = heuristic.get("primary_type") or merged_types[0]

    if primary == "mixed" and merged_types:
        secondary = merged_types[0]
    else:
        secondary = merged_types[1] if len(merged_types) > 1 else None

    heuristic_primary = _normalize_label(heuristic.get("primary_type") or "")
    heuristic_types = set(heuristic.get("content_types") or [])
    llm_types_set = set(llm_types)
    agreement = 0.0
    if primary and primary == heuristic_primary:
        agreement += 0.12
    if heuristic_types and llm_types_set:
        overlap = len(heuristic_types.intersection(llm_types_set)) / max(len(heuristic_types.union(llm_types_set)), 1)
        agreement += overlap * 0.1
    base_confidence = float(heuristic.get("classification_confidence") or 0.62)
    classification_confidence = max(0.2, min(0.98, base_confidence + agreement - (0.06 if primary == "mixed" else 0.0)))

    return {
        "primary_type": primary or heuristic.get("primary_type") or "lore",
        "content_types": merged_types,
        "secondary_type": secondary,
        "classification_confidence": classification_confidence,
        "classification_method": "heuristic+llm",
    }


def classify_chunks(chunks: List[SectionChunk], model: str | None = None) -> List[SectionChunk]:
    """
    Pass 1 classification over section-aware chunks.

    Each chunk receives:
    - content_type: primary type or "mixed" when ambiguous
    - secondary_type: optional additional type
    - content_types: the concrete extractor types to run in pass 2
    """
    if not chunks:
        return chunks

    heuristics = [_heuristic_classification(chunk) for chunk in chunks]

    llm_results: list[dict[str, Any]] | None = None
    try:
        llm_results = _classify_chunks_with_llm(chunks, heuristics, model or "")
    except Exception as exc:
        logging.warning("Chunk classification LLM refinement failed; using heuristics only: %s", exc)

    for index, chunk in enumerate(chunks):
        merged = _merge_classification(
            heuristics[index],
            llm_results[index] if llm_results and index < len(llm_results) else None,
        )
        chunk.content_type = _normalize_label(merged["primary_type"]) or "lore"
        if chunk.content_type not in CONTENT_TYPES:
            chunk.content_type = "lore"
        chunk.secondary_type = _normalize_label(merged.get("secondary_type") or "") or None
        if chunk.secondary_type not in CONTENT_TYPES:
            chunk.secondary_type = None
        chunk.content_types = _dedupe_labels(merged.get("content_types") or [])
        if not chunk.content_types:
            fallback = chunk.content_type if chunk.content_type != "mixed" else "lore"
            chunk.content_types = [fallback]
        chunk.classification_confidence = float(merged.get("classification_confidence") or 0.0)
        chunk.classification_method = str(merged.get("classification_method") or "heuristic")

    return chunks
