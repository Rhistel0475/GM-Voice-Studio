"""Pass 2: run type-specific extraction on classified chunks."""
import re
from typing import Any, Iterable

from app.services.parsing.extractors import (
    extract_clues,
    extract_consequences,
    extract_codex_entries,
    extract_hooks,
    extract_items,
    extract_locations,
    extract_npcs,
    extract_quests,
    extract_read_aloud,
    extract_rewards,
    extract_rumors,
    extract_scene_seeds,
    extract_secrets,
)
from app.services.parsing.evidence import attach_entity_evidence
from app.services.parsing.models import SectionChunk


_SCENE_SIGNAL_RE = re.compile(
    r"\b(?:read aloud|boxed text|paraphrase|when the pcs|when the investigators|"
    r"when the players|upon entering|as you enter|initiative|attack|ambush|"
    r"social encounter|combat encounter|conversation|speaks|greets|"
    r"search check|spot hidden|perception check|survival check)\b",
    re.IGNORECASE,
)


def _slug(value: str, prefix: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return f"{prefix}_{clean}" if clean else prefix


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _attach_source(items: Iterable[dict[str, Any]], chunk: SectionChunk) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        decorated.append(attach_entity_evidence(item, chunk))
    return decorated


def _build_encounters(scene_items: Iterable[dict[str, Any]], chunk: SectionChunk) -> list[dict[str, Any]]:
    encounters: list[dict[str, Any]] = []
    for scene in scene_items:
        if not isinstance(scene, dict):
            continue
        title = str(scene.get("title") or "").strip()
        if not title:
            continue
        encounters.append(
            {
                "id": _slug(title, "encounter"),
                "name": title,
                "intro_text": str(scene.get("read_aloud") or "").strip(),
                "ambience": "combat" if str(scene.get("type") or "").strip().lower() == "combat" else "",
                "enemies": [name for name in scene.get("npcs", []) if str(name).strip()],
                "difficulty": str(scene.get("difficulty") or "").strip(),
                "treasure_or_rewards": [
                    str(scene.get("rewards") or "").strip()
                ] if str(scene.get("rewards") or "").strip() else [],
                "scene_id": str(scene.get("id") or "").strip() or _slug(title, "scene"),
                "source": dict(scene.get("source") or chunk.source_metadata()),
            }
        )
    return encounters


def _scene_has_distinct_detail(scene: dict[str, Any]) -> bool:
    if any(str(scene.get(field) or "").strip() for field in ("read_aloud", "difficulty", "rewards", "notes", "act")):
        return True

    npcs = scene.get("npcs") or []
    if isinstance(npcs, list) and any(str(name).strip() for name in npcs):
        return True

    scene_type = str(scene.get("type") or "").strip().lower()
    if scene_type and scene_type != "exploration":
        return True

    title = _normalized_name(scene.get("title"))
    location = _normalized_name(scene.get("location"))
    return bool(location and title and location != title)


def _filter_location_echo_scenes(
    chunk: SectionChunk,
    location_items: Iterable[dict[str, Any]],
    scene_items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    if chunk.chunk_type_guess != "location_section":
        return [dict(item) for item in scene_items if isinstance(item, dict)]

    location_names = {
        _normalized_name(item.get("name"))
        for item in location_items
        if isinstance(item, dict) and _normalized_name(item.get("name"))
    }
    heading_name = _normalized_name(chunk.heading)
    if heading_name:
        location_names.add(heading_name)
    if not location_names:
        return [dict(item) for item in scene_items if isinstance(item, dict)]

    filtered: list[dict[str, Any]] = []
    for scene in scene_items:
        if not isinstance(scene, dict):
            continue
        title = _normalized_name(scene.get("title"))
        location = _normalized_name(scene.get("location"))
        looks_like_location_echo = (
            title in location_names
            and (not location or location in location_names)
            and not _scene_has_distinct_detail(scene)
        )
        if looks_like_location_echo:
            continue
        filtered.append(dict(scene))
    return filtered


def _chunk_supports_scene_extraction(chunk: SectionChunk, labels: set[str]) -> bool:
    if chunk.chunk_type_guess in {"boxed_text", "encounter_section"}:
        return True
    if "scene" in labels and chunk.content_type == "scene":
        return True
    if "encounter" in labels and chunk.content_type == "encounter":
        return True
    if chunk.chunk_type_guess == "location_section":
        text = " ".join(
            part for part in (chunk.subheading, chunk.raw_text[:1600]) if str(part).strip()
        )
        return bool(_SCENE_SIGNAL_RE.search(text))
    if "scene" in labels:
        text = " ".join(
            part for part in (chunk.subheading, chunk.raw_text[:1600]) if str(part).strip()
        )
        return bool(_SCENE_SIGNAL_RE.search(text))
    return False


def extract_typed_entities(
    chunks: list[SectionChunk],
    model: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Pass 2 extraction orchestration.

    Keeps routing explicit so the parser remains easy to debug:
    - npc chunks -> NPC extractor
    - location/scene chunks -> location and/or scene extractor
    - encounter chunks -> scene extractor plus derived encounter payloads
    - quest chunks -> quest extractor
    - item chunks -> item extractor
    - lore/faction/mixed chunks -> codex extractor fallback
    """
    results: dict[str, list[dict[str, Any]]] = {
        "npcs": [],
        "locations": [],
        "scenes": [],
        "quests": [],
        "items": [],
        "codex_entries": [],
        "encounters": [],
        "clues": [],
        "secrets": [],
        "rumors": [],
        "read_alouds": [],
        "consequences": [],
        "rewards": [],
        "hooks": [],
    }

    for chunk in chunks:
        labels = set(chunk.content_types or [])
        if not labels:
            labels = {chunk.content_type or "lore"}

        extracted_any = False
        location_items: list[dict[str, Any]] = []

        if "npc" in labels:
            npc_items = _attach_source(extract_npcs(chunk, model=model), chunk)
            results["npcs"].extend(npc_items)
            extracted_any = extracted_any or bool(npc_items)

        if "location" in labels:
            location_items = _attach_source(extract_locations(chunk, model=model), chunk)
            results["locations"].extend(location_items)
            extracted_any = extracted_any or bool(location_items)

        should_extract_scene = _chunk_supports_scene_extraction(chunk, labels)
        if should_extract_scene:
            scene_items = _attach_source(extract_scene_seeds(chunk, model=model), chunk)
            scene_items = _filter_location_echo_scenes(chunk, location_items, scene_items)
            results["scenes"].extend(scene_items)
            if chunk.chunk_type_guess == "encounter_section" or chunk.content_type == "encounter":
                results["encounters"].extend(_build_encounters(scene_items, chunk))
            extracted_any = extracted_any or bool(scene_items)

        if "quest" in labels:
            quest_items = _attach_source(extract_quests(chunk, model=model), chunk)
            results["quests"].extend(quest_items)
            extracted_any = extracted_any or bool(quest_items)

        if "item" in labels:
            item_items = _attach_source(extract_items(chunk, model=model), chunk)
            results["items"].extend(item_items)
            extracted_any = extracted_any or bool(item_items)

        codex_type = None
        if "faction" in labels:
            codex_type = "faction"
        elif "lore" in labels or "mixed" in labels:
            codex_type = "lore"

        if codex_type is not None:
            codex_items = _attach_source(
                extract_codex_entries(chunk, content_type=codex_type, model=model),
                chunk,
            )
            results["codex_entries"].extend(codex_items)
            extracted_any = extracted_any or bool(codex_items)

        high_value_extractors = (
            ("clues", extract_clues),
            ("secrets", extract_secrets),
            ("rumors", extract_rumors),
            ("read_alouds", extract_read_aloud),
            ("consequences", extract_consequences),
            ("rewards", extract_rewards),
            ("hooks", extract_hooks),
        )
        for result_key, extractor in high_value_extractors:
            extracted_items = _attach_source(extractor(chunk, model=model), chunk)
            if extracted_items:
                results[result_key].extend(extracted_items)
                extracted_any = True

        if not extracted_any and chunk.body.strip():
            fallback_codex = _attach_source(
                extract_codex_entries(chunk, content_type="lore", model=model),
                chunk,
            )
            results["codex_entries"].extend(fallback_codex)

    return results
