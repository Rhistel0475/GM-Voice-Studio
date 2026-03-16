"""Suggest likely next scenes from the campaign graph and recent player intent."""
from __future__ import annotations

import re
from typing import Any, Optional

from app.repositories import campaign_repository

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_DIRECTION_TOKENS = {"north", "south", "east", "west", "road", "gate", "inn"}
_ACTION_THEMES = {
    "combat": {"attack", "ambush", "fight", "battle", "bandit", "combat", "strike"},
    "travel": {"go", "travel", "ride", "road", "journey", "path", "north", "south", "east", "west", "continue"},
    "rest": {"rest", "sleep", "camp", "inn", "tavern", "drink", "recover"},
    "town": {"town", "city", "market", "merchant", "shop", "village", "plaza"},
    "watch": {"guard", "watch", "tower", "gate", "captain", "patrol"},
    "dungeon": {"dungeon", "crypt", "ruin", "cave", "cavern", "underground"},
}


def _scene_label(scene: dict[str, Any]) -> str:
    return str(scene.get("title") or scene.get("name") or scene.get("id") or "Scene").strip() or "Scene"


def _scene_description(scene: dict[str, Any]) -> str:
    for candidate in (
        scene.get("description"),
        scene.get("summary"),
        scene.get("read_aloud"),
        scene.get("notes"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _connected_refs(scene: dict[str, Any]) -> list[str]:
    raw = scene.get("connected_scenes") or scene.get("connectedScenes") or []
    refs: list[str] = []
    seen: set[str] = set()
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        candidate = item
        if isinstance(item, dict):
            candidate = item.get("id") or item.get("scene_id") or item.get("title") or item.get("name")
        value = str(candidate or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        refs.append(value)
    return refs


def _scene_ref_keys(scene: dict[str, Any]) -> set[str]:
    keys = {
        str(scene.get("id") or "").strip().casefold(),
        str(scene.get("title") or "").strip().casefold(),
        str(scene.get("name") or "").strip().casefold(),
    }
    return {key for key in keys if key}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(str(text or "").lower()):
        if len(token) >= 3 or token in _DIRECTION_TOKENS:
            tokens.add(token)
    return tokens


def _scene_search_blob(scene: dict[str, Any]) -> str:
    return " ".join(
        part for part in (
            _scene_label(scene),
            _scene_description(scene),
            str(scene.get("location") or "").strip(),
            str(scene.get("type") or "").strip(),
            str(scene.get("act") or "").strip(),
        )
        if part
    )


def _resolve_ref_scene(ref: str, scenes: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    wanted = str(ref or "").strip().casefold()
    if not wanted:
        return None
    for scene in scenes:
        if wanted in _scene_ref_keys(scene):
            return scene
    return None


def _graph_weights(current_scene: dict[str, Any], scenes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    current_keys = _scene_ref_keys(current_scene)
    current_id = str(current_scene.get("id") or "").strip()
    weights: dict[str, dict[str, Any]] = {}

    def register(scene: dict[str, Any], weight: float, reason: str) -> None:
        scene_id = str(scene.get("id") or _scene_label(scene)).strip()
        if not scene_id or scene_id == current_id:
            return
        existing = weights.get(scene_id)
        if existing is None or weight > existing["weight"]:
            weights[scene_id] = {"scene": scene, "weight": weight, "reason": reason}

    for ref in _connected_refs(current_scene):
        candidate = _resolve_ref_scene(ref, scenes)
        if candidate is not None:
            register(candidate, 6.0, "Connected to the current scene")

    for scene in scenes:
        scene_id = str(scene.get("id") or "").strip()
        if not scene_id or scene_id == current_id:
            continue
        refs = {ref.casefold() for ref in _connected_refs(scene)}
        if current_keys & refs:
            register(scene, 4.5, "Linked through the campaign graph")

    try:
        current_index = next(index for index, scene in enumerate(scenes) if str(scene.get("id") or "").strip() == current_id)
    except StopIteration:
        current_index = -1

    if current_index >= 0:
        for offset, bonus in ((1, 2.25), (2, 1.25), (-1, 1.0)):
            idx = current_index + offset
            if 0 <= idx < len(scenes):
                register(scenes[idx], bonus, "Adjacent in the campaign flow")

    return weights


def _score_action_match(scene: dict[str, Any], action_tokens: set[str]) -> tuple[float, Optional[str]]:
    if not action_tokens:
        return 0.0, None

    scene_tokens = _tokenize(_scene_search_blob(scene))
    overlap = action_tokens & scene_tokens
    score = min(len(overlap) * 1.4, 4.2)
    reason: Optional[str] = None
    if overlap:
        reason = "Matches the party's recent action"

    for theme, theme_tokens in _ACTION_THEMES.items():
        if not action_tokens.intersection(theme_tokens):
            continue
        if theme == "combat" and str(scene.get("type") or "").strip().lower() == "combat":
            score += 2.4
            reason = reason or "Fits the current conflict"
        elif theme == "rest" and any(token in scene_tokens for token in {"inn", "tavern", "rest"}):
            score += 2.0
            reason = reason or "Supports a rest or tavern stop"
        elif theme == "travel" and any(token in scene_tokens for token in {"road", "north", "south", "east", "west", "path"}):
            score += 2.0
            reason = reason or "Follows the party's travel direction"
        elif theme == "town" and any(token in scene_tokens for token in {"town", "city", "market", "merchant", "village"}):
            score += 2.0
            reason = reason or "Matches the destination the party mentioned"
        elif theme == "watch" and any(token in scene_tokens for token in {"watch", "guard", "tower", "gate"}):
            score += 2.0
            reason = reason or "Lines up with the guards or watch"
        elif theme == "dungeon" and any(token in scene_tokens for token in {"dungeon", "crypt", "ruin", "cave"}):
            score += 2.0
            reason = reason or "Points toward the dungeon route"

    return score, reason


def suggest_next_scenes(
    current_scene: dict[str, Any],
    player_action: str,
    campaign_scenes: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Suggest likely next scenes from explicit links, graph neighbors, and player intent."""
    if not isinstance(current_scene, dict):
        return {"suggested_scenes": []}

    scenes = [scene for scene in (campaign_scenes or []) if isinstance(scene, dict)]
    if not scenes:
        campaign_id = current_scene.get("campaign_id")
        if campaign_id is not None:
            try:
                campaign = campaign_repository.get_by_id(int(campaign_id))
            except (TypeError, ValueError):
                campaign = None
            scenes = [scene for scene in (campaign or {}).get("scenes", []) if isinstance(scene, dict)]

    current_id = str(current_scene.get("id") or "").strip()
    action_tokens = _tokenize(player_action)
    graph_scores = _graph_weights(current_scene, scenes)
    ranked: list[tuple[float, dict[str, Any], str]] = []

    for scene in scenes:
        scene_id = str(scene.get("id") or "").strip()
        if not scene_id or scene_id == current_id:
            continue

        score = 0.0
        reasons: list[str] = []
        graph_hit = graph_scores.get(scene_id)
        if graph_hit is not None:
            score += float(graph_hit["weight"])
            reasons.append(str(graph_hit["reason"]))

        action_score, action_reason = _score_action_match(scene, action_tokens)
        score += action_score
        if action_reason:
            reasons.append(action_reason)

        if score <= 0:
            continue

        ranked.append((score, scene, reasons[0] if reasons else "Suggested next step"))

    ranked.sort(key=lambda item: (-item[0], _scene_label(item[1]).casefold()))
    suggestions = [
        {
            **scene,
            "title": _scene_label(scene),
            "name": str(scene.get("name") or _scene_label(scene)).strip(),
            "description": _scene_description(scene),
            "connected_scenes": _connected_refs(scene),
            "suggestion_reason": reason,
            "suggestion_score": round(score, 2),
        }
        for score, scene, reason in ranked[:4]
    ]
    return {"suggested_scenes": suggestions}


def suggest_next_scenes_for_scene(current_scene_id: str, player_action: str) -> dict[str, Any]:
    """Load a scene by id/title and return suggested next scenes."""
    scene = campaign_repository.get_scene_record(current_scene_id)
    if scene is None:
        raise FileNotFoundError("Scene not found")
    campaign = campaign_repository.get_by_id(int(scene["campaign_id"]))
    scenes = [item for item in (campaign or {}).get("scenes", []) if isinstance(item, dict)]
    return suggest_next_scenes(scene, player_action, campaign_scenes=scenes)
