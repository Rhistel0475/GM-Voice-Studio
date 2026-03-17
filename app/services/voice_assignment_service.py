from __future__ import annotations

import re
from typing import Any, Optional

from app.services import tts_service
from app.services.voice_store_service import list_voices, load_embedding_path

_SUPPORTED_TAGS: tuple[str, ...] = (
    "male",
    "female",
    "old",
    "young",
    "rough",
    "noble",
    "merchant",
    "villain",
    "guard",
    "scholar",
)

_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "male": ("male", "man", "boy", "he", "him", "his", "gentleman", "sir"),
    "female": ("female", "woman", "girl", "she", "her", "hers", "lady", "madam"),
    "old": ("old", "elder", "elderly", "aged", "ancient", "grandfather", "grandmother", "veteran"),
    "young": ("young", "youth", "teen", "child", "boy", "girl", "apprentice"),
    "rough": ("rough", "gruff", "raspy", "harsh", "scarred", "hard-bitten", "gravelly"),
    "noble": ("noble", "lord", "lady", "duke", "duchess", "regal", "refined", "aristocrat", "courtier"),
    "merchant": ("merchant", "trader", "vendor", "shopkeeper", "innkeeper", "barkeep", "apothecary"),
    "villain": ("villain", "tyrant", "cruel", "evil", "menacing", "bandit", "assassin", "cultist", "warlord"),
    "guard": ("guard", "captain", "soldier", "watch", "warden", "marshal", "knight", "mercenary"),
    "scholar": ("scholar", "sage", "professor", "academic", "librarian", "scribe", "researcher", "wizard"),
}

_NPC_FIELD_WEIGHTS: dict[str, int] = {
    "role": 5,
    "description": 3,
    "personality": 3,
    "name": 1,
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_normalize_text(item) for item in value)
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9\s]+", " ", text)


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = r"(?:^|\b)" + re.escape(keyword) + r"(?:\b|$)"
    return re.search(pattern, text) is not None


def _voice_search_text(voice: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            _normalize_text(voice.get("name")),
            _normalize_text(voice.get("description")),
            _normalize_text(voice.get("tone")),
            _normalize_text(voice.get("style")),
            _normalize_text(voice.get("provider_kind")),
            _normalize_text(voice.get("source")),
            _normalize_text(voice.get("faction")),
            _normalize_text(voice.get("tags")),
        )
        if part
    )


def _explicit_voice_tags(voice: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    for raw_tag in voice.get("tags") or []:
        normalized = _normalize_text(raw_tag).replace(" ", "_")
        if normalized in _SUPPORTED_TAGS:
            tags.add(normalized)
            continue
        for tag, keywords in _TAG_KEYWORDS.items():
            if normalized == tag or any(_contains_keyword(normalized, keyword) for keyword in keywords):
                tags.add(tag)
                break
    return tags


def _derive_voice_tags(voice: dict[str, Any]) -> set[str]:
    search = _voice_search_text(voice)
    tags = set(_explicit_voice_tags(voice))
    for tag, keywords in _TAG_KEYWORDS.items():
        if tag in tags:
            continue
        if any(_contains_keyword(search, keyword) for keyword in keywords):
            tags.add(tag)
    return tags


def _infer_npc_tag_weights(npc: dict[str, Any]) -> dict[str, int]:
    field_texts = {
        field: _normalize_text(npc.get(field))
        for field in _NPC_FIELD_WEIGHTS
    }
    weights: dict[str, int] = {}
    for tag, keywords in _TAG_KEYWORDS.items():
        score = 0
        for field, weight in _NPC_FIELD_WEIGHTS.items():
            text = field_texts.get(field, "")
            if text and any(_contains_keyword(text, keyword) for keyword in keywords):
                score += weight
        if score > 0:
            weights[tag] = score
    return weights


def _score_voice(
    voice: dict[str, Any],
    npc_tag_weights: dict[str, int],
) -> tuple[int, int, int, str]:
    derived_tags = _derive_voice_tags(voice)
    explicit_tags = _explicit_voice_tags(voice)
    matched_tags = [tag for tag in _SUPPORTED_TAGS if tag in npc_tag_weights and tag in derived_tags]

    weighted_score = sum(npc_tag_weights[tag] for tag in matched_tags)
    weighted_score += sum(2 for tag in matched_tags if tag in explicit_tags)
    weighted_score += len(matched_tags)

    is_custom = 1 if _normalize_text(voice.get("source")) == "custom" else 0
    has_explicit_tags = len(explicit_tags)
    name = _normalize_text(voice.get("name")) or _normalize_text(voice.get("voice_id"))
    return (weighted_score, has_explicit_tags, is_custom, name)


def _list_assignable_voices(owner_id: Optional[str] = None) -> list[dict[str, Any]]:
    if tts_service.is_hume_provider():
        return [
            voice
            for voice in tts_service.list_hume_voices()
            if str((voice or {}).get("voice_id") or "").strip()
        ]

    usable: list[dict[str, Any]] = []
    for voice in list_voices(owner_id=owner_id):
        voice_id = str((voice or {}).get("voice_id") or "").strip()
        if not voice_id:
            continue
        if not load_embedding_path(voice_id):
            continue
        usable.append(tts_service.normalize_stored_voice(voice))
    return usable


def _confidence_for_match(matched_tags: list[str], npc_tag_weights: dict[str, int]) -> float:
    total_possible = sum(npc_tag_weights.values())
    if total_possible <= 0:
        return 0.1
    matched_weight = sum(npc_tag_weights.get(tag, 0) for tag in matched_tags)
    if matched_weight <= 0:
        return 0.1
    confidence = matched_weight / total_possible
    return round(min(0.99, max(0.1, confidence)), 3)


def suggest_voice_for_npc(
    npc: dict[str, Any],
    owner_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    voices = _list_assignable_voices(owner_id=owner_id)
    if not voices:
        return None

    npc_tag_weights = _infer_npc_tag_weights(npc)
    selected = max(voices, key=lambda voice: _score_voice(voice, npc_tag_weights))
    voice_id = str(selected.get("voice_id") or "").strip()
    if not voice_id:
        return None

    matched_tags = [
        tag
        for tag in _SUPPORTED_TAGS
        if tag in npc_tag_weights and tag in _derive_voice_tags(selected)
    ]

    return {
        "voice_id": voice_id,
        "provider": str(selected.get("provider") or tts_service.get_tts_provider()),
        "confidence": _confidence_for_match(matched_tags, npc_tag_weights),
        "voice_name": str(selected.get("name") or voice_id),
        "matched_tags": matched_tags,
    }
