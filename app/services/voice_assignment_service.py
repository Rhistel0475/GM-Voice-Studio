from __future__ import annotations

from typing import Any, Optional

from app.services import tts_service
from app.services.voice_store_service import list_voices, load_embedding_path

_ARCHETYPES: tuple[dict[str, Any], ...] = (
    {
        "id": "villain",
        "keywords": (
            "villain",
            "tyrant",
            "warlord",
            "cultist",
            "assassin",
            "crime lord",
            "cruel",
            "ruthless",
            "menacing",
        ),
        "voice_keywords": ("villain", "dark", "grim", "cold", "sinister", "menacing", "ominous", "raspy"),
        "tone": "menacing",
        "style": "controlled and intimidating",
    },
    {
        "id": "guard",
        "keywords": (
            "guard",
            "soldier",
            "captain",
            "watch",
            "warden",
            "knight",
            "mercenary",
            "commander",
            "veteran",
        ),
        "voice_keywords": ("guard", "soldier", "captain", "marshal", "steady", "authoritative", "veteran"),
        "tone": "authoritative",
        "style": "crisp and disciplined",
    },
    {
        "id": "noble",
        "keywords": (
            "noble",
            "lord",
            "lady",
            "duke",
            "duchess",
            "baron",
            "queen",
            "king",
            "aristocrat",
            "courtier",
            "ambassador",
            "scholar",
        ),
        "voice_keywords": ("noble", "court", "refined", "regal", "elegant", "scholar", "measured"),
        "tone": "refined",
        "style": "formal and poised",
    },
    {
        "id": "merchant",
        "keywords": (
            "merchant",
            "trader",
            "vendor",
            "shopkeeper",
            "innkeeper",
            "barkeep",
            "bartender",
            "apothecary",
            "artisan",
        ),
        "voice_keywords": ("merchant", "trader", "innkeeper", "barkeep", "shop", "warm", "friendly", "welcoming"),
        "tone": "warm",
        "style": "chatty and inviting",
    },
    {
        "id": "mystic",
        "keywords": (
            "oracle",
            "seer",
            "prophet",
            "witch",
            "wizard",
            "mage",
            "cleric",
            "priest",
            "mystic",
            "sage",
            "spirit",
            "ancient",
        ),
        "voice_keywords": ("oracle", "seer", "whisper", "mystic", "sage", "echo", "ancient", "ethereal"),
        "tone": "mysterious",
        "style": "measured and otherworldly",
    },
    {
        "id": "rogue",
        "keywords": (
            "rogue",
            "smuggler",
            "thief",
            "spy",
            "bandit",
            "pirate",
            "scoundrel",
            "outlaw",
            "raider",
        ),
        "voice_keywords": ("rogue", "bandit", "smuggler", "sly", "quick", "rascal", "street", "mocking"),
        "tone": "wry",
        "style": "quick and sly",
    },
    {
        "id": "default",
        "keywords": (),
        "voice_keywords": ("clear", "versatile", "neutral", "storyteller", "companion"),
        "tone": "grounded",
        "style": "conversational and adaptable",
    },
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_normalize_text(item) for item in value)
    return str(value).strip().lower()


def _npc_text(npc: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            _normalize_text(npc.get("name")),
            _normalize_text(npc.get("role")),
            _normalize_text(npc.get("personality")),
            _normalize_text(npc.get("personality_traits")),
            _normalize_text(npc.get("description")),
            _normalize_text(npc.get("motivation")),
            _normalize_text(npc.get("secrets")),
            _normalize_text(npc.get("faction")),
        )
        if part
    )


def _voice_text(voice: dict[str, Any]) -> str:
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


def _count_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword and keyword in text)


def _select_archetype(npc_text: str) -> dict[str, Any]:
    best = _ARCHETYPES[-1]
    best_score = 0
    for archetype in _ARCHETYPES:
        score = _count_hits(npc_text, archetype["keywords"])
        if score > best_score:
            best = archetype
            best_score = score
    return best


def _score_voice(voice: dict[str, Any], npc_text: str, archetype: dict[str, Any]) -> tuple[int, int, int, str]:
    search = _voice_text(voice)
    score = 0
    score += _count_hits(search, archetype["voice_keywords"]) * 5
    score += _count_hits(search, tuple(_normalize_text(archetype["tone"]).split())) * 2
    score += _count_hits(search, tuple(_normalize_text(archetype["style"]).split())) * 1

    role_text = _normalize_text(voice.get("name"))
    npc_role = _normalize_text(npc_text)
    if role_text and role_text in npc_role:
        score += 2

    is_custom = 1 if _normalize_text(voice.get("source")) == "custom" else 0
    is_non_featured = 1 if not voice.get("featured") else 0
    if voice.get("featured"):
        score -= 1

    name = _normalize_text(voice.get("name")) or _normalize_text(voice.get("voice_id"))
    return (score, is_custom, is_non_featured, name)


def _list_assignable_voices(owner_id: Optional[str] = None) -> list[dict[str, Any]]:
    if tts_service.is_hume_provider():
        return [voice for voice in tts_service.list_hume_voices() if str((voice or {}).get("voice_id") or "").strip()]

    usable: list[dict[str, Any]] = []
    for voice in list_voices(owner_id=owner_id):
        voice_id = str((voice or {}).get("voice_id") or "").strip()
        if not voice_id:
            continue
        if not load_embedding_path(voice_id):
            continue
        usable.append(tts_service.normalize_stored_voice(voice))
    return usable


def suggest_voice_for_npc(npc: dict[str, Any], owner_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    npc_text = _npc_text(npc)
    voices = _list_assignable_voices(owner_id=owner_id)
    if not voices:
        return None

    archetype = _select_archetype(npc_text)
    selected = max(voices, key=lambda voice: _score_voice(voice, npc_text, archetype))
    voice_id = str(selected.get("voice_id") or "").strip()
    if not voice_id:
        return None

    return {
        "provider": str(selected.get("provider") or tts_service.get_tts_provider()),
        "voice_id": voice_id,
        "voice_name": str(selected.get("name") or voice_id),
        "tone": archetype["tone"],
        "style": archetype["style"],
    }
