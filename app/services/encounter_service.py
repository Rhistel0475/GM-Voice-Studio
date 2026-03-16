"""One-click encounter launch: intro narration, combat ambience, and enemy battle voice."""
from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass
from typing import Any, Optional

import soundfile as sf

from app.core.config import DEFAULT_VOICE_ID
from app.domain.live.scene_triggers import resolve_scene_npcs
from app.repositories import campaign_repository
from app.services import ai_service, scene_activation_service, tts_service
from app.services.atmosphere_service import get_atmosphere_audio
from app.services.session_memory_service import get_session_context

_HOSTILE_TOKENS = (
    "enemy",
    "villain",
    "bandit",
    "hostile",
    "monster",
    "foe",
    "raider",
    "assassin",
    "antagonist",
)


@dataclass
class EncounterRecord:
    id: str
    name: str
    enemies: list[dict[str, Any]]
    intro_text: str
    ambience: str
    campaign_id: int
    scene_id: Optional[str] = None


def _audio_to_wav_base64(audio, sample_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _normalize_ref(value: Any) -> str:
    return str(value or "").strip().casefold()


def _match_ref(candidate: Any, *values: Any) -> bool:
    wanted = {_normalize_ref(value) for value in values if _normalize_ref(value)}
    return bool(wanted) and _normalize_ref(candidate) in wanted


def _campaign_npcs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return [npc for npc in campaign.get("npcs", []) if isinstance(npc, dict)]


def _find_scene_in_campaign(campaign: dict[str, Any], scene_ref: Any) -> Optional[dict[str, Any]]:
    for scene in campaign.get("scenes", []) or []:
        if not isinstance(scene, dict):
            continue
        if _match_ref(scene.get("id"), scene_ref) or _match_ref(scene.get("title"), scene_ref) or _match_ref(scene.get("name"), scene_ref):
            return scene
    return None


def _lookup_npc(campaign: dict[str, Any], npc_ref: Any) -> Optional[dict[str, Any]]:
    for npc in _campaign_npcs(campaign):
        if _match_ref(npc.get("id"), npc_ref) or _match_ref(npc.get("name"), npc_ref):
            return npc
    return None


def _encounter_enemy_records(
    campaign: dict[str, Any],
    scene: Optional[dict[str, Any]],
    encounter_payload: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    explicit_refs = []
    if isinstance(encounter_payload, dict):
        explicit_refs = encounter_payload.get("enemies") or encounter_payload.get("enemy_npcs") or encounter_payload.get("enemyNpcRefs") or []
        explicit_refs = explicit_refs if isinstance(explicit_refs, list) else [explicit_refs]

    enemies: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in explicit_refs:
        candidate = ref
        if isinstance(ref, dict):
            candidate = ref.get("id") or ref.get("npc_id") or ref.get("ref") or ref.get("refId") or ref.get("name") or ref.get("refName")
        npc = _lookup_npc(campaign, candidate)
        npc_id = str((npc or {}).get("id") or "").strip()
        if npc is not None and npc_id and npc_id not in seen:
            seen.add(npc_id)
            enemies.append(npc)
    if enemies:
        return enemies

    resolved_scene_npcs = resolve_scene_npcs(scene or {}, npcs=_campaign_npcs(campaign))
    hostile_scene_npcs: list[dict[str, Any]] = []
    for npc in resolved_scene_npcs:
        blob = " ".join(
            str((npc or {}).get(key) or "").strip().lower()
            for key in ("role", "description", "personality", "name")
        )
        if any(token in blob for token in _HOSTILE_TOKENS):
            hostile_scene_npcs.append(npc)

    return hostile_scene_npcs or resolved_scene_npcs


def _derive_encounter_from_scene(scene: dict[str, Any], campaign: dict[str, Any]) -> EncounterRecord:
    intro_text = str(scene.get("read_aloud") or scene.get("notes") or "").strip()
    if not intro_text:
        intro_text = f"{str(scene.get('title') or 'An encounter').strip()} erupts into violence."
    enemies = _encounter_enemy_records(campaign, scene)
    return EncounterRecord(
        id=f"scene:{str(scene.get('id') or scene.get('title') or 'encounter').strip()}",
        name=str(scene.get("title") or scene.get("name") or "Encounter").strip() or "Encounter",
        enemies=enemies,
        intro_text=intro_text,
        ambience="combat",
        campaign_id=int(scene["campaign_id"]),
        scene_id=str(scene.get("id") or "").strip() or None,
    )


def _explicit_encounter_record(campaign: dict[str, Any], encounter_payload: dict[str, Any]) -> Optional[EncounterRecord]:
    encounter_id = str(encounter_payload.get("id") or "").strip()
    name = str(encounter_payload.get("name") or encounter_payload.get("title") or "").strip()
    if not encounter_id and not name:
        return None

    scene_ref = (
        encounter_payload.get("scene_id")
        or encounter_payload.get("sceneId")
        or encounter_payload.get("scene")
    )
    scene = _find_scene_in_campaign(campaign, scene_ref) if scene_ref else None
    intro_text = str(
        encounter_payload.get("intro_text")
        or encounter_payload.get("introText")
        or encounter_payload.get("narrativeSetup")
        or encounter_payload.get("summary")
        or (scene or {}).get("read_aloud")
        or (scene or {}).get("notes")
        or ""
    ).strip()
    enemies = _encounter_enemy_records(campaign, scene, encounter_payload)
    return EncounterRecord(
        id=encounter_id or f"encounter:{name or 'encounter'}",
        name=name or (str((scene or {}).get("title") or "Encounter").strip() or "Encounter"),
        enemies=enemies,
        intro_text=intro_text or f"{name or 'An encounter'} begins.",
        ambience=str(encounter_payload.get("ambience") or encounter_payload.get("ambience_track") or "combat").strip() or "combat",
        campaign_id=int(campaign["id"]),
        scene_id=str((scene or {}).get("id") or "").strip() or None,
    )


def _resolve_encounter(encounter_id: str) -> EncounterRecord:
    scene = campaign_repository.get_scene_record(encounter_id)
    if scene is not None:
        campaign = campaign_repository.get_by_id(int(scene["campaign_id"]))
        if campaign is None:
            raise FileNotFoundError("Campaign not found")
        for encounter_payload in campaign.get("encounters", []) or []:
            if not isinstance(encounter_payload, dict):
                continue
            linked_scene = encounter_payload.get("scene_id") or encounter_payload.get("sceneId") or encounter_payload.get("scene")
            if _match_ref(linked_scene, scene.get("id"), scene.get("title"), scene.get("name")):
                explicit = _explicit_encounter_record(campaign, encounter_payload)
                if explicit is not None:
                    return explicit
        return _derive_encounter_from_scene(scene, campaign)

    for summary in campaign_repository.list_all():
        campaign = campaign_repository.get_by_id(int(summary["id"]))
        if campaign is None:
            continue
        for encounter_payload in campaign.get("encounters", []) or []:
            if not isinstance(encounter_payload, dict):
                continue
            if _match_ref(encounter_payload.get("id"), encounter_id) or _match_ref(encounter_payload.get("name"), encounter_id) or _match_ref(encounter_payload.get("title"), encounter_id):
                explicit = _explicit_encounter_record(campaign, encounter_payload)
                if explicit is not None:
                    return explicit

    raise FileNotFoundError("Encounter not found")


def _synthesize_text(*, text: str, voice_id: str) -> tuple[Any, int]:
    speaker_emb_path = tts_service.resolve_voice_target(voice_id)
    return tts_service.generate(text.strip(), language_tag="en", speaker_emb_path=speaker_emb_path)


def _audio_payload(text: str, voice_id: str) -> dict[str, Any]:
    audio, sample_rate = _synthesize_text(text=text, voice_id=voice_id)
    return {
        "text": text,
        "voice_id": voice_id,
        "audio_base64": _audio_to_wav_base64(audio, sample_rate),
        "mime_type": "audio/wav",
    }


def _narrator_voice_id(encounter: EncounterRecord) -> str:
    if encounter.scene_id:
        scene = campaign_repository.get_scene_record(encounter.scene_id)
        if scene is not None:
            voice_id = str(scene.get("narrator_voice_id") or "").strip()
            if voice_id:
                return voice_id
    return (
        campaign_repository.get_narrator_voice_id(encounter.campaign_id)
        or str(DEFAULT_VOICE_ID or "").strip()
    )


def _enemy_battle_line(encounter: EncounterRecord, enemy: dict[str, Any]) -> str:
    personality = (
        str(enemy.get("description") or "").strip()
        or str(enemy.get("personality") or "").strip()
        or str(enemy.get("role") or "").strip()
        or f"A hostile combatant in the encounter {encounter.name}."
    )
    situation = (
        f"Combat is starting in {encounter.name}. "
        f"Deliver a short battle taunt, threat, or shouted order as the first clash begins. "
        f"Intro: {encounter.intro_text}"
    )
    session_context = get_session_context(
        campaign_id=encounter.campaign_id,
        npc_id=str(enemy.get("id") or "").strip() or None,
    )
    return ai_service.generate_dialogue(
        npc_name=str(enemy.get("name") or "Enemy"),
        personality=personality,
        situation=situation,
        conversation_history=[],
        faction=str(enemy.get("faction") or "").strip(),
        session_context=str(session_context.get("summary") or "").strip(),
        npc_memory_summary=str(session_context.get("npc_memory_summary") or "").strip(),
    )


def launch_encounter(encounter_id: str) -> dict[str, Any]:
    """Launch an encounter with intro narration, combat ambience, and one enemy battle voice."""
    encounter = _resolve_encounter(encounter_id)

    if encounter.scene_id:
        combat_payload = scene_activation_service.start_scene_combat(encounter.scene_id)
        scene = combat_payload.get("scene")
        ambience_audio = combat_payload.get("ambience_audio")
    else:
        scene = None
        ambience_audio = get_atmosphere_audio({"atmosphere_type": encounter.ambience}).get("ambience_track")

    narrator_voice_id = _narrator_voice_id(encounter)
    narration_audio = _audio_payload(encounter.intro_text, narrator_voice_id)

    enemy_dialogue_audio = None
    enemy_dialogue_text = ""
    enemy_npc_name = None

    primary_enemy = next((enemy for enemy in encounter.enemies if isinstance(enemy, dict)), None)
    if primary_enemy is not None:
        enemy_dialogue_text = _enemy_battle_line(encounter, primary_enemy)
        enemy_voice_id = (
            str(primary_enemy.get("voice_id") or "").strip()
            or narrator_voice_id
        )
        enemy_dialogue_audio = _audio_payload(enemy_dialogue_text, enemy_voice_id)
        enemy_dialogue_audio["npc_name"] = str(primary_enemy.get("name") or "Enemy").strip() or "Enemy"
        enemy_npc_name = enemy_dialogue_audio["npc_name"]

    return {
        "encounter": asdict(encounter),
        "scene": scene,
        "ambience_audio": ambience_audio,
        "narration_audio": narration_audio,
        "enemy_dialogue_audio": enemy_dialogue_audio,
        "enemy_dialogue_text": enemy_dialogue_text,
        "enemy_npc_name": enemy_npc_name,
    }
