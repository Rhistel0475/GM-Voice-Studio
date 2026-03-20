"""
Campaign persistence: list, get, delete, create, voice assignment, session events.
Uses campaign DB (codm.db) via SessionLocal; session handling is internal.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.domain.campaign.systems import (
    DEFAULT_CAMPAIGN_SYSTEM_ID,
    get_campaign_system_preset,
    normalize_campaign_system,
)
from app.domain.live.scene_triggers import normalize_scene_triggers, resolve_scene_npcs
from app.services.entity_normalization_service import normalize_campaign_entities
from app.infrastructure.database import SessionLocal
from app.infrastructure.db_models import (
    Campaign,
    CampaignDocument,
    Location,
    NPC,
    Scene,
    SessionEvent,
)


_ATMOSPHERE_SCENE_TYPE_FALLBACKS = {
    "combat": "combat",
    "social": "tavern",
    "exploration": "forest",
    "mystery": "dungeon",
    "travel": "town",
}

_ATMOSPHERE_ALIASES = {
    "forest": ("forest", "woods", "grove", "wild", "jungle"),
    "tavern": ("tavern", "inn", "alehouse", "pub", "meadhall"),
    "town": ("town", "city", "street", "market", "village", "plaza"),
    "dungeon": ("dungeon", "crypt", "cavern", "cave", "catacomb", "ruin", "underground"),
    "combat": ("combat", "battle", "fight", "skirmish", "ambush", "war"),
    "mystery": ("mystery", "eerie", "ominous", "suspense", "investigation"),
}

_DEFAULT_AMBIENCE_TRACKS = {
    "forest": "forest.wav",
    "tavern": "tavern.wav",
    "town": "town.wav",
    "dungeon": "dungeon.wav",
    "combat": "combat.wav",
    "mystery": "dungeon.wav",
}


def _normalize_atmosphere_type(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return None
    for atmosphere_type, aliases in _ATMOSPHERE_ALIASES.items():
        if raw == atmosphere_type or any(alias in raw for alias in aliases):
            return atmosphere_type
    return None


def _normalize_scene_title(scene_payload: Optional[dict[str, Any]], *, relation_scene: Optional[Scene] = None) -> str:
    if isinstance(scene_payload, dict):
        title = str(scene_payload.get("title") or scene_payload.get("name") or "").strip()
        if title:
            return title
    if relation_scene is not None:
        return str(relation_scene.title or "").strip()
    return ""


def _normalize_scene_description(scene_payload: Optional[dict[str, Any]], *, relation_scene: Optional[Scene] = None) -> str:
    if isinstance(scene_payload, dict):
        for candidate in (
            scene_payload.get("description"),
            scene_payload.get("summary"),
            scene_payload.get("read_aloud"),
            scene_payload.get("notes"),
        ):
            text = str(candidate or "").strip()
            if text:
                return text
    if relation_scene is not None:
        for candidate in (relation_scene.read_aloud, relation_scene.notes):
            text = str(candidate or "").strip()
            if text:
                return text
    return ""


def _normalize_connected_scene_refs(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        candidate = item
        if isinstance(item, dict):
            candidate = (
                item.get("id")
                or item.get("scene_id")
                or item.get("sceneId")
                or item.get("title")
                or item.get("name")
            )
        ref = str(candidate or "").strip()
        if not ref:
            continue
        key = ref.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(ref)
    return normalized


def _normalize_scene_graph_fields(
    scene_payload: dict[str, Any],
    *,
    relation_scene: Optional[Scene] = None,
) -> dict[str, Any]:
    title = _normalize_scene_title(scene_payload, relation_scene=relation_scene)
    description = _normalize_scene_description(scene_payload, relation_scene=relation_scene)
    scene_payload["title"] = title
    scene_payload["name"] = str(scene_payload.get("name") or title).strip() or title
    scene_payload["description"] = description
    scene_payload["connected_scenes"] = _normalize_connected_scene_refs(
        scene_payload.get("connected_scenes") or scene_payload.get("connectedScenes")
    )
    return scene_payload


def _normalize_ambience_track(value: Any, atmosphere_type: str) -> Optional[str]:
    raw = str(value or "").strip()
    if raw:
        return raw
    if atmosphere_type:
        return _DEFAULT_AMBIENCE_TRACKS.get(atmosphere_type, f"{atmosphere_type}.wav")
    return None


def _resolve_scene_atmosphere_type(
    scene_payload: Optional[dict[str, Any]],
    *,
    relation_scene: Optional[Scene] = None,
) -> str:
    if isinstance(scene_payload, dict):
        direct = _normalize_atmosphere_type(scene_payload.get("atmosphere_type") or scene_payload.get("atmosphereType"))
        if direct:
            return direct

        raw_atmosphere = scene_payload.get("atmosphere")
        if isinstance(raw_atmosphere, (list, tuple)):
            for item in raw_atmosphere:
                candidate = _normalize_atmosphere_type(item)
                if candidate:
                    return candidate
        else:
            candidate = _normalize_atmosphere_type(raw_atmosphere)
            if candidate:
                return candidate

        for candidate_source in (
            scene_payload.get("type"),
            scene_payload.get("location"),
            scene_payload.get("title"),
        ):
            candidate = _normalize_atmosphere_type(candidate_source)
            if candidate:
                return candidate

        scene_type = str(scene_payload.get("type") or "").strip().lower()
        if scene_type in _ATMOSPHERE_SCENE_TYPE_FALLBACKS:
            return _ATMOSPHERE_SCENE_TYPE_FALLBACKS[scene_type]

    if relation_scene is not None:
        scene_type = str(relation_scene.type or "").strip().lower()
        if scene_type in _ATMOSPHERE_SCENE_TYPE_FALLBACKS:
            return _ATMOSPHERE_SCENE_TYPE_FALLBACKS[scene_type]

    return "town"


def _normalize_session_payload(session: Any, campaign_id: int) -> Optional[dict[str, Any]]:
    if not isinstance(session, dict):
        return None

    normalized_session = {
        "id": str(session.get("id") or session.get("session_id") or "").strip() or str(uuid.uuid4()),
        "campaign_id": campaign_id,
        "title": str(session.get("title") or "Session").strip() or "Session",
        "active_scene_id": str(
            session.get("active_scene_id")
            or session.get("activeSceneId")
            or session.get("scene_id")
            or ""
        ).strip()
        or None,
        "started_at": str(session.get("started_at") or session.get("startedAt") or "").strip() or None,
        "status": str(session.get("status") or "prep").strip() or "prep",
        "narrator_voice": str(
            session.get("narrator_voice")
            or session.get("narratorVoice")
            or session.get("narrator_voice_id")
            or ""
        ).strip()
        or None,
    }
    override = _normalize_atmosphere_type(
        session.get("atmosphere_override_type") or session.get("atmosphereOverrideType")
    )
    if override:
        normalized_session["atmosphere_override_type"] = override
    return normalized_session


def _apply_campaign_system_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    system_id = normalize_campaign_system(payload.get("system_id") or payload.get("systemId"))
    payload["system_id"] = system_id
    payload["systemId"] = system_id
    payload["system"] = get_campaign_system_preset(system_id)
    return payload


def _campaign_payload_from_json_record(campaign: Campaign) -> Optional[dict[str, Any]]:
    """
    Parse and normalize Campaign.data_json payload.
    Returns None when payload is missing/invalid so callers can use relational fallback.
    """
    raw = (getattr(campaign, "data_json", "") or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("Campaign %s has invalid data_json; using relational fallback", campaign.id)
        return None
    if not isinstance(payload, dict):
        return None
    payload["id"] = campaign.id
    payload["title"] = campaign.title or payload.get("title", "")
    payload["summary"] = campaign.summary or payload.get("summary", "")
    for key in (
        "npcs",
        "party",
        "scenes",
        "locations",
        "reveals",
        "items",
        "images",
        "quests",
        "factions",
        "lore",
        "clues",
        "secrets",
        "rumors",
        "read_alouds",
        "consequences",
        "rewards",
        "hooks",
        "parse_candidates",
    ):
        if not isinstance(payload.get(key), list):
            payload[key] = []
    if not isinstance(payload.get("encounters"), list):
        payload["encounters"] = []
    if not isinstance(payload.get("sessions"), list):
        payload["sessions"] = []
    payload["active_session_id"] = str(
        payload.get("active_session_id")
        or payload.get("activeSessionId")
        or ""
    ).strip() or None
    for scene in payload.get("scenes", []):
        if isinstance(scene, dict):
            if not isinstance(scene.get("triggers"), list):
                scene["triggers"] = []
            _normalize_scene_graph_fields(scene)
            scene["atmosphere_type"] = _resolve_scene_atmosphere_type(scene)
            scene["ambience_track"] = _normalize_ambience_track(
                scene.get("ambience_track") or scene.get("ambienceTrack"),
                scene["atmosphere_type"],
            )
    normalized_sessions: list[dict[str, Any]] = []
    for session in payload.get("sessions", []):
        normalized_session = _normalize_session_payload(session, campaign.id)
        if normalized_session is not None:
            normalized_sessions.append(normalized_session)
    payload["sessions"] = normalized_sessions
    for key in ("codex_entries", "relationships"):
        if not isinstance(payload.get(key), list):
            payload[key] = []
    if not isinstance(payload.get("review_summary"), dict):
        payload["review_summary"] = {}
    if not isinstance(payload.get("coverage_report"), dict):
        payload["coverage_report"] = {"summary": {"total_gaps": 0}}
    if not isinstance(payload.get("documents"), list):
        payload["documents"] = []
    return _apply_campaign_system_metadata(payload)

def _campaign_payload_from_relations(campaign: Campaign) -> dict[str, Any]:
    """Legacy fallback payload built from normalized relational tables."""
    return _apply_campaign_system_metadata({
        "id": campaign.id,
        "title": campaign.title,
        "summary": campaign.summary,
        "system_id": DEFAULT_CAMPAIGN_SYSTEM_ID,
        "active_session_id": None,
        "sessions": [],
        "npcs": [
            {
                "id": n.id,
                "name": n.name,
                "role": n.role,
                "personality": n.personality,
                "faction": n.faction,
                "description": n.description,
                "motivation": n.motivation,
                "secrets": n.secrets,
                "hp": n.hp,
                "ac": n.ac,
                "cr": n.cr,
                "image_url": n.image_url,
                "voice_id": n.voice_id,
            }
            for n in campaign.npcs
        ],
        "party": [],
        "scenes": [
            {
                "id": s.id,
                "title": s.title,
                "name": s.title,
                "description": str(s.read_aloud or s.notes or "").strip(),
                "act": s.act,
                "type": s.type,
                "atmosphere_type": _resolve_scene_atmosphere_type(None, relation_scene=s),
                "ambience_track": _normalize_ambience_track(None, _resolve_scene_atmosphere_type(None, relation_scene=s)),
                "read_aloud": s.read_aloud,
                "difficulty": s.difficulty,
                "rewards": s.rewards,
                "notes": s.notes,
                "image_url": s.image_url,
                "location": "",
                "connected_scenes": [],
                "npcs": [],
                "reveals": [],
                "items": [],
                "triggers": [],
            }
            for s in campaign.scenes
        ],
        "locations": [
            {"id": loc.id, "name": loc.name, "description": loc.description, "image_url": loc.image_url}
            for loc in campaign.locations
        ],
        "encounters": [],
        "reveals": [],
        "items": [],
        "quests": [],
        "factions": [],
        "lore": [],
        "images": [],
        "codex_entries": [],
        "relationships": [],
        "documents": [],
    })


def _campaign_documents_payload(db, campaign_id: int) -> list[dict[str, Any]]:
    docs = (
        db.query(CampaignDocument)
        .filter(CampaignDocument.campaign_id == campaign_id)
        .order_by(CampaignDocument.created_at.desc(), CampaignDocument.id.desc())
        .all()
    )
    payload: list[dict[str, Any]] = []
    for doc in docs:
        try:
            metadata = json.loads(doc.metadata_json or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except json.JSONDecodeError:
            metadata = {}
        payload.append(
            {
                "id": doc.id,
                "title": doc.filename,
                "filename": doc.filename,
                "summary": doc.summary or metadata.get("summary", ""),
                "file_type": doc.file_type,
                "mime_type": doc.mime_type,
                "chunk_count": doc.chunk_count,
                "created_at": doc.created_at,
                "metadata": metadata,
            }
        )
    return payload


def _find_scene_payload(
    payload: Optional[dict[str, Any]],
    scene_ref: str,
    relation_scene: Optional[Scene] = None,
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    wanted: list[str] = []
    raw_ref = str(scene_ref or "").strip()
    if raw_ref:
        wanted.append(raw_ref)
    if relation_scene is not None:
        wanted.append(str(relation_scene.id))
        if relation_scene.title:
            wanted.append(str(relation_scene.title).strip())

    wanted = [candidate for candidate in wanted if candidate]
    if not wanted:
        return None

    for scene_payload in payload.get("scenes", []):
        if not isinstance(scene_payload, dict):
            continue
        payload_id = str(scene_payload.get("id") or "").strip()
        payload_title = _normalize_scene_title(scene_payload)
        if any(candidate in {payload_id, payload_title} for candidate in wanted):
            return scene_payload
    return None


def _build_scene_record(
    *,
    campaign: Campaign,
    scene_ref: str,
    relation_scene: Optional[Scene] = None,
    scene_payload: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = payload or _enrich_campaign_payload(campaign, _campaign_payload_from_json_record(campaign) or _campaign_payload_from_relations(campaign))
    payload_scene = scene_payload or _find_scene_payload(payload, scene_ref, relation_scene=relation_scene) or {}

    narrator_voice_id = (
        str(payload_scene.get("narrator_voice_id") or "").strip()
        or str(payload_scene.get("voice_id") or "").strip()
        or str(payload.get("narrator_voice_id") or payload.get("narrator_voice") or "").strip()
        or None
    )

    triggers = payload_scene.get("triggers") if isinstance(payload_scene.get("triggers"), list) else []
    npcs = payload_scene.get("npcs") if isinstance(payload_scene.get("npcs"), list) else []

    return {
        "id": str(
            payload_scene.get("id")
            or (relation_scene.id if relation_scene is not None else "")
            or scene_ref
        ),
        "campaign_id": campaign.id,
        "title": _normalize_scene_title(payload_scene, relation_scene=relation_scene),
        "name": str(payload_scene.get("name") or _normalize_scene_title(payload_scene, relation_scene=relation_scene)).strip(),
        "description": _normalize_scene_description(payload_scene, relation_scene=relation_scene),
        "read_aloud": str(payload_scene.get("read_aloud") or (relation_scene.read_aloud if relation_scene is not None else "") or "").strip(),
        "notes": str(payload_scene.get("notes") or (relation_scene.notes if relation_scene is not None else "") or "").strip(),
        "type": str(payload_scene.get("type") or (relation_scene.type if relation_scene is not None else "") or "").strip(),
        "atmosphere_type": _resolve_scene_atmosphere_type(payload_scene, relation_scene=relation_scene),
        "ambience_track": _normalize_ambience_track(
            payload_scene.get("ambience_track") or payload_scene.get("ambienceTrack"),
            _resolve_scene_atmosphere_type(payload_scene, relation_scene=relation_scene),
        ),
        "location": str(payload_scene.get("location") or "").strip(),
        "connected_scenes": _normalize_connected_scene_refs(
            payload_scene.get("connected_scenes") or payload_scene.get("connectedScenes")
        ),
        "npcs": [item for item in npcs if item],
        "triggers": [item for item in triggers if isinstance(item, dict)],
        "narrator_voice_id": narrator_voice_id,
    }


def _enrich_campaign_payload(campaign: Campaign, payload: dict[str, Any]) -> dict[str, Any]:
    _apply_campaign_system_metadata(payload)
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        payload["scenes"] = []
        scenes = payload["scenes"]

    relation_scenes = list(campaign.scenes or [])
    by_title: dict[str, list[Scene]] = {}
    by_id: dict[str, Scene] = {}
    for relation_scene in relation_scenes:
        by_id[str(relation_scene.id)] = relation_scene
        title_key = str(relation_scene.title or "").strip().casefold()
        if title_key:
            by_title.setdefault(title_key, []).append(relation_scene)

    used_relation_ids: set[int] = set()
    payload_npcs = payload.get("npcs") if isinstance(payload.get("npcs"), list) else []

    for index, scene_entry in enumerate(scenes):
        if not isinstance(scene_entry, dict):
            continue

        relation_scene: Optional[Scene] = None
        raw_id = str(scene_entry.get("id") or "").strip()
        if raw_id and raw_id in by_id:
            relation_scene = by_id[raw_id]

        if relation_scene is None:
            title_key = str(scene_entry.get("title") or "").strip().casefold()
            for candidate in by_title.get(title_key, []):
                if candidate.id not in used_relation_ids:
                    relation_scene = candidate
                    break

        if relation_scene is None and index < len(relation_scenes):
            candidate = relation_scenes[index]
            if candidate.id not in used_relation_ids:
                relation_scene = candidate

        if relation_scene is not None:
            used_relation_ids.add(relation_scene.id)
            scene_entry["id"] = str(scene_entry.get("id") or relation_scene.id)
            scene_entry["title"] = str(scene_entry.get("title") or relation_scene.title or "").strip()
            scene_entry["act"] = str(scene_entry.get("act") or relation_scene.act or "").strip()
            scene_entry["type"] = str(scene_entry.get("type") or relation_scene.type or "").strip()
            scene_entry["read_aloud"] = str(scene_entry.get("read_aloud") or relation_scene.read_aloud or "").strip()
            scene_entry["difficulty"] = str(scene_entry.get("difficulty") or relation_scene.difficulty or "").strip()
            scene_entry["rewards"] = str(scene_entry.get("rewards") or relation_scene.rewards or "").strip()
            scene_entry["notes"] = str(scene_entry.get("notes") or relation_scene.notes or "").strip()
            if relation_scene.image_url and not scene_entry.get("image_url"):
                scene_entry["image_url"] = relation_scene.image_url
        elif raw_id:
            scene_entry["id"] = raw_id

        for key in ("npcs", "reveals", "items"):
            if not isinstance(scene_entry.get(key), list):
                scene_entry[key] = []
        _normalize_scene_graph_fields(scene_entry, relation_scene=relation_scene)
        scene_entry["atmosphere_type"] = _resolve_scene_atmosphere_type(scene_entry, relation_scene=relation_scene)
        scene_entry["ambience_track"] = _normalize_ambience_track(
            scene_entry.get("ambience_track") or scene_entry.get("ambienceTrack"),
            scene_entry["atmosphere_type"],
        )
        scene_entry["triggers"] = normalize_scene_triggers(scene_entry, npcs=payload_npcs)

    return _apply_campaign_system_metadata(payload)


def list_all() -> list[dict[str, Any]]:
    """Return all campaigns (id, title, summary) newest first."""
    db = SessionLocal()
    try:
        campaigns = db.query(Campaign).order_by(Campaign.id.desc()).all()
        payload: list[dict[str, Any]] = []
        for campaign in campaigns:
            system_payload = _campaign_payload_from_json_record(campaign) or {}
            system_id = normalize_campaign_system(system_payload.get("system_id"))
            system = get_campaign_system_preset(system_id)
            payload.append(
                {
                    "id": campaign.id,
                    "title": campaign.title,
                    "summary": campaign.summary,
                    "system_id": system_id,
                    "system_label": system["label"],
                }
            )
        return payload
    finally:
        db.close()


def get_by_id(campaign_id: int) -> Optional[dict[str, Any]]:
    """Return full campaign payload (from data_json or relational fallback), or None."""
    db = SessionLocal()
    try:
        c = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if c is None:
            return None
        payload = _campaign_payload_from_json_record(c)
        if payload is None:
            payload = _campaign_payload_from_relations(c)
        payload = _enrich_campaign_payload(c, payload)
        payload["documents"] = _campaign_documents_payload(db, campaign_id)
        return payload
    finally:
        db.close()


def delete(campaign_id: int) -> bool:
    """Delete campaign and related NPCs/scenes/locations. Returns True if deleted."""
    db = SessionLocal()
    try:
        c = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if c is None:
            return False
        db.delete(c)
        db.commit()
        return True
    finally:
        db.close()


def create_from_parse_result(result: dict[str, Any]) -> int:
    """
    Persist a parsed adventure result as Campaign + NPCs, Scenes, Locations.
    Returns the new campaign id.
    """
    db = SessionLocal()
    try:
        result = normalize_campaign_entities(result)
        system_id = normalize_campaign_system(result.get("system_id") or result.get("systemId"))
        quests_payload = result.get("quests", []) if isinstance(result.get("quests"), list) else []
        factions_payload = result.get("factions", []) if isinstance(result.get("factions"), list) else []
        lore_payload = result.get("lore", []) if isinstance(result.get("lore"), list) else []
        campaign = Campaign(title=result.get("title", ""), summary=result.get("summary", ""), data_json="{}")
        db.add(campaign)
        db.flush()

        npcs_payload = result.get("npcs", []) if isinstance(result.get("npcs"), list) else []
        scenes_payload: list[dict[str, Any]] = []

        for n in npcs_payload:
            db.add(
                NPC(
                    campaign_id=campaign.id,
                    name=n.get("name", ""),
                    role=n.get("role", ""),
                    personality=n.get("personality", ""),
                    faction=n.get("faction", ""),
                    description=n.get("description", ""),
                    motivation=n.get("motivation", ""),
                    secrets=n.get("secrets", ""),
                    hp=str(n.get("hp", "")),
                    ac=n.get("ac") or None,
                    cr=n.get("cr", ""),
                    image_url=n.get("image_url"),
                    voice_id=n.get("voice_id"),
                )
            )
        for s in result.get("scenes", []):
            scene_source = s if isinstance(s, dict) else {"title": str(s or "").strip()}
            scene_row = Scene(
                campaign_id=campaign.id,
                title=scene_source.get("title", ""),
                act=scene_source.get("act", ""),
                type=scene_source.get("type", ""),
                read_aloud=scene_source.get("read_aloud", ""),
                difficulty=scene_source.get("difficulty", ""),
                rewards=scene_source.get("rewards", ""),
                notes=scene_source.get("notes", ""),
                image_url=scene_source.get("image_url"),
            )
            db.add(scene_row)
            db.flush()

            scene_payload = dict(scene_source)
            scene_payload["id"] = str(scene_row.id)
            _normalize_scene_graph_fields(scene_payload, relation_scene=scene_row)
            scene_payload["atmosphere_type"] = _resolve_scene_atmosphere_type(scene_payload, relation_scene=scene_row)
            scene_payload["ambience_track"] = _normalize_ambience_track(
                scene_payload.get("ambience_track") or scene_payload.get("ambienceTrack"),
                scene_payload["atmosphere_type"],
            )
            scene_payload["triggers"] = normalize_scene_triggers(scene_payload, npcs=npcs_payload)
            scenes_payload.append(scene_payload)

        for loc in result.get("locations", []):
            db.add(
                Location(
                    campaign_id=campaign.id,
                    name=loc.get("name", ""),
                    description=loc.get("description", ""),
                    image_url=loc.get("image_url"),
                )
            )

        canonical_payload = {
            "title": result.get("title", ""),
            "summary": result.get("summary", ""),
            "system_id": system_id,
            "active_session_id": None,
            "sessions": [],
            "npcs": npcs_payload,
            "party": result.get("party", []),
            "scenes": scenes_payload,
            "locations": result.get("locations", []),
            "encounters": result.get("encounters", []) if isinstance(result.get("encounters"), list) else [],
            "reveals": result.get("reveals", []),
            "items": result.get("items", []),
            "quests": quests_payload,
            "factions": factions_payload,
            "lore": lore_payload,
            "clues": result.get("clues", []),
            "secrets": result.get("secrets", []),
            "rumors": result.get("rumors", []),
            "read_alouds": result.get("read_alouds", []),
            "consequences": result.get("consequences", []),
            "rewards": result.get("rewards", []),
            "hooks": result.get("hooks", []),
            "parse_candidates": result.get("parse_candidates", []),
            "images": result.get("images", []),
            "codex_entries": result.get("codex_entries", []),
            "relationships": result.get("relationships", []),
            "review_summary": result.get("review_summary", {}),
            "coverage_report": result.get("coverage_report", {"summary": {"total_gaps": 0}}),
        }
        campaign.data_json = json.dumps(canonical_payload, ensure_ascii=False)
        result["id"] = campaign.id
        result["scenes"] = scenes_payload
        result["system_id"] = system_id
        result["systemId"] = system_id
        result["system"] = get_campaign_system_preset(system_id)
        result["quests"] = quests_payload
        result["factions"] = factions_payload
        result["lore"] = lore_payload

        db.commit()
        return campaign.id
    finally:
        db.close()


def assign_npc_voice(campaign_id: int, npc_name: str, voice_id: str) -> bool:
    """
    Set voice_id on the NPC with the given name within campaign_id.
    Updates both the relational NPC row and the NPC entry in data_json.
    Returns True if at least one NPC was updated.
    """
    db = SessionLocal()
    try:
        c = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if c is None:
            return False

        updated = False

        # Update relational NPC row(s) matching the name
        for npc in db.query(NPC).filter(NPC.campaign_id == campaign_id, NPC.name == npc_name).all():
            npc.voice_id = voice_id
            updated = True

        # Update data_json NPC entry
        raw = (c.data_json or "").strip()
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and isinstance(payload.get("npcs"), list):
                    for npc_entry in payload["npcs"]:
                        if isinstance(npc_entry, dict) and npc_entry.get("name") == npc_name:
                            npc_entry["voice_id"] = voice_id
                            updated = True
                    c.data_json = json.dumps(payload, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                logging.warning("Campaign %s data_json invalid during voice assignment", campaign_id)

        if updated:
            db.commit()
        return updated
    finally:
        db.close()


def get_npc_record(npc_id: str, campaign_id: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Return a normalized NPC record by id, with name fallback for legacy callers."""
    db = SessionLocal()
    try:
        npc = None
        query = db.query(NPC)
        if campaign_id is not None:
            query = query.filter(NPC.campaign_id == campaign_id)
        if str(npc_id).isdigit():
            npc = query.filter(NPC.id == int(npc_id)).first()
        if npc is None:
            npc = query.filter(NPC.name == str(npc_id)).first()
        if npc is None:
            return None
        return {
            "id": str(npc.id),
            "campaign_id": npc.campaign_id,
            "name": npc.name,
            "role": npc.role,
            "personality": npc.personality,
            "faction": npc.faction,
            "description": npc.description,
            "motivation": npc.motivation,
            "secrets": npc.secrets,
            "voice_id": npc.voice_id,
        }
    finally:
        db.close()


def get_scene_record(scene_id: str) -> Optional[dict[str, Any]]:
    """Return a normalized scene record by id, with title fallback for legacy callers."""
    db = SessionLocal()
    try:
        scene_ref = str(scene_id).strip()
        scene = None
        if scene_ref.isdigit():
            scene = db.query(Scene).filter(Scene.id == int(scene_ref)).first()
        if scene is None:
            scene = db.query(Scene).filter(Scene.title == scene_ref).first()

        if scene is not None:
            campaign = db.query(Campaign).filter(Campaign.id == scene.campaign_id).first()
            if campaign is None:
                return None
            payload = _enrich_campaign_payload(campaign, _campaign_payload_from_json_record(campaign) or _campaign_payload_from_relations(campaign))
            return _build_scene_record(campaign=campaign, scene_ref=scene_ref, relation_scene=scene, payload=payload)

        for campaign in db.query(Campaign).all():
            payload = _campaign_payload_from_json_record(campaign)
            scene_payload = _find_scene_payload(payload, scene_ref)
            if scene_payload is not None:
                payload = _enrich_campaign_payload(campaign, payload)
                return _build_scene_record(
                    campaign=campaign,
                    scene_ref=scene_ref,
                    relation_scene=None,
                    scene_payload=_find_scene_payload(payload, scene_ref),
                    payload=payload,
                )
        return None
    finally:
        db.close()


def get_scene_bundle(scene_id: str) -> Optional[dict[str, Any]]:
    """Return a scene plus campaign/NPC context for scene control execution."""
    db = SessionLocal()
    try:
        scene_ref = str(scene_id).strip()
        relation_scene = None
        if scene_ref.isdigit():
            relation_scene = db.query(Scene).filter(Scene.id == int(scene_ref)).first()
        if relation_scene is None:
            relation_scene = db.query(Scene).filter(Scene.title == scene_ref).first()

        campaign: Optional[Campaign] = None
        payload: Optional[dict[str, Any]] = None
        scene_payload: Optional[dict[str, Any]] = None

        if relation_scene is not None:
            campaign = db.query(Campaign).filter(Campaign.id == relation_scene.campaign_id).first()
            if campaign is None:
                return None
            payload = _enrich_campaign_payload(campaign, _campaign_payload_from_json_record(campaign) or _campaign_payload_from_relations(campaign))
            scene_payload = _find_scene_payload(payload, scene_ref, relation_scene=relation_scene)
        else:
            for candidate_campaign in db.query(Campaign).all():
                candidate_payload = _campaign_payload_from_json_record(candidate_campaign)
                candidate_scene = _find_scene_payload(candidate_payload, scene_ref)
                if candidate_scene is None:
                    continue
                campaign = candidate_campaign
                payload = _enrich_campaign_payload(candidate_campaign, candidate_payload or _campaign_payload_from_relations(candidate_campaign))
                scene_payload = _find_scene_payload(payload, scene_ref)
                break

        if campaign is None or payload is None:
            return None

        scene_record = _build_scene_record(
            campaign=campaign,
            scene_ref=scene_ref,
            relation_scene=relation_scene,
            scene_payload=scene_payload,
            payload=payload,
        )
        payload_npcs = payload.get("npcs") if isinstance(payload.get("npcs"), list) else []
        return {
            "campaign_id": campaign.id,
            "scene": scene_record,
            "npcs": payload_npcs,
            "scene_npcs": resolve_scene_npcs(scene_record, npcs=payload_npcs),
        }
    finally:
        db.close()


def get_scene_trigger_record(scene_id: str, trigger_name: str) -> Optional[dict[str, Any]]:
    """Return a scene + trigger bundle for the requested trigger name."""
    scene = get_scene_record(scene_id)
    if scene is None:
        return None

    wanted = str(trigger_name or "").strip().casefold()
    if not wanted:
        return {"scene": scene, "trigger": None}

    for trigger in scene.get("triggers") or []:
        if not isinstance(trigger, dict):
            continue
        if str(trigger.get("name") or "").strip().casefold() == wanted:
            return {"scene": scene, "trigger": trigger}
    return {"scene": scene, "trigger": None}


def get_narrator_voice_id(campaign_id: int) -> Optional[str]:
    """Return narrator voice id from campaign JSON payload when available."""
    db = SessionLocal()
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if campaign is None:
            return None
        raw = (getattr(campaign, "data_json", "") or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logging.warning("Campaign %s data_json invalid during narrator voice lookup", campaign_id)
            return None
        if not isinstance(payload, dict):
            return None
        return str(payload.get("narrator_voice_id") or payload.get("narrator_voice") or "").strip() or None
    finally:
        db.close()


def append_session_event(
    campaign_id: int,
    event_type: str,
    text: str,
    scene_id: Optional[str] = None,
    session_id: Optional[str] = None,
    event_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> str:
    """
    Append a session event for the given campaign. Returns the event id.
    """
    db = SessionLocal()
    try:
        eid = event_id or str(uuid.uuid4())
        ts = created_at or datetime.now(timezone.utc).isoformat()
        event = SessionEvent(
            id=eid,
            campaign_id=campaign_id,
            scene_id=scene_id,
            session_id=session_id,
            type=event_type,
            text=text,
            created_at=ts,
        )
        db.add(event)
        db.commit()
        return eid
    finally:
        db.close()


def start_session(campaign_id: int, scene_id: str, narrator_voice: str) -> dict[str, Any]:
    """
    Create a new active session for a campaign, initialize its log, and activate the chosen scene.
    Returns the created session plus the refreshed campaign payload.
    """
    db = SessionLocal()
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if campaign is None:
            raise FileNotFoundError("Campaign not found")

        payload = _campaign_payload_from_json_record(campaign) or _campaign_payload_from_relations(campaign)
        payload = _enrich_campaign_payload(campaign, payload)

        scene_ref = str(scene_id or "").strip()
        if not scene_ref:
            raise ValueError("Scene id is required")

        relation_scene = None
        if scene_ref.isdigit():
            relation_scene = (
                db.query(Scene)
                .filter(Scene.campaign_id == campaign_id, Scene.id == int(scene_ref))
                .first()
            )
        if relation_scene is None:
            relation_scene = (
                db.query(Scene)
                .filter(Scene.campaign_id == campaign_id, Scene.title == scene_ref)
                .first()
            )

        scene_payload = _find_scene_payload(payload, scene_ref, relation_scene=relation_scene)
        if scene_payload is None:
            raise FileNotFoundError("Scene not found")

        scene_title = str(
            scene_payload.get("title")
            or (relation_scene.title if relation_scene is not None else "")
            or scene_ref
        ).strip()
        resolved_scene_id = str(
            scene_payload.get("id")
            or (relation_scene.id if relation_scene is not None else "")
            or scene_ref
        ).strip()
        if not resolved_scene_id:
            raise ValueError("Scene id is required")

        narrator_voice_id = str(narrator_voice or "").strip()
        if not narrator_voice_id:
            raise ValueError("Narrator voice is required")

        existing_sessions = payload.get("sessions") if isinstance(payload.get("sessions"), list) else []
        normalized_sessions: list[dict[str, Any]] = []
        for session in existing_sessions:
            normalized_session = _normalize_session_payload(session, campaign_id)
            if normalized_session is None:
                continue
            if normalized_session["status"] == "active":
                normalized_session["status"] = "closed"
            normalized_session.pop("atmosphere_override_type", None)
            normalized_sessions.append(normalized_session)

        session_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        session_title = f"Session - {scene_title}" if scene_title else "Session"
        session_record = {
            "id": session_id,
            "campaign_id": campaign_id,
            "title": session_title,
            "active_scene_id": resolved_scene_id,
            "started_at": started_at,
            "status": "active",
            "narrator_voice": narrator_voice_id,
        }
        normalized_sessions.append(session_record)

        payload["sessions"] = normalized_sessions
        payload["active_session_id"] = session_id
        payload["narrator_voice_id"] = narrator_voice_id
        payload["narrator_voice"] = narrator_voice_id
        scene_payload["narrator_voice_id"] = narrator_voice_id
        scene_payload["voice_id"] = scene_payload.get("voice_id") or narrator_voice_id
        campaign.data_json = json.dumps(payload, ensure_ascii=False)

        log_text = f"Session started in {scene_title}."
        db.add(
            SessionEvent(
                id=str(uuid.uuid4()),
                campaign_id=campaign_id,
                scene_id=resolved_scene_id,
                session_id=session_id,
                type="system",
                text=log_text,
                created_at=started_at,
            )
        )
        db.commit()
    finally:
        db.close()

    refreshed_campaign = get_by_id(campaign_id)
    if refreshed_campaign is None:
        raise FileNotFoundError("Campaign not found")

    return {
        "campaign_id": campaign_id,
        "scene_id": resolved_scene_id,
        "session": session_record,
        "campaign": refreshed_campaign,
    }


def activate_scene(
    scene_id: str,
    *,
    atmosphere_override_type: Optional[str] = None,
    reset_atmosphere_override: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Activate a scene for the current active session when available.
    Returns the normalized scene record.
    """
    db = SessionLocal()
    try:
        scene_ref = str(scene_id or "").strip()
        if not scene_ref:
            return None

        relation_scene = None
        if scene_ref.isdigit():
            relation_scene = db.query(Scene).filter(Scene.id == int(scene_ref)).first()
        if relation_scene is None:
            relation_scene = db.query(Scene).filter(Scene.title == scene_ref).first()

        campaign: Optional[Campaign] = None
        payload: Optional[dict[str, Any]] = None
        scene_payload: Optional[dict[str, Any]] = None

        if relation_scene is not None:
            campaign = db.query(Campaign).filter(Campaign.id == relation_scene.campaign_id).first()
            if campaign is None:
                return None
            payload = _enrich_campaign_payload(
                campaign,
                _campaign_payload_from_json_record(campaign) or _campaign_payload_from_relations(campaign),
            )
            scene_payload = _find_scene_payload(payload, scene_ref, relation_scene=relation_scene)
        else:
            for candidate_campaign in db.query(Campaign).all():
                candidate_payload = _campaign_payload_from_json_record(candidate_campaign)
                candidate_scene = _find_scene_payload(candidate_payload, scene_ref)
                if candidate_scene is None:
                    continue
                campaign = candidate_campaign
                payload = _enrich_campaign_payload(
                    candidate_campaign,
                    candidate_payload or _campaign_payload_from_relations(candidate_campaign),
                )
                scene_payload = _find_scene_payload(payload, scene_ref)
                break

        if campaign is None or payload is None:
            return None

        scene_record = _build_scene_record(
            campaign=campaign,
            scene_ref=scene_ref,
            relation_scene=relation_scene,
            scene_payload=scene_payload,
            payload=payload,
        )
        resolved_scene_id = str(scene_record.get("id") or scene_ref).strip() or scene_ref

        active_session_id = str(payload.get("active_session_id") or "").strip()
        existing_sessions = payload.get("sessions") if isinstance(payload.get("sessions"), list) else []

        target_session_id = active_session_id
        if not target_session_id:
            for session in existing_sessions:
                normalized_session = _normalize_session_payload(session, campaign.id)
                if normalized_session and str(normalized_session.get("status") or "").lower() == "active":
                    target_session_id = normalized_session["id"]
                    break

        normalized_sessions: list[dict[str, Any]] = []
        updated_session: Optional[dict[str, Any]] = None
        override = _normalize_atmosphere_type(atmosphere_override_type)

        for session in existing_sessions:
            normalized_session = _normalize_session_payload(session, campaign.id)
            if normalized_session is None:
                continue
            if target_session_id and normalized_session["id"] == target_session_id:
                previous_scene_id = str(normalized_session.get("active_scene_id") or "").strip()
                normalized_session["active_scene_id"] = resolved_scene_id
                if override:
                    normalized_session["atmosphere_override_type"] = override
                elif reset_atmosphere_override or previous_scene_id != resolved_scene_id:
                    normalized_session.pop("atmosphere_override_type", None)
                updated_session = normalized_session
            normalized_sessions.append(normalized_session)

        payload["sessions"] = normalized_sessions
        payload["active_session_id"] = updated_session["id"] if updated_session is not None else (target_session_id or None)
        campaign.data_json = json.dumps(payload, ensure_ascii=False)
        db.commit()

        if updated_session and updated_session.get("atmosphere_override_type"):
            scene_record["atmosphere_override_type"] = updated_session["atmosphere_override_type"]
        return scene_record
    finally:
        db.close()


def get_session_events(
    campaign_id: int,
    scene_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Return session events for a campaign, optionally filtered by scene_id.
    Ordered oldest-first; capped at limit.
    """
    db = SessionLocal()
    try:
        q = db.query(SessionEvent).filter(SessionEvent.campaign_id == campaign_id)
        if scene_id:
            q = q.filter(SessionEvent.scene_id == scene_id)
        events = q.order_by(SessionEvent.created_at).limit(limit).all()
        return [
            {
                "id": e.id,
                "campaign_id": e.campaign_id,
                "scene_id": e.scene_id,
                "session_id": e.session_id,
                "type": e.type,
                "text": e.text,
                "created_at": e.created_at,
            }
            for e in events
        ]
    finally:
        db.close()
