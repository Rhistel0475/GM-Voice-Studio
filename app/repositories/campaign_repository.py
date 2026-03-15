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

from app.infrastructure.database import SessionLocal
from app.infrastructure.db_models import (
    Campaign,
    CampaignDocument,
    Location,
    NPC,
    Scene,
    SessionEvent,
)


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
    for key in ("npcs", "party", "scenes", "locations", "reveals", "items", "images"):
        if not isinstance(payload.get(key), list):
            payload[key] = []
    for key in ("codex_entries", "relationships"):
        if not isinstance(payload.get(key), list):
            payload[key] = []
    if not isinstance(payload.get("documents"), list):
        payload["documents"] = []
    return payload


def _campaign_payload_from_relations(campaign: Campaign) -> dict[str, Any]:
    """Legacy fallback payload built from normalized relational tables."""
    return {
        "id": campaign.id,
        "title": campaign.title,
        "summary": campaign.summary,
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
                "act": s.act,
                "type": s.type,
                "read_aloud": s.read_aloud,
                "difficulty": s.difficulty,
                "rewards": s.rewards,
                "notes": s.notes,
                "image_url": s.image_url,
                "location": "",
                "npcs": [],
                "reveals": [],
                "items": [],
            }
            for s in campaign.scenes
        ],
        "locations": [
            {"id": loc.id, "name": loc.name, "description": loc.description, "image_url": loc.image_url}
            for loc in campaign.locations
        ],
        "reveals": [],
        "items": [],
        "images": [],
        "codex_entries": [],
        "relationships": [],
        "documents": [],
    }


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


def list_all() -> list[dict[str, Any]]:
    """Return all campaigns (id, title, summary) newest first."""
    db = SessionLocal()
    try:
        campaigns = db.query(Campaign).order_by(Campaign.id.desc()).all()
        return [{"id": c.id, "title": c.title, "summary": c.summary} for c in campaigns]
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
        campaign = Campaign(
            title=result.get("title", ""),
            summary=result.get("summary", ""),
            data_json=json.dumps(
                {
                    "title": result.get("title", ""),
                    "summary": result.get("summary", ""),
                    "npcs": result.get("npcs", []),
                    "party": result.get("party", []),
                    "scenes": result.get("scenes", []),
                    "locations": result.get("locations", []),
                    "reveals": result.get("reveals", []),
                    "items": result.get("items", []),
                    "images": result.get("images", []),
                    "codex_entries": result.get("codex_entries", []),
                    "relationships": result.get("relationships", []),
                },
                ensure_ascii=False,
            ),
        )
        db.add(campaign)
        db.flush()

        for n in result.get("npcs", []):
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
            db.add(
                Scene(
                    campaign_id=campaign.id,
                    title=s.get("title", ""),
                    act=s.get("act", ""),
                    type=s.get("type", ""),
                    read_aloud=s.get("read_aloud", ""),
                    difficulty=s.get("difficulty", ""),
                    rewards=s.get("rewards", ""),
                    notes=s.get("notes", ""),
                    image_url=s.get("image_url"),
                )
            )
        for loc in result.get("locations", []):
            db.add(
                Location(
                    campaign_id=campaign.id,
                    name=loc.get("name", ""),
                    description=loc.get("description", ""),
                    image_url=loc.get("image_url"),
                )
            )
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


def get_npc_record(npc_id: str) -> Optional[dict[str, Any]]:
    """Return a normalized NPC record by id, with name fallback for legacy callers."""
    db = SessionLocal()
    try:
        npc = None
        if str(npc_id).isdigit():
            npc = db.query(NPC).filter(NPC.id == int(npc_id)).first()
        if npc is None:
            npc = db.query(NPC).filter(NPC.name == str(npc_id)).first()
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
        scene = None
        if str(scene_id).isdigit():
            scene = db.query(Scene).filter(Scene.id == int(scene_id)).first()
        if scene is None:
            scene = db.query(Scene).filter(Scene.title == str(scene_id)).first()
        if scene is None:
            return None

        narrator_voice_id = None
        campaign = db.query(Campaign).filter(Campaign.id == scene.campaign_id).first()
        raw = (getattr(campaign, "data_json", "") or "").strip() if campaign is not None else ""
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    narrator_voice_id = (
                        payload.get("narrator_voice_id")
                        or payload.get("narrator_voice")
                        or None
                    )
            except (json.JSONDecodeError, TypeError):
                logging.warning("Campaign %s data_json invalid during scene lookup", scene.campaign_id)

        return {
            "id": str(scene.id),
            "campaign_id": scene.campaign_id,
            "title": scene.title,
            "read_aloud": scene.read_aloud,
            "notes": scene.notes,
            "type": scene.type,
            "narrator_voice_id": narrator_voice_id,
        }
    finally:
        db.close()


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
