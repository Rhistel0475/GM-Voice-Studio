"""
Campaign persistence: list, get, delete, create.
Uses campaign DB (codm.db) via SessionLocal; session handling is internal.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.infrastructure.database import SessionLocal
from app.infrastructure.db_models import Campaign, NPC, Scene, Location


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
    }


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
        if payload is not None:
            return payload
        return _campaign_payload_from_relations(c)
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
