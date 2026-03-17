"""Session memory tracking for important live-play events."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.infrastructure.database import SessionLocal
from app.infrastructure.db_models import Campaign, SessionMemory
from app.repositories import campaign_repository


def _normalize_event_type(event_type: str) -> str:
    return str(event_type or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_iso_timestamp(timestamp: Optional[str]) -> str:
    raw = str(timestamp or "").strip()
    if raw:
        return raw
    return datetime.now(timezone.utc).isoformat()


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    raw_items = tags if isinstance(tags, (list, tuple, set)) else [tags]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = _normalize_event_type(item)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _deserialize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return _normalize_tags(value)
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = [part.strip() for part in raw.split(",")]
    return _normalize_tags(parsed)


def _parse_sortable_timestamp(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()
    except ValueError:
        return raw


def _normalize_session_record(session: Any, campaign_id: int) -> Optional[dict[str, Any]]:
    if not isinstance(session, dict):
        return None
    session_id = str(session.get("id") or session.get("session_id") or "").strip()
    if not session_id:
        return None
    return {
        "id": session_id,
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
        "status": str(session.get("status") or "prep").strip().lower() or "prep",
    }


def _campaign_payload(campaign: Campaign) -> dict[str, Any]:
    raw = (getattr(campaign, "data_json", "") or "").strip()
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, TypeError):
            logging.warning("Campaign %s has invalid data_json during session memory lookup", campaign.id)

    fallback = campaign_repository.get_by_id(campaign.id)
    return fallback if isinstance(fallback, dict) else {}


def _load_candidate_sessions(
    *,
    campaign_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        query = db.query(Campaign)
        if campaign_id is not None:
            query = query.filter(Campaign.id == campaign_id)
        campaigns = query.all()
    finally:
        db.close()

    wanted_session_id = str(session_id or "").strip()
    candidates: list[dict[str, Any]] = []
    for campaign in campaigns:
        payload = _campaign_payload(campaign)
        active_session_id = str(payload.get("active_session_id") or "").strip()
        sessions = payload.get("sessions") if isinstance(payload.get("sessions"), list) else []
        for index, session in enumerate(sessions):
            normalized = _normalize_session_record(session, campaign.id)
            if normalized is None:
                continue
            normalized["is_active"] = (
                normalized["id"] == active_session_id
                or normalized["status"] == "active"
            )
            normalized["_sort_key"] = (
                1 if normalized["is_active"] else 0,
                _parse_sortable_timestamp(normalized.get("started_at")),
                index,
            )
            if wanted_session_id and normalized["id"] != wanted_session_id:
                continue
            candidates.append(normalized)
    return candidates


def _resolve_session_reference(
    *,
    campaign_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    wanted_session_id = str(session_id or "").strip()
    if wanted_session_id:
        matches = _load_candidate_sessions(campaign_id=campaign_id, session_id=wanted_session_id)
        if matches:
            return sorted(matches, key=lambda item: item["_sort_key"], reverse=True)[0]
        return {
            "id": wanted_session_id,
            "campaign_id": campaign_id,
            "title": "Session",
            "active_scene_id": None,
            "started_at": None,
            "status": "active",
            "is_active": True,
        }

    candidates = _load_candidate_sessions(campaign_id=campaign_id)
    active_candidates = [candidate for candidate in candidates if candidate.get("is_active")]
    if not active_candidates:
        return None
    return sorted(active_candidates, key=lambda item: item["_sort_key"], reverse=True)[0]


def _label_for_event(event_type: str) -> str:
    mapping = {
        "npc_interaction": "NPC Interaction",
        "player_decision": "Player Decision",
        "quest_progress": "Quest Progress",
        "combat_outcome": "Combat Outcome",
        "important_dialogue": "Important Dialogue",
    }
    normalized = _normalize_event_type(event_type)
    return mapping.get(normalized, normalized.replace("_", " ").title() or "Event")


def _format_event_line(event: dict[str, Any]) -> str:
    label = _label_for_event(str(event.get("event_type") or ""))
    description = str(event.get("description") or "").strip()
    npc_id = str(event.get("npc_id") or "").strip()
    tags = _normalize_tags(event.get("tags"))
    tags_text = f" [tags: {', '.join(tags)}]" if tags else ""
    if npc_id:
        return f"- {label} ({npc_id}): {description}{tags_text}"
    return f"- {label}: {description}{tags_text}"


def _summarize_events(events: list[dict[str, Any]], *, npc_id: Optional[str] = None) -> tuple[str, str]:
    ordered_events = events[-12:]
    summary = "\n".join(_format_event_line(event) for event in ordered_events if event.get("description"))

    wanted_npc_id = str(npc_id or "").strip()
    npc_events = [
        event for event in ordered_events
        if wanted_npc_id and str(event.get("npc_id") or "").strip() == wanted_npc_id
    ]
    npc_summary = "\n".join(_format_event_line(event) for event in npc_events if event.get("description"))

    return summary, npc_summary


def record_event(
    *,
    event_type: str,
    description: str,
    npc_id: Optional[str] = None,
    tags: Any = None,
    campaign_id: Optional[int] = None,
    scene_id: Optional[str] = None,
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """Store an important event against the active or specified session."""
    normalized_type = _normalize_event_type(event_type)
    if not normalized_type:
        raise ValueError("Event type is required.")

    normalized_description = str(description or "").strip()
    if not normalized_description:
        raise ValueError("Description is required.")

    resolved_session = _resolve_session_reference(campaign_id=campaign_id, session_id=session_id)
    if resolved_session is None:
        raise ValueError("No active session found.")

    db = SessionLocal()
    try:
        normalized_tags = _normalize_tags(tags)
        record = SessionMemory(
            session_id=str(resolved_session["id"]),
            timestamp=_normalize_iso_timestamp(timestamp),
            event_type=normalized_type,
            npc_id=str(npc_id or "").strip() or None,
            description=normalized_description,
            tags=json.dumps(normalized_tags, ensure_ascii=True),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return {
            "id": record.id,
            "session_id": record.session_id,
            "campaign_id": resolved_session.get("campaign_id"),
            "scene_id": str(scene_id or "").strip() or resolved_session.get("active_scene_id"),
            "timestamp": record.timestamp,
            "event_type": record.event_type,
            "npc_id": record.npc_id,
            "description": record.description,
            "tags": normalized_tags,
        }
    finally:
        db.close()


def get_session_summary(
    *,
    campaign_id: Optional[int] = None,
    session_id: Optional[str] = None,
    npc_id: Optional[str] = None,
    limit: int = 24,
) -> dict[str, Any]:
    """Return recent session memory and compact summaries for AI prompts."""
    try:
        resolved_session = _resolve_session_reference(campaign_id=campaign_id, session_id=session_id)
        if resolved_session is None:
            return {
                "session_id": None,
                "campaign_id": campaign_id,
                "events": [],
                "summary": "",
                "npc_memory_summary": "",
            }

        db = SessionLocal()
        try:
            rows = (
                db.query(SessionMemory)
                .filter(SessionMemory.session_id == str(resolved_session["id"]))
                .order_by(SessionMemory.timestamp.desc(), SessionMemory.id.desc())
                .limit(max(1, min(int(limit or 24), 100)))
                .all()
            )
        finally:
            db.close()
    except Exception as exc:
        logging.warning("Failed to load session memory context: %s", exc)
        return {
            "session_id": None,
            "campaign_id": campaign_id,
            "events": [],
            "summary": "",
            "npc_memory_summary": "",
        }

    events = [
        {
            "id": row.id,
            "session_id": row.session_id,
            "timestamp": row.timestamp,
            "event_type": row.event_type,
            "npc_id": row.npc_id,
            "description": row.description,
            "tags": _deserialize_tags(getattr(row, "tags", None)),
        }
        for row in reversed(rows)
    ]
    summary, npc_summary = _summarize_events(events, npc_id=npc_id)

    return {
        "session_id": resolved_session["id"],
        "campaign_id": resolved_session.get("campaign_id"),
        "events": events,
        "summary": summary,
        "npc_memory_summary": npc_summary,
    }


def get_session_context(
    *,
    campaign_id: Optional[int] = None,
    session_id: Optional[str] = None,
    npc_id: Optional[str] = None,
    limit: int = 24,
) -> dict[str, Any]:
    """Backward-compatible alias for callers expecting session context."""
    return get_session_summary(
        campaign_id=campaign_id,
        session_id=session_id,
        npc_id=npc_id,
        limit=limit,
    )
