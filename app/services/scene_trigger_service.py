"""Execution service for LiveBoard scene control triggers."""
from __future__ import annotations

import base64
import io
from typing import Any

import soundfile as sf

from app.core.config import DEFAULT_VOICE_ID
from app.domain.live.scene_triggers import (
    TRIGGER_TYPE_AI_ACTION,
    TRIGGER_TYPE_DIALOGUE,
    TRIGGER_TYPE_NARRATION,
    normalize_scene_triggers,
    resolve_scene_npcs,
)
from app.repositories import campaign_repository
from app.services import tts_service


def _audio_to_wav_base64(audio, sample_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _scene_text(scene: dict[str, Any]) -> str:
    return (
        str(scene.get("read_aloud") or "").strip()
        or str(scene.get("notes") or "").strip()
        or str(scene.get("summary") or "").strip()
        or str(scene.get("title") or "").strip()
    )


def _find_trigger(triggers: list[dict[str, Any]], trigger_name: str) -> dict[str, Any] | None:
    lookup = str(trigger_name or "").strip().lower()
    if not lookup:
        return None
    for trigger in triggers:
        if str(trigger.get("name") or "").strip().lower() == lookup:
            return trigger
    return None


def _resolve_npc(bundle: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
    action = trigger.get("action") or {}
    candidate_name = str(action.get("npc_name") or action.get("npc") or "").strip()
    if not candidate_name:
        name = str(trigger.get("name") or "").strip()
        if name.lower().startswith("speak as "):
            candidate_name = name[9:].strip()

    scene_npcs = bundle.get("scene_npcs") or resolve_scene_npcs(bundle.get("scene") or {}, npcs=bundle.get("npcs") or [])
    all_npcs = [*scene_npcs, *(bundle.get("npcs") or [])]

    if candidate_name:
        for npc in all_npcs:
            if str((npc or {}).get("name") or "").strip().lower() == candidate_name.lower():
                return npc

    if scene_npcs:
        return scene_npcs[0]
    raise ValueError("This trigger does not have an NPC to speak through.")


def _npc_personality(npc: dict[str, Any]) -> str:
    return (
        str(npc.get("description") or "").strip()
        or str(npc.get("personality") or "").strip()
        or str(npc.get("role") or "").strip()
        or "An NPC in a tabletop RPG campaign."
    )


def _resolve_brain_response(action: dict[str, Any], *, fallback: str = "") -> dict[str, Any]:
    from app.services.llm_orchestrator import handle_query

    prompt = (
        str(action.get("prompt") or "").strip()
        or str(action.get("query") or "").strip()
        or fallback
    )
    if not prompt:
        raise ValueError("This trigger does not include enough prompt text to run.")
    return handle_query(prompt)


def _resolve_brain_text(action: dict[str, Any], *, fallback: str = "") -> str:
    result = _resolve_brain_response(action, fallback=fallback)
    return str(result.get("content") or result.get("text") or "").strip()


def _resolve_narration_text(trigger: dict[str, Any], scene: dict[str, Any]) -> str:
    text = str(trigger.get("text") or "").strip()
    if text:
        return text

    action = trigger.get("action") or {}
    action_kind = str(action.get("kind") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if action_kind in {"brain_query", "query", "generate_narration", ""}:
        generated = _resolve_brain_text(action, fallback=_scene_text(scene))
        if generated:
            return generated

    fallback = _scene_text(scene)
    if fallback:
        return fallback
    raise ValueError("This narration trigger does not have any text to speak.")


def _resolve_dialogue_text(trigger: dict[str, Any], bundle: dict[str, Any], npc: dict[str, Any]) -> str:
    text = str(trigger.get("text") or "").strip()
    if text:
        return text

    action = trigger.get("action") or {}
    action_kind = str(action.get("kind") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not action_kind and (action.get("prompt") or action.get("situation")):
        action_kind = "generate_dialogue"

    if action_kind in {"generate_dialogue", "npc_dialogue", "dialogue"}:
        from app.services import ai_service

        situation = (
            str(action.get("prompt") or "").strip()
            or str(action.get("situation") or "").strip()
            or _scene_text(bundle.get("scene") or {})
        )
        return ai_service.generate_dialogue(
            npc_name=str(npc.get("name") or "NPC"),
            personality=_npc_personality(npc),
            situation=situation,
            conversation_history=[],
            faction=str(npc.get("faction") or "").strip(),
        )

    if action_kind in {"brain_query", "query"}:
        return _resolve_brain_text(action, fallback=_scene_text(bundle.get("scene") or {}))

    raise ValueError("This dialogue trigger is missing text or a supported action.")


def _resolve_ai_action_payload(trigger: dict[str, Any], bundle: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None]:
    text = str(trigger.get("text") or "").strip()
    if text:
        return text, None, None

    action = trigger.get("action") or {}
    action_kind = str(action.get("kind") or "").strip().lower().replace("-", "_").replace(" ", "_")

    if action_kind in {"generate_dialogue", "npc_dialogue", "dialogue"}:
        npc = _resolve_npc(bundle, trigger)
        return _resolve_dialogue_text(trigger, bundle, npc), str(npc.get("name") or "").strip() or None, None

    brain_response = _resolve_brain_response(action, fallback=_scene_text(bundle.get("scene") or {}))
    text = str(brain_response.get("content") or brain_response.get("text") or "").strip()
    return text, None, brain_response


def _synthesize_text(*, text: str, voice_id: str) -> tuple[Any, int]:
    speaker_emb_path = tts_service.resolve_voice_target(voice_id)
    return tts_service.generate(text.strip(), language_tag="en", speaker_emb_path=speaker_emb_path)


def get_scene_triggers(scene_id: str) -> list[dict[str, Any]]:
    bundle = campaign_repository.get_scene_bundle(scene_id)
    if not bundle:
        raise FileNotFoundError("Scene not found")
    scene = bundle.get("scene") or {}
    return normalize_scene_triggers(scene, npcs=bundle.get("scene_npcs") or bundle.get("npcs") or [])


def execute_scene_trigger(scene_id: str, trigger_name: str) -> dict[str, Any]:
    bundle = campaign_repository.get_scene_bundle(scene_id)
    if not bundle:
        raise FileNotFoundError("Scene not found")

    scene = bundle.get("scene") or {}
    triggers = normalize_scene_triggers(scene, npcs=bundle.get("scene_npcs") or bundle.get("npcs") or [])
    trigger = _find_trigger(triggers, trigger_name)
    if trigger is None:
        raise FileNotFoundError("Trigger not found")

    trigger_type = str(trigger.get("type") or "").strip()
    base_result: dict[str, Any] = {
        "scene_id": str(scene.get("id") or scene_id),
        "campaign_id": bundle.get("campaign_id"),
        "trigger_name": str(trigger.get("name") or trigger_name).strip(),
        "trigger_type": trigger_type,
        "trigger": {
            "name": str(trigger.get("name") or trigger_name).strip(),
            "type": trigger_type,
        },
        "text": "",
        "display_text": "",
        "audio_base64": None,
        "mime_type": None,
        "voice_id": None,
        "npc_name": None,
        "log_type": "system",
        "event_type": "system",
        "ai_response": None,
    }

    if trigger_type == TRIGGER_TYPE_NARRATION:
        text = _resolve_narration_text(trigger, scene)
        voice_id = (
            str((trigger.get("action") or {}).get("voice_id") or "").strip()
            or str(scene.get("narrator_voice_id") or "").strip()
            or str(DEFAULT_VOICE_ID or "").strip()
        )
        if not voice_id:
            raise ValueError("No narrator voice configured for this scene.")
        audio, sample_rate = _synthesize_text(text=text, voice_id=voice_id)
        base_result.update({
            "text": text,
            "display_text": text,
            "audio_base64": _audio_to_wav_base64(audio, sample_rate),
            "mime_type": "audio/wav",
            "voice_id": voice_id,
            "log_type": "narration",
            "event_type": "narration",
        })
        return base_result

    if trigger_type == TRIGGER_TYPE_DIALOGUE:
        npc = _resolve_npc(bundle, trigger)
        voice_id = (
            str((trigger.get("action") or {}).get("voice_id") or "").strip()
            or str(npc.get("voice_id") or "").strip()
        )
        if not voice_id:
            raise ValueError(f"{str(npc.get('name') or 'This NPC').strip()} has no assigned voice.")
        text = _resolve_dialogue_text(trigger, bundle, npc)
        audio, sample_rate = _synthesize_text(text=text, voice_id=voice_id)
        base_result.update({
            "text": text,
            "display_text": f"{str(npc.get('name') or 'NPC').strip()}: {text}",
            "audio_base64": _audio_to_wav_base64(audio, sample_rate),
            "mime_type": "audio/wav",
            "voice_id": voice_id,
            "npc_name": str(npc.get("name") or "").strip() or None,
            "log_type": "npc",
            "event_type": "npc",
        })
        return base_result

    if trigger_type == TRIGGER_TYPE_AI_ACTION:
        text, npc_name, ai_response = _resolve_ai_action_payload(trigger, bundle)
        base_result.update({
            "text": text,
            "display_text": text if not npc_name else f"{npc_name}: {text}",
            "npc_name": npc_name,
            "log_type": "gm_note",
            "event_type": "gm_note",
            "ai_response": ai_response,
        })
        return base_result

    raise ValueError("Unsupported trigger type.")
