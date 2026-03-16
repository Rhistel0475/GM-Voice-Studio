import numpy as np

from app.services import scene_trigger_service


def _sample_bundle(*, scene=None, npcs=None):
    scene_payload = {
        "id": "1",
        "title": "Frozen Crossroads",
        "read_aloud": "A cold mist rolls over the crossroads as lantern light flickers in the dark.",
        "notes": "The ruined milestone points toward the old barrow.",
        "narrator_voice_id": "narrator-1",
        "npcs": ["Oleg"],
        "triggers": [],
    }
    if scene:
        scene_payload.update(scene)

    npc_payloads = npcs or [
        {
            "id": "npc-1",
            "name": "Oleg",
            "description": "A wary scout with a dry sense of humor.",
            "faction": "North Road Wardens",
            "voice_id": "voice-oleg",
        }
    ]

    return {
        "campaign_id": 7,
        "scene": scene_payload,
        "npcs": npc_payloads,
        "scene_npcs": npc_payloads,
    }


def test_execute_scene_trigger_uses_default_narration(monkeypatch):
    monkeypatch.setattr(
        scene_trigger_service.campaign_repository,
        "get_scene_bundle",
        lambda scene_id: _sample_bundle(),
    )
    monkeypatch.setattr(scene_trigger_service.tts_service, "resolve_voice_target", lambda voice_id: f"resolved:{voice_id}")
    monkeypatch.setattr(
        scene_trigger_service.tts_service,
        "generate",
        lambda text, **kwargs: (np.zeros(32, dtype=np.float32), 24000),
    )

    result = scene_trigger_service.execute_scene_trigger("1", "Narrate Scene")

    assert result["trigger_type"] == "narration"
    assert result["text"].startswith("A cold mist rolls over the crossroads")
    assert result["voice_id"] == "narrator-1"
    assert result["audio_base64"]


def test_execute_scene_trigger_generates_dialogue_for_default_npc(monkeypatch):
    monkeypatch.setattr(
        scene_trigger_service.campaign_repository,
        "get_scene_bundle",
        lambda scene_id: _sample_bundle(),
    )
    monkeypatch.setattr(
        scene_trigger_service.ai_service,
        "generate_dialogue",
        lambda **kwargs: "Keep your voices down. The dead carry sound in this fog.",
    )
    monkeypatch.setattr(scene_trigger_service.tts_service, "resolve_voice_target", lambda voice_id: f"resolved:{voice_id}")
    monkeypatch.setattr(
        scene_trigger_service.tts_service,
        "generate",
        lambda text, **kwargs: (np.zeros(32, dtype=np.float32), 24000),
    )

    result = scene_trigger_service.execute_scene_trigger("1", "Speak as Oleg")

    assert result["trigger_type"] == "dialogue"
    assert result["npc_name"] == "Oleg"
    assert "Keep your voices down" in result["text"]
    assert result["voice_id"] == "voice-oleg"
    assert result["audio_base64"]


def test_execute_scene_trigger_runs_ai_action_without_audio(monkeypatch):
    bundle = _sample_bundle(
        scene={
            "triggers": [
                {
                    "name": "Quest Hook",
                    "type": "ai_action",
                    "action": {"prompt": "Offer a short quest hook for this crossroads scene."},
                }
            ]
        }
    )
    monkeypatch.setattr(
        scene_trigger_service.campaign_repository,
        "get_scene_bundle",
        lambda scene_id: bundle,
    )
    monkeypatch.setattr(
        scene_trigger_service,
        "handle_query",
        lambda prompt: {"content": "A rider begs the party to recover a missing relic before dawn."},
    )

    result = scene_trigger_service.execute_scene_trigger("1", "Quest Hook")

    assert result["trigger_type"] == "ai_action"
    assert result["text"] == "A rider begs the party to recover a missing relic before dawn."
    assert result["audio_base64"] is None

