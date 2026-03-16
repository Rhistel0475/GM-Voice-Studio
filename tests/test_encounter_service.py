import numpy as np

from app.services import encounter_service


def test_launch_encounter_from_scene_returns_narration_enemy_audio_and_ambience(monkeypatch):
    scene = {
        "id": "scene-ambush",
        "campaign_id": 7,
        "title": "Bandit Ambush",
        "read_aloud": "From the treeline, three bandits emerge with blades drawn.",
        "notes": "The bandits try to surround the party.",
        "narrator_voice_id": "voice-narrator",
        "npcs": ["Happs Bydon", "Bandit Grunt"],
    }
    campaign = {
        "id": 7,
        "scenes": [scene],
        "encounters": [],
        "npcs": [
            {
                "id": "npc-happs",
                "name": "Happs Bydon",
                "role": "bandit leader",
                "description": "A swaggering brigand leader.",
                "voice_id": "voice-happs",
            },
            {
                "id": "npc-bandit",
                "name": "Bandit Grunt",
                "role": "bandit",
                "description": "A nervous raider with a rusty spear.",
                "voice_id": "voice-bandit",
            },
        ],
    }

    monkeypatch.setattr(encounter_service.campaign_repository, "get_scene_record", lambda encounter_id: scene)
    monkeypatch.setattr(encounter_service.campaign_repository, "get_by_id", lambda campaign_id: campaign)
    monkeypatch.setattr(
        encounter_service.scene_activation_service,
        "start_scene_combat",
        lambda scene_id: {
            "scene": {**scene, "atmosphere_override_type": "combat"},
            "ambience_audio": {"url": "/static/audio/atmosphere/combat.wav", "atmosphere_type": "combat"},
        },
    )
    monkeypatch.setattr(encounter_service.tts_service, "resolve_voice_target", lambda voice_id: f"resolved:{voice_id}")
    monkeypatch.setattr(
        encounter_service.tts_service,
        "generate",
        lambda text, **kwargs: (np.zeros(32, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        encounter_service.ai_service,
        "generate_dialogue",
        lambda **kwargs: "Take them alive if you can!",
    )
    monkeypatch.setattr(
        encounter_service,
        "get_session_context",
        lambda **kwargs: {"summary": "", "npc_memory_summary": ""},
    )

    payload = encounter_service.launch_encounter("scene-ambush")

    assert payload["encounter"]["name"] == "Bandit Ambush"
    assert payload["ambience_audio"]["url"] == "/static/audio/atmosphere/combat.wav"
    assert payload["narration_audio"]["text"].startswith("From the treeline")
    assert payload["narration_audio"]["audio_base64"]
    assert payload["enemy_dialogue_audio"]["npc_name"] == "Happs Bydon"
    assert payload["enemy_dialogue_audio"]["text"] == "Take them alive if you can!"
    assert payload["enemy_dialogue_audio"]["audio_base64"]
