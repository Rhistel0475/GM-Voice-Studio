from app.services import voice_assignment_service


def test_suggest_voice_for_guard_prefers_guard_library_voice(monkeypatch):
    monkeypatch.setattr(
        voice_assignment_service,
        "_list_assignable_voices",
        lambda owner_id=None: [
            {
                "voice_id": "voice-merchant",
                "provider": "hume",
                "name": "Friendly Shopkeeper",
                "description": "Warm and welcoming market voice.",
                "source": "custom",
            },
            {
                "voice_id": "voice-guard",
                "provider": "hume",
                "name": "City Watch Captain",
                "description": "Steady, authoritative veteran commander.",
                "source": "custom",
            },
        ],
    )

    suggested = voice_assignment_service.suggest_voice_for_npc(
        {
            "name": "Sergeant Vale",
            "role": "guard captain",
            "personality": "disciplined, loyal, and watchful",
            "description": "A veteran officer who keeps order at the city gate.",
        }
    )

    assert suggested == {
        "provider": "hume",
        "voice_id": "voice-guard",
        "voice_name": "City Watch Captain",
        "tone": "authoritative",
        "style": "crisp and disciplined",
    }


def test_suggest_voice_for_npc_returns_none_without_available_library(monkeypatch):
    monkeypatch.setattr(voice_assignment_service, "_list_assignable_voices", lambda owner_id=None: [])

    suggested = voice_assignment_service.suggest_voice_for_npc(
        {
            "name": "Mira",
            "role": "merchant",
            "personality": "cheerful and persuasive",
            "description": "A spice trader with a bright smile.",
        }
    )

    assert suggested is None
