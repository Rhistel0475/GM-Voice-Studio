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
                "tags": ["merchant", "female", "young"],
                "source": "custom",
            },
            {
                "voice_id": "voice-guard",
                "provider": "hume",
                "name": "City Watch Captain",
                "description": "Steady, authoritative veteran commander.",
                "tags": ["guard", "rough", "male"],
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

    assert suggested["provider"] == "hume"
    assert suggested["voice_id"] == "voice-guard"
    assert suggested["voice_name"] == "City Watch Captain"
    assert suggested["confidence"] > 0.5
    assert set(suggested["matched_tags"]) >= {"guard", "rough"}


def test_suggest_voice_for_scholar_matches_voice_tags_from_description(monkeypatch):
    monkeypatch.setattr(
        voice_assignment_service,
        "_list_assignable_voices",
        lambda owner_id=None: [
            {
                "voice_id": "voice-bandit",
                "provider": "hume",
                "name": "Highway Bandit",
                "tags": ["villain", "rough", "male"],
            },
            {
                "voice_id": "voice-scholar",
                "provider": "hume",
                "name": "Archive Lecturer",
                "tags": ["scholar", "old"],
            },
        ],
    )

    suggested = voice_assignment_service.suggest_voice_for_npc(
        {
            "name": "Professor Elsin",
            "role": "scholar",
            "personality": "patient and observant",
            "description": "An elderly historian who catalogs forbidden texts.",
        }
    )

    assert suggested["voice_id"] == "voice-scholar"
    assert set(suggested["matched_tags"]) >= {"scholar", "old"}
    assert suggested["confidence"] > 0.5


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
