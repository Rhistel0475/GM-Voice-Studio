import json

from app.services import tts_service


def test_parse_hume_voice_id_round_trip():
    voice_id = tts_service.make_hume_voice_id("HUME_AI", "voice_123")
    assert voice_id == "hume:HUME_AI:voice_123"
    assert tts_service.parse_hume_voice_id(voice_id) == ("HUME_AI", "voice_123")


def test_list_hume_voices_maps_provider_payload(monkeypatch):
    def fake_hume_request(path: str, payload=None, method="GET"):
        if "provider=HUME_AI" in path:
            body = {"voices": [{"id": "builtin_1", "name": "Narrator", "description": "Warm and clear"}]}
        else:
            body = {"voices": [{"id": "custom_1", "name": "Shopkeeper"}]}
        return json.dumps(body).encode("utf-8")

    monkeypatch.setattr(tts_service, "HUME_API_KEY", "test-key")
    monkeypatch.setattr(tts_service, "_hume_request", fake_hume_request)

    voices = tts_service.list_hume_voices()

    assert voices == [
        {
            "voice_id": "hume:HUME_AI:builtin_1",
            "provider_voice_id": "builtin_1",
            "provider": "HUME_AI",
            "name": "Narrator",
            "source": "system",
            "status": "ready",
            "description": "Warm and clear",
        },
        {
            "voice_id": "hume:CUSTOM_VOICE:custom_1",
            "provider_voice_id": "custom_1",
            "provider": "CUSTOM_VOICE",
            "name": "Shopkeeper",
            "source": "custom",
            "status": "ready",
            "description": "",
        },
    ]
