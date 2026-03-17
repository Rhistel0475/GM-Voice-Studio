from app.services import atmosphere_service, scene_activation_service


def test_get_scene_atmosphere_returns_static_track(tmp_path, monkeypatch):
    audio_dir = tmp_path / "atmosphere"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "forest.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(atmosphere_service, "_ATMOSPHERE_DIR", audio_dir)

    payload = atmosphere_service.get_scene_atmosphere({"atmosphere_type": "forest"})

    assert payload == {
        "ambience_audio": {
            "track_id": "forest",
            "atmosphere_type": "forest",
            "label": "Forest Ambience",
            "filename": "forest.wav",
            "url": "/static/audio/atmosphere/forest.wav",
            "loop": True,
            "volume": 0.34,
        },
        "loop": True,
    }


def test_start_scene_combat_returns_combat_ambience(monkeypatch):
    monkeypatch.setattr(
        scene_activation_service.campaign_repository,
        "activate_scene",
        lambda scene_id, atmosphere_override_type=None: {
            "id": scene_id,
            "title": "Ambush",
            "atmosphere_type": "forest",
            "atmosphere_override_type": atmosphere_override_type,
        },
    )
    monkeypatch.setattr(
        scene_activation_service,
        "get_scene_atmosphere",
        lambda scene: {"ambience_audio": {"url": "/static/audio/atmosphere/combat.wav", "atmosphere_type": scene.get("atmosphere_override_type")}},
    )

    payload = scene_activation_service.start_scene_combat("42")

    assert payload == {
        "scene": {
            "id": "42",
            "title": "Ambush",
            "atmosphere_type": "forest",
            "atmosphere_override_type": "combat",
            "ambience_track": {
                "url": "/static/audio/atmosphere/combat.wav",
                "atmosphere_type": "combat",
            },
        },
        "ambience_audio": {
            "url": "/static/audio/atmosphere/combat.wav",
            "atmosphere_type": "combat",
        },
    }


def test_get_scene_atmosphere_supports_mystery_track_alias(tmp_path, monkeypatch):
    audio_dir = tmp_path / "atmosphere"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "dungeon.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(atmosphere_service, "_ATMOSPHERE_DIR", audio_dir)

    payload = atmosphere_service.get_scene_atmosphere({"atmosphere_type": "mystery"})

    assert payload["ambience_audio"]["track_id"] == "mystery"
    assert payload["ambience_audio"]["filename"] == "dungeon.wav"
    assert payload["ambience_audio"]["atmosphere_type"] == "mystery"
