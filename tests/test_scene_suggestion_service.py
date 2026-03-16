from app.services import scene_suggestion_service


def test_suggest_next_scenes_prefers_connected_scene_that_matches_player_action():
    current_scene = {
        "id": "scene-roadside",
        "title": "Crossroads",
        "description": "A muddy crossroads with tracks heading north.",
        "connected_scenes": ["scene-road", "scene-tavern"],
        "campaign_id": 7,
    }
    scenes = [
        current_scene,
        {
            "id": "scene-road",
            "title": "Northern Road",
            "description": "The northern road winds toward a ruined watchtower.",
            "type": "exploration",
        },
        {
            "id": "scene-tavern",
            "title": "Old Lantern Tavern",
            "description": "A warm tavern where travelers rest and trade rumors.",
            "type": "social",
        },
    ]

    payload = scene_suggestion_service.suggest_next_scenes(
        current_scene,
        "The party heads north toward the watchtower.",
        campaign_scenes=scenes,
    )

    assert [scene["id"] for scene in payload["suggested_scenes"][:2]] == [
        "scene-road",
        "scene-tavern",
    ]
    assert payload["suggested_scenes"][0]["suggestion_reason"] in {
        "Connected to the current scene",
        "Follows the party's travel direction",
        "Matches the party's recent action",
    }


def test_suggest_next_scenes_uses_reverse_links_and_campaign_flow_when_needed():
    current_scene = {
        "id": "scene-gate",
        "title": "South Gate",
        "description": "The city gate opens onto the trade road.",
        "campaign_id": 9,
    }
    scenes = [
        current_scene,
        {
            "id": "scene-market",
            "title": "Market Square",
            "description": "A bustling plaza full of merchants and guards.",
            "connected_scenes": ["scene-gate"],
            "type": "social",
        },
        {
            "id": "scene-ambush",
            "title": "Bandit Ambush",
            "description": "Bandits spring from the tree line beside the road.",
            "type": "combat",
        },
    ]

    payload = scene_suggestion_service.suggest_next_scenes(
        current_scene,
        "",
        campaign_scenes=scenes,
    )

    assert [scene["id"] for scene in payload["suggested_scenes"][:2]] == [
        "scene-market",
        "scene-ambush",
    ]
