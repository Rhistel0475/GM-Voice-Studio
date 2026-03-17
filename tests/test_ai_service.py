from unittest.mock import MagicMock, patch

from app.services.ai_service import _finalize_parse_payload, assign_images_to_entities


class TestAssignImages:
    def test_assign_images_to_entities_salvages_valid_prefix_from_truncated_json(self):
        images = [
            {"idx": 1, "page": 2, "url": "https://example.com/vane.png"},
            {"idx": 2, "page": 3, "url": "https://example.com/gate.png"},
        ]
        campaign = {
            "title": "South Gate Trouble",
            "npcs": [{"name": "Captain Vane"}],
            "scenes": [],
            "locations": [{"name": "South Gate"}],
        }
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "idx":1,
                "type":"portrait",
                "assigned_to":"Captain Vane",
                "label":"grim guard captain"
            },
            {
                "idx":2,
                "type":"map",
                "assigned_to":"South Gate",
                "label":"frontier gate ma
        """)]

        with patch("app.infrastructure.llm.anthropic_client.get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = assign_images_to_entities(images, campaign, total_pages=12)

        assert result[0]["type"] == "portrait"
        assert result[0]["assigned_to"] == "Captain Vane"
        assert result[1]["type"] == "illustration"
        assert result[1]["assigned_to"] is None
        assert campaign["npcs"][0]["image_url"] == "https://example.com/vane.png"

    def test_assign_images_to_entities_recovers_missing_comma_inside_object(self):
        images = [
            {"idx": 1, "page": 2, "url": "https://example.com/vane.png"},
        ]
        campaign = {
            "title": "South Gate Trouble",
            "npcs": [{"name": "Captain Vane"}],
            "scenes": [],
            "locations": [{"name": "South Gate"}],
        }
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "idx":1,
                "type":"portrait",
                "assigned_to":"Captain Vane"
                "label":"grim guard captain"
            }
        ]""")]

        with patch("app.infrastructure.llm.anthropic_client.get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = assign_images_to_entities(images, campaign, total_pages=12)

        assert result[0]["type"] == "portrait"
        assert result[0]["assigned_to"] == "Captain Vane"
        assert result[0]["label"] == "grim guard captain"
        assert campaign["npcs"][0]["image_url"] == "https://example.com/vane.png"


def test_finalize_parse_payload_keeps_existing_review_metadata_stable():
    payload = {
        "title": "South Gate Trouble",
        "summary": "A short test adventure.",
        "npcs": [
            {
                "name": "Captain Vane",
                "description": "Watch captain guarding the gate.",
                "confidence": 0.91,
                "confidence_score": 0.91,
                "confidence_label": "high",
                "needs_review": False,
                "review_priority": "auto_approve",
            }
        ],
        "party": [],
        "scenes": [],
        "locations": [],
        "encounters": [],
        "reveals": [],
        "items": [],
        "quests": [],
        "factions": [],
        "lore": [],
        "codex_entries": [],
        "relationships": [],
        "review_summary": {"auto_approve": 1, "review_queue": 0, "hidden": 0},
    }

    result = _finalize_parse_payload(payload)

    assert result["npcs"][0]["confidence_score"] == 0.91
    assert result["npcs"][0]["confidence_label"] == "high"
    assert result["npcs"][0]["review_priority"] == "auto_approve"
    assert result["review_summary"] == {"auto_approve": 1, "review_queue": 0, "hidden": 0}
