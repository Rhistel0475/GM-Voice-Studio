from app.services.entity_normalization_service import normalize_campaign_entities
from app.services.live_context_service import build_scene_live_context
from app.services.parsing.confidence import annotate_campaign_confidence
from app.services.parsing.models import SectionChunk
from app.services.parsing.coverage_audit import audit_coverage
from app.services.parsing.importance import score_importance


def test_annotate_campaign_confidence_sets_review_priority_from_chunk_signals():
    chunk = SectionChunk(
        heading="South Gate",
        level=1,
        body="Captain Vane challenges the party at the gate and explains the current danger.",
        content_type="npc",
        content_types=["npc", "scene"],
        classification_confidence=0.92,
        classification_method="heuristic+llm",
        document_id="doc_1",
        page_number=3,
    )
    payload = {
        "npcs": [
            {
                "name": "Captain Vane",
                "role": "ally",
                "description": "Veteran gate captain",
                "motivation": "Keep the gate secure",
                "faction": "Town Guard",
                "confidence": 0.88,
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                },
            }
        ],
        "scenes": [],
        "locations": [],
        "encounters": [],
        "items": [],
        "quests": [],
        "factions": [],
        "lore": [],
        "codex_entries": [],
    }

    result = annotate_campaign_confidence(payload, [chunk])

    assert result["npcs"][0]["confidence_label"] == "high"
    assert result["npcs"][0]["review_priority"] == "auto_approve"
    assert result["npcs"][0]["needs_review"] is False
    assert result["review_summary"]["auto_approve_count"] == 1


def test_normalize_campaign_entities_preserves_relationship_sets_after_merging():
    payload = {
        "npcs": [
            {
                "name": "Gavel Thuldrin Kreed",
                "description": "Leader of the Lumber Consortium.",
                "confidence": 0.8,
                "source": {"document_id": "doc_1", "heading": "Authority Figures", "page_number": 5},
            },
            {
                "name": "Thuldrin Kreed",
                "description": "Ruthless consortium boss.",
                "confidence": 0.9,
                "source": {"document_id": "doc_1", "heading": "Authority Figures", "page_number": 5},
            },
        ],
        "locations": [
            {
                "name": "Lumber Consortium Hall",
                "description": "A severe guildhall.",
                "confidence": 0.9,
                "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
            }
        ],
        "scenes": [
            {
                "title": "Audience with the Consortium",
                "npcs": ["Gavel Thuldrin Kreed"],
                "location": "Lumber Consortium Hall",
                "confidence": 0.8,
                "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
            }
        ],
        "quests": [
            {
                "name": "Win Kreed's Support",
                "description": "Negotiate with Kreed.",
                "objective": "Win Gavel Thuldrin Kreed's support.",
                "related_npcs": ["Gavel Thuldrin Kreed"],
                "related_locations": ["Lumber Consortium Hall"],
                "confidence": 0.86,
                "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
            }
        ],
        "factions": [],
        "lore": [],
        "items": [],
        "encounters": [],
        "codex_entries": [],
        "relationships": [
            {
                "from_type": "scene",
                "from_id": "Audience with the Consortium",
                "relation": "advances",
                "to_type": "quest",
                "to_id": "Win Kreed's Support",
            }
        ],
    }

    result = normalize_campaign_entities(payload)

    assert len(result["npcs"]) == 1
    assert result["npcs"][0]["name"] == "Thuldrin Kreed"
    assert result["scenes"][0]["npcs"] == ["Thuldrin Kreed"]
    assert result["quests"][0]["related_npcs"] == ["Thuldrin Kreed"]
    assert result["scenes"][0]["related_quests"] == ["Win Kreed's Support"]


def test_build_scene_live_context_collects_location_npcs_and_related_quests(monkeypatch):
    bundle = {
        "campaign_id": 42,
        "scene": {
            "id": "scene_1",
            "title": "At the South Gate",
            "location": "South Gate",
        },
        "scene_npcs": [{"id": "npc_1", "name": "Captain Vane"}],
    }
    campaign = {
        "id": 42,
        "locations": [{"id": "loc_1", "name": "South Gate", "description": "The guarded entrance to town."}],
        "quests": [
            {
                "id": "quest_1",
                "name": "Gain Entry",
                "description": "Convince the captain to open the gate.",
                "related_npcs": ["Captain Vane"],
                "related_locations": ["South Gate"],
            }
        ],
        "codex_entries": [],
        "relationships": [
            {
                "from_type": "scene",
                "from_id": "At the South Gate",
                "relation": "advances",
                "to_type": "quest",
                "to_id": "Gain Entry",
            }
        ],
    }

    monkeypatch.setattr("app.services.live_context_service.campaign_repository.get_scene_bundle", lambda scene_id: bundle)
    monkeypatch.setattr("app.services.live_context_service.campaign_repository.get_by_id", lambda campaign_id: campaign)

    context = build_scene_live_context(scene_id="scene_1")

    assert context["location"]["name"] == "South Gate"
    assert context["scene_npcs"][0]["name"] == "Captain Vane"
    assert context["related_quests"][0]["name"] == "Gain Entry"
    assert "Related quests: Gain Entry" in context["summary"]


def test_recall_upgrade_scores_and_coverage_can_coexist_with_payload():
    payload = {
        "npcs": [{"name": "Archivist Mel", "mention_count": 3, "confidence": 0.8, "source": {"heading": "Mel"}}],
        "locations": [{"name": "Vault of Ash", "description": ""}],
        "scenes": [{"title": "Vault Entry", "npcs": []}],
        "quests": [{"name": "Open the Vault", "objective": "", "stakes": ""}],
        "encounters": [],
        "items": [],
        "factions": [],
        "lore": [],
        "codex_entries": [],
        "relationships": [],
    }
    scored = score_importance(payload)
    report = audit_coverage(scored, [SectionChunk(heading="Vault of Ash", level=1, body="The vault waits.", page_number=2)])
    scored["coverage_report"] = report
    assert isinstance(scored["coverage_report"], dict)
    assert "summary" in scored["coverage_report"]
    assert "importance_score" in scored["npcs"][0]
