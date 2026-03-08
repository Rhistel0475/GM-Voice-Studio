"""
Campaign domain: adventure parsing, campaign persistence, NPCs/scenes/locations.

Ownership:
- Repository: app.repositories.campaign_repository
- Models: app.infrastructure.db_models (Campaign, NPC, Scene, Location)
- Routes (in routes_legacy): GET/POST/DELETE /api/campaigns, /api/campaigns/{id},
  POST /adventure/parse, POST /adventure/ai-parse, POST /adventure/images (persist step)
- AI parsing (campaign structure): app.services.ai_service.ai_full_parse, assign_images_to_entities
"""

from app.repositories import campaign_repository

list_all = campaign_repository.list_all
get_by_id = campaign_repository.get_by_id
delete = campaign_repository.delete
create_from_parse_result = campaign_repository.create_from_parse_result

__all__ = [
    "campaign_repository",
    "list_all",
    "get_by_id",
    "delete",
    "create_from_parse_result",
]
