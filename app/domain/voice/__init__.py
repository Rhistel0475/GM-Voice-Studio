"""
Voice domain: cloning, storage, TTS, and voice metadata.

Ownership:
- Services: app.services.tts_service, voice_clone_service, voice_store_service
- Repository: app.repositories.voice_repository
- Routes (in routes_legacy): GET/POST /voices, /voices/clone, /voices/list, /voices/{id},
  PATCH/DELETE /voices/{id}, DELETE /admin/voices/{id}, POST /tts, POST /tts/narrate,
  GET /jobs/{id}, GET /jobs/{id}/result (clone/narrate jobs)
- Infrastructure: Celery tasks for async clone/narrate (app.infrastructure.tasks.celery_app)
"""

from app.services.voice_clone_service import clone_voice
from app.services.voice_store_service import (
    delete_voice,
    get_metadata,
    list_voices,
    load_embedding_path,
    update_metadata,
)
from app.services.tts_service import (
    generate as tts_generate,
    get_preset_voices,
    get_supported_language_tags,
    is_model_loaded,
)

__all__ = [
    "clone_voice",
    "delete_voice",
    "get_metadata",
    "list_voices",
    "load_embedding_path",
    "update_metadata",
    "tts_generate",
    "get_preset_voices",
    "get_supported_language_tags",
    "is_model_loaded",
]
