"""
Voice storage adapter: abstract interface and default (local/S3 + voice_repository) implementation.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class VoiceStorageAdapter(Protocol):
    """Interface for voice embedding and metadata storage (local or S3)."""

    def create_voice_id(self) -> str: ...

    def save_embedding(
        self,
        voice_id: str,
        embedding: Any,
        *,
        consent_scope: str = "tts",
        name: Optional[str] = None,
        owner_id: Optional[str] = None,
        faction: Optional[str] = None,
    ) -> None: ...

    def load_embedding_path(self, voice_id: str) -> Optional[str]: ...

    def get_metadata(self, voice_id: str, owner_id: Optional[str] = None) -> Optional[dict]: ...

    def list_voices(self, owner_id: Optional[str] = None) -> list[dict]: ...

    def update_metadata(
        self,
        voice_id: str,
        name: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> bool: ...

    def delete_voice(self, voice_id: str, owner_id: Optional[str] = None) -> bool: ...


class DefaultVoiceStorageAdapter:
    """Delegates to app.services.voice_store_service (local or S3 + optional DB)."""

    def create_voice_id(self) -> str:
        from app.services.voice_store_service import create_voice_id
        return create_voice_id()

    def save_embedding(
        self,
        voice_id: str,
        embedding: Any,
        *,
        consent_scope: str = "tts",
        name: Optional[str] = None,
        owner_id: Optional[str] = None,
        faction: Optional[str] = None,
    ) -> None:
        from app.services.voice_store_service import save_embedding
        save_embedding(
            voice_id, embedding,
            consent_scope=consent_scope,
            name=name, owner_id=owner_id, faction=faction,
        )

    def load_embedding_path(self, voice_id: str) -> Optional[str]:
        from app.services.voice_store_service import load_embedding_path
        return load_embedding_path(voice_id)

    def get_metadata(self, voice_id: str, owner_id: Optional[str] = None) -> Optional[dict]:
        from app.services.voice_store_service import get_metadata
        return get_metadata(voice_id, owner_id=owner_id)

    def list_voices(self, owner_id: Optional[str] = None) -> list[dict]:
        from app.services.voice_store_service import list_voices
        return list_voices(owner_id=owner_id)

    def update_metadata(
        self,
        voice_id: str,
        name: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> bool:
        from app.services.voice_store_service import update_metadata
        return update_metadata(voice_id, name=name, owner_id=owner_id)

    def delete_voice(self, voice_id: str, owner_id: Optional[str] = None) -> bool:
        from app.services.voice_store_service import delete_voice
        return delete_voice(voice_id, owner_id=owner_id)


_default_voice_storage: DefaultVoiceStorageAdapter | None = None


def get_default_voice_storage() -> VoiceStorageAdapter:
    """Return the default voice storage adapter (local/S3 + optional DB)."""
    global _default_voice_storage
    if _default_voice_storage is None:
        _default_voice_storage = DefaultVoiceStorageAdapter()
    return _default_voice_storage
