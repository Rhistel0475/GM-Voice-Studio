"""
External service adapter protocols (Phase 6).
Implementations live in this package; services can depend on these abstractions.
"""

from app.infrastructure.adapters.llm import LLMAdapter, get_default_llm_adapter
from app.infrastructure.adapters.retriever import RetrieverAdapter, get_default_retriever
from app.infrastructure.adapters.indexer import IndexerAdapter, get_default_indexer
from app.infrastructure.adapters.transcription import TranscriptionAdapter, get_default_transcription_adapter
from app.infrastructure.adapters.storage import VoiceStorageAdapter, get_default_voice_storage

__all__ = [
    "LLMAdapter",
    "get_default_llm_adapter",
    "RetrieverAdapter",
    "get_default_retriever",
    "IndexerAdapter",
    "get_default_indexer",
    "TranscriptionAdapter",
    "get_default_transcription_adapter",
    "VoiceStorageAdapter",
    "get_default_voice_storage",
]
