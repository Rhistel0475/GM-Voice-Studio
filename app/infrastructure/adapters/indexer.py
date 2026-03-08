"""
Indexer adapter: abstract interface and Pinecone ingest implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexerAdapter(Protocol):
    """Interface for ingesting documents into the vector store."""

    def ingest(self, paths: list[Path], doc_type: str) -> int:
        """Ingest files and return the number of chunks indexed."""
        ...


class PineconeIndexerAdapter:
    """Pinecone + OpenAI embeddings implementation of IndexerAdapter."""

    def ingest(self, paths: list[Path], doc_type: str) -> int:
        from app.infrastructure.retrieval.indexer import ingest
        return ingest(paths, doc_type)


_default_indexer: PineconeIndexerAdapter | None = None


def get_default_indexer() -> IndexerAdapter:
    """Return the default indexer (Pinecone ingest)."""
    global _default_indexer
    if _default_indexer is None:
        _default_indexer = PineconeIndexerAdapter()
    return _default_indexer
