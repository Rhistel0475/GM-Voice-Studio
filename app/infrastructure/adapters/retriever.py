"""
Retriever adapter: abstract interface and Pinecone implementation.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class RetrieverAdapter(Protocol):
    """Interface for semantic retrieval (e.g. RAG top-k chunks)."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Return top_k relevant chunks. Each dict: text, source, page, doc_type, score.
        """
        ...


class PineconeRetrieverAdapter:
    """Pinecone + OpenAI embeddings implementation of RetrieverAdapter."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        from app.infrastructure.retrieval.pinecone_retriever import retrieve
        return retrieve(query, top_k=top_k, doc_type=doc_type)


_default_retriever: PineconeRetrieverAdapter | None = None


def get_default_retriever() -> RetrieverAdapter:
    """Return the default retriever (Pinecone RAG)."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = PineconeRetrieverAdapter()
    return _default_retriever
