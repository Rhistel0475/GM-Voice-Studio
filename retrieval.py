"""
Co-DM RAG retrieval service.
Connects to Pinecone on first call (lazy init) and reuses the connection.
"""
import logging
from typing import Optional

from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

_vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file.")
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        pc = Pinecone(api_key=PINECONE_API_KEY)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
        )
        _vector_store = PineconeVectorStore(
            index=pc.Index(PINECONE_INDEX_NAME),
            embedding=embeddings,
        )
        logging.info("RAG: connected to Pinecone index '%s'", PINECONE_INDEX_NAME)
    return _vector_store


def retrieve(
    query: str,
    top_k: int = 5,
    doc_type: Optional[str] = None,
) -> list[dict]:
    """
    Return top_k semantically relevant chunks for query.
    Each result: {text, source, page, doc_type, score}
    Optionally filter by doc_type (rulebook | faction | bestiary | session).
    """
    if not query or not query.strip():
        return []

    store = _get_vector_store()
    search_kwargs: dict = {"k": top_k}
    if doc_type:
        search_kwargs["filter"] = {"doc_type": doc_type}

    try:
        results = store.similarity_search_with_score(query.strip(), **search_kwargs)
    except Exception as e:
        logging.exception("RAG retrieval failed")
        raise RuntimeError(f"Retrieval failed: {e!s}") from e

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", 0),
            "doc_type": doc.metadata.get("doc_type", ""),
            "score": float(score),
        }
        for doc, score in results
    ]
