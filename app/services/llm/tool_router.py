"""
Tool routing: given intent and query, decide RAG vs direct LLM and build user message.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.services.llm.response_planner import build_rag_context

logger = logging.getLogger(__name__)


def get_route_result(query: str, intent: str) -> tuple[Optional[str], list[dict]]:
    """
    Return (user_message, sources).
    user_message is None for npc_request (caller should return redirect message).
    Otherwise user_message is the prompt to send to the LLM; sources are RAG chunk refs.
    """
    if intent == "npc_request":
        return None, []

    if intent == "rule_lookup":
        try:
            from app.infrastructure.adapters.retriever import get_default_retriever
            retriever = get_default_retriever()
            chunks = retriever.retrieve(query.strip(), top_k=5)
        except Exception:
            logger.exception("RAG retrieval failed; falling back to general chat")
            chunks = []
        sources = [
            {"source": c.get("source", ""), "page": c.get("page", 0), "score": c.get("score", 0.0)}
            for c in chunks
        ]
        context_block = build_rag_context(chunks)
        user_message = f"{context_block}\n\nQUESTION: {query}" if context_block else f"QUESTION: {query}"
        return user_message, sources

    return query.strip(), []
