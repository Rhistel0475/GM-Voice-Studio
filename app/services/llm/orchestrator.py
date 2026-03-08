"""
Co-DM query handler: classify intent → route (RAG or direct) → call LLM → return structured result.
"""
from __future__ import annotations

from app.services.llm.intent import classify_intent
from app.services.llm.tool_router import get_route_result
from app.services.llm.response_planner import call_claude

NPC_REDIRECT_MESSAGE = (
    "Use the NPC Gen tab in the middle panel to generate a full character profile (genre, location, name, role)."
)


def handle_query(query: str) -> dict:
    """
    Classify intent, optionally fetch RAG context, call Claude, return structured result.

    Returns:
        {
            "type":    "stat_block" | "lore" | "chat",
            "intent":  "rule_lookup" | "npc_request" | "general_chat",
            "content": "<markdown text>",
            "sources": [{"source": str, "page": int, "score": float}, ...]
        }
    """
    query = query.strip()
    if not query:
        return {"type": "chat", "intent": "general_chat", "content": "I didn't catch that.", "sources": []}

    intent = classify_intent(query)
    user_message, sources = get_route_result(query, intent)

    if user_message is None:
        return {
            "type": "chat",
            "intent": intent,
            "content": NPC_REDIRECT_MESSAGE,
            "sources": [],
        }

    response_type, content = call_claude(user_message)
    return {
        "type": response_type,
        "intent": intent,
        "content": content,
        "sources": sources,
    }
