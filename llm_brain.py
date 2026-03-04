"""
Co-DM LLM Brain — Sprint 3
============================
Intent classification + RAG-augmented Claude responses.

Entry point: handle_query(query) -> dict
Returns: {type, intent, content, sources}
"""
import logging
import re
from typing import Optional

from config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Anthropic client
# ---------------------------------------------------------------------------

_anthropic_client = None


def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


# ---------------------------------------------------------------------------
# Intent classification (keyword-based, no API call)
# ---------------------------------------------------------------------------

_NPC_PATTERNS = re.compile(
    r"\b(generate|create|make|build|give me|invent)\b.{0,30}\b(npc|character|villain|ally|enemy|patron|contact)\b"
    r"|\b(npc|character)\b.{0,20}\b(generate|create|make|build)\b",
    re.IGNORECASE,
)

_RULE_PATTERNS = re.compile(
    r"\b(rule|mechanic|how does|how do|what (is|are) the|stat|stats|AC|HP|CR|DC|attack|damage|ability|skill|"
    r"saving throw|spell|grapple|flanking|initiative|cover|condition|resistance|immunity|proficiency)\b",
    re.IGNORECASE,
)


def classify_intent(query: str) -> str:
    """
    Fast keyword-based intent classification.
    Returns one of: 'rule_lookup', 'npc_request', 'general_chat'
    """
    if _NPC_PATTERNS.search(query):
        return "npc_request"
    if _RULE_PATTERNS.search(query):
        return "rule_lookup"
    return "general_chat"


# ---------------------------------------------------------------------------
# Co-DM system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Co-DM, an expert assistant for a 1930s noir fantasy tabletop RPG campaign.
Answer concisely — DMs read this behind a screen during live play.

Rules:
- If CONTEXT is provided, base your answer on it. If the answer isn't in the context, say so briefly.
- For stats, monsters, or NPCs: output a compact markdown stat block.
- For lore, factions, or history: output a short prose summary (2–4 sentences max).
- For general questions: respond conversationally in 1–3 sentences.

Begin your reply with exactly one of these tags on its own line:
  [STAT_BLOCK]
  [LORE]
  [CHAT]

Then provide your answer immediately after."""

# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

_TAG_TO_TYPE = {
    "[STAT_BLOCK]": "stat_block",
    "[LORE]": "lore",
    "[CHAT]": "chat",
}


def _call_claude(user_message: str) -> tuple[str, str]:
    """
    Call Claude haiku with the Co-DM system prompt.
    Returns (response_type, content) where response_type is 'stat_block'|'lore'|'chat'.
    """
    client = _get_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()

    # Parse the leading [TAG]
    for tag, response_type in _TAG_TO_TYPE.items():
        if raw.startswith(tag):
            content = raw[len(tag):].strip()
            return response_type, content

    # No tag found — default to chat
    logger.warning("Claude response missing type tag; defaulting to chat")
    return "chat", raw


# ---------------------------------------------------------------------------
# RAG context builder
# ---------------------------------------------------------------------------

def _build_rag_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a CONTEXT block for the prompt."""
    if not chunks:
        return ""
    lines = ["CONTEXT (from campaign documents):"]
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "")
        ref = f"{source}, p.{page}" if page else source
        lines.append(f"\n[{i}] ({ref})\n{chunk['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    sources: list[dict] = []

    if intent == "npc_request":
        # Sprint 4 placeholder
        return {
            "type": "chat",
            "intent": intent,
            "content": "NPC generation is coming in Sprint 4. For now, try asking me about rules or lore.",
            "sources": [],
        }

    if intent == "rule_lookup":
        try:
            from retrieval import retrieve
            chunks = retrieve(query, top_k=5)
        except Exception:
            logger.exception("RAG retrieval failed; falling back to general chat")
            chunks = []

        sources = [
            {"source": c.get("source", ""), "page": c.get("page", 0), "score": c.get("score", 0.0)}
            for c in chunks
        ]
        context_block = _build_rag_context(chunks)
        user_message = f"{context_block}\n\nQUESTION: {query}" if context_block else f"QUESTION: {query}"
    else:
        user_message = query

    response_type, content = _call_claude(user_message)
    return {
        "type": response_type,
        "intent": intent,
        "content": content,
        "sources": sources,
    }
