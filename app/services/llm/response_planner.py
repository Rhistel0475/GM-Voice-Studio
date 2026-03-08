"""
Response planning: Co-DM system prompt, RAG context formatting, and Claude call with tag parsing.
"""
from __future__ import annotations

import logging
from typing import Any

from app.infrastructure.adapters.llm import get_default_llm_adapter

logger = logging.getLogger(__name__)

CO_DM_SYSTEM_PROMPT = """\
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

TAG_TO_TYPE = {
    "[STAT_BLOCK]": "stat_block",
    "[LORE]": "lore",
    "[CHAT]": "chat",
}


def build_rag_context(chunks: list[dict[str, Any]]) -> str:
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


def call_claude(user_message: str) -> tuple[str, str]:
    """
    Call Claude with the Co-DM system prompt.
    Returns (response_type, content) where response_type is 'stat_block'|'lore'|'chat'.
    """
    adapter = get_default_llm_adapter()
    raw = adapter.complete(
        system=CO_DM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
    )
    raw = (raw or "").strip()

    for tag, response_type in TAG_TO_TYPE.items():
        if raw.startswith(tag):
            content = raw[len(tag):].strip()
            return response_type, content

    logger.warning("Claude response missing type tag; defaulting to chat")
    return "chat", raw
