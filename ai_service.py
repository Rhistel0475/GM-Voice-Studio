"""
AI service: NPC dialogue generation via Anthropic Claude.
Callers pass NPC profile + situation + history; receive an in-character line.
Mirrors the pattern of tts_service.py — lazy client init, clear public API.
"""
import logging
from typing import Optional

import anthropic

from config import AI_MODEL, ANTHROPIC_API_KEY

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def build_npc_system_prompt(
    npc_name: str,
    personality: str,
    faction: str = "",
    situation: str = "",
) -> str:
    """
    Build a system prompt that casts Claude as an NPC character.
    Enforces short responses (1-3 sentences) for live table use.
    """
    faction_line = f"\nFaction/Allegiance: {faction.strip()}" if faction.strip() else ""
    situation_line = f"\nCurrent situation: {situation.strip()}" if situation.strip() else ""
    return (
        f"You are {npc_name.strip()}, a character in a tabletop RPG session. "
        f"Personality: {personality.strip()}{faction_line}{situation_line}\n\n"
        "Speak ONLY as this character. Do NOT break character. Do NOT explain or narrate. "
        "Do NOT say you are an AI. Respond as if you are actually speaking the words out loud "
        "at the game table. Keep every response to 1-3 sentences maximum — this is live "
        "dialogue, not prose. Use the character's voice, vocabulary, and emotional state."
    )


def generate_dialogue(
    npc_name: str,
    personality: str,
    situation: str,
    conversation_history: list[dict],
    faction: str = "",
) -> str:
    """
    Generate a short in-character NPC line using Claude.

    Args:
        npc_name: NPC's name (e.g. "Captain Aldric Vane")
        personality: Brief description (e.g. "gruff, loyal to the crown, hiding a secret")
        situation: What is happening right now (e.g. "Players are demanding to pass the gate")
        conversation_history: list of {"role": "user"|"assistant", "content": "..."}
        faction: Optional allegiance (e.g. "Silver Court Mages")

    Returns:
        The NPC's spoken line as a string.

    Raises:
        RuntimeError: on API connection, auth, rate limit, or unexpected errors.
    """
    client = _get_client()
    system_prompt = build_npc_system_prompt(npc_name, personality, faction, situation)

    # If no history, inject an opening nudge so Claude has something to respond to.
    messages = list(conversation_history)
    if not messages:
        messages = [{"role": "user", "content": f"[Scene begins. Situation: {situation}]"}]

    try:
        response = client.messages.create(
            model=AI_MODEL,
            max_tokens=256,  # ~3 sentences max; hard cap for speed and cost
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text.strip()
    except anthropic.APIConnectionError as e:
        logging.error("Anthropic connection error: %s", e)
        raise RuntimeError("Could not reach Anthropic API. Check your network connection.") from e
    except anthropic.AuthenticationError as e:
        logging.error("Anthropic auth error: %s", e)
        raise RuntimeError("Invalid ANTHROPIC_API_KEY. Check your .env file.") from e
    except anthropic.RateLimitError as e:
        logging.error("Anthropic rate limit: %s", e)
        raise RuntimeError("Anthropic rate limit hit; try again in a moment.") from e
    except Exception as e:
        logging.exception("Unexpected error calling Anthropic API")
        raise RuntimeError(f"Dialogue generation failed: {e!s}") from e
