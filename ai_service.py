"""
AI service: NPC dialogue generation and adventure parsing via Anthropic Claude.
Mirrors the pattern of tts_service.py — lazy client init, clear public API.
"""
import json
import logging
from typing import Optional

import anthropic

from config import AI_MODEL, ANTHROPIC_API_KEY, MAX_ADVENTURE_CHARS

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


def extract_text_from_file(path: str, suffix: str) -> str:
    """
    Extract plain text from PDF, DOCX, or text files.
    Lazy-imports optional libraries so they're not required at startup.

    Args:
        path: Absolute path to the temp file.
        suffix: Lowercase file extension including dot (e.g. ".pdf", ".docx").

    Returns:
        Raw text string.

    Raises:
        RuntimeError: If a required library is missing or extraction fails.
    """
    suffix = suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as e:
            raise RuntimeError(
                "pdfplumber is required for PDF parsing. Run: pip install pdfplumber"
            ) from e
        try:
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages)
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e!s}") from e

    elif suffix in (".docx",):
        try:
            import docx
        except ImportError as e:
            raise RuntimeError(
                "python-docx is required for DOCX parsing. Run: pip install python-docx"
            ) from e
        try:
            doc = docx.Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            raise RuntimeError(f"DOCX extraction failed: {e!s}") from e

    else:
        # Plain text, markdown, or any other text format
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Text file read failed: {e!s}") from e


def parse_adventure(text: str) -> dict:
    """
    Use Claude to extract read-aloud passages and NPCs from adventure text.

    Args:
        text: Raw adventure text (will be truncated to MAX_ADVENTURE_CHARS).

    Returns:
        {"read_alouds": [...], "npcs": [...]}
        Each read_aloud: {"title": str, "text": str, "scene": str}
        Each npc: {"name": str, "personality": str, "faction": str, "description": str, "scene": str}

    Raises:
        RuntimeError: on API error or JSON parse failure.
    """
    client = _get_client()
    truncated = text[:MAX_ADVENTURE_CHARS]

    system_prompt = (
        "You are a tabletop RPG game prep assistant. "
        "Extract structured data from adventure module text and return ONLY valid JSON — "
        "no markdown, no explanation, no preamble. Just the JSON object."
    )

    user_prompt = (
        "Analyze this adventure text and extract two things:\n\n"
        '1. "read_alouds": Boxed text or passages meant to be read aloud to players. '
        "Look for: text marked as boxed, italicized description blocks, passages starting with "
        "'Read the following aloud', or descriptive scene-setting text written in second person. "
        'Each item: {"title": "brief scene name (5 words max)", "text": "the exact passage", "scene": "chapter or area name if known"}\n\n'
        '2. "npcs": Named non-player characters, monsters with personalities, and key figures. '
        'Each item: {"name": "full name or title", "personality": "personality traits, motivation, and speech style in 1-3 sentences", '
        '"faction": "organization or group affiliation if any", "description": "brief physical description", "scene": "where they appear"}\n\n'
        "Return JSON in exactly this format:\n"
        '{"read_alouds": [...], "npcs": [...]}\n\n'
        "Adventure text:\n---\n"
        f"{truncated}\n---"
    )

    try:
        response = client.messages.create(
            model=AI_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if Claude wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        # Normalise: ensure both keys exist
        result.setdefault("read_alouds", [])
        result.setdefault("npcs", [])
        return result
    except json.JSONDecodeError as e:
        logging.error("Claude returned non-JSON for adventure parse: %s", e)
        raise RuntimeError("Claude returned invalid JSON. Try a shorter or cleaner text input.") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("Could not reach Anthropic API.") from e
    except anthropic.AuthenticationError as e:
        raise RuntimeError("Invalid ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError("Anthropic rate limit hit; try again in a moment.") from e
    except Exception as e:
        logging.exception("Adventure parse failed")
        raise RuntimeError(f"Adventure parsing failed: {e!s}") from e
