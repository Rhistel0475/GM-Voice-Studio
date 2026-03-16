"""
AI service: NPC dialogue generation and adventure parsing via Anthropic Claude.
Mirrors the pattern of tts_service.py — lazy client init, clear public API.
"""
import json
import logging
from typing import Any, Optional

try:
    import anthropic
except ModuleNotFoundError:
    class _AnthropicImportFallback:
        Anthropic = None

        class APIConnectionError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        class RateLimitError(Exception):
            pass

    anthropic = _AnthropicImportFallback()

from app.core.config import AI_MODEL, ANTHROPIC_API_KEY, MAX_ADVENTURE_CHARS

_client: Optional[Any] = None


def _get_client() -> Any:
    global _client
    if _client is None:
        if getattr(anthropic, "Anthropic", None) is None:
            raise RuntimeError("Anthropic SDK is not installed. Install the 'anthropic' package in the active environment.")
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def build_npc_system_prompt(
    npc_name: str,
    personality: str,
    faction: str = "",
    situation: str = "",
    session_context: str = "",
    npc_memory_summary: str = "",
) -> str:
    """
    Build a system prompt that casts Claude as an NPC character.
    Enforces short responses (1-3 sentences) for live table use.
    """
    faction_line = f"\nFaction/Allegiance: {faction.strip()}" if faction.strip() else ""
    situation_line = f"\nCurrent situation: {situation.strip()}" if situation.strip() else ""
    npc_memory_line = (
        f"\nNPC memory from this session:\n{npc_memory_summary.strip()}"
        if npc_memory_summary.strip()
        else ""
    )
    session_context_line = (
        f"\nRecent session events:\n{session_context.strip()}"
        if session_context.strip()
        else ""
    )
    return (
        f"You are {npc_name.strip()}, a character in a tabletop RPG session. "
        f"Personality: {personality.strip()}{faction_line}{situation_line}"
        f"{npc_memory_line}{session_context_line}\n\n"
        "Speak ONLY as this character. Do NOT break character. Do NOT explain or narrate. "
        "Do NOT say you are an AI. Respond as if you are actually speaking the words out loud "
        "at the game table. Keep every response to 1-3 sentences maximum — this is live "
        "dialogue, not prose. Use the character's voice, vocabulary, and emotional state. "
        "Stay consistent with the session history and how the characters have treated each other."
    )


def generate_dialogue(
    npc_name: str,
    personality: str,
    situation: str,
    conversation_history: list[dict],
    faction: str = "",
    session_context: str = "",
    npc_memory_summary: str = "",
) -> str:
    """
    Generate a short in-character NPC line using Claude.

    Args:
        npc_name: NPC's name (e.g. "Captain Aldric Vane")
        personality: Brief description (e.g. "gruff, loyal to the crown, hiding a secret")
        situation: What is happening right now (e.g. "Players are demanding to pass the gate")
        conversation_history: list of {"role": "user"|"assistant", "content": "..."}
        faction: Optional allegiance (e.g. "Silver Court Mages")
        session_context: Important recent session events formatted for the model
        npc_memory_summary: NPC-specific session memory lines

    Returns:
        The NPC's spoken line as a string.

    Raises:
        RuntimeError: on API connection, auth, rate limit, or unexpected errors.
    """
    client = _get_client()
    system_prompt = build_npc_system_prompt(
        npc_name,
        personality,
        faction,
        situation,
        session_context,
        npc_memory_summary,
    )

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


def _parse_json_payload(raw_text: str) -> Any:
    raw = (raw_text or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


def analyze_session_context(
    transcript_entries: list[str],
    scene_title: str = "",
    scene_summary: str = "",
    npcs: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, str]]:
    """
    Analyze recent live-session transcript entries and return actionable GM suggestions.

    Suggestion types:
      - npc_dialogue
      - narration
      - rule_check
      - lore_reference
    """
    client = _get_client()
    transcript_lines = [str(entry).strip() for entry in transcript_entries if str(entry).strip()]
    if not transcript_lines:
        return []

    npc_rows: list[str] = []
    npc_name_set: set[str] = set()
    for npc in npcs or []:
        if not isinstance(npc, dict):
            continue
        name = str(npc.get("name") or "").strip()
        if not name:
            continue
        npc_name_set.add(name)
        role = str(npc.get("role") or "").strip()
        summary = str(npc.get("description") or npc.get("personality") or "").strip()
        parts = [name]
        if role:
            parts.append(f"role={role}")
        if summary:
            parts.append(f"notes={summary}")
        npc_rows.append("- " + " | ".join(parts))

    system_prompt = (
        "You are Session Assistant for a tabletop RPG GM. "
        "Analyze recent table conversation and return ONLY valid JSON. "
        "Keep suggestions practical, brief, and immediately usable during live play."
    )
    user_prompt = (
        "Return JSON with this exact shape:\n"
        '{'
        '"suggestions": ['
        '{"type":"npc_dialogue|narration|rule_check|lore_reference",'
        '"title":"short label",'
        '"text":"short actionable suggestion",'
        '"npc_name":"exact NPC name or empty string",'
        '"spoken_text":"what should be spoken aloud if relevant, else empty string",'
        '"action_prompt":"question to send to the GM assistant for rule/lore help, else empty string"}'
        "]}\n\n"
        "Rules:\n"
        "- Return 0 to 4 suggestions.\n"
        "- Prefer concrete opportunities over vague advice.\n"
        "- npc_dialogue must use an NPC name exactly from the provided NPC list.\n"
        "- narration should be one or two spoken sentences.\n"
        "- rule_check should identify a likely check, save, or ruling question.\n"
        "- lore_reference should identify a likely lore callback or world detail.\n"
        "- Do not include markdown fences.\n\n"
        f"Scene title: {scene_title.strip() or 'Unknown scene'}\n"
        f"Scene summary: {scene_summary.strip() or 'No summary provided.'}\n"
        "NPCs in play:\n"
        f"{chr(10).join(npc_rows) if npc_rows else '- None provided'}\n\n"
        "Recent transcript entries:\n"
        + "\n".join(f"- {line}" for line in transcript_lines[-8:])
    )

    try:
        response = client.messages.create(
            model=AI_MODEL,
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        payload = _parse_json_payload(raw)
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
        logging.exception("Unexpected error calling Anthropic for session analysis")
        raise RuntimeError(f"Session analysis failed: {e!s}") from e

    raw_suggestions = payload.get("suggestions") if isinstance(payload, dict) else payload
    if not isinstance(raw_suggestions, list):
        return []

    allowed_types = {"npc_dialogue", "narration", "rule_check", "lore_reference"}
    normalized: list[dict[str, str]] = []

    for item in raw_suggestions[:4]:
        if not isinstance(item, dict):
            continue
        suggestion_type = str(item.get("type") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if suggestion_type not in allowed_types or not text:
            continue

        npc_name = str(item.get("npc_name") or "").strip()
        if suggestion_type == "npc_dialogue" and npc_name_set and npc_name not in npc_name_set:
            continue

        normalized.append({
            "type": suggestion_type,
            "title": str(item.get("title") or suggestion_type.replace("_", " ").title()).strip(),
            "text": text,
            "npc_name": npc_name,
            "spoken_text": str(item.get("spoken_text") or "").strip(),
            "action_prompt": str(item.get("action_prompt") or "").strip(),
        })

    return normalized


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


def ai_full_parse(text: str) -> dict:
    """
    Use the staged parsing pipeline to extract a complete campaign data object from adventure text.
    Falls back to single-shot Claude extraction if the pipeline fails.
    Returns structured data for NPCs, party, scenes, locations, reveals, codex_entries, relationships.
    """
    try:
        from app.services.parsing.pipeline import run_parsing_pipeline
        result = run_parsing_pipeline(text)
        for key in ("title", "summary"):
            result.setdefault(key, "")
        for key in ("npcs", "party", "scenes", "locations", "reveals", "items"):
            result.setdefault(key, [])
        result.setdefault("codex_entries", [])
        result.setdefault("relationships", [])
        for npc in result.get("npcs", []):
            npc.setdefault("hp", "")
            npc.setdefault("ac", 0)
            npc.setdefault("cr", "")
        for scene in result.get("scenes", []):
            scene.setdefault("difficulty", "")
            scene.setdefault("rewards", "")
            scene.setdefault("notes", "")
        return result
    except Exception as e:
        logging.warning("Parsing pipeline failed, falling back to single-shot parse: %s", e)
        return _ai_full_parse_fallback(text)


def _ai_full_parse_fallback(text: str) -> dict:
    """Legacy single-shot Claude extraction when pipeline is unavailable or fails."""
    from app.infrastructure.llm.anthropic_client import get_client
    client = get_client()
    # Cap at 60k chars to leave ample room for the large JSON response within 8192 tokens
    truncated = text[:min(MAX_ADVENTURE_CHARS, 60_000)]

    system_prompt = (
        "You are a tabletop RPG game prep assistant. "
        "Extract structured data from adventure module text and return ONLY valid JSON — "
        "no markdown, no explanation, no preamble. Just the JSON object."
    )

    user_prompt = (
        "Extract GM prep data from this adventure text. "
        "Return ONLY compact JSON — no markdown, no prose. "
        "Keep ALL string values short (≤15 words each). "
        "Limits: max 10 npcs, max 10 scenes, max 8 locations, max 8 reveals, max 8 items.\n\n"
        'JSON keys required:\n'
        '"title": adventure title\n'
        '"summary": 2-sentence premise\n'
        '"npcs": [{"name","role"(villain|ally|quest-giver|neutral),"personality","faction","motivation","secrets","hp"(e.g."45" or "3d8"),"ac"(int),"cr"(e.g."CR 3")}]\n'
        '"party": [{"name","class_","race","level"(int),"hp","ac"(int)}] or []\n'
        '"scenes": [{"title","act","type"(combat|social|exploration|mystery),"read_aloud"(≤30 words),"npcs":[str],"location","difficulty"(easy|medium|hard|deadly|none),"rewards"(≤15 words),"notes"(≤20 words)}]\n'
        '"locations": [{"name","description"}]\n'
        '"reveals": [{"name","when","type"(hook|secret|clue|twist)}]\n'
        '"items": [{"name","description"(≤15 words),"scene"(scene title or ""),"magical"(true|false)}]\n\n'
        "Adventure text:\n---\n"
        f"{truncated}\n---"
    )

    try:
        response = client.messages.create(
            model=AI_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        logging.debug("ai_full_parse raw response (%d chars): %s...", len(raw), raw[:200])

        # Strip markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # If the JSON was truncated mid-stream, find the last complete top-level key
        brace_start = raw.find("{")
        if brace_start != -1 and not raw.endswith("}"):
            raw = raw[:raw.rfind(",")].rstrip() + "\n}"
            logging.warning("ai_full_parse: JSON was truncated; attempted auto-close.")

        result = json.loads(raw)
        for key in ("title", "summary"):
            result.setdefault(key, "")
        for key in ("npcs", "party", "scenes", "locations", "reveals", "items"):
            result.setdefault(key, [])
        result.setdefault("codex_entries", [])
        result.setdefault("relationships", [])
        for npc in result.get("npcs", []):
            npc.setdefault("hp", "")
            npc.setdefault("ac", 0)
            npc.setdefault("cr", "")
        for scene in result.get("scenes", []):
            scene.setdefault("difficulty", "")
            scene.setdefault("rewards", "")
            scene.setdefault("notes", "")
        return result
    except json.JSONDecodeError as e:
        logging.error("Claude returned non-JSON for ai_full_parse: %s", e)
        raise RuntimeError("Claude returned invalid JSON. Try a shorter or cleaner text input.") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("Could not reach Anthropic API.") from e
    except anthropic.AuthenticationError as e:
        raise RuntimeError("Invalid ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError("Anthropic rate limit hit; try again in a moment.") from e
    except Exception as e:
        logging.exception("ai_full_parse failed")
        raise RuntimeError(f"Adventure AI parse failed: {e!s}") from e


def assign_images_to_entities(images: list[dict], campaign: dict, total_pages: int) -> list[dict]:
    """
    Text-only Claude call to assign each extracted image (identified by page number)
    to the most likely NPC, scene, or location in the campaign.

    Args:
        images: [{"idx": int, "page": int, "url": str}]
        campaign: parsed campaign dict with npcs, scenes, locations
        total_pages: total pages in the source PDF

    Returns:
        Updated list with "type" and "assigned_to" and "label" added to each entry.
        Also sets image_url on matching entities in campaign (mutates campaign).
    """
    from app.infrastructure.llm.anthropic_client import get_client
    client = get_client()

    npc_names = [n["name"] for n in campaign.get("npcs", [])]
    scene_titles = [s["title"] for s in campaign.get("scenes", [])]
    location_names = [l["name"] for l in campaign.get("locations", [])]

    img_list = [{"idx": img["idx"], "page": img["page"]} for img in images]

    prompt = (
        f"Adventure: \"{campaign.get('title', 'Unknown')}\", {total_pages} pages.\n"
        f"NPCs: {npc_names}\n"
        f"Scenes: {scene_titles}\n"
        f"Locations: {location_names}\n\n"
        f"Images extracted from the PDF by page:\n{json.dumps(img_list)}\n\n"
        "For each image, decide:\n"
        "- type: portrait | map | handout | illustration | decoration\n"
        "- assigned_to: exact NPC name, scene title, or location name from the lists above — or null\n"
        "- label: 5-word description of what the image likely shows\n\n"
        "Return ONLY a JSON array, one entry per image, in order:\n"
        '[{"idx":1,"type":"...","assigned_to":"...","label":"..."},...]\n'
        "No markdown, no prose."
    )

    try:
        response = client.messages.create(
            model=AI_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        assignments = json.loads(raw)
    except Exception as e:
        logging.warning("assign_images_to_entities failed: %s", e)
        # Return images without assignments rather than failing the whole parse
        return [{"idx": img["idx"], "page": img["page"], "url": img["url"],
                 "type": "illustration", "assigned_to": None, "label": ""} for img in images]

    # Merge assignments back onto original image list
    assign_map = {a["idx"]: a for a in assignments if isinstance(a, dict)}
    result = []
    for img in images:
        a = assign_map.get(img["idx"], {})
        entry = {
            "idx": img["idx"],
            "page": img["page"],
            "url": img["url"],
            "type": a.get("type", "illustration"),
            "assigned_to": a.get("assigned_to") or None,
            "label": a.get("label", ""),
        }
        result.append(entry)

    # Stamp image_url onto matching NPCs, scenes, and locations (first match wins)
    assigned_to_used: set = set()
    for entry in result:
        target = entry.get("assigned_to")
        if not target or target in assigned_to_used:
            continue
        for npc in campaign.get("npcs", []):
            if npc["name"] == target and "image_url" not in npc:
                npc["image_url"] = entry["url"]
                assigned_to_used.add(target)
                break
        for scene in campaign.get("scenes", []):
            if scene["title"] == target and "image_url" not in scene:
                scene["image_url"] = entry["url"]
                assigned_to_used.add(target)
                break
        for loc in campaign.get("locations", []):
            if loc["name"] == target and "image_url" not in loc:
                loc["image_url"] = entry["url"]
                assigned_to_used.add(target)
                break

    return result


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
