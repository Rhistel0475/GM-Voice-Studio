"""
Co-DM NPC Generator — Sprint 4
================================
Streams a full NPC profile from Claude using the Anthropic SDK streaming API.
Entry point: generate_npc_stream(genre, location, name, role) -> Iterator[str]
"""
from typing import Iterator

_NPC_SYSTEM_PROMPT = """\
You are a masterful character development assistant for tabletop RPGs.
Given a genre, location, character name or type, and their role in the story,
output a vivid and immediately usable NPC profile.

Use exactly these five markdown headers (in this order):
## Basic Information & Stats
## Physical Appearance
## Personality Traits & Quirks
## Motivations & Driving Forces
## Secrets / Plot Hooks

Rules:
- Stats (AC, HP, attacks) should fit the genre and role — estimate believably.
- Physical appearance: sensory, cinematic details a GM can describe at the table.
- Personality: 3–4 specific traits, not generic adjectives.
- Motivations: the one thing this character wants most, and what they fear.
- Secrets / Plot Hooks: 2–3 hooks that could drive a session forward.
- Total length: fit on one side of an index card when printed. Be dense, not padded.
- Format for quick reading behind a DM screen."""


def generate_npc_stream(
    genre: str,
    location: str,
    name: str,
    role: str,
) -> Iterator[str]:
    """
    Stream NPC profile text tokens from Claude.
    Yields raw text chunks (strings) as they arrive.
    """
    from app.infrastructure.llm.anthropic_client import get_client as _get_client

    user_message = (
        f"Genre: {genre.strip()}\n"
        f"Location: {location.strip()}\n"
        f"Character: {name.strip()}\n"
        f"Role: {role.strip()}"
    )

    client = _get_client()
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_NPC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text
