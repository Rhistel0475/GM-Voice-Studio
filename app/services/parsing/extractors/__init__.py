"""
Type-specific structured extractors for parsed sections.
"""
from app.services.parsing.extractors.npc import extract_npcs
from app.services.parsing.extractors.location import extract_locations
from app.services.parsing.extractors.scene_seed import extract_scene_seeds
from app.services.parsing.extractors.codex import extract_codex_entries
from app.services.parsing.extractors.quest import extract_quests
from app.services.parsing.extractors.item import extract_items
from app.services.parsing.extractors.clue import extract_clues
from app.services.parsing.extractors.secret import extract_secrets
from app.services.parsing.extractors.rumor import extract_rumors
from app.services.parsing.extractors.read_aloud import extract_read_aloud
from app.services.parsing.extractors.consequence import extract_consequences
from app.services.parsing.extractors.reward import extract_rewards
from app.services.parsing.extractors.hook import extract_hooks

__all__ = [
    "extract_npcs",
    "extract_locations",
    "extract_scene_seeds",
    "extract_codex_entries",
    "extract_quests",
    "extract_items",
    "extract_clues",
    "extract_secrets",
    "extract_rumors",
    "extract_read_aloud",
    "extract_consequences",
    "extract_rewards",
    "extract_hooks",
]
