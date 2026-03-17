"""
Type-specific structured extractors for parsed sections.
"""
from app.services.parsing.extractors.npc import extract_npcs
from app.services.parsing.extractors.location import extract_locations
from app.services.parsing.extractors.scene_seed import extract_scene_seeds
from app.services.parsing.extractors.codex import extract_codex_entries
from app.services.parsing.extractors.quest import extract_quests
from app.services.parsing.extractors.item import extract_items

__all__ = [
    "extract_npcs",
    "extract_locations",
    "extract_scene_seeds",
    "extract_codex_entries",
    "extract_quests",
    "extract_items",
]
