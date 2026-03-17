"""
Campaign system presets and registry helpers.

The parser stays system-agnostic and extracts universal structures. These presets
describe how a campaign should be flavored elsewhere in the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_CAMPAIGN_SYSTEM_ID = "dnd"


@dataclass(frozen=True)
class CampaignSystemPreset:
    id: str
    label: str
    rules_flavor: str
    skill_check_terminology: dict[str, str]
    encounter_assumptions: str
    thematic_guidance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "rules_flavor": self.rules_flavor,
            "skill_check_terminology": dict(self.skill_check_terminology),
            "encounter_assumptions": self.encounter_assumptions,
            "thematic_guidance": self.thematic_guidance,
        }


_CAMPAIGN_SYSTEMS: dict[str, CampaignSystemPreset] = {
    "dnd": CampaignSystemPreset(
        id="dnd",
        label="D&D",
        rules_flavor="Heroic fantasy with class-driven abilities, spellcasting, and escalating encounter stakes.",
        skill_check_terminology={
            "skill_term": "skill",
            "check_term": "ability check",
            "difficulty_term": "DC",
        },
        encounter_assumptions="Set-piece encounters, tactical combat, and clear monster-threat expectations are common.",
        thematic_guidance="Lean into adventure hooks, faction conflict, treasure, exploration, and cinematic hero moments.",
    ),
    "pathfinder": CampaignSystemPreset(
        id="pathfinder",
        label="Pathfinder",
        rules_flavor="High-detail fantasy adventure with crunchy tactical options, feats, and strong character builds.",
        skill_check_terminology={
            "skill_term": "skill",
            "check_term": "check",
            "difficulty_term": "DC",
        },
        encounter_assumptions="Expect structured encounters, robust stat blocks, and mechanical precision around challenge.",
        thematic_guidance="Support layered prep, faction agendas, rich setting detail, and tactically expressive conflicts.",
    ),
    "coc": CampaignSystemPreset(
        id="coc",
        label="Call of Cthulhu",
        rules_flavor="Investigative cosmic horror centered on clues, sanity pressure, fragile investigators, and dread.",
        skill_check_terminology={
            "skill_term": "skill",
            "check_term": "roll",
            "difficulty_term": "difficulty level",
        },
        encounter_assumptions="Combat is dangerous, mysteries unfold through clues, and psychological consequences matter.",
        thematic_guidance="Favor ominous atmosphere, secrets, helplessness, clue chains, and unsettling revelations.",
    ),
    "homebrew": CampaignSystemPreset(
        id="homebrew",
        label="Homebrew",
        rules_flavor="Flexible custom campaign with system-defined terminology, assumptions, and table norms set by the GM.",
        skill_check_terminology={
            "skill_term": "skill or trait",
            "check_term": "check or roll",
            "difficulty_term": "difficulty",
        },
        encounter_assumptions="Do not assume specific combat math, stat formats, or class structures unless the source text says so.",
        thematic_guidance="Follow the campaign text closely and preserve its own terminology, tone, and world logic.",
    ),
}

_SYSTEM_ALIASES = {
    "d&d": "dnd",
    "dnd5e": "dnd",
    "5e": "dnd",
    "pathfinder2e": "pathfinder",
    "pathfinder1e": "pathfinder",
    "callofcthulhu": "coc",
    "call_of_cthulhu": "coc",
    "call-of-cthulhu": "coc",
}


def normalize_campaign_system(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    if not raw:
        return DEFAULT_CAMPAIGN_SYSTEM_ID
    if raw in _CAMPAIGN_SYSTEMS:
        return raw
    alias = _SYSTEM_ALIASES.get(raw)
    if alias in _CAMPAIGN_SYSTEMS:
        return alias
    normalized = raw.replace("_", "")
    alias = _SYSTEM_ALIASES.get(normalized)
    if alias in _CAMPAIGN_SYSTEMS:
        return alias
    return DEFAULT_CAMPAIGN_SYSTEM_ID


def get_campaign_system_preset(system_id: Any) -> dict[str, Any]:
    normalized = normalize_campaign_system(system_id)
    return _CAMPAIGN_SYSTEMS[normalized].to_dict()


def list_campaign_system_presets() -> list[dict[str, Any]]:
    return [preset.to_dict() for preset in _CAMPAIGN_SYSTEMS.values()]
