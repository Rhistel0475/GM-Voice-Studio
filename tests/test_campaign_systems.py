from app.domain.campaign.systems import (
    DEFAULT_CAMPAIGN_SYSTEM_ID,
    get_campaign_system_preset,
    list_campaign_system_presets,
    normalize_campaign_system,
)


def test_normalize_campaign_system_accepts_aliases_and_defaults():
    assert normalize_campaign_system("D&D") == "dnd"
    assert normalize_campaign_system("call_of_cthulhu") == "coc"
    assert normalize_campaign_system("unknown-system") == DEFAULT_CAMPAIGN_SYSTEM_ID


def test_list_campaign_system_presets_contains_supported_systems():
    systems = list_campaign_system_presets()

    assert [system["id"] for system in systems] == [
        "dnd",
        "pathfinder",
        "coc",
        "homebrew",
    ]


def test_get_campaign_system_preset_returns_homebrew_fallback():
    preset = get_campaign_system_preset("not-real")

    assert preset["id"] == "homebrew"
    assert "Do not assume" in preset["encounter_assumptions"]
