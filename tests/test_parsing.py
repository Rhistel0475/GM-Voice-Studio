"""Unit tests for the document parsing pipeline (normalize, sections, dedupe, extractors)."""
import pytest
from unittest.mock import patch

from app.services.parsing.normalize import normalize_text
from app.services.parsing.sections import split_into_sections
from app.services.parsing.models import SectionChunk
from app.services.parsing.classify import classify_chunks
from app.services.parsing.dedupe import (
    dedupe_npcs,
    dedupe_locations,
    dedupe_scenes,
    dedupe_codex_entries,
)
from app.services.parsing.relationships import extract_relationships
from app.services.parsing.candidates import build_candidates
from app.services.parsing.aggregation import fuse_candidates
from app.services.parsing.coverage_audit import audit_coverage
from app.services.parsing.importance import score_importance


class TestNormalize:
    """Tests for normalize_text."""

    def test_empty_returns_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   \n\n  ") == ""

    def test_collapses_blank_lines(self):
        out = normalize_text("a\n\n\n\nb")
        assert out == "a\n\nb"

    def test_collapses_spaces(self):
        out = normalize_text("a   b\t\tc")
        assert "  " not in out
        assert "\t" not in out

    def test_max_chars_truncates_at_word(self):
        text = "one two three four five"
        out = normalize_text(text, max_chars=12)
        # 12 chars: "one two three" is 15, so truncate to last full word within 12 -> "one two"
        assert out == "one two"
        assert len(out) <= 12

    def test_preserves_single_newlines(self):
        out = normalize_text("line1\nline2")
        assert out == "line1\nline2"


class TestSections:
    """Tests for split_into_sections."""

    def test_empty_returns_empty(self):
        assert split_into_sections("") == []
        assert split_into_sections("   ") == []

    def test_atx_headings_split(self):
        text = "# Chapter One\n\nBody one.\n\n## Scene A\n\nBody A."
        chunks = split_into_sections(text)
        assert len(chunks) >= 1
        headings = [c.heading for c in chunks]
        assert "Chapter One" in headings
        assert any("Scene A" in c.heading for c in chunks)

    def test_single_chunk_no_headings(self):
        text = "Just a paragraph with no markdown headings."
        chunks = split_into_sections(text)
        assert len(chunks) == 1
        assert chunks[0].heading == ""
        assert chunks[0].level == 0
        assert "paragraph" in chunks[0].body

    def test_bold_heading_fallback(self):
        text = "**Villain: The Shadow**\n\nHe lurks in the dark."
        chunks = split_into_sections(text)
        assert len(chunks) >= 1
        assert any("Shadow" in c.heading or "Villain" in c.heading for c in chunks)

    def test_section_metadata_and_semantic_subchunks(self):
        text = """
<<<DOCUMENT: Lost Mine PDF>>>
# Goblin Ambush

Read Aloud: "Two goblins burst from the thicket."

Goblin Boss
AC 15
HP 27

Quest Hook: Recover the stolen wagon.
""".strip()
        chunks = split_into_sections(text)
        assert len(chunks) >= 3
        assert all(chunk.document_id == "lost_mine_pdf" for chunk in chunks)
        assert all(chunk.page_number == 1 for chunk in chunks)
        assert any(chunk.subheading == "Read Aloud" and chunk.chunk_type_guess == "boxed_text" for chunk in chunks)
        assert any(chunk.subheading == "Stat Block" and chunk.chunk_type_guess == "stat_block" for chunk in chunks)
        assert any(chunk.subheading == "Quest Hook" and chunk.chunk_type_guess == "quest_section" for chunk in chunks)

    def test_front_matter_cleanup_drops_cover_credits_and_keeps_intro(self):
        text = """
A DARK AND
STORMY KNIGHT
A Short Adventure for Four
1st-Level Player Characters

CREDITS
Design:
Editing:
Typesetting:

Owen K.C. Stephens
Penny Williams
Nancy Walker

This material is protected under the copyright laws of the United
States of America.
visit www.wizards.com/dnd

The unusually violent storms in these parts often drive
motley collections of intelligent beings to take shelter
together for a time.

PREPARATION
You (the DM) need the D&D core rulebooks.
""".strip()
        chunks = split_into_sections(text)
        combined = "\n".join(
            f"{chunk.heading}\n{chunk.subheading}\n{chunk.body}".strip()
            for chunk in chunks
        )

        assert "CREDITS" not in combined
        assert "wizards.com" not in combined.lower()
        assert "copyright laws" not in combined.lower()
        assert "violent storms" in combined.lower()
        assert any(chunk.heading == "PREPARATION" for chunk in chunks)

    def test_front_matter_cleanup_removes_toc_and_running_headers(self):
        text = """
Table of Contents
Overview ............................................................ 4
Background ......................................................... 4
Set Up ............................................................. 5
\f
What's in the Cellar?

4

What's in the Cellar?
Will your investigators escape from the mysterious tomb?

Overview

This demonstration scenario is intended for use at conventions.

Background

Arthur Blackwood has contacted his cousin in a desperate plea.

Call of Cthulhu Demo Game
\f
What's in the Cellar?

5

What's in the Cellar?

Set Up

The investigators arrive at the Blackwood house.

Call of Cthulhu Demo Game
""".strip()
        chunks = split_into_sections(text)
        combined = "\n".join(
            f"{chunk.heading}\n{chunk.subheading}\n{chunk.body}".strip()
            for chunk in chunks
        )
        headings = [chunk.heading for chunk in chunks if chunk.heading]

        assert "Table of Contents" not in combined
        assert "................................" not in combined
        assert "Call of Cthulhu Demo Game" not in combined
        assert "What's in the Cellar?" not in combined
        assert "Overview" in headings
        assert "Background" in headings
        assert "Set Up" in headings


class TestClassification:
    def test_classify_chunks_uses_heuristics_when_llm_unavailable(self):
        chunks = [
            SectionChunk(
                heading="Town Guard",
                level=1,
                body="Guard Captain. AC 16. HP 45. Gruff and suspicious.",
                chunk_type_guess="stat_block",
            ),
            SectionChunk(
                heading="Quest Hook",
                level=2,
                body="The investigators must recover the silver idol from the flooded crypt.",
                chunk_type_guess="quest_section",
            ),
        ]

        with patch("app.services.parsing.classify._get_client", side_effect=RuntimeError("offline")):
            classify_chunks(chunks)

        assert "npc" in chunks[0].content_types
        assert "encounter" in chunks[0].content_types
        assert chunks[1].content_type in {"quest", "mixed"}
        assert "quest" in chunks[1].content_types

    def test_location_section_without_scene_cues_stays_location_only(self):
        chunk = SectionChunk(
            heading="Roots and Remedies",
            level=2,
            body=(
                "Creeping ivy and full window boxes cover the facade of the rugged-looking, "
                "two-story shop bearing the faded sign Roots and Remedies."
            ),
            chunk_type_guess="location_section",
        )

        with patch("app.services.parsing.classify._get_client", side_effect=RuntimeError("offline")):
            classify_chunks([chunk])

        assert chunk.content_type == "location"
        assert chunk.content_types == ["location"]

    def test_location_section_with_entry_cues_keeps_scene_label(self):
        chunk = SectionChunk(
            heading="Hightower Main Entrance",
            level=2,
            body=(
                "Read or paraphrase the following when the PCs come within 20 feet of Hightower. "
                "The heavy door swings shut on its own unless propped open."
            ),
            chunk_type_guess="location_section",
        )

        with patch("app.services.parsing.classify._get_client", side_effect=RuntimeError("offline")):
            classify_chunks([chunk])

        assert chunk.content_type in {"location", "mixed", "scene"}
        assert "location" in chunk.content_types
        assert "scene" in chunk.content_types


class TestDedupe:
    """Tests for deduplication."""

    def test_dedupe_npcs_by_name(self):
        npcs = [
            {"name": "Bob", "role": "ally", "confidence": 0.9},
            {"name": "Bob", "role": "neutral", "confidence": 0.5},
        ]
        out = dedupe_npcs(npcs)
        assert len(out) == 1
        assert out[0]["name"] == "Bob"
        assert out[0]["confidence"] == 0.9

    def test_dedupe_locations_by_name(self):
        locs = [
            {"name": "Tavern", "description": "A cozy inn", "confidence": 0.8},
            {"name": "Tavern", "description": "The local pub", "confidence": 0.6},
        ]
        out = dedupe_locations(locs)
        assert len(out) == 1
        assert out[0]["confidence"] == 0.8

    def test_dedupe_scenes_merge_npcs(self):
        scenes = [
            {"title": "Encounter", "npcs": ["Alice"], "confidence": 0.9},
            {"title": "Encounter", "npcs": ["Bob"], "confidence": 0.7},
        ]
        out = dedupe_scenes(scenes)
        assert len(out) == 1
        assert set(out[0]["npcs"]) == {"Alice", "Bob"}

    def test_dedupe_codex_by_id(self):
        entries = [
            {"id": "codex_foo", "title": "Foo", "confidence": 0.9},
            {"id": "codex_foo", "title": "Foo", "confidence": 0.5},
        ]
        out = dedupe_codex_entries(entries)
        assert len(out) == 1

    def test_dedupe_npcs_merges_alias_variant(self):
        npcs = [
            {
                "name": "Gavel Thuldrin Kreed",
                "role": "villain",
                "description": "Leader of the Lumber Consortium.",
                "confidence": 0.8,
                "source": {"document_id": "doc_1", "heading": "Authority Figures", "page_number": 1},
            },
            {
                "name": "Thuldrin Kreed",
                "role": "",
                "description": "Ruthless consortium boss.",
                "confidence": 0.9,
                "source": {"document_id": "doc_1", "heading": "Authority Figures", "page_number": 1},
            },
        ]
        out = dedupe_npcs(npcs)
        assert len(out) == 1
        assert out[0]["name"] == "Thuldrin Kreed"
        assert "consortium" in out[0]["description"].lower()

    def test_dedupe_locations_merges_canonical_variant_but_keeps_distinct_subplace(self):
        locations = [
            {
                "name": "Joseph Klein's Law Office",
                "description": "High-rise office with skyline view.",
                "confidence": 0.8,
                "source": {"document_id": "doc_1", "heading": "Set Up", "page_number": 1},
            },
            {
                "name": "Joseph Klein's Law Office, New York",
                "description": "Office in New York with a striking skyline view.",
                "confidence": 0.9,
                "source": {"document_id": "doc_1", "heading": "Set Up", "page_number": 1},
            },
            {
                "name": "Hightower Main Entrance",
                "description": "The heavy front doorway.",
                "confidence": 0.85,
            },
            {
                "name": "Hightower",
                "description": "The broader tor complex.",
                "confidence": 0.8,
            },
        ]
        out = dedupe_locations(locations)
        names = {entry["name"] for entry in out}
        assert len(out) == 3
        assert "Joseph Klein's Law Office, New York" in names
        assert "Hightower Main Entrance" in names
        assert "Hightower" in names

    def test_dedupe_scenes_merges_parenthetical_title_variant(self):
        scenes = [
            {
                "title": "Rat Race",
                "npcs": ["Giant Rat"],
                "read_aloud": "",
                "confidence": 0.7,
                "source": {"document_id": "doc_1", "heading": "Rat Race", "page_number": 2},
            },
            {
                "title": "Rat Race (EL 2)",
                "npcs": [],
                "read_aloud": "Eight rats swarm into the chamber.",
                "notes": "Occurs in area 2.",
                "confidence": 0.9,
                "source": {"document_id": "doc_1", "heading": "Rat Race", "page_number": 2},
            },
        ]
        out = dedupe_scenes(scenes)
        assert len(out) == 1
        assert out[0]["title"] == "Rat Race (EL 2)"
        assert out[0]["npcs"] == ["Giant Rat"]
        assert "Eight rats" in out[0]["read_aloud"]


class TestExtractNpcs:
    """Tests for NPC extractor (with mocked LLM)."""

    def test_extract_npcs_returns_list_of_dicts(self):
        from app.services.parsing.extractors.npc import extract_npcs
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(
            heading="Captain Vane",
            level=1,
            body="Human fighter. AC 16, HP 45. Gruff, loyal to the crown.",
        )
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text='[{"name":"Captain Vane","role":"ally","personality":"gruff, loyal","faction":"","description":"human fighter","motivation":"","secrets":"","hp":"45","ac":16,"cr":"","confidence":0.85}]')]

        with patch("app.services.parsing.extractors.npc._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_npcs(chunk)
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["name"] == "Captain Vane"
            assert result[0]["role"] == "ally"
            assert result[0].get("confidence", 0) > 0

    def test_extract_npcs_empty_section_returns_empty(self):
        from app.services.parsing.extractors.npc import extract_npcs
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(heading="", level=0, body="No character here.")
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="[]")]

        with patch("app.services.parsing.extractors.npc._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_npcs(chunk)
            assert result == []

    def test_extract_npcs_salvages_valid_prefix_from_truncated_json(self):
        from app.services.parsing.extractors.npc import extract_npcs
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(
            heading="Captain Vane",
            level=1,
            body="Captain Vane briefs the guards while another officer is only half-described.",
        )
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "name":"Captain Vane",
                "role":"ally",
                "personality":"gruff, loyal",
                "faction":"Town Guard",
                "description":"Veteran watch captain",
                "motivation":"Keep the gate secure",
                "secrets":"",
                "hp":"45",
                "ac":16,
                "cr":"",
                "confidence":0.91
            },
            {
                "name":"Broken Officer",
                "role":"ally",
                "personality":"nervous
        """)]

        with patch("app.services.parsing.extractors.npc._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_npcs(chunk)

        assert len(result) == 1
        assert result[0]["name"] == "Captain Vane"


class TestExtractorJsonRecovery:
    def test_extract_scene_seeds_repairs_trailing_comma_json(self):
        from app.services.parsing.extractors.scene_seed import extract_scene_seeds
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(
            heading="South Gate",
            level=1,
            body="Read aloud as the gate opens and the guards challenge the party.",
        )
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "title":"At the South Gate",
                "act":"",
                "type":"social",
                "read_aloud":"The gate creaks open a handspan.",
                "npcs":["Captain Vane"],
                "location":"South Gate",
                "difficulty":"",
                "rewards":"",
                "notes":"Questions the party",
                "confidence":0.89,
            }
        ]""")]

        with patch("app.services.parsing.extractors.scene_seed._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_scene_seeds(chunk)

        assert len(result) == 1
        assert result[0]["title"] == "At the South Gate"
        assert result[0]["location"] == "South Gate"

    def test_extract_codex_entries_salvages_valid_prefix_from_truncated_json(self):
        from app.services.parsing.extractors.codex import extract_codex_entries
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(
            heading="Black Stair",
            level=1,
            body="Ancient lore describes the stair beneath the manor and a second, incomplete note.",
        )
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "title":"The Black Stair",
                "summary":"Ancient steps beneath the manor.",
                "content":"Basalt stairs descend into a sealed crypt.",
                "tags":["lore","crypt"],
                "confidence":0.9
            },
            {
                "title":"Broken Note",
                "summary":"This one fails
        """)]

        with patch("app.services.parsing.extractors.codex._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_codex_entries(chunk, content_type="lore")

        assert len(result) == 1
        assert result[0]["title"] == "The Black Stair"
        assert result[0]["type"] == "lore"


class TestExtractQuests:
    def test_extract_quests_merges_overlapping_background_objectives(self):
        from app.services.parsing.extractors.quest import extract_quests
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(
            heading="Background",
            level=1,
            body="Arthur Blackwood needs the investigators to clear his name and search the cabin for evidence.",
            chunk_type_guess="quest_section",
        )
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "name":"Clear Arthur Blackwood's Name",
                "description":"Find evidence to prove Arthur's innocence before trial.",
                "objective":"Investigate the Blackwood cabin and uncover the truth behind Rose's disappearance.",
                "stakes":"Arthur faces execution.",
                "related_npcs":["Arthur Blackwood","Joseph Klein"],
                "related_locations":["Blackwood Cabin","Whitehall"],
                "tags":["investigation","trial"],
                "confidence":0.95
            },
            {
                "name":"Investigate the Blackwood Cabin",
                "description":"Search the summer cabin for overlooked evidence.",
                "objective":"Examine the cabin for clues the police missed.",
                "stakes":"Critical evidence may clear Arthur.",
                "related_npcs":["Arthur Blackwood","Joseph Klein"],
                "related_locations":["Blackwood Cabin","Whitehall"],
                "tags":["investigation","evidence"],
                "confidence":0.92
            },
            {
                "name":"Determine Arthur's Mental State",
                "description":"Assess whether Arthur's strange claims reflect truth or delusion.",
                "objective":"Investigate Arthur's claims about darkness and Rose's disappearance.",
                "stakes":"The defense depends on Arthur's credibility.",
                "related_npcs":["Arthur Blackwood","Joseph Klein"],
                "related_locations":["Blackwood Cabin"],
                "tags":["investigation","mystery"],
                "confidence":0.88
            }
        ]""")]

        with patch("app.services.parsing.extractors.quest._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_quests(chunk)
            assert len(result) == 1
            assert result[0]["name"] == "Clear Arthur Blackwood's Name"
            assert "Blackwood Cabin" in result[0]["related_locations"]
            assert "Arthur Blackwood" in result[0]["related_npcs"]

    def test_extract_quests_keeps_distinct_hook_options(self):
        from app.services.parsing.extractors.quest import extract_quests
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(
            heading="Adventure Hooks",
            level=1,
            body="The party can seek shelter, scout the tower, or recover a family heirloom.",
            chunk_type_guess="quest_section",
        )
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "name":"Seek Shelter from the Storm",
                "description":"Reach the tower before the storm worsens.",
                "objective":"Get inside Hightower safely.",
                "stakes":"Survive the storm.",
                "related_npcs":[],
                "related_locations":["Hightower"],
                "tags":["intro"],
                "confidence":0.94
            },
            {
                "name":"Recover the Family Heirloom",
                "description":"Retrieve a buried heirloom for a patron.",
                "objective":"Locate and recover the heirloom from Hightower.",
                "stakes":"Earn the patron's reward.",
                "related_npcs":["Wealthy patron"],
                "related_locations":["Hightower"],
                "tags":["recovery"],
                "confidence":0.9
            }
        ]""")]

        with patch("app.services.parsing.extractors.quest._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_quests(chunk)
            assert len(result) == 2
            assert {item["name"] for item in result} == {
                "Seek Shelter from the Storm",
                "Recover the Family Heirloom",
            }

    def test_extract_quests_filters_rumor_flavor_without_player_directive(self):
        from app.services.parsing.extractors.quest import extract_quests
        from unittest.mock import patch, MagicMock

        chunk = SectionChunk(
            heading="Rumors",
            level=1,
            body=(
                "Villagers whisper that cold blue lights drift over the sunken shrine at night. "
                "Some say the marsh remembers every trespass."
            ),
            chunk_type_guess="quest_section",
        )
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="""[
            {
                "name":"Investigate the Sunken Shrine",
                "description":"Look into the lights over the shrine.",
                "objective":"Visit the shrine and learn why it glows.",
                "stakes":"The marsh remains unsettling.",
                "related_npcs":[],
                "related_locations":["Sunken Shrine"],
                "tags":["rumor"],
                "confidence":0.67
            }
        ]""")]

        with patch("app.services.parsing.extractors.quest._get_client") as get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            get_client.return_value = mock_client

            result = extract_quests(chunk)

        assert result == []


class TestQuestCanonicalization:
    def test_canonicalize_quests_merges_duplicate_restatement_across_nearby_chunks(self):
        from app.services.parsing.extractors.quest import canonicalize_quests

        quests = [
            {
                "id": "quest_find_surveyor",
                "name": "Find the Missing Surveyor",
                "description": "Locate the lost surveyor before nightfall.",
                "objective": "Search Blackfen Marsh and bring the surveyor back alive.",
                "stakes": "The expedition cannot continue without the surveyor.",
                "related_npcs": ["Lysa Dorn"],
                "related_locations": ["Blackfen Marsh"],
                "tags": ["rescue"],
                "confidence": 0.92,
                "source": {
                    "document_id": "doc_1",
                    "page_number": 5,
                    "heading": "Blackfen Marsh",
                    "subheading": "Quest Hook",
                    "heading_path": ["Blackfen Marsh", "Quest Hook"],
                },
            },
            {
                "id": "quest_rescue_surveyor",
                "name": "Rescue the Lost Surveyor",
                "description": "Track down the surveyor who vanished in the marsh.",
                "objective": "Track the missing surveyor through Blackfen Marsh.",
                "stakes": "Without the surveyor, the stone markers stay hidden.",
                "related_npcs": ["Lysa Dorn"],
                "related_locations": ["Blackfen Marsh"],
                "tags": ["search"],
                "confidence": 0.88,
                "source": {
                    "document_id": "doc_1",
                    "page_number": 6,
                    "heading": "Blackfen Marsh",
                    "subheading": "Development",
                    "heading_path": ["Blackfen Marsh", "Development"],
                },
            },
        ]

        result = canonicalize_quests(quests)

        assert len(result) == 1
        assert result[0]["name"] == "Find the Missing Surveyor"
        assert "Lysa Dorn" in result[0]["related_npcs"]
        assert "Blackfen Marsh" in result[0]["related_locations"]
        assert {"rescue", "search"}.issubset(set(result[0]["tags"]))


class TestRelationships:
    def test_extract_relationships_links_by_names_and_source_proximity(self):
        npcs = [
            {
                "name": "Oleg",
                "description": "A suspicious gatekeeper.",
                "faction": "Town Watch",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "NPCs",
                    "heading_path": ["South Gate", "NPCs"],
                },
            }
        ]
        locations = [
            {
                "name": "South Gate",
                "description": "The town's guarded entrance.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "",
                    "heading_path": ["South Gate"],
                },
            }
        ]
        scenes = [
            {
                "title": "At the South Gate",
                "npcs": ["Oleg"],
                "location": "South Gate",
                "read_aloud": "Oleg blocks the way.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "Encounter",
                    "heading_path": ["South Gate", "Encounter"],
                },
            }
        ]
        codex_entries = []
        quests = [
            {
                "name": "Gain Entry",
                "description": "Convince Oleg to let the party through.",
                "related_npcs": ["Oleg"],
                "related_locations": ["South Gate"],
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "Quest Hook",
                    "heading_path": ["South Gate", "Quest Hook"],
                },
            }
        ]
        factions = [
            {
                "name": "Town Watch",
                "description": "The guards who patrol South Gate.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "Faction",
                    "heading_path": ["South Gate", "Faction"],
                },
            }
        ]

        relationships = extract_relationships(
            npcs,
            locations,
            scenes,
            codex_entries,
            quests=quests,
            factions=factions,
        )

        relation_tuples = {
            (rel["from_type"], rel["relation"], rel["to_type"], rel["to_id"])
            for rel in relationships
        }
        assert ("npc", "appears_in", "scene", "At the South Gate") in relation_tuples
        assert ("npc", "located_at", "location", "South Gate") in relation_tuples
        assert ("scene", "occurs_at", "location", "South Gate") in relation_tuples
        assert ("scene", "advances", "quest", "Gain Entry") in relation_tuples
        assert ("faction", "includes", "npc", "Oleg") in relation_tuples

    def test_extract_relationships_uses_aliases_and_heading_inheritance(self):
        npcs = [
            {
                "name": "Thuldrin Kreed",
                "description": "Ruthless consortium boss.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 5,
                    "heading": "Authority Figures",
                    "subheading": "",
                    "heading_path": ["Lumber Consortium Hall", "Authority Figures"],
                },
            }
        ]
        locations = [
            {
                "name": "Lumber Consortium Hall",
                "description": "A severe guildhall of ledgers and timber maps.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 5,
                    "heading": "Lumber Consortium Hall",
                    "subheading": "",
                    "heading_path": ["Lumber Consortium Hall"],
                },
            }
        ]
        scenes = [
            {
                "title": "Audience with the Consortium",
                "npcs": ["Gavel Thuldrin Kreed"],
                "location": "",
                "read_aloud": "Gavel Thuldrin Kreed looks up from his ledger.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 5,
                    "heading": "Lumber Consortium Hall",
                    "subheading": "Audience",
                    "heading_path": ["Lumber Consortium Hall", "Audience"],
                },
            }
        ]
        quests = [
            {
                "name": "Win Kreed's Support",
                "description": "Negotiate with Thuldrin Kreed for safe passage.",
                "related_npcs": ["Thuldrin Kreed"],
                "related_locations": ["Lumber Consortium Hall"],
                "source": {
                    "document_id": "doc_1",
                    "page_number": 5,
                    "heading": "Lumber Consortium Hall",
                    "subheading": "Negotiation",
                    "heading_path": ["Lumber Consortium Hall", "Negotiation"],
                },
            }
        ]

        relationships = extract_relationships(
            npcs,
            locations,
            scenes,
            [],
            quests=quests,
        )

        relation_tuples = {
            (rel["from_type"], rel["relation"], rel["to_type"], rel["to_id"])
            for rel in relationships
        }
        assert ("npc", "appears_in", "scene", "Audience with the Consortium") in relation_tuples
        assert ("scene", "occurs_at", "location", "Lumber Consortium Hall") in relation_tuples
        assert ("scene", "advances", "quest", "Win Kreed's Support") in relation_tuples

    def test_extract_relationships_does_not_link_npc_to_every_same_heading_location(self):
        npcs = [
            {
                "name": "Oleg",
                "description": "A suspicious gatekeeper.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "NPCs",
                    "heading_path": ["South Gate", "NPCs"],
                },
            }
        ]
        locations = [
            {
                "name": "South Gate",
                "description": "The guarded entrance to town.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "",
                    "heading_path": ["South Gate"],
                },
            },
            {
                "name": "Gate Barracks",
                "description": "The barracks adjacent to the gate.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "Features",
                    "heading_path": ["South Gate", "Features"],
                },
            },
        ]

        relationships = extract_relationships(npcs, locations, [], [])
        relation_tuples = {
            (rel["from_type"], rel["relation"], rel["to_type"], rel["to_id"])
            for rel in relationships
        }

        assert ("npc", "located_at", "location", "South Gate") in relation_tuples
        assert ("npc", "located_at", "location", "Gate Barracks") not in relation_tuples

    def test_extract_relationships_does_not_link_scene_to_every_same_heading_location(self):
        scenes = [
            {
                "title": "At the South Gate",
                "npcs": [],
                "location": "South Gate",
                "read_aloud": "The gate creaks open a handspan.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "Encounter",
                    "heading_path": ["South Gate", "Encounter"],
                },
            }
        ]
        locations = [
            {
                "name": "South Gate",
                "description": "The guarded entrance to town.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "",
                    "heading_path": ["South Gate"],
                },
            },
            {
                "name": "Gate Barracks",
                "description": "The barracks adjacent to the gate.",
                "source": {
                    "document_id": "doc_1",
                    "page_number": 3,
                    "heading": "South Gate",
                    "subheading": "Features",
                    "heading_path": ["South Gate", "Features"],
                },
            },
        ]

        relationships = extract_relationships([], locations, scenes, [])
        relation_tuples = {
            (rel["from_type"], rel["relation"], rel["to_type"], rel["to_id"])
            for rel in relationships
        }

        assert ("scene", "occurs_at", "location", "South Gate") in relation_tuples
        assert ("scene", "occurs_at", "location", "Gate Barracks") not in relation_tuples


class TestPipelineLinking:
    def test_run_parsing_pipeline_links_after_dedupe(self):
        from app.services.parsing.pipeline import run_parsing_pipeline

        captured: dict[str, list[dict]] = {}
        sections = [SectionChunk(heading="Authority Figures", level=1, body="Thuldrin Kreed in the hall.")]
        extracted = {
            "npcs": [
                {
                    "name": "Gavel Thuldrin Kreed",
                    "description": "Leader of the Lumber Consortium.",
                    "confidence": 0.8,
                    "source": {"document_id": "doc_1", "heading": "Authority Figures", "page_number": 5},
                },
                {
                    "name": "Thuldrin Kreed",
                    "description": "Ruthless consortium boss.",
                    "confidence": 0.9,
                    "source": {"document_id": "doc_1", "heading": "Authority Figures", "page_number": 5},
                },
            ],
            "locations": [
                {
                    "name": "Lumber Consortium Hall",
                    "description": "A severe guildhall.",
                    "confidence": 0.9,
                    "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
                },
                {
                    "name": "The Lumber Consortium Hall",
                    "description": "A severe guildhall.",
                    "confidence": 0.8,
                    "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
                },
            ],
            "scenes": [
                {
                    "title": "Audience with the Consortium",
                    "npcs": ["Gavel Thuldrin Kreed"],
                    "location": "The Lumber Consortium Hall",
                    "confidence": 0.8,
                    "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
                },
                {
                    "title": "Audience with the Consortium",
                    "npcs": ["Thuldrin Kreed"],
                    "location": "Lumber Consortium Hall",
                    "confidence": 0.9,
                    "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
                },
            ],
            "quests": [
                {
                    "id": "quest_win_support",
                    "name": "Win Kreed's Support",
                    "description": "Negotiate with the consortium boss.",
                    "objective": "Convince Gavel Thuldrin Kreed to grant safe passage.",
                    "stakes": "Without support the party is delayed.",
                    "related_npcs": ["Gavel Thuldrin Kreed"],
                    "related_locations": ["The Lumber Consortium Hall"],
                    "tags": ["social"],
                    "confidence": 0.9,
                    "source": {"document_id": "doc_1", "heading": "Lumber Consortium Hall", "page_number": 5},
                },
            ],
            "items": [],
            "codex_entries": [],
            "encounters": [],
        }

        def _capture_relationships(npcs, locations, scenes, codex_entries, **kwargs):
            captured["npcs"] = npcs
            captured["locations"] = locations
            captured["scenes"] = scenes
            captured["quests"] = kwargs.get("quests") or []
            captured["codex_entries"] = codex_entries
            return []

        with patch("app.services.parsing.pipeline.normalize_text", return_value="Authority Figures"):
            with patch("app.services.parsing.pipeline.split_into_sections", return_value=sections):
                with patch("app.services.parsing.pipeline.classify_chunks"):
                    with patch("app.services.parsing.pipeline._extract_title_summary", return_value=("Title", "Summary")):
                        with patch("app.services.parsing.pipeline.extract_typed_entities", return_value=extracted):
                            with patch("app.services.parsing.pipeline.extract_relationships", side_effect=_capture_relationships):
                                result = run_parsing_pipeline("Authority Figures")

        assert result["title"] == "Title"
        assert len(captured["npcs"]) == 1
        assert captured["npcs"][0]["name"] == "Thuldrin Kreed"
        assert len(captured["locations"]) == 1
        assert captured["locations"][0]["name"] == "Lumber Consortium Hall"
        assert len(captured["scenes"]) == 1
        assert captured["scenes"][0]["npcs"] == ["Thuldrin Kreed"]
        assert captured["scenes"][0]["location"] == "Lumber Consortium Hall"
        assert captured["quests"][0]["related_npcs"] == ["Thuldrin Kreed"]
        assert captured["quests"][0]["related_locations"] == ["Lumber Consortium Hall"]

    def test_run_parsing_pipeline_merges_valid_quest_supporting_chunks(self):
        from app.services.parsing.pipeline import run_parsing_pipeline

        sections = [SectionChunk(heading="South Gate", level=1, body="Quest text.")]
        extracted = {
            "npcs": [],
            "locations": [],
            "scenes": [],
            "quests": [
                {
                    "id": "quest_win_kreeds_support",
                    "name": "Win Kreed's Support",
                    "description": "Negotiate for safe passage.",
                    "objective": "Secure Kreed's approval for travel papers.",
                    "stakes": "Without his approval the party is delayed.",
                    "related_npcs": ["Thuldrin Kreed"],
                    "related_locations": [],
                    "tags": ["negotiation"],
                    "confidence": 0.9,
                    "source": {
                        "document_id": "doc_1",
                        "page_number": 5,
                        "heading": "Lumber Consortium Hall",
                        "subheading": "Quest Hook",
                        "heading_path": ["Lumber Consortium Hall", "Quest Hook"],
                    },
                },
                {
                    "id": "quest_win_kreeds_support_supporting",
                    "name": "Win Kreed's Support",
                    "description": "The negotiation takes place in the consortium hall.",
                    "objective": "Present the timber ledger in the hall to earn passage.",
                    "stakes": "",
                    "related_npcs": [],
                    "related_locations": ["Lumber Consortium Hall"],
                    "tags": ["social"],
                    "confidence": 0.84,
                    "source": {
                        "document_id": "doc_1",
                        "page_number": 6,
                        "heading": "Lumber Consortium Hall",
                        "subheading": "Development",
                        "heading_path": ["Lumber Consortium Hall", "Development"],
                    },
                },
            ],
            "items": [],
            "codex_entries": [],
            "encounters": [],
        }

        with patch("app.services.parsing.pipeline.normalize_text", return_value="South Gate"):
            with patch("app.services.parsing.pipeline.split_into_sections", return_value=sections):
                with patch("app.services.parsing.pipeline.classify_chunks"):
                    with patch("app.services.parsing.pipeline._extract_title_summary", return_value=("Title", "Summary")):
                        with patch("app.services.parsing.pipeline.extract_typed_entities", return_value=extracted):
                            with patch("app.services.parsing.pipeline.extract_relationships", return_value=[]):
                                result = run_parsing_pipeline("South Gate")

        assert len(result["quests"]) == 1
        assert result["quests"][0]["name"] == "Win Kreed's Support"
        assert "Thuldrin Kreed" in result["quests"][0]["related_npcs"]
        assert "Lumber Consortium Hall" in result["quests"][0]["related_locations"]
        assert {"negotiation", "social"}.issubset(set(result["quests"][0]["tags"]))


class TestExtractionRouting:
    def test_pure_location_chunk_does_not_run_scene_extractor(self):
        from app.services.parsing.extraction import extract_typed_entities

        chunk = SectionChunk(
            heading="Roots and Remedies",
            level=2,
            body=(
                "Creeping ivy and full window boxes cover the facade of the rugged-looking, "
                "two-story shop bearing the faded sign Roots and Remedies."
            ),
            chunk_type_guess="location_section",
            content_type="location",
            content_types=["location"],
        )

        with patch("app.services.parsing.extraction.extract_locations", return_value=[{"name": "Roots and Remedies"}]), \
             patch("app.services.parsing.extraction.extract_scene_seeds", return_value=[{"title": "Should Not Exist"}]):
            result = extract_typed_entities([chunk])

        assert len(result["locations"]) == 1
        assert result["scenes"] == []

    def test_location_chunk_with_scene_cues_runs_both_extractors(self):
        from app.services.parsing.extraction import extract_typed_entities

        chunk = SectionChunk(
            heading="Hightower Main Entrance",
            level=2,
            body=(
                "Read or paraphrase the following when the PCs come within 20 feet of Hightower. "
                "The heavy door swings shut on its own unless propped open."
            ),
            chunk_type_guess="location_section",
            content_type="location",
            content_types=["location", "scene"],
        )

        with patch("app.services.parsing.extraction.extract_locations", return_value=[{"name": "Hightower Main Entrance"}]), \
             patch(
                 "app.services.parsing.extraction.extract_scene_seeds",
                 return_value=[
                     {
                         "title": "Hightower Main Entrance",
                         "read_aloud": "The heavy door swings shut on its own unless propped open.",
                         "npcs": [],
                         "location": "Hightower Main Entrance",
                         "difficulty": "",
                         "rewards": "",
                         "notes": "",
                         "act": "",
                         "type": "exploration",
                     }
                 ],
             ):
            result = extract_typed_entities([chunk])

        assert len(result["locations"]) == 1
        assert len(result["scenes"]) == 1

    def test_location_chunk_drops_scene_that_only_repeats_location(self):
        from app.services.parsing.extraction import extract_typed_entities

        chunk = SectionChunk(
            heading="Roots and Remedies",
            level=2,
            body=(
                "Creeping ivy and full window boxes cover the facade of the rugged-looking, "
                "two-story shop bearing the faded sign Roots and Remedies."
            ),
            chunk_type_guess="location_section",
            content_type="location",
            content_types=["location", "scene"],
        )

        with patch("app.services.parsing.extraction.extract_locations", return_value=[{"name": "Roots and Remedies"}]), \
             patch(
                 "app.services.parsing.extraction.extract_scene_seeds",
                 return_value=[
                     {
                         "title": "Roots and Remedies",
                         "read_aloud": "",
                         "npcs": [],
                         "location": "Roots and Remedies",
                         "difficulty": "",
                         "rewards": "",
                         "notes": "",
                         "act": "",
                         "type": "exploration",
                     }
                 ],
             ):
            result = extract_typed_entities([chunk])

        assert len(result["locations"]) == 1
        assert result["scenes"] == []


class TestPipelineIntegration:
    """Integration-style test: run pipeline on a small fixture (requires ANTHROPIC_API_KEY)."""

    @pytest.mark.integration
    def test_run_parsing_pipeline_returns_expected_keys(self):
        from app.services.parsing.pipeline import run_parsing_pipeline

        fixture = """
# The Lost Mine

A short adventure for beginners.

## Goblin Ambush

Goblins attack on the road. Read aloud: "You see goblins ahead."

## Villain: Nezznar

Nezznar is a drow. AC 15, HP 44. He seeks the forge.
"""
        result = run_parsing_pipeline(fixture)
        assert "title" in result
        assert "summary" in result
        assert "npcs" in result
        assert "scenes" in result
        assert "locations" in result
        assert "encounters" in result
        assert "quests" in result
        assert "factions" in result
        assert "lore" in result
        assert "codex_entries" in result
        assert "relationships" in result
        assert isinstance(result["npcs"], list)
        assert isinstance(result["scenes"], list)
        assert isinstance(result["encounters"], list)
        assert isinstance(result["quests"], list)
        assert isinstance(result["factions"], list)
        assert isinstance(result["lore"], list)
        assert isinstance(result["codex_entries"], list)
        assert isinstance(result["relationships"], list)


class TestRecallUpgrade:
    def test_extraction_attaches_evidence_fields(self):
        from app.services.parsing.extraction import extract_typed_entities

        chunk = SectionChunk(
            heading="The Drowned Library",
            level=1,
            body="Clue: The silver key bears the mark of House Vey.",
            document_id="doc_alpha",
            page_number=12,
            chunk_type_guess="quest_section",
            content_type="quest",
            content_types=["quest"],
        )
        with patch("app.services.parsing.extraction.extract_quests", return_value=[{
            "name": "Find the Silver Key",
            "description": "Investigate who forged the marked key.",
            "objective": "Trace the sigil on the key.",
            "stakes": "Without it, the vault remains sealed.",
            "related_npcs": ["Archivist Mel"],
            "related_locations": ["The Drowned Library"],
            "tags": ["investigation"],
            "confidence": 0.86,
        }]):
            result = extract_typed_entities([chunk])
        quest = result["quests"][0]
        assert quest["source_document_id"] == "doc_alpha"
        assert quest["page_number"] == 12
        assert quest["source_chunk_id"]
        assert quest["evidence_text"]
        assert isinstance(quest.get("evidence"), dict)

    def test_cross_chunk_fusion_merges_split_npc_details(self):
        chunk_a = SectionChunk(
            heading="Captain Ilvara",
            level=2,
            body="Captain Ilvara commands the harbor guard.",
            document_id="doc_beta",
            page_number=9,
            chunk_type_guess="stat_block",
        )
        chunk_b = SectionChunk(
            heading="Captain Ilvara",
            level=2,
            body="Secret: Ilvara is smuggling relics for the Cinder Choir.",
            document_id="doc_beta",
            page_number=10,
            chunk_type_guess="quest_section",
        )
        extracted = {
            "npcs": [
                {"name": "Captain Ilvara", "description": "Commands the harbor guard.", "confidence": 0.8, "source_chunk_id": chunk_a.chunk_id(), "source_document_id": "doc_beta", "page_number": 9},
                {"name": "Ilvara", "description": "Smuggling relics for the Cinder Choir.", "confidence": 0.83, "source_chunk_id": chunk_b.chunk_id(), "source_document_id": "doc_beta", "page_number": 10},
            ]
        }
        candidates = build_candidates(extracted, [chunk_a, chunk_b])
        fused = fuse_candidates(candidates)
        assert len(fused["npcs"]) == 1
        assert "smuggling relics" in fused["npcs"][0]["description"].lower()
        assert fused["npcs"][0]["mention_count"] >= 2

    def test_high_value_extractors_capture_read_aloud_and_secrets(self):
        from app.services.parsing.extraction import extract_typed_entities

        chunk = SectionChunk(
            heading="Moonwell Chapel",
            level=2,
            body=(
                "Read Aloud: Moonlight pours over broken pews.\n\n"
                "Secret: The altar conceals a blood-ink pact.\n\n"
                "Rumor: Villagers whisper the chapel bells ring at dawn."
            ),
            chunk_type_guess="boxed_text",
            content_type="scene",
            content_types=["scene", "lore"],
        )
        with patch("app.services.parsing.extraction.extract_scene_seeds", return_value=[]):
            result = extract_typed_entities([chunk])
        assert len(result["read_alouds"]) >= 1
        assert len(result["secrets"]) >= 1
        assert len(result["rumors"]) >= 1

    def test_coverage_audit_reports_missing_quest_details(self):
        chunk = SectionChunk(
            heading="Old Quarry",
            level=1,
            body="Objective: reach the quarry before sunset.",
            document_id="doc_gamma",
            page_number=4,
            chunk_type_guess="quest_section",
        )
        payload = {
            "npcs": [],
            "locations": [{"name": "Old Quarry", "description": ""}],
            "scenes": [{"title": "At the Quarry", "npcs": []}],
            "quests": [{"name": "Reach the Quarry", "objective": "", "stakes": ""}],
            "encounters": [{"name": "Quarry Wolves", "scene_id": ""}],
            "clues": [],
            "secrets": [],
            "rumors": [],
            "hooks": [],
            "rewards": [],
            "consequences": [],
        }
        report = audit_coverage(payload, [chunk])
        assert report["summary"]["quest_detail_gaps"] >= 1
        assert report["summary"]["encounter_link_gaps"] >= 1

    def test_importance_scoring_prefers_repeated_linked_entities(self):
        scored = score_importance(
            {
                "npcs": [
                    {
                        "name": "Marshal Tovin",
                        "mention_count": 4,
                        "related_locations": ["Bastion Gate", "Watch Barracks"],
                        "confidence": 0.9,
                        "source": {"heading": "Marshal Tovin", "subheading": "Quest Hook"},
                    },
                    {
                        "name": "Dockhand Neri",
                        "mention_count": 1,
                        "related_locations": [],
                        "confidence": 0.55,
                        "source": {"heading": "", "subheading": ""},
                    },
                ]
            }
        )
        assert scored["npcs"][0]["importance_score"] > scored["npcs"][1]["importance_score"]
