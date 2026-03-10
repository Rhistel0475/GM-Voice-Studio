"""Unit tests for the document parsing pipeline (normalize, sections, dedupe, extractors)."""
import pytest

from app.services.parsing.normalize import normalize_text
from app.services.parsing.sections import split_into_sections
from app.services.parsing.models import SectionChunk
from app.services.parsing.dedupe import (
    dedupe_npcs,
    dedupe_locations,
    dedupe_scenes,
    dedupe_codex_entries,
)


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
        assert "codex_entries" in result
        assert "relationships" in result
        assert isinstance(result["npcs"], list)
        assert isinstance(result["scenes"], list)
        assert isinstance(result["codex_entries"], list)
        assert isinstance(result["relationships"], list)
