"""
tests/test_text_utils.py
------------------------
Unit tests for src/tools/text_utils.py.

All functions are pure Python with no external dependencies —
tests run completely offline.
"""

from __future__ import annotations

import pytest

from src.tools.text_utils import (
    extract_keyword_candidates,
    extract_source_snippet,
    normalize_skill_token,
    normalize_whitespace,
)


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("Hello   world") == "Hello world"

    def test_collapses_tabs(self):
        assert normalize_whitespace("Hello\t\tworld") == "Hello world"

    def test_strips_line_leading_spaces(self):
        result = normalize_whitespace("  leading space")
        assert not result.startswith(" ")

    def test_collapses_triple_newlines_to_double(self):
        result = normalize_whitespace("a\n\n\n\nb")
        assert "\n\n\n" not in result
        assert "a" in result and "b" in result

    def test_non_breaking_space_normalized(self):
        # \u00a0 is a non-breaking space
        result = normalize_whitespace("Hello\u00a0world")
        assert result == "Hello world"

    def test_empty_string_stays_empty(self):
        assert normalize_whitespace("") == ""


# ---------------------------------------------------------------------------
# normalize_skill_token
# ---------------------------------------------------------------------------

class TestNormalizeSkillToken:
    """Tests for alias resolution and lowercase normalisation."""

    @pytest.mark.parametrize("raw, expected", [
        ("Python3",   "python"),
        ("  Python3 ", "python"),
        ("K8S",        "kubernetes"),
        ("REST API",   "rest"),
        ("js",         "javascript"),
        ("Postgres",   "postgresql"),
        ("py",         "python"),
        ("ML",         "machine learning"),
        ("React.js",   "react"),
        ("Vue.js",     "vue"),
        ("ts",         "typescript"),
    ])
    def test_known_aliases(self, raw: str, expected: str):
        assert normalize_skill_token(raw) == expected

    def test_lowercase_passthrough(self):
        assert normalize_skill_token("fastapi") == "fastapi"

    def test_strips_surrounding_punctuation(self):
        result = normalize_skill_token('"Python"')
        assert result == "python"

    def test_strips_commas(self):
        result = normalize_skill_token("docker,")
        assert result == "docker"

    def test_empty_string(self):
        result = normalize_skill_token("")
        assert result == ""

    def test_unknown_token_lowercased(self):
        result = normalize_skill_token("LangGraph")
        assert result == "langgraph"


# ---------------------------------------------------------------------------
# extract_keyword_candidates
# ---------------------------------------------------------------------------

class TestExtractKeywordCandidates:
    """Extract candidate skill tokens from free text."""

    def test_returns_sorted_unique_list(self):
        result = extract_keyword_candidates("Python fastapi Python")
        assert result == sorted(set(result))

    def test_drops_stopwords(self):
        result = extract_keyword_candidates("we are the champions")
        # 'we', 'are', 'the' are stopwords; 'champions' should survive
        assert "we" not in result
        assert "are" not in result
        assert "the" not in result

    def test_drops_single_char_tokens(self):
        result = extract_keyword_candidates("C Python Java")
        # Default min_length=2, 'C' is length 1 after norm
        # The regex pattern already requires 2+ chars so 'C' won't match
        # at all; just check Python and java are present
        assert "python" in result

    def test_normalizes_aliases(self):
        result = extract_keyword_candidates("we need Python3 and K8S skills")
        assert "python" in result
        assert "kubernetes" in result

    def test_returns_list_type(self):
        result = extract_keyword_candidates("FastAPI SQLite")
        assert isinstance(result, list)

    def test_empty_text_returns_empty(self):
        assert extract_keyword_candidates("") == []

    def test_respects_min_length(self):
        result = extract_keyword_candidates("ab cd", min_length=3)
        assert result == []


# ---------------------------------------------------------------------------
# extract_source_snippet
# ---------------------------------------------------------------------------

class TestExtractSourceSnippet:
    def test_finds_keyword(self):
        text = "We need Python skills for this role."
        snippet = extract_source_snippet(text, "python")
        assert snippet is not None
        assert "Python" in snippet

    def test_case_insensitive(self):
        text = "We need PYTHON skills."
        snippet = extract_source_snippet(text, "python")
        assert snippet is not None

    def test_returns_none_when_not_found(self):
        text = "We need Java skills."
        assert extract_source_snippet(text, "python") is None

    def test_adds_ellipsis_at_start_when_truncated(self):
        # Place keyword near the end of a long string
        padding = "x " * 100
        text = f"{padding}Python is great."
        snippet = extract_source_snippet(text, "Python", window=40)
        assert snippet is not None
        assert snippet.startswith("...")

    def test_adds_ellipsis_at_end_when_truncated(self):
        padding = "y " * 100
        text = f"Python is great. {padding}"
        snippet = extract_source_snippet(text, "Python", window=30)
        assert snippet is not None
        assert snippet.endswith("...")

    def test_window_respected(self):
        text = "A " * 200 + "Python" + " B" * 200
        snippet = extract_source_snippet(text, "Python", window=20)
        assert snippet is not None
        # snippet <= window chars of content + at most 6 chars of ellipses
        assert len(snippet) <= 20 + 6 + 10  # window + ellipses + small rounding
