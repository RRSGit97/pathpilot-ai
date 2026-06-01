"""
tests/test_job_parser.py
------------------------
Tests for src/tools/job_parser.py.

Covers:
- Single JD parsing (title inference, cleaning, keyword extraction)
- Multiple JD parsing (batch API, empty input skipping, MAX_JD_COUNT guard)
- Auto-split detection
- Batch summary helper
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.tools.job_parser import (
    MAX_JD_COUNT,
    ParsedJobDescription,
    parse_job_description,
    parse_multiple_jds,
    summarise_jd_batch,
)

# ---------------------------------------------------------------------------
# Minimal valid JD (>= 50 chars) used in many tests
# ---------------------------------------------------------------------------

_MINIMAL_JD = (
    "Senior AI Engineer at Acme Corp.\n"
    "We are looking for a Python expert with LangGraph experience.\n"
    "Remote position."
)

_FULL_JD = """\
Job Title: AI Engineer
Company: Acme Corp
Location: Remote

We are looking for a mid-level AI Engineer to join our team.
The ideal candidate has strong Python and LangGraph skills.

Requirements:
- Python (required)
- LangGraph (required)
- FastAPI (preferred)
- Docker

Responsibilities:
- Build and deploy multi-agent pipelines.
- Collaborate with product teams.
"""


# ---------------------------------------------------------------------------
# parse_job_description: single JD
# ---------------------------------------------------------------------------

class TestParseJobDescription:
    def test_returns_parsed_jd_object(self):
        result = parse_job_description(_FULL_JD)
        assert isinstance(result, ParsedJobDescription)

    def test_title_extracted_from_explicit_label(self):
        result = parse_job_description(_FULL_JD)
        assert result.inferred_title.lower() == "ai engineer"

    def test_company_extracted(self):
        result = parse_job_description(_FULL_JD)
        assert "Acme Corp" in result.inferred_company

    def test_location_extracted(self):
        result = parse_job_description(_FULL_JD)
        assert result.inferred_location  # not empty

    def test_cleaned_text_has_no_double_spaces(self):
        # Use a full-length JD so min-50-chars validator passes after normalisation
        raw = "Python   skills   are   required for this role.   We need   FastAPI and Docker too."
        result = parse_job_description(raw)
        assert "  " not in result.cleaned_text

    def test_keyword_candidates_are_normalised_lowercase(self):
        result = parse_job_description(_FULL_JD)
        for kw in result.keyword_candidates:
            assert kw == kw.lower(), f"Non-lowercase keyword: {kw!r}"

    def test_keyword_candidates_sorted(self):
        result = parse_job_description(_FULL_JD)
        assert result.keyword_candidates == sorted(result.keyword_candidates)

    def test_title_hint_overrides_inferred(self):
        result = parse_job_description(_FULL_JD, title_hint="Custom Title")
        assert result.inferred_title == "Custom Title"

    def test_source_url_stored(self):
        url = "https://example.com/job/123"
        result = parse_job_description(_FULL_JD, source_url=url)
        assert result.jd.source_url == url

    def test_too_short_jd_raises_validation_error(self):
        with pytest.raises(ValidationError):
            parse_job_description("Too short.")  # < 50 chars

    def test_source_snippets_map_to_keywords(self):
        result = parse_job_description(_FULL_JD)
        # At least some keywords should have snippets
        assert len(result.source_snippets) > 0
        for kw, snippet in result.source_snippets.items():
            assert isinstance(snippet, str)
            assert len(snippet) > 0


# ---------------------------------------------------------------------------
# parse_multiple_jds: batch parsing
# ---------------------------------------------------------------------------

class TestParseMultipleJds:
    def test_parses_batch_of_two(self):
        result = parse_multiple_jds([_MINIMAL_JD, _FULL_JD])
        assert len(result) == 2

    def test_skips_empty_strings(self):
        result = parse_multiple_jds(["", _FULL_JD, "  \n  "])
        assert len(result) == 1

    def test_respects_max_jd_count(self):
        # Provide MAX_JD_COUNT + 1 valid JDs
        over_limit = [_MINIMAL_JD] * (MAX_JD_COUNT + 1)
        # The 6th should be silently dropped (warning logged)
        result = parse_multiple_jds(over_limit)
        assert len(result) == MAX_JD_COUNT

    def test_returns_empty_list_for_all_empty_inputs(self):
        result = parse_multiple_jds(["", ""])
        assert result == []

    def test_title_hints_applied_per_jd(self):
        hints = ["First Hint", "Second Hint"]
        result = parse_multiple_jds([_MINIMAL_JD, _FULL_JD], title_hints=hints)
        assert result[0].inferred_title == "First Hint"

    def test_auto_split_single_blob_fallback(self):
        # When no split delimiter is present, the single blob is parsed as-is
        result = parse_multiple_jds([_FULL_JD], attempt_auto_split=True)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# summarise_jd_batch
# ---------------------------------------------------------------------------

class TestSummariseJdBatch:
    def test_returns_expected_keys(self):
        parsed = parse_multiple_jds([_FULL_JD])
        summary = summarise_jd_batch(parsed)
        assert "count" in summary
        assert "titles" in summary
        assert "all_keywords" in summary
        assert "total_chars" in summary

    def test_count_matches_input(self):
        parsed = parse_multiple_jds([_MINIMAL_JD, _FULL_JD])
        summary = summarise_jd_batch(parsed)
        assert summary["count"] == 2

    def test_total_chars_is_sum(self):
        parsed = parse_multiple_jds([_MINIMAL_JD, _FULL_JD])
        summary = summarise_jd_batch(parsed)
        expected = sum(len(p.cleaned_text) for p in parsed)
        assert summary["total_chars"] == expected

    def test_all_keywords_is_sorted_and_unique(self):
        parsed = parse_multiple_jds([_FULL_JD])
        summary = summarise_jd_batch(parsed)
        kws = summary["all_keywords"]
        assert kws == sorted(set(kws))

    def test_empty_batch_returns_zero_count(self):
        summary = summarise_jd_batch([])
        assert summary["count"] == 0
        assert summary["total_chars"] == 0
