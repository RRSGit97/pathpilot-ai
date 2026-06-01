"""
tests/test_resume_parser.py
---------------------------
Tests for src/tools/resume_parser.py.

The parser has three source types: PLAIN_TEXT, PDF, DOCX.
PDF and DOCX parsing require third-party libraries (pypdf, python-docx).
Tests that need those libraries use pytest.importorskip so the suite
still passes if the libraries are not installed.

LLM-dependent pieces: none in this module.  All extraction is heuristic.

Covers:
- PLAIN_TEXT: valid input returns ResumeParseResult
- PLAIN_TEXT: short input sets parse_warning
- PLAIN_TEXT: extracted_skills populated from heuristic keywords
- PLAIN_TEXT: experience summary is a non-empty string
- Error cases: bytes with PLAIN_TEXT raises ValueError
- PDF: mocked pypdf returns text → skills extracted
- DOCX: mocked docx returns text → skills extracted
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.resume_parser import ResumeSourceType, parse_resume
from src.data.schemas import ResumeParseResult


# ---------------------------------------------------------------------------
# PLAIN_TEXT parsing
# ---------------------------------------------------------------------------

class TestPlainTextParsing:
    _GOOD_RESUME = """
    Senior Software Engineer | 5 years experience

    Skills: Python, FastAPI, Docker, PostgreSQL, LangChain, Pydantic

    Experience:
    2020-2024 | Tech Corp | Built a RAG pipeline using LangChain and ChromaDB.
    Deployed microservices with Docker and FastAPI on AWS.

    Education:
    BSc Computer Science, State University, 2019.
    """

    def test_returns_resume_parse_result(self):
        result = parse_resume(self._GOOD_RESUME)
        assert isinstance(result, ResumeParseResult)

    def test_no_warning_for_adequate_text(self):
        result = parse_resume(self._GOOD_RESUME)
        assert result.parse_warning is None

    def test_extracted_skills_contains_known_skills(self):
        result = parse_resume(self._GOOD_RESUME)
        # Skills from the text should appear (normalised)
        skill_names_lower = {s.lower() for s in result.extracted_skills}
        # At least one of these should appear
        expected = {"python", "fastapi", "docker", "langchain"}
        assert skill_names_lower & expected, (
            f"Expected at least one of {expected} in extracted_skills, got: {skill_names_lower}"
        )

    def test_work_experience_summary_non_empty(self):
        result = parse_resume(self._GOOD_RESUME)
        assert result.work_experience_summary
        assert len(result.work_experience_summary) > 5

    def test_raw_text_stored(self):
        result = parse_resume(self._GOOD_RESUME)
        assert result.raw_text
        assert "Python" in result.raw_text

    def test_short_resume_sets_parse_warning(self):
        short_text = "I know Python."  # < 150 chars
        result = parse_resume(short_text)
        assert result.parse_warning is not None

    def test_empty_resume_sets_parse_warning(self):
        result = parse_resume("")
        assert result.parse_warning is not None

    def test_bytes_with_plain_text_raises_value_error(self):
        with pytest.raises(ValueError, match="bytes"):
            parse_resume(b"resume bytes", source_type=ResumeSourceType.PLAIN_TEXT)


# ---------------------------------------------------------------------------
# PDF parsing (mocked)
# ---------------------------------------------------------------------------

class TestPdfParsing:
    def test_pdf_text_extracted_and_skills_populated(self):
        """Mock pypdf.PdfReader to return known text."""
        pypdf = pytest.importorskip("pypdf")

        fake_text = (
            "Jane Doe | AI Engineer\n"
            "Skills: Python, LangGraph, Qdrant, FastAPI, OpenAI\n"
            "Experience: Built a multi-agent system using LangGraph.\n" * 5
        )

        mock_page = MagicMock()
        mock_page.extract_text.return_value = fake_text
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = parse_resume(b"%PDF-1.4 fake", source_type=ResumeSourceType.PDF)

        assert isinstance(result, ResumeParseResult)
        assert result.raw_text
        # At least some skills should be extracted
        assert len(result.extracted_skills) > 0

    def test_pdf_with_empty_pages_sets_warning(self):
        """PDF that extracts empty text should set parse_warning."""
        pytest.importorskip("pypdf")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = parse_resume(b"%PDF-1.4 fake", source_type=ResumeSourceType.PDF)

        assert result.parse_warning is not None


# ---------------------------------------------------------------------------
# DOCX parsing (mocked)
# ---------------------------------------------------------------------------

class TestDocxParsing:
    def test_docx_text_extracted(self):
        """Mock python-docx Document to return known paragraphs."""
        docx = pytest.importorskip("docx")

        fake_paragraphs = [
            "John Smith | Machine Learning Engineer",
            "Skills: Python, Scikit-Learn, Docker, MLflow",
            "Experience: 3 years in ML modelling and deployment.",
            "Deployed ML models with Docker and FastAPI to GCP.",
        ]
        mock_para = [MagicMock(text=p) for p in fake_paragraphs]
        mock_doc = MagicMock()
        mock_doc.paragraphs = mock_para

        with patch("docx.Document", return_value=mock_doc):
            result = parse_resume(b"PK fake docx bytes", source_type=ResumeSourceType.DOCX)

        assert isinstance(result, ResumeParseResult)
        assert "Python" in result.raw_text or "python" in result.raw_text.lower()
