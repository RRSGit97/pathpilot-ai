"""
src/tools/job_parser.py
-----------------------
Clean and pre-process pasted job description text for PathPilot AI.

Responsibilities
----------------
- Accept 1–5 raw JD strings from the Streamlit UI.
- Normalise whitespace and Unicode.
- Optionally attempt to split a single blob into multiple JDs when the user
  pastes them all at once.
- Extract lightweight metadata (title, company, location) from the first
  few lines using simple heuristics — no LLM calls here.
- Extract keyword candidates for downstream evidence / traceability.
- Return :class:`ParsedJobDescription` objects that wrap
  :class:`~src.data.schemas.JobDescriptionInput` with the extra fields the
  parser adds.

What this does NOT do
---------------------
- No LLM-powered extraction (that lives in the graph nodes).
- No network requests or live scraping.
- No parsing of HTML or structured job-board formats.

Public API
----------
.. code-block:: python

    from src.tools.job_parser import parse_job_description, parse_multiple_jds

    result = parse_job_description(raw_text)
    batch  = parse_multiple_jds([raw1, raw2, raw3])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.data.schemas import JobDescriptionInput
from src.tools.text_utils import (
    extract_keyword_candidates,
    extract_source_snippet,
    normalize_skill_token,
    normalize_whitespace,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Minimum character count for a JD to be accepted (matches schema validator).
MIN_JD_CHARS: int = 50

#: Maximum number of JDs PathPilot accepts per session.
MAX_JD_COUNT: int = 5

#: Common patterns that signal the start of a new job posting when multiple
#: are pasted into a single text area.
#: Pre-compiled with re.MULTILINE for Python 3.13 compatibility
#: (inline (?m) flags in combined | patterns are not allowed in Python 3.13+).
_JD_SPLIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^-{5,}$",            re.MULTILINE),  # -----
    re.compile(r"^={5,}$",            re.MULTILINE),  # =====
    re.compile(r"^#{1,3}\s+Job\b",    re.MULTILINE),  # ## Job Title
    re.compile(r"^Position\s*:\s*\S", re.MULTILINE),  # Position: Something
    re.compile(r"^Job\s+Title\s*:\s*\S", re.MULTILINE),  # Job Title: Something
    re.compile(r"^\*{3,}$",           re.MULTILINE),  # ***
]

#: Regex for lines that look like a job title (first non-empty line heuristic).
_TITLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)^(?:job\s+)?title\s*[:\-]\s*(.+)$"),
    re.compile(r"(?i)^(?:position|role)\s*[:\-]\s*(.+)$"),
    re.compile(r"(?i)^hiring\s+(?:for|a)\s*[:\-]?\s*(.+)$"),
]

#: Regex for lines that look like a company name.
_COMPANY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)^company\s*[:\-]\s*(.+)$"),
    re.compile(r"(?i)^(?:at|@)\s+([A-Z][A-Za-z0-9\s&,\.]+)$"),
    re.compile(r"(?i)^employer\s*[:\-]\s*(.+)$"),
]

#: Regex for lines that look like a location.
_LOCATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)^location\s*[:\-]\s*(.+)$"),
    re.compile(r"(?i)^(?:remote|hybrid|on-?site)\b(.*)$"),
    re.compile(r"(?i)^based\s+in\s+(.+)$"),
]


# ---------------------------------------------------------------------------
# Output type (richer than the plain schema model)
# ---------------------------------------------------------------------------

@dataclass
class ParsedJobDescription:
    """
    Cleaned and enriched job description ready for the LLM extraction node.

    Fields
    ------
    jd:
        The validated :class:`~src.data.schemas.JobDescriptionInput` with
        ``raw_text`` set to the normalised content.
    cleaned_text:
        Same as ``jd.raw_text`` — exposed for convenience.
    inferred_title:
        Best-guess title from the first few lines, or ``""`` if not found.
    inferred_company:
        Best-guess company name, or ``""`` if not found.
    inferred_location:
        Best-guess location / work-mode, or ``""`` if not found.
    keyword_candidates:
        Sorted list of normalised skill tokens found in the text.
    source_snippets:
        Mapping of ``keyword → context snippet`` for traceability.
    parse_notes:
        Human-readable notes about what the parser did (e.g. split info).
    """

    jd: JobDescriptionInput
    cleaned_text: str
    inferred_title: str = ""
    inferred_company: str = ""
    inferred_location: str = ""
    keyword_candidates: list[str] = field(default_factory=list)
    source_snippets: dict[str, str] = field(default_factory=dict)
    parse_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_jd_text(raw: str) -> str:
    """
    Normalise whitespace then strip leading/trailing blank lines.

    Preserves single blank lines between paragraphs (important for section
    detection).
    """
    text = normalize_whitespace(raw)
    # Strip leading/trailing whitespace from the whole document
    return text.strip()


def _attempt_split(raw: str) -> list[str]:
    """
    Try to split a single blob into multiple JDs using delimiter heuristics.

    Returns a list of candidate strings.  If no split is detected, returns
    ``[raw]``.  Callers should still validate each segment for minimum length.
    """
    # Try each pattern in turn; use the first that splits the text.
    for pat in _JD_SPLIT_PATTERNS:
        segments = pat.split(raw)
        segments = [s.strip() for s in segments if s.strip()]
        if len(segments) > 1:
            logger.debug("Auto-split detected %d JD segments.", len(segments))
            return segments

    return [raw]


def _extract_metadata(text: str) -> tuple[str, str, str]:
    """
    Scan the first 20 lines for title, company, and location signals.

    Returns (title, company, location) — each may be ``""`` if not found.
    """
    lines = text.splitlines()[:20]

    title = company = location = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        for pat in _TITLE_PATTERNS:
            m = pat.match(line)
            if m and not title:
                title = m.group(1).strip()
                break

        for pat in _COMPANY_PATTERNS:
            m = pat.match(line)
            if m and not company:
                company = m.group(1).strip()
                break

        for pat in _LOCATION_PATTERNS:
            m = pat.match(line)
            if m and not location:
                location = m.group(0).strip()
                break

    # Fallback: if no explicit title found, treat the first non-empty line
    # that is short (< 80 chars) and title-cased as the title.
    if not title:
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 80 and stripped[0].isupper():
                title = stripped
                break

    return title, company, location


def _build_snippets(
    text: str,
    keywords: list[str],
    max_keywords: int = 20,
) -> dict[str, str]:
    """
    Build a ``{keyword: snippet}`` map for the first *max_keywords* terms.

    Keeps the map small — this is for traceability display only.
    """
    snippets: dict[str, str] = {}
    for kw in keywords[:max_keywords]:
        snippet = extract_source_snippet(text, kw, window=100)
        if snippet:
            snippets[kw] = snippet
    return snippets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_job_description(
    raw_text: str,
    title_hint: str = "",
    source_url: str | None = None,
) -> ParsedJobDescription:
    """
    Clean and enrich a single pasted job description.

    Parameters
    ----------
    raw_text:
        The raw copy-pasted JD text.
    title_hint:
        Optional title the user typed in (used if metadata extraction fails).
    source_url:
        Optional URL the user pasted from.

    Returns
    -------
    ParsedJobDescription
        Always returns a result — validation errors from the schema are
        propagated as ``pydantic.ValidationError`` so the UI can surface them.

    Raises
    ------
    pydantic.ValidationError
        If the cleaned text is shorter than 50 characters.
    """
    notes: list[str] = []

    cleaned = _clean_jd_text(raw_text)

    title, company, location = _extract_metadata(cleaned)

    # Prefer user-provided hint over inferred title
    effective_title = title_hint.strip() or title

    keywords = extract_keyword_candidates(cleaned)
    # Normalise each candidate
    norm_keywords = sorted({normalize_skill_token(k) for k in keywords if k})

    snippets = _build_snippets(cleaned, norm_keywords)

    if company:
        notes.append(f"Inferred company: {company!r}")
    if location:
        notes.append(f"Inferred location: {location!r}")
    if not effective_title:
        notes.append("No title could be inferred from the first 20 lines.")

    jd = JobDescriptionInput(
        title=effective_title,
        raw_text=cleaned,
        source_url=source_url,
    )

    return ParsedJobDescription(
        jd=jd,
        cleaned_text=cleaned,
        inferred_title=effective_title,
        inferred_company=company,
        inferred_location=location,
        keyword_candidates=norm_keywords,
        source_snippets=snippets,
        parse_notes=notes,
    )


def parse_multiple_jds(
    raw_inputs: list[str],
    title_hints: list[str] | None = None,
    source_urls: list[str | None] | None = None,
    attempt_auto_split: bool = False,
) -> list[ParsedJobDescription]:
    """
    Parse and clean a batch of 1–5 pasted job descriptions.

    Parameters
    ----------
    raw_inputs:
        One string per JD.  If the user pasted everything into one text area,
        pass a single-element list and set *attempt_auto_split=True*.
    title_hints:
        Optional parallel list of title strings typed by the user.
    source_urls:
        Optional parallel list of source URLs.
    attempt_auto_split:
        When ``True`` and ``len(raw_inputs) == 1``, the parser will try to
        detect multiple JDs in the single blob and split them automatically.

    Returns
    -------
    list[ParsedJobDescription]
        One item per successfully parsed JD.  Empty inputs are silently
        skipped (not an error).

    Raises
    ------
    ValueError
        If more than :data:`MAX_JD_COUNT` valid descriptions are provided.
    """
    if title_hints is None:
        title_hints = [""] * len(raw_inputs)
    if source_urls is None:
        source_urls = [None] * len(raw_inputs)

    # Auto-split mode: single blob containing multiple JDs
    if attempt_auto_split and len(raw_inputs) == 1:
        segments = _attempt_split(raw_inputs[0])
        if len(segments) > 1:
            logger.info("Auto-split produced %d JD segments.", len(segments))
            raw_inputs = segments
            title_hints = [""] * len(segments)
            source_urls = [None] * len(segments)

    results: list[ParsedJobDescription] = []

    for i, (raw, hint, url) in enumerate(zip(raw_inputs, title_hints, source_urls)):
        # Skip blank entries silently (user left a text area empty)
        if not raw or not raw.strip():
            continue

        if len(results) >= MAX_JD_COUNT:
            logger.warning(
                "Reached MAX_JD_COUNT (%d); ignoring remaining inputs.", MAX_JD_COUNT
            )
            break

        try:
            parsed = parse_job_description(raw, title_hint=hint, source_url=url)
            results.append(parsed)
            logger.debug("JD %d parsed: title=%r, keywords=%d", i + 1, parsed.inferred_title, len(parsed.keyword_candidates))
        except Exception as exc:
            # Let the caller decide whether to surface or skip bad entries
            logger.warning("JD %d failed to parse: %s", i + 1, exc)
            raise

    if len(results) > MAX_JD_COUNT:
        raise ValueError(
            f"Too many job descriptions: {len(results)} provided, maximum is {MAX_JD_COUNT}."
        )

    return results


def summarise_jd_batch(parsed: list[ParsedJobDescription]) -> dict[str, object]:
    """
    Return a lightweight summary dict for the whole JD batch.

    Useful for logging, debugging, and the Streamlit preview panel.

    Returns
    -------
    dict with keys:
        ``count``, ``titles``, ``all_keywords``, ``total_chars``
    """
    all_keywords: set[str] = set()
    for p in parsed:
        all_keywords.update(p.keyword_candidates)

    return {
        "count": len(parsed),
        "titles": [p.inferred_title or "(no title)" for p in parsed],
        "all_keywords": sorted(all_keywords),
        "total_chars": sum(len(p.cleaned_text) for p in parsed),
    }
