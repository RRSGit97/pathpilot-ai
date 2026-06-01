"""
src/tools/text_utils.py
-----------------------
Shared, pure-Python text helpers used by both resume_parser and job_parser.

No external dependencies beyond the Python standard library, so this module
is safe to import anywhere — including test files that run without network
access.

Functions
---------
normalize_whitespace        Clean up messy spacing and line endings.
normalize_skill_token       Lowercase, strip, apply common aliases.
extract_keyword_candidates  Simple stopword-filtered token list.
extract_source_snippet      Context window around a keyword occurrence.
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Small hand-curated stopword set
# (avoids pulling in NLTK/spaCy for an MVP-scoped tool)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "not", "no", "nor", "so", "yet", "both", "either",
    "its", "it", "this", "that", "these", "those", "we", "our", "you", "your",
    "they", "their", "he", "she", "his", "her", "as", "if", "then", "than",
    "too", "very", "just", "more", "also", "such", "each", "how", "what",
    "when", "where", "who", "which", "while", "after", "before", "between",
    "under", "over", "again", "any", "all", "both", "few", "most", "other",
    "some", "same", "only", "own", "off", "out", "once",
})

# ---------------------------------------------------------------------------
# Common skill alias normalisations
# (extend this dict as needed — alphabetical order for easy review)
# ---------------------------------------------------------------------------

_SKILL_ALIASES: dict[str, str] = {
    "aws lambda":         "aws",
    "css3":               "css",
    "html5":              "html",
    "js":                 "javascript",
    "k8s":                "kubernetes",
    "ml":                 "machine learning",
    "nlp":                "natural language processing",
    "oop":                "object-oriented programming",
    "postgres":           "postgresql",
    "py":                 "python",
    "python3":            "python",
    "react.js":           "react",
    "reactjs":            "react",
    "rest api":           "rest",
    "restful":            "rest",
    "restful api":        "rest",
    "typescript":         "typescript",  # keep as-is but entry ensures no alias
    "ts":                 "typescript",
    "vue.js":             "vue",
    "vuejs":              "vue",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """
    Collapse runs of whitespace (spaces, tabs, non-breaking spaces) to single
    spaces and runs of blank lines to a single blank line.

    Does NOT strip leading/trailing whitespace from the returned string so the
    caller decides how to handle document boundaries.

    Examples
    --------
    >>> normalize_whitespace("Hello   world\\n\\n\\n  Foo")
    'Hello world\\n\\nFoo'
    """
    # Normalise unicode (e.g. \u00a0 → space)
    text = unicodedata.normalize("NFKC", text)
    # Replace all horizontal whitespace characters with a plain space
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    # Strip leading spaces from every line (tabs converted above leave lone spaces)
    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
    # Collapse 3+ consecutive newlines to exactly two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces at end of each line
    text = re.sub(r" +\n", "\n", text)
    return text


def normalize_skill_token(token: str) -> str:
    """
    Return a canonical, lower-cased form of a skill name.

    Steps
    -----
    1. NFKC normalise and lowercase.
    2. Strip surrounding punctuation / whitespace.
    3. Apply alias lookup (see ``_SKILL_ALIASES``).

    Examples
    --------
    >>> normalize_skill_token("  Python3 ")
    'python'
    >>> normalize_skill_token("K8S")
    'kubernetes'
    >>> normalize_skill_token("REST API")
    'rest'
    """
    cleaned = unicodedata.normalize("NFKC", token).lower().strip(" \t\n.,;:/'\"()")
    return _SKILL_ALIASES.get(cleaned, cleaned)


def extract_keyword_candidates(
    text: str,
    min_length: int = 2,
    max_length: int = 40,
) -> list[str]:
    """
    Return a de-duplicated, sorted list of candidate skill tokens from *text*.

    Strategy
    --------
    - Split on word boundaries; keep alphanumeric tokens and tokens containing
      common tech separators (``-``, ``+``, ``.``, ``/``).
    - Drop English stopwords.
    - Drop tokens shorter than *min_length* or longer than *max_length*.
    - Normalise each surviving token via :func:`normalize_skill_token`.
    - Return sorted unique values.

    This is intentionally conservative — precision over recall.  Downstream
    LLM extraction handles the nuanced cases.

    Parameters
    ----------
    text:
        Raw or lightly normalised text.
    min_length:
        Minimum character count for a token to be kept.
    max_length:
        Maximum character count (guards against garbled text producing huge
        tokens).
    """
    # Tokenise: split on whitespace then strip surrounding punctuation
    raw_tokens: list[str] = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9\-\+\.\/]*[A-Za-z0-9]|[A-Za-z0-9]{2,}",
        text,
    )

    seen: set[str] = set()
    result: list[str] = []

    for tok in raw_tokens:
        normalised = normalize_skill_token(tok)
        if (
            normalised not in _STOPWORDS
            and min_length <= len(normalised) <= max_length
            and normalised not in seen
        ):
            seen.add(normalised)
            result.append(normalised)

    return sorted(result)


def extract_source_snippet(
    text: str,
    keyword: str,
    window: int = 120,
) -> str | None:
    """
    Return a short context snippet around the first occurrence of *keyword*.

    The snippet is *window* characters wide, centred on the match.  Returns
    ``None`` if *keyword* is not found.  Useful for traceability — lets the UI
    show *why* a skill was extracted.

    Parameters
    ----------
    text:
        Source document text (any length).
    keyword:
        The skill or term to locate (case-insensitive).
    window:
        Total character width of the returned snippet.

    Examples
    --------
    >>> extract_source_snippet("We need Python skills.", "python", window=30)
    '...We need Python skills....'
    """
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return None

    half = window // 2
    start = max(0, match.start() - half)
    end = min(len(text), match.end() + half)

    snippet = text[start:end].replace("\n", " ").strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"
