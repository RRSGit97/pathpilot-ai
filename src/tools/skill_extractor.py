"""
src/tools/skill_extractor.py
-----------------------------
LLM-powered role requirement extraction from cleaned job descriptions.

How the pipeline works (step by step)
--------------------------------------
1. Each :class:`~src.tools.job_parser.ParsedJobDescription` is sent to the
   LLM individually.  Keeping them separate avoids context bleed between
   different roles and makes it easy to trace which JD contributed each skill.

2. The LLM returns a ``SingleJDExtraction`` Pydantic object using LangChain's
   ``with_structured_output()`` — this enforces a JSON schema and eliminates
   free-form hallucination.

3. All per-JD extractions are passed to the deterministic aggregator which:
   - Normalises every skill token (``K8S`` → ``kubernetes``)
   - Counts how many JDs mention each skill
   - Promotes any skill that is *required* in at least one JD to the
     ``required_skills`` list (conservative rule: required anywhere = required)
   - Deduplicates and sorts lists by frequency
   - Pulls evidence snippets from the original JD text

4. Returns one :class:`~src.data.schemas.AggregatedRoleAnalysis` object.

LLM calls are only in step 2.  All other work is pure Python and is
independently testable without an API key.

Quick usage::

    from src.tools.job_parser import parse_multiple_jds
    from src.tools.skill_extractor import extract_role_requirements

    parsed_jds = parse_multiple_jds([raw_jd_1, raw_jd_2, raw_jd_3])
    analysis = extract_role_requirements(parsed_jds)
    print(analysis.required_skills)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.config import settings
from src.data.enums import LLMProvider
from src.data.prompts import JD_EXTRACTION_SYSTEM_PROMPT, JD_EXTRACTION_USER_TEMPLATE
from src.data.schemas import AggregatedRoleAnalysis
from src.tools.job_parser import ParsedJobDescription
from src.tools.text_utils import extract_source_snippet, normalize_skill_token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — LLM output schema (per single JD)
# ---------------------------------------------------------------------------

class SingleJDExtraction(BaseModel):
    """
    Pydantic schema that the LLM must return for each job description.

    LangChain's ``with_structured_output()`` validates the response against
    this schema automatically.  Fields that the LLM leaves blank default to
    empty lists / strings — we never crash on a missing key.
    """

    inferred_title: str = Field(default="")
    seniority: str = Field(default="unknown")
    required_skills: list[str] = Field(default_factory=list)
    optional_skills: list[str] = Field(default_factory=list)
    tools_and_frameworks: list[str] = Field(default_factory=list)
    project_expectations: list[str] = Field(default_factory=list)
    seniority_signals: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 2 — Build the LLM client
# ---------------------------------------------------------------------------

def build_llm(temperature: float = 0.0):
    """
    Create a LangChain chat model from the app settings.

    Temperature is set to 0 by default to make extraction as deterministic
    as possible.  Low temperature = less creativity = fewer hallucinations.

    Supported providers
    -------------------
    - ``openai``   → ChatOpenAI (langchain-openai must be installed)
    - ``gemini``   → ChatGoogleGenerativeAI (langchain-google-genai needed)
    - ``anthropic`` → ChatAnthropic (langchain-anthropic needed)

    Raises
    ------
    ValueError
        If the configured provider is not supported or the API key is missing.
    ImportError
        If the required provider package is not installed.
    """
    # Use the runtime key (checks session override first, then .env)
    api_key = settings.get_runtime_api_key()
    if not api_key:
        raise ValueError(
            f"No API key configured for provider '{settings.llm_provider}'. "
            f"Enter your key in the PathPilot app or set it in your .env file."
        )

    if settings.llm_provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OpenAI. "
                "Run: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=api_key,
            temperature=temperature,
        )

    if settings.llm_provider == LLMProvider.GEMINI:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for Gemini. "
                "Run: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=api_key,
            temperature=temperature,
        )

    if settings.llm_provider == LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for Anthropic. "
                "Run: pip install langchain-anthropic"
            ) from exc
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=api_key,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{settings.llm_provider}'. "
        "Expected one of: openai, gemini, anthropic."
    )


# ---------------------------------------------------------------------------
# Step 3 — Extract from a single JD
# ---------------------------------------------------------------------------

def extract_from_single_jd(
    parsed_jd: ParsedJobDescription,
    llm,
) -> SingleJDExtraction:
    """
    Run LLM extraction on one cleaned job description.

    Parameters
    ----------
    parsed_jd:
        A job description already cleaned by ``job_parser.parse_job_description``.
    llm:
        A LangChain chat model (from :func:`build_llm`).

    Returns
    -------
    SingleJDExtraction
        Structured extraction result.  All fields default to empty if the LLM
        didn't find relevant content.

    Notes
    -----
    - ``with_structured_output`` forces the LLM to return JSON matching the
      schema.  This eliminates free-form responses.
    - We include the inferred title in the prompt so the LLM can use it as
      context when splitting required vs optional skills.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    user_content = JD_EXTRACTION_USER_TEMPLATE.format(jd_text=parsed_jd.cleaned_text)

    structured_llm = llm.with_structured_output(SingleJDExtraction)

    logger.debug(
        "Calling LLM for JD '%s' (%d chars).",
        parsed_jd.inferred_title or "(no title)",
        len(parsed_jd.cleaned_text),
    )

    result: SingleJDExtraction = structured_llm.invoke([
        SystemMessage(content=JD_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    logger.debug(
        "LLM returned: required=%d optional=%d tools=%d",
        len(result.required_skills),
        len(result.optional_skills),
        len(result.tools_and_frameworks),
    )
    return result


# ---------------------------------------------------------------------------
# Step 4 — Deterministic post-processing helpers
# ---------------------------------------------------------------------------

def _normalize_list(items: list[str]) -> list[str]:
    """
    Normalize, strip, and deduplicate a list of skill strings.

    - Applies ``normalize_skill_token`` (lowercases, applies alias dict).
    - Removes empty strings.
    - Deduplicates while preserving first-seen order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalised = normalize_skill_token(item.strip())
        if normalised and normalised not in seen:
            seen.add(normalised)
            result.append(normalised)
    return result


def _sort_by_count(items: list[str], counts: Counter) -> list[str]:
    """Sort a list of skill strings by their count, highest first."""
    return sorted(items, key=lambda s: counts.get(s, 0), reverse=True)


def _infer_seniority_consensus(extractions: list[SingleJDExtraction]) -> str:
    """
    Return the most common seniority level across all JD extractions.

    Strategy
    --------
    - Ignore "unknown" values when counting.
    - If the result is still a tie, prefer "senior" > "mid" > "junior" > "staff".
    - If every extraction returned "unknown", return "unknown".
    """
    priority = {"senior": 4, "mid": 3, "junior": 2, "staff": 1, "unknown": 0}
    known = [e.seniority for e in extractions if e.seniority != "unknown"]
    if not known:
        return "unknown"
    # Most common, with priority as tiebreaker
    counter = Counter(known)
    return max(counter, key=lambda s: (counter[s], priority.get(s, 0)))


def _consensus_title(extractions: list[SingleJDExtraction]) -> tuple[str, list[str]]:
    """
    Pick one representative title and collect all distinct titles.

    Tiebreaker: when two titles have equal counts, prefer the longer one
    (e.g. "Senior AI Engineer" over "AI Engineer") — specificity is better.
    This makes the selection stable and deterministic.

    Returns
    -------
    (consensus_title, all_distinct_titles)
    """
    titles = [e.inferred_title.strip() for e in extractions if e.inferred_title.strip()]
    if not titles:
        return "", []
    counter = Counter(titles)
    # Sort by (count DESC, length DESC) for stable, deterministic result
    best = max(counter, key=lambda t: (counter[t], len(t)))
    distinct = list(dict.fromkeys(titles))  # preserve insertion order, deduplicate
    return best, distinct


def _build_evidence_snippets(
    skill: str,
    parsed_jds: list[ParsedJobDescription],
    max_snippets: int = 3,
) -> list[str]:
    """
    Find up to *max_snippets* context windows for *skill* across all JDs.

    First checks each JD's precomputed ``source_snippets`` dict (fast).
    Falls back to ``extract_source_snippet`` on the full text (slower, handles
    multi-word skills not caught during keyword tokenisation).

    Parameters
    ----------
    skill:
        The normalised skill name to look up.
    parsed_jds:
        Source job descriptions with their cleaned text.
    max_snippets:
        Maximum number of evidence snippets to return.
    """
    snippets: list[str] = []
    for jd in parsed_jds:
        if len(snippets) >= max_snippets:
            break
        # Fast path: precomputed single-token snippets
        if skill in jd.source_snippets:
            snippets.append(jd.source_snippets[skill])
            continue
        # Slow path: scan for multi-word skills or aliases
        snippet = extract_source_snippet(jd.cleaned_text, skill, window=120)
        if snippet:
            snippets.append(snippet)
    return snippets


# ---------------------------------------------------------------------------
# Step 5 — Aggregate all per-JD extractions
# ---------------------------------------------------------------------------

def aggregate_extractions(
    extractions: list[SingleJDExtraction],
    parsed_jds: list[ParsedJobDescription],
) -> AggregatedRoleAnalysis:
    """
    Deterministically merge per-JD extractions into one unified analysis.

    This function contains zero LLM calls — it is pure Python and fully
    testable without an API key.

    Aggregation rules
    -----------------
    required_skills
        Union of required skills from ALL JDs.  A skill is required if ANY
        single JD lists it as required (conservative — avoids under-counting).

    optional_skills
        Union of optional skills that did NOT appear as required in any JD.
        Skill counts are tracked so the UI can show "2 / 3 JDs mention this".

    tools_and_frameworks
        Union across all JDs, deduplicated.

    project_expectations
        Deduplicated by normalised text.  Long phrases kept as-is.

    seniority
        Majority vote, ignoring "unknown" responses.
    """
    # ── Collect required/optional skill sets ──────────────────────────────
    required_raw: list[str] = []
    optional_raw: list[str] = []
    tools_raw: list[str] = []
    expectations_raw: list[str] = []
    seniority_signals_raw: list[str] = []

    # Track which skills appear as required in at least one JD
    required_anywhere: set[str] = set()

    # Count how many JDs mention each normalised skill
    skill_counts: Counter = Counter()

    for extraction in extractions:
        norm_req = _normalize_list(extraction.required_skills)
        norm_opt = _normalize_list(extraction.optional_skills)
        norm_tools = _normalize_list(extraction.tools_and_frameworks)

        required_anywhere.update(norm_req)

        # Count this JD's contribution (each skill counts once per JD)
        for s in set(norm_req + norm_opt + norm_tools):
            skill_counts[s] += 1

        required_raw.extend(norm_req)
        optional_raw.extend(norm_opt)
        tools_raw.extend(norm_tools)
        expectations_raw.extend(extraction.project_expectations)
        seniority_signals_raw.extend(extraction.seniority_signals)

    # ── Build final deduplicated skill lists ──────────────────────────────

    # Required: anything required in at least one JD
    required_final = _sort_by_count(
        list(dict.fromkeys(required_raw)),   # deduplicate, preserve first-seen order
        skill_counts,
    )

    # Optional: only skills that were NEVER in any required section
    optional_deduped = list(dict.fromkeys(optional_raw))
    optional_final = _sort_by_count(
        [s for s in optional_deduped if s not in required_anywhere],
        skill_counts,
    )

    # Tools: union of all tools, sorted by frequency
    tools_final = _sort_by_count(
        list(dict.fromkeys(tools_raw)),
        skill_counts,
    )

    # Project expectations: deduplicate by normalised text (lowercased, stripped)
    seen_exp: set[str] = set()
    expectations_final: list[str] = []
    for exp in expectations_raw:
        key = exp.strip().lower()
        if key and key not in seen_exp:
            seen_exp.add(key)
            expectations_final.append(exp.strip())

    # Seniority signals: deduplicate verbatim phrases
    seen_sig: set[str] = set()
    seniority_signals_final: list[str] = []
    for sig in seniority_signals_raw:
        key = sig.strip().lower()
        if key and key not in seen_sig:
            seen_sig.add(key)
            seniority_signals_final.append(sig.strip())

    # ── Evidence snippets ─────────────────────────────────────────────────
    all_skills = list(dict.fromkeys(required_final + optional_final + tools_final))
    evidence: dict[str, list[str]] = {}
    for skill in all_skills:
        snippets = _build_evidence_snippets(skill, parsed_jds, max_snippets=3)
        if snippets:
            evidence[skill] = snippets

    # ── Seniority & title consensus ───────────────────────────────────────
    seniority = _infer_seniority_consensus(extractions)
    consensus_title, all_titles = _consensus_title(extractions)

    return AggregatedRoleAnalysis(
        source_jd_count=len(extractions),
        consensus_title=consensus_title,
        all_titles=all_titles,
        seniority=seniority,
        seniority_signals=seniority_signals_final,
        required_skills=required_final,
        optional_skills=optional_final,
        tools_and_frameworks=tools_final,
        project_expectations=expectations_final,
        skill_jd_counts=dict(skill_counts),
        evidence_snippets=evidence,
    )


# ---------------------------------------------------------------------------
# Public API — main pipeline entry point
# ---------------------------------------------------------------------------

def extract_role_requirements(
    parsed_jds: list[ParsedJobDescription],
    llm=None,
) -> AggregatedRoleAnalysis:
    """
    Run the full extraction pipeline on 1–5 cleaned job descriptions.

    This is the main function callers should use.

    Parameters
    ----------
    parsed_jds:
        List of :class:`~src.tools.job_parser.ParsedJobDescription` objects
        produced by ``parse_multiple_jds()``.  Must not be empty.
    llm:
        A LangChain chat model.  If ``None``, one is created automatically
        from ``src.config.settings``.  Pass an explicit model in tests or
        when you want to use a specific configuration.

    Returns
    -------
    AggregatedRoleAnalysis
        Merged extraction across all provided JDs.

    Raises
    ------
    ValueError
        If ``parsed_jds`` is empty or if the API key is missing.

    Example
    -------
    ::

        parsed = parse_multiple_jds([raw_jd_1, raw_jd_2])
        analysis = extract_role_requirements(parsed)
        print(analysis.required_skills[:5])
        print(analysis.seniority)
    """
    if not parsed_jds:
        raise ValueError("parsed_jds must not be empty.")

    if llm is None:
        llm = build_llm()

    extractions: list[SingleJDExtraction] = []
    for i, jd in enumerate(parsed_jds, start=1):
        logger.info("Extracting requirements from JD %d/%d: '%s'", i, len(parsed_jds), jd.inferred_title or "(no title)")
        extraction = extract_from_single_jd(jd, llm)
        extractions.append(extraction)

    logger.info("Aggregating %d JD extractions...", len(extractions))
    return aggregate_extractions(extractions, parsed_jds)
