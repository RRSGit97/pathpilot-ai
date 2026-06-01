"""
src/tools
---------
Stateless parsing and text-processing utilities.

All functions here are pure Python — no LLM calls, no DB access.
They take raw user input and return cleaned structures ready for the
LangGraph nodes to consume.

Public surface::

    from src.tools.text_utils import normalize_skill_token, extract_keyword_candidates
    from src.tools.resume_parser import parse_resume
    from src.tools.job_parser import parse_job_description, parse_multiple_jds
"""
