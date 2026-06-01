from src.schemas.project import ProjectIdea, ProjectScores, ScoredProject

def score_project(idea: ProjectIdea, user_profile: dict) -> ScoredProject:
    # Placeholder project scorer logic
    scores = ProjectScores(
        resume_value=0.0,
        agentic_fit=0.0,
        buildability_4_6_weeks=0.0,
        personal_relevance=0.0,
        technical_depth=0.0,
        differentiation=0.0,
        recruiter_explainability=0.0
    )
    return ScoredProject(
        idea=idea,
        scores=scores,
        composite_score=0.0
    )
