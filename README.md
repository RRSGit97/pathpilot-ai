# PathPilot AI

PathPilot AI is an agentic career and project-planning assistant that takes a target AI role, current profile, and job descriptions to produce a personalized learning roadmap, scored project ideas, and resume-ready builder artifacts.

## Quick Demo (Sample Data)

To demonstrate and test PathPilot AI without uploading your own resume or writing job descriptions from scratch:

1. Launch the Streamlit application (see instructions below).
2. Click the **💾 Load Demo (Sample Data)** button in the sidebar. This loads a curated profile of a junior developer transitioning to AI engineering, along with 4 realistic job descriptions (AI Engineer, Applied AI Engineer, ML Engineer, and Data Scientist).
3. Navigate to **2. Job Descriptions** (Tab 2) and click the **🚀 Run Full Analysis** button to run the agentic pipeline.
4. The sample files used for this demo path are stored in:
   - `sample_data/sample_profile.json` — Transitioning AI Engineering profile
   - `sample_data/sample_job_descriptions.json` — 4 realistic job descriptions
   - `sample_data/sample_resources.json` — Curated local resources mapped to missing skills (automatically ingested to the vector store if enabled)

## Features

- **Profile & Resume Parsing**: Input current skills, upload PDF resume.
- **JD Analyzer**: Paste 3–5 job descriptions to extract required and optional skills.
- **Skill Gap Mapping**: Visual categorization of skill alignments.
- **Personalized Roadmap**: Week-by-week instruction program.
- **Scored Project Generation**: Agentic pipeline suggesting ranked projects.
- **Human-in-the-loop Controls**: Review and refine target role, roadmap, and artifacts.

## Architecture

PathPilot AI is powered by:
- **LangGraph**: Orchestrates a 14-node career discovery graph with human-in-the-loop review checkpoints.
- **SQLite**: Manages application memory (Pydantic models mapping to session IDs) and workflow checkpointer state.
- **Qdrant (Optional)**: Provides local semantic search for target resources and job descriptions.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

## Testing

```bash
pytest tests/
```

