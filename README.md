# PathPilot AI

**PathPilot AI** is an agentic AI career and project-planning assistant that turns a user profile, target AI roles, and job descriptions into a personalized skill-gap analysis, learning roadmap, ranked project pipeline, six-week build plan, and portfolio-ready outputs.

It was built as a practical AI engineering project to demonstrate **LLM application development, LangGraph workflow orchestration, structured outputs, human-in-the-loop checkpoints, SQLite memory, optional vector search, and Streamlit product design**.

---

## Links

- **Live App:** https://pathpilot-ai.streamlit.app
- **Video Demo:** https://app.guidde.com/share/playbooks/m3JM19fMYzvBQHLxE4Cf2n?origin=EMw3SaL2yhSrH41cueQA4v6xWVo2&t=0&mode=videoAndDoc
- **GitHub Repository:** https://github.com/RRSGit97/pathpilot-ai

---

## App Preview

> The app opens with a secure API-key setup flow and then guides the user through profile input, job-description analysis, role analysis, skill-gap mapping, roadmap generation, project scoring, build planning, portfolio output generation, and memory/progress tracking.

![PathPilot AI App Preview](assets/pathpilot-ai-app-preview.png)

If the screenshot does not render yet, upload the app screenshot to `assets/pathpilot-ai-app-preview.png` in this repository.

---

## What PathPilot AI Does

PathPilot AI helps an aspiring AI engineer answer four practical questions:

1. **What skills do target AI roles actually require?**
2. **Which of those skills do I already have, partially have, or still need to build?**
3. **What projects should I build to close those gaps and strengthen my resume?**
4. **How do I turn the chosen project into a structured build plan and portfolio story?**

Instead of acting like a generic chatbot, PathPilot AI uses a structured agentic workflow to move through a career-planning pipeline with saved state, deterministic scoring, and human review checkpoints.

---

## Quick Demo Flow

To demonstrate PathPilot AI without uploading a real resume or writing job descriptions from scratch:

1. Open the live app or run it locally.
2. Enter an OpenAI API key in the secure session-only API-key screen.
3. Click **Load Demo (Sample Data)** in the sidebar.
4. Go to the **Job Descriptions** tab.
5. Click **Run Full Analysis**.
6. Review the generated outputs across:
   - Role Analysis
   - Skill Gap
   - Roadmap
   - Project Ideas
   - Scoring
   - Build Plan
   - Portfolio
   - Memory

The sample demo files are stored in:

- `sample_data/sample_profile.json`
- `sample_data/sample_job_descriptions.json`
- `sample_data/sample_resources.json`

---

## Key Features

- **Profile and Resume Input**: Enter background, skills, prior projects, goals, time availability, and learning preferences.
- **Resume Upload Support**: Upload PDF, TXT, or DOCX resumes for profile enrichment.
- **Job Description Analysis**: Paste multiple job descriptions and extract required skills, optional skills, tools, frameworks, project expectations, and seniority signals.
- **Skill-Gap Mapping**: Compare the user's current profile against target-role requirements.
- **Personalized Learning Roadmap**: Generate a week-by-week roadmap focused on missing or partial skills.
- **Project Idea Generation**: Produce practical, resume-worthy AI engineering project ideas.
- **Deterministic Project Scoring**: Rank project ideas using a transparent scoring rubric.
- **Six-Week Build Plan**: Convert a selected project into weekly milestones, deliverables, tests, and demo outputs.
- **Portfolio Output Generator**: Generate README outline, resume bullet, architecture summary, demo script, and interview explanation.
- **Human-in-the-Loop Controls**: Pause for user review before approving key outputs.
- **Session Memory**: Store progress and generated outputs through SQLite-backed memory.
- **Optional Vector Store Layer**: Use Qdrant for semantic retrieval when enabled.

---

## Why This Is Agentic AI

PathPilot AI is not a single-turn chatbot. It follows a multi-step workflow with explicit state, specialized nodes, structured outputs, and review checkpoints.

The system:

- Maintains a shared pipeline state across the workflow.
- Separates role extraction, skill-gap mapping, roadmap generation, project ideation, project scoring, build planning, and portfolio generation.
- Uses deterministic scoring in addition to LLM-generated reasoning.
- Pauses at human review checkpoints before continuing to later stages.
- Persists user/session progress through SQLite-backed memory.

---

## Architecture

PathPilot AI is built around a Streamlit UI, a LangGraph workflow, structured Pydantic data models, and local persistence.

```text
User
  |
  v
Streamlit UI
  |
  v
LangGraph Career Planning Workflow
  |
  |-- Profile ingestion
  |-- Resume parsing
  |-- Job-description extraction
  |-- Skill-gap mapping
  |-- Human review: role and gap approval
  |-- Roadmap generation
  |-- Human review: roadmap approval
  |-- Project ideation
  |-- Deterministic project scoring
  |-- Human review: project selection
  |-- Project critique
  |-- Six-week build-plan generation
  |-- Portfolio output generation
  |-- Human review: portfolio approval
  |
  v
SQLite Memory + Optional Qdrant Vector Store
```

### Core Components

- **Streamlit**: Frontend and product interface.
- **LangGraph**: Agentic workflow orchestration with checkpointing and interrupts.
- **LangChain / LangChain OpenAI**: LLM integration utilities.
- **Pydantic**: Structured schemas for profile, role analysis, skill gaps, project ideas, build plans, and portfolio outputs.
- **SQLite**: Local structured memory and workflow checkpoint persistence.
- **Qdrant**: Optional vector store for semantic retrieval over resources and notes.
- **pytest**: Tests for parsing, scoring, memory, and workflow behavior.

---

## Human-in-the-Loop Checkpoints

PathPilot AI includes human review points so the user stays in control of important decisions.

The workflow pauses for approval before:

1. Accepting the target role analysis and skill-gap interpretation.
2. Finalizing the learning roadmap.
3. Selecting the final project idea.
4. Approving portfolio-facing outputs such as resume bullets and README/demo wording.

This makes the app more reliable because career and resume outputs should not be blindly accepted from an LLM.

---

## Project Scoring Rubric

Each generated project idea is scored using criteria designed to reflect real portfolio value:

| Criterion | What it Measures |
|---|---|
| Resume value | How strongly the project supports AI engineering applications |
| Agentic AI fit | Whether the project demonstrates workflows, tools, memory, and multi-step reasoning |
| Buildability | Whether the project can realistically be built in four to six weeks |
| Personal relevance | Whether it connects to the user's background, goals, or real pain points |
| Technical depth | Whether it demonstrates meaningful engineering beyond a simple chatbot |
| Differentiation | Whether it stands out from generic AI toy projects |
| Recruiter explainability | Whether the project can be clearly explained in interviews |

The scoring system combines structured reasoning with deterministic ranking so the final recommendation is easier to inspect and justify.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/RRSGit97/pathpilot-ai.git
cd pathpilot-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: create a `.env` file

The app can accept an OpenAI API key directly inside the Streamlit UI, so a local `.env` file is optional.

If using `.env`, create it from the example:

```bash
cp .env.example .env
```

Then add your own key locally:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-api-key-here
VECTOR_STORE_ENABLED=false
```

Do not commit `.env` to GitHub.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser, usually at:

```text
http://localhost:8501
```

---

## Running Tests

```bash
pytest tests/
```

or:

```bash
pytest -q
```

---

## Repository Structure

```text
pathpilot-ai/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── sample_data/
│   ├── sample_profile.json
│   ├── sample_job_descriptions.json
│   └── sample_resources.json
├── src/
│   ├── config/
│   ├── data/
│   ├── graph/
│   ├── services/
│   ├── storage/
│   ├── tools/
│   └── ui/
└── tests/
```

---

## Security Notes

- The real `.env` file is ignored by Git.
- `.env.example` contains only placeholder values.
- OpenAI API keys entered in the Streamlit app are stored only in session memory.
- API keys are not written to SQLite, logs, README files, sample data, or persistent files.
- The app masks the key after it is entered.

---

## Limitations

- The app currently requires the user to provide an OpenAI API key or configure one locally.
- Live job scraping is intentionally excluded from the MVP to avoid reliability and compliance issues.
- Qdrant vector storage is optional and disabled by default.
- The project uses local SQLite storage, so it is designed for demo and personal use rather than multi-user production deployment.
- LLM-generated outputs should be reviewed before being used in resumes or public-facing materials.

---

## Future Improvements

- Add hosted authentication and user accounts.
- Add persistent cloud storage for multi-user deployments.
- Add richer evaluation metrics for roadmap and project quality.
- Add export to PDF or DOCX for roadmap and portfolio outputs.
- Add more LLM provider options in the UI.
- Add Docker deployment support.
- Add CI checks with GitHub Actions.
- Add a polished project landing page with screenshots and video embeds.

---

## Resume Bullet

Built **PathPilot AI**, an agentic AI career-planning assistant that analyzes target AI roles and job descriptions, maps user skill gaps, generates personalized learning roadmaps, scores resume-worthy project ideas, and produces six-week build plans and portfolio outputs using **Python, Streamlit, LangGraph, LangChain, Pydantic, SQLite memory, optional Qdrant vector search, and human-in-the-loop checkpoints**.

---

## Demo

Watch the full walkthrough here:

**Video Demo:** https://app.guidde.com/share/playbooks/m3JM19fMYzvBQHLxE4Cf2n?origin=EMw3SaL2yhSrH41cueQA4v6xWVo2&t=0&mode=videoAndDoc
