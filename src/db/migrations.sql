-- migrations.sql

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    name            TEXT,
    current_skills  TEXT,   -- JSON array
    prior_projects  TEXT,   -- free text
    target_roles    TEXT,   -- JSON array
    weekly_hours    INTEGER,
    learning_style  TEXT,
    pain_points     TEXT,
    resume_text     TEXT,   -- extracted PDF text, nullable
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_descriptions (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    raw_text    TEXT NOT NULL,
    extracted   TEXT,       -- JSON blob of ExtractedJD
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_gaps (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    report      TEXT NOT NULL,  -- JSON blob of SkillGapReport
    approved    INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roadmaps (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    roadmap     TEXT NOT NULL,  -- JSON blob of LearningRoadmap
    approved    INTEGER DEFAULT 0,
    version     INTEGER DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(id),
    title               TEXT NOT NULL,
    description         TEXT,
    scores              TEXT,   -- JSON blob of ProjectScores
    composite_score     REAL,
    chosen              INTEGER DEFAULT 0,
    approved            INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS build_plans (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    plan        TEXT NOT NULL,  -- JSON blob of BuildPlan
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id              TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects(id),
    artifact_type   TEXT NOT NULL,  -- readme | bullets | arch | demo | interview
    content         TEXT NOT NULL,
    approved        INTEGER DEFAULT 0,
    version         INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL
);
