"""
tests/test_roadmap_validator.py
--------------------------------
Tests for LearningRoadmap schema validation.

NOTE: The old stub at this path imported from non-existent
`src.logic.roadmap_validator` and `src.schemas.roadmap` paths,
and referenced a `validate_roadmap` function that doesn't exist.

The real validation logic lives in the `LearningRoadmap` Pydantic
model's `@model_validator`. This file now tests that directly.

For comprehensive schema validation tests, see: tests/test_schemas.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data.schemas import LearningRoadmap, LearningRoadmapWeek


def _week(n: int) -> LearningRoadmapWeek:
    return LearningRoadmapWeek(week_number=n, focus_topic=f"Week {n} topic", estimated_hours=8)


def test_valid_roadmap_validates_without_error():
    """Replaces broken placeholder — LearningRoadmap validates correctly."""
    roadmap = LearningRoadmap(
        title="AI Engineer Roadmap",
        target_role="AI Engineer",
        total_weeks=4,
        weeks=[_week(i) for i in range(1, 5)],
    )
    assert roadmap.total_weeks == 4
    assert len(roadmap.weeks) == 4


def test_mismatched_week_count_raises():
    """total_weeks=4 but 3 weeks provided must raise ValidationError."""
    with pytest.raises(ValidationError):
        LearningRoadmap(
            title="Bad Roadmap",
            target_role="AI Engineer",
            total_weeks=4,
            weeks=[_week(i) for i in range(1, 4)],  # 3 weeks only
        )


def test_total_weeks_above_12_raises():
    with pytest.raises(ValidationError):
        LearningRoadmap(
            title="Too Long",
            target_role="AI Engineer",
            total_weeks=13,  # exceeds schema max of 12
        )


def test_empty_weeks_list_is_allowed():
    """Empty weeks list skips the week-count validator (draft state)."""
    roadmap = LearningRoadmap(
        title="Draft",
        target_role="AI Engineer",
        total_weeks=6,
    )
    assert roadmap.weeks == []


def test_roadmap_has_version_field():
    """Version should default to 1."""
    roadmap = LearningRoadmap(
        title="Versioned",
        target_role="AI Engineer",
        total_weeks=1,
        weeks=[_week(1)],
    )
    assert roadmap.version == 1
