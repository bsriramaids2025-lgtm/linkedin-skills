"""Tests for LinkedIn optimization supporting modules."""

import pytest

from lib.profile_schemas import LinkedInProfile, Experience
from lib.profile_optimizer_prompts import (
    SYSTEM_PROMPT,
    build_optimization_prompt,
)
from lib.profile_optimizer_scorer import ProfileScorer
from lib.profile_optimizer import ProfileOptimization, ExperienceRewrite
from lib.profile_comparator import ProfileComparator


@pytest.fixture
def sample_profile():
    return LinkedInProfile(
        full_name="Test Engineer",
        headline="AI Engineer | Python | Machine Learning",
        about=(
            "I develop machine learning applications and build software "
            "solutions. I have experience with Python, data analysis, and "
            "deep learning. I enjoy designing and implementing practical "
            "AI projects that solve real problems."
        ),
        experiences=[
            Experience(
                title="AI Intern",
                company="Example Company",
                description="Developed a Python application for data analysis.",
            )
        ],
        skills=["Python", "Machine Learning", "SQL", "Git", "Deep Learning"],
        education=["B.Tech Artificial Intelligence"],
        target_roles=["AI Engineer"],
    )


def test_prompt_builder_includes_profile_and_job_description(sample_profile):
    prompt = build_optimization_prompt(
        sample_profile,
        "Looking for a Python AI Engineer.",
    )
    assert "Test Engineer" in prompt
    assert "Python AI Engineer" in prompt
    assert "Generate five distinct headline options" in prompt


def test_system_prompt_has_factuality_rules():
    assert "Never invent" in SYSTEM_PROMPT
    assert "valid JSON" in SYSTEM_PROMPT


def test_scorer_returns_valid_scores(sample_profile):
    result = ProfileScorer().score(sample_profile)
    assert 0 <= result.overall_score <= 100
    assert 0 <= result.completeness_score <= 100
    assert result.headline_score > 0


def test_empty_profile_scores_zero():
    result = ProfileScorer().score(LinkedInProfile())
    assert result.overall_score == 0
    assert result.completeness_score == 0


def test_scorer_accepts_dictionary(sample_profile):
    result = ProfileScorer().score(sample_profile.model_dump())
    assert result.overall_score > 0


def test_comparator_detects_improvement(sample_profile):
    optimization = ProfileOptimization(
        recommended_headline=(
            "AI Engineer | Python | Machine Learning | Deep Learning"
        ),
        recommended_about=(
            "I develop and implement practical AI solutions using Python. "
            "My interests include machine learning, deep learning, and "
            "data analysis. I build projects to solve real-world problems."
        ),
        experience_rewrites=[
            ExperienceRewrite(
                index=0,
                original=sample_profile.experiences[0].description,
                optimized=(
                    "Developed and implemented a Python application "
                    "for data analysis and reporting."
                ),
                explanation="Clarifies the action and deliverable.",
            )
        ],
    )

    result = ProfileComparator().compare(sample_profile, optimization)
    assert result.score_change >= 0
    assert "Headline updated" in result.changes
    assert "About section updated" in result.changes
    assert "Experience descriptions updated" in result.changes


def test_comparator_rejects_wrong_original_text(sample_profile):
    optimization = ProfileOptimization(
        experience_rewrites=[
            ExperienceRewrite(
                index=0,
                original="Fabricated original description",
                optimized="New description",
            )
        ]
    )

    with pytest.raises(ValueError, match="does not match source"):
        ProfileComparator().compare(sample_profile, optimization)


def test_comparator_rejects_duplicate_indices(sample_profile):
    original = sample_profile.experiences[0].description
    optimization = ProfileOptimization(
        experience_rewrites=[
            ExperienceRewrite(index=0, original=original, optimized="Version A"),
            ExperienceRewrite(index=0, original=original, optimized="Version B"),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate"):
        ProfileComparator().compare(sample_profile, optimization)


def test_comparator_does_not_mutate_original_profile(sample_profile):
    original_description = sample_profile.experiences[0].description
    optimization = ProfileOptimization(
        experience_rewrites=[
            ExperienceRewrite(
                index=0,
                original=original_description,
                optimized="Improved description",
            )
        ]
    )

    ProfileComparator().compare(sample_profile, optimization)
    assert sample_profile.experiences[0].description == original_description
