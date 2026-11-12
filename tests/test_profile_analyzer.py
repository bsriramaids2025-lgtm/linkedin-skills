import pytest

from lib.profile_analyzer import ProfileAnalyzer, ProfileAnalysisError
from lib.profile_schemas import LinkedInProfile


class FakeOllama:
    def __init__(self, response):
        self.response = response

    def generate(self, **kwargs):
        return self.response

    def close(self):
        pass


VALID_RESPONSE = """
{
  "overall_score": 82,
  "headline_score": 85,
  "about_score": 80,
  "experience_score": 75,
  "skills_score": 88,
  "summary": "Strong technical profile with room for improvement.",
  "strengths": ["Python", "Machine Learning"],
  "weaknesses": ["Limited measurable achievements"],
  "recommended_skills": ["Docker", "AWS"],
  "career_paths": ["AI Engineer", "ML Engineer"],
  "recommendations": [
    {
      "category": "Experience",
      "priority": "high",
      "issue": "Missing measurable results",
      "action": "Add genuine project metrics"
    }
  ]
}
"""


def test_profile_schema():
    profile = LinkedInProfile(
        full_name="Test User",
        headline="AI and Data Science Student",
        skills=["Python", "Machine Learning"],
    )

    assert profile.full_name == "Test User"
    assert len(profile.skills) == 2


def test_profile_analysis():
    analyzer = ProfileAnalyzer(
        ollama_client=FakeOllama(VALID_RESPONSE)
    )

    result = analyzer.analyze({
        "full_name": "Test User",
        "headline": "AI Engineer",
        "skills": ["Python", "SQL"],
    })

    assert result.overall_score == 82
    assert result.headline_score == 85
    assert "AI Engineer" in result.career_paths


def test_invalid_ai_json():
    analyzer = ProfileAnalyzer(
        ollama_client=FakeOllama("not valid json")
    )

    with pytest.raises(ProfileAnalysisError):
        analyzer.analyze({"headline": "Developer"})


def test_invalid_score():
    invalid_response = VALID_RESPONSE.replace(
        '"overall_score": 82',
        '"overall_score": 150'
    )

    analyzer = ProfileAnalyzer(
        ollama_client=FakeOllama(invalid_response)
    )

    with pytest.raises(ProfileAnalysisError):
        analyzer.analyze({"headline": "Developer"})


def test_recommendations():
    analyzer = ProfileAnalyzer(
        ollama_client=FakeOllama(VALID_RESPONSE)
    )

    result = analyzer.analyze({"headline": "AI Engineer"})

    assert len(result.recommendations) == 1
    assert result.recommendations[0].priority == "high"
