"""Local AI-powered LinkedIn profile analysis."""

import json
import re

from pydantic import ValidationError

from .ollama_client import OllamaClient, DEFAULT_MODEL
from .profile_schemas import LinkedInProfile, ProfileAnalysis


class ProfileAnalysisError(RuntimeError):
    """Raised when profile analysis fails."""


SYSTEM_PROMPT = """
You are CareerPilot AI, an expert LinkedIn profile and career strategist.

Analyze LinkedIn profiles honestly and professionally.

Evaluate:
1. Headline clarity and keyword relevance.
2. About section quality and professional positioning.
3. Experience descriptions and measurable achievements.
4. Skills relevance and missing technical skills.
5. Overall profile completeness and employability.

Provide practical recommendations.

Important rules:
- Never invent qualifications, experience or achievements.
- Distinguish missing information from actual weaknesses.
- Do not guarantee ATS scores, recruiter attention or employment.
- Consider the user's stated target roles.
- Give realistic recommendations.
- Return ONLY valid JSON.
- Do not include Markdown code fences.

Required JSON structure:
{
  "overall_score": 0,
  "headline_score": 0,
  "about_score": 0,
  "experience_score": 0,
  "skills_score": 0,
  "summary": "Analysis summary",
  "strengths": [],
  "weaknesses": [],
  "recommended_skills": [],
  "career_paths": [],
  "recommendations": [
    {
      "category": "Headline",
      "priority": "high",
      "issue": "Description",
      "action": "Recommended action"
    }
  ]
}
"""


class ProfileAnalyzer:
    """Analyze LinkedIn profiles using local Ollama inference."""

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.ollama = ollama_client or OllamaClient(model=model)
        self.model = model

    def analyze(self, profile: LinkedInProfile | dict) -> ProfileAnalysis:
        """Analyze a profile and return validated structured results."""

        try:
            if isinstance(profile, dict):
                profile = LinkedInProfile.model_validate(profile)

            profile_data = profile.model_dump_json(indent=2)

            prompt = f"""
Analyze this LinkedIn profile.

PROFILE DATA:
{profile_data}

Return the complete JSON analysis using the required schema.
Scores must be integers between 0 and 100.
"""

            response = self.ollama.generate(
                prompt=prompt,
                model=self.model,
                system=SYSTEM_PROMPT,
                options={"temperature": 0.2},
            )

            cleaned = re.sub(
                r"<think>.*?</think>",
                "",
                response,
                flags=re.DOTALL,
            ).strip()

            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

            result = json.loads(cleaned)

            return ProfileAnalysis.model_validate(result)

        except (json.JSONDecodeError, ValidationError) as exc:
            raise ProfileAnalysisError(
                f"Invalid AI analysis response: {exc}"
            ) from exc

        except Exception as exc:
            if isinstance(exc, ProfileAnalysisError):
                raise
            raise ProfileAnalysisError(
                f"Profile analysis failed: {exc}"
            ) from exc

    def close(self):
        """Close the Ollama connection if owned by this analyser."""
        self.ollama.close()
