"""Local AI-powered LinkedIn profile optimization engine."""

import json
import re

from pydantic import BaseModel, Field, ValidationError

from .ollama_client import OllamaClient, DEFAULT_MODEL
from .profile_schemas import LinkedInProfile
from .profile_optimizer_prompts import (
    SYSTEM_PROMPT as OPTIMIZER_SYSTEM_PROMPT,
    build_optimization_prompt,
)
from .profile_optimizer_scorer import ProfileScorer


class ExperienceRewrite(BaseModel):
    index: int
    original: str
    optimized: str
    explanation: str = ""


class ProfileOptimization(BaseModel):
    headline_options: list[str] = Field(default_factory=list)
    recommended_headline: str = ""

    about_versions: list[str] = Field(default_factory=list)
    recommended_about: str = ""

    experience_rewrites: list[ExperienceRewrite] = Field(
        default_factory=list
    )

    recommended_skills: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)

    target_roles: list[str] = Field(default_factory=list)
    role_alignment: str = ""

    strengths: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)

    overall_score: int = Field(default=0, ge=0, le=100)
    improvement_summary: str = ""


class ProfileOptimizationError(RuntimeError):
    """Raised when profile optimization fails."""


SYSTEM_PROMPT = """
You are CareerPilot AI, a professional LinkedIn profile optimization engine.

Your task is to improve LinkedIn profiles using factual information.

Optimize:
1. Professional headline.
2. About section.
3. Experience descriptions.
4. Technical and professional skills.
5. Relevant industry keywords.
6. Alignment with target job roles.
7. Before-and-after improvements.

STRICT RULES:

- Never invent employment, internships or certifications.
- Never invent numerical achievements or project results.
- Never claim skills that are not supported by profile information.
- Preserve the user's actual experience.
- Use professional, natural human language.
- Avoid generic AI-generated phrases.
- Make recommendations relevant to the target roles.
- Mark missing information as missing rather than inventing it.
- Keep LinkedIn headline within 220 characters.
- Return valid JSON only.
- Do not use Markdown code fences.

Return this JSON structure:

{
  "headline_options": [],
  "recommended_headline": "",
  "about_versions": [],
  "recommended_about": "",
  "experience_rewrites": [
    {
      "index": 0,
      "original": "",
      "optimized": "",
      "explanation": ""
    }
  ],
  "recommended_skills": [],
  "missing_keywords": [],
  "target_roles": [],
  "role_alignment": "",
  "strengths": [],
  "improvement_areas": [],
  "overall_score": 0,
  "improvement_summary": ""
}
"""


class ProfileOptimizer:
    """Optimize LinkedIn profiles using local Ollama inference."""

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.ollama = ollama_client or OllamaClient(model=model)
        self.model = model
        self.scorer = ProfileScorer()

    def optimize(
        self,
        profile: LinkedInProfile | dict,
        job_description: str = "",
    ) -> ProfileOptimization:

        try:
            if isinstance(profile, dict):
                profile = LinkedInProfile.model_validate(profile)

            if not isinstance(profile, LinkedInProfile):
                raise TypeError("profile must be LinkedInProfile or dict")

            prompt = build_optimization_prompt(profile, job_description)

            response = self.ollama.generate(
                prompt=prompt,
                model=self.model,
                system=OPTIMIZER_SYSTEM_PROMPT,
                options={
                    "temperature": 0.3,
                    "num_ctx": 8192,
                },
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

            optimization = ProfileOptimization.model_validate(result)

            if len(optimization.recommended_headline) > 220:
                raise ProfileOptimizationError(
                    "Recommended headline exceeds 220 characters."
                )

            if len(optimization.experience_rewrites) > len(profile.experiences):
                raise ProfileOptimizationError(
                    "AI returned more experience entries than the profile contains."
                )

            for item in optimization.experience_rewrites:
                if item.index < 0 or item.index >= len(profile.experiences):
                    raise ProfileOptimizationError(
                        "Invalid experience index returned by AI."
                    )

                expected = profile.experiences[item.index].description

                if item.original != expected:
                    raise ProfileOptimizationError(
                        "Experience original does not match source profile."
                    )

            return optimization

        except (json.JSONDecodeError, ValidationError) as exc:
            raise ProfileOptimizationError(
                f"Invalid optimization response: {exc}"
            ) from exc

        except ProfileOptimizationError:
            raise

        except Exception as exc:
            raise ProfileOptimizationError(
                f"Profile optimization failed: {exc}"
            ) from exc

    def score(self, profile: LinkedInProfile | dict):
        """Return a deterministic profile quality score."""
        return self.scorer.score(profile)

    def close(self):
        self.ollama.close()
