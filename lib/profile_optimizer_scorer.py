"""Deterministic LinkedIn profile scoring."""

import re
from pydantic import BaseModel, Field

from .profile_schemas import LinkedInProfile


class ProfileScore(BaseModel):
    headline_score: int = Field(ge=0, le=100)
    about_score: int = Field(ge=0, le=100)
    experience_score: int = Field(ge=0, le=100)
    skills_score: int = Field(ge=0, le=100)
    overall_score: int = Field(ge=0, le=100)
    completeness_score: int = Field(ge=0, le=100)


class ProfileScorer:
    """Score profile completeness and presentation using transparent heuristics."""

    WEIGHTS = {
        "headline": 0.20,
        "about": 0.25,
        "experience": 0.35,
        "skills": 0.20,
    }

    ACTION_VERBS = {
        "built", "created", "developed", "designed", "implemented",
        "managed", "led", "improved", "analyzed", "automated",
        "deployed", "optimized", "delivered", "engineered",
    }

    def score(self, profile: LinkedInProfile | dict) -> ProfileScore:
        if isinstance(profile, dict):
            profile = LinkedInProfile.model_validate(profile)
        if not isinstance(profile, LinkedInProfile):
            raise TypeError("profile must be LinkedInProfile or dict")

        headline = self._headline_score(profile.headline)
        about = self._about_score(profile.about)
        experience = self._experience_score(profile.experiences)
        skills = self._skills_score(profile.skills)

        overall = round(
            headline * self.WEIGHTS["headline"]
            + about * self.WEIGHTS["about"]
            + experience * self.WEIGHTS["experience"]
            + skills * self.WEIGHTS["skills"]
        )

        fields = [
            bool(profile.full_name.strip()),
            bool(profile.headline.strip()),
            bool(profile.about.strip()),
            bool(profile.experiences),
            bool(profile.skills),
            bool(profile.education),
        ]

        completeness = round(sum(fields) / len(fields) * 100)

        return ProfileScore(
            headline_score=headline,
            about_score=about,
            experience_score=experience,
            skills_score=skills,
            overall_score=overall,
            completeness_score=completeness,
        )

    def _headline_score(self, value: str) -> int:
        text = value.strip()
        if not text:
            return 0

        length = len(text)
        score = 35
        if 40 <= length <= 180:
            score += 35
        elif 20 <= length < 40 or 180 < length <= 220:
            score += 20
        else:
            score += 5

        if "|" in text or "•" in text or ":" in text:
            score += 10
        if len(text.split()) >= 4:
            score += 10
        if length > 220:
            score -= 30

        return max(0, min(100, score))

    def _about_score(self, value: str) -> int:
        text = value.strip()
        if not text:
            return 0

        words = text.split()
        score = 25
        if 80 <= len(words) <= 300:
            score += 45
        elif 40 <= len(words) < 80 or 300 < len(words) <= 500:
            score += 30
        else:
            score += 10

        if re.search(r"[.!?]", text):
            score += 10
        if len(re.findall(r"\n", text)) >= 2:
            score += 10
        if any(word.lower() in text.lower() for word in self.ACTION_VERBS):
            score += 10

        return min(100, score)

    def _experience_score(self, experiences) -> int:
        if not experiences:
            return 0

        scores = []
        for item in experiences:
            text = item.description.strip()
            score = 20 if text else 0
            words = text.split()

            if len(words) >= 20:
                score += 30
            elif len(words) >= 8:
                score += 20

            if any(v in text.lower() for v in self.ACTION_VERBS):
                score += 25
            if re.search(r"\b\d+([.,]\d+)?%?\b", text):
                score += 15
            if item.title.strip() and item.company.strip():
                score += 10

            scores.append(min(100, score))

        return round(sum(scores) / len(scores))

    def _skills_score(self, skills: list[str]) -> int:
        clean = {item.strip().lower() for item in skills if item.strip()}
        count = len(clean)
        if count == 0:
            return 0
        if 10 <= count <= 20:
            return 100
        if 6 <= count < 10:
            return 80
        if 3 <= count < 6:
            return 60
        if count < 3:
            return 35
        return 85
