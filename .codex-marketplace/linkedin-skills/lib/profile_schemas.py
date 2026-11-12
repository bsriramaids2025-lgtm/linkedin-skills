from typing import Literal
from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: str = ""
    company: str = ""
    description: str = ""


class LinkedInProfile(BaseModel):
    full_name: str = ""
    headline: str = ""
    about: str = ""
    experiences: list[Experience] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    category: str
    priority: Literal["high", "medium", "low"]
    issue: str
    action: str


class ProfileAnalysis(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    headline_score: int = Field(ge=0, le=100)
    about_score: int = Field(ge=0, le=100)
    experience_score: int = Field(ge=0, le=100)
    skills_score: int = Field(ge=0, le=100)

    summary: str
    strengths: list[str]
    weaknesses: list[str]
    recommended_skills: list[str]
    career_paths: list[str]
    recommendations: list[Recommendation]
