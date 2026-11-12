"""Before-and-after LinkedIn profile comparison."""

from pydantic import BaseModel, Field

from .profile_schemas import LinkedInProfile
from .profile_optimizer_scorer import ProfileScore, ProfileScorer


class ProfileComparison(BaseModel):
    before: ProfileScore
    after: ProfileScore
    score_change: int
    improved: bool
    changes: list[str] = Field(default_factory=list)


class ProfileComparator:
    """Compare a source profile with an optimized profile proposal."""

    def __init__(self, scorer: ProfileScorer | None = None):
        self.scorer = scorer or ProfileScorer()

    def compare(self, before, after) -> ProfileComparison:
        from .profile_optimizer import ProfileOptimization

        source = (
            LinkedInProfile.model_validate(before)
            if isinstance(before, dict)
            else before
        )
        if not isinstance(source, LinkedInProfile):
            raise TypeError("before must be LinkedInProfile or dict")

        if isinstance(after, dict):
            after = ProfileOptimization.model_validate(after)

        if isinstance(after, LinkedInProfile):
            optimized_profile = after
        elif isinstance(after, ProfileOptimization):
            optimized_profile = source.model_copy(deep=True)

            if after.recommended_headline:
                optimized_profile.headline = after.recommended_headline
            if after.recommended_about:
                optimized_profile.about = after.recommended_about

            seen_indices = set()
            for rewrite in after.experience_rewrites:
                if rewrite.index < 0 or rewrite.index >= len(source.experiences):
                    raise ValueError("Experience rewrite index is invalid.")
                if rewrite.index in seen_indices:
                    raise ValueError("Duplicate experience rewrite index.")
                if rewrite.original != source.experiences[rewrite.index].description:
                    raise ValueError("Experience rewrite does not match source.")
                seen_indices.add(rewrite.index)
                optimized_profile.experiences[rewrite.index].description = (
                    rewrite.optimized
                )
        else:
            raise TypeError(
                "after must be LinkedInProfile, ProfileOptimization or dict"
            )

        before_score = self.scorer.score(source)
        after_score = self.scorer.score(optimized_profile)
        changes = []

        if source.headline != optimized_profile.headline:
            changes.append("Headline updated")
        if source.about != optimized_profile.about:
            changes.append("About section updated")
        if source.experiences != optimized_profile.experiences:
            changes.append("Experience descriptions updated")

        delta = after_score.overall_score - before_score.overall_score

        return ProfileComparison(
            before=before_score,
            after=after_score,
            score_change=delta,
            improved=delta > 0,
            changes=changes,
        )
