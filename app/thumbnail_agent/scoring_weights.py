"""Niche-specific scoring weight configurations for thumbnail selection.

Different YouTube niches prioritize different factors:
- Beauty/Lifestyle: Aesthetics matter most
- Tech/Education: Psychology and clarity
- Gaming: High energy and action
- Commentary: Authenticity and expression
"""

from __future__ import annotations

from typing import Any


class ScoringWeights:
    """Default and niche-specific scoring weights."""

    # Default balanced weights
    DEFAULT = {
        "creator_alignment": 0.22,
        "aesthetic_quality": 0.18,
        "psychology_score": 0.18,
        "editability": 0.15,  # NEW: Workability for cropping, text, edits
        "face_quality": 0.13,
        "originality": 0.08,
        "composition": 0.04,
        "technical_quality": 0.02,
    }

    # Niche-specific weight presets
    NICHE_PRESETS = {
        # Beauty, fashion, lifestyle, vlogging
        "beauty_lifestyle": {
            "creator_alignment": 0.18,
            "aesthetic_quality": 0.28,  # ⬆️ Aesthetics critical
            "psychology_score": 0.18,
            "editability": 0.13,  # Moderate - less text, more visual
            "face_quality": 0.13,
            "originality": 0.06,
            "composition": 0.03,
            "technical_quality": 0.01,
        },
        # Tech reviews, tutorials, how-to
        "tech_educational": {
            "creator_alignment": 0.22,
            "aesthetic_quality": 0.13,  # ⬇️ Less critical
            "psychology_score": 0.22,  # ⬆️ Curiosity/clarity key
            "editability": 0.18,  # ⬆️ High - text overlays critical
            "face_quality": 0.13,
            "originality": 0.09,
            "composition": 0.02,
            "technical_quality": 0.01,
        },
        # Gaming, let's plays, gaming news
        "gaming": {
            "creator_alignment": 0.19,
            "aesthetic_quality": 0.15,
            "psychology_score": 0.19,  # ⬆️ Hype and excitement
            "editability": 0.20,  # ⬆️ Highest - heavy text overlays
            "face_quality": 0.15,  # ⬆️ Reactions important
            "originality": 0.09,
            "composition": 0.02,
            "technical_quality": 0.01,
        },
        # Commentary, reaction, opinion videos
        "commentary": {
            "creator_alignment": 0.22,
            "aesthetic_quality": 0.13,
            "psychology_score": 0.18,
            "editability": 0.14,  # Moderate - some text for context
            "face_quality": 0.18,  # ⬆️ Expression is key
            "originality": 0.10,
            "composition": 0.03,
            "technical_quality": 0.02,
        },
        # Cooking, recipes, food content
        "cooking_food": {
            "creator_alignment": 0.18,
            "aesthetic_quality": 0.25,  # ⬆️ Food must look good
            "psychology_score": 0.18,
            "editability": 0.16,  # Recipe text overlays common
            "face_quality": 0.10,  # ⬇️ Food can be focus
            "originality": 0.09,
            "composition": 0.03,  # Plating matters
            "technical_quality": 0.01,
        },
        # Fitness, health, wellness
        "fitness_health": {
            "creator_alignment": 0.20,
            "aesthetic_quality": 0.22,  # ⬆️ Results/transformation
            "psychology_score": 0.20,  # Aspiration
            "editability": 0.15,  # Motivational text overlays
            "face_quality": 0.11,
            "originality": 0.08,
            "composition": 0.03,
            "technical_quality": 0.01,
        },
        # Business, finance, motivation
        "business_finance": {
            "creator_alignment": 0.22,
            "aesthetic_quality": 0.16,
            "psychology_score": 0.20,  # ⬆️ Authority/credibility
            "editability": 0.17,  # ⬆️ Professional text overlays
            "face_quality": 0.15,  # ⬆️ Trust signals
            "originality": 0.07,
            "composition": 0.02,
            "technical_quality": 0.01,
        },
        # Entertainment, comedy, sketches
        "entertainment": {
            "creator_alignment": 0.20,
            "aesthetic_quality": 0.16,
            "psychology_score": 0.21,  # ⬆️ Curiosity/surprise
            "editability": 0.15,  # Moderate - context text
            "face_quality": 0.16,  # Expressions
            "originality": 0.09,  # ⬆️ Stand out
            "composition": 0.02,
            "technical_quality": 0.01,
        },
        # Music, covers, performances
        "music": {
            "creator_alignment": 0.18,
            "aesthetic_quality": 0.26,  # ⬆️ Visual appeal
            "psychology_score": 0.18,
            "editability": 0.13,  # ⬇️ Lower - visual dominates
            "face_quality": 0.10,
            "originality": 0.10,
            "composition": 0.04,  # ⬆️ Artistic composition
            "technical_quality": 0.01,
        },
        # News, journalism, documentary
        "news_journalism": {
            "creator_alignment": 0.25,  # ⬆️ Brief match critical
            "aesthetic_quality": 0.13,
            "psychology_score": 0.19,
            "editability": 0.17,  # ⬆️ Headlines critical
            "face_quality": 0.15,  # Authority
            "originality": 0.08,
            "composition": 0.02,
            "technical_quality": 0.01,
        },
    }

    # Niche detection mapping (from channel profile niche to preset)
    NICHE_MAPPING = {
        # Beauty & Lifestyle
        "beauty": "beauty_lifestyle",
        "fashion": "beauty_lifestyle",
        "lifestyle": "beauty_lifestyle",
        "vlog": "beauty_lifestyle",
        "vlogging": "beauty_lifestyle",
        "makeup": "beauty_lifestyle",
        "skincare": "beauty_lifestyle",
        # Tech & Educational
        "tech": "tech_educational",
        "technology": "tech_educational",
        "tech reviews": "tech_educational",
        "tutorial": "tech_educational",
        "how-to": "tech_educational",
        "educational": "tech_educational",
        "education": "tech_educational",
        "science": "tech_educational",
        "programming": "tech_educational",
        "coding": "tech_educational",
        # Gaming
        "gaming": "gaming",
        "games": "gaming",
        "let's play": "gaming",
        "esports": "gaming",
        "game reviews": "gaming",
        # Commentary
        "commentary": "commentary",
        "reaction": "commentary",
        "opinion": "commentary",
        "discussion": "commentary",
        "podcast": "commentary",
        # Cooking & Food
        "cooking": "cooking_food",
        "food": "cooking_food",
        "recipes": "cooking_food",
        "baking": "cooking_food",
        "culinary": "cooking_food",
        # Fitness & Health
        "fitness": "fitness_health",
        "health": "fitness_health",
        "wellness": "fitness_health",
        "workout": "fitness_health",
        "yoga": "fitness_health",
        # Business & Finance
        "business": "business_finance",
        "finance": "business_finance",
        "investing": "business_finance",
        "entrepreneurship": "business_finance",
        "motivation": "business_finance",
        "self-help": "business_finance",
        # Entertainment
        "entertainment": "entertainment",
        "comedy": "entertainment",
        "funny": "entertainment",
        "sketches": "entertainment",
        "pranks": "entertainment",
        # Music
        "music": "music",
        "covers": "music",
        "performance": "music",
        "singing": "music",
        "musician": "music",
        # News & Journalism
        "news": "news_journalism",
        "journalism": "news_journalism",
        "documentary": "news_journalism",
        "politics": "news_journalism",
    }

    @classmethod
    def get_weights_for_niche(cls, niche: str | None) -> dict[str, float]:
        """
        Get scoring weights based on channel niche.

        Args:
            niche: Channel niche string (e.g., "tech reviews", "beauty", "gaming")

        Returns:
            Dictionary of scoring weights

        Example:
            >>> weights = ScoringWeights.get_weights_for_niche("tech reviews")
            >>> weights["aesthetic_quality"]
            0.15
        """
        if not niche:
            return cls.DEFAULT.copy()

        # Normalize niche string
        niche_normalized = niche.lower().strip()

        # Check direct mapping
        preset_key = cls.NICHE_MAPPING.get(niche_normalized)

        if preset_key:
            return cls.NICHE_PRESETS[preset_key].copy()

        # Fallback: check if niche contains any mapped keywords
        for keyword, preset_key in cls.NICHE_MAPPING.items():
            if keyword in niche_normalized:
                return cls.NICHE_PRESETS[preset_key].copy()

        # Default if no match
        print(f"[ScoringWeights] No preset found for niche '{niche}', using defaults")
        return cls.DEFAULT.copy()

    @classmethod
    def get_weights_for_profile(
        cls, channel_profile: dict[str, Any], custom_weights: dict[str, float] | None = None
    ) -> dict[str, float]:
        """
        Get scoring weights based on full channel profile.

        Allows for:
        1. Niche-based preset selection
        2. Custom weight overrides
        3. Content-type based adjustments

        Args:
            channel_profile: Full channel profile dict
            custom_weights: Optional custom weight overrides

        Returns:
            Final scoring weights

        Example:
            >>> profile = {
            ...     "niche": "tech reviews",
            ...     "content_type": "educational",
            ...     "visual_consistency": "high"
            ... }
            >>> weights = ScoringWeights.get_weights_for_profile(profile)
        """
        # Start with niche-based weights
        niche = channel_profile.get("niche")
        weights = cls.get_weights_for_niche(niche)

        # Adjust based on content type
        content_type = channel_profile.get("content_type")
        if content_type == "educational":
            # Educational content: boost psychology (clarity) and reduce aesthetics
            weights["psychology_score"] = min(weights["psychology_score"] + 0.05, 1.0)
            weights["aesthetic_quality"] = max(weights["aesthetic_quality"] - 0.05, 0.0)

        elif content_type == "entertainment":
            # Entertainment: boost originality and psychology
            weights["originality"] = min(weights["originality"] + 0.03, 1.0)
            weights["psychology_score"] = min(weights["psychology_score"] + 0.03, 1.0)

        elif content_type == "lifestyle":
            # Lifestyle: boost aesthetics
            weights["aesthetic_quality"] = min(weights["aesthetic_quality"] + 0.05, 1.0)

        # Adjust based on visual consistency preference
        visual_consistency = channel_profile.get("visual_consistency")
        if visual_consistency == "high":
            # High consistency: boost creator_alignment (brand match)
            weights["creator_alignment"] = min(weights["creator_alignment"] + 0.03, 1.0)
            weights["originality"] = max(weights["originality"] - 0.03, 0.0)

        # Apply custom weight overrides
        if custom_weights:
            for key, value in custom_weights.items():
                if key in weights:
                    weights[key] = value

        # Normalize to ensure sum = 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point error
            weights = {k: v / total for k, v in weights.items()}

        return weights

    @classmethod
    def explain_weights(cls, weights: dict[str, float], niche: str | None = None) -> str:
        """
        Generate human-readable explanation of weight choices.

        Args:
            weights: Scoring weights dictionary
            niche: Optional niche for context

        Returns:
            Explanation string

        Example:
            >>> weights = ScoringWeights.get_weights_for_niche("beauty")
            >>> explanation = ScoringWeights.explain_weights(weights, "beauty")
            >>> print(explanation)
        """
        explanation = f"Scoring weights for {niche or 'default'} niche:\n\n"

        # Sort by weight (highest first)
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for key, value in sorted_weights:
            percentage = value * 100
            explanation += f"  • {key.replace('_', ' ').title()}: {percentage:.1f}%\n"

            # Add reasoning
            if key == "aesthetic_quality" and value >= 0.25:
                explanation += "    → High priority: Visual appeal is critical in this niche\n"
            elif key == "psychology_score" and value >= 0.23:
                explanation += "    → High priority: Engagement triggers drive clicks\n"
            elif key == "face_quality" and value >= 0.18:
                explanation += "    → High priority: Authentic expressions matter\n"
            elif key == "creator_alignment" and value >= 0.25:
                explanation += "    → High priority: Must match creator's brand and goals\n"

        return explanation


# Example usage functions
def get_scoring_weights(
    channel_profile: dict[str, Any], custom_weights: dict[str, float] | None = None
) -> dict[str, float]:
    """
    Main function to get scoring weights for a channel.

    Args:
        channel_profile: Channel profile with niche, content_type, etc.
        custom_weights: Optional custom overrides

    Returns:
        Scoring weights dictionary

    Example:
        >>> profile = {"niche": "tech reviews", "content_type": "educational"}
        >>> weights = get_scoring_weights(profile)
        >>> weights["psychology_score"]
        0.25
    """
    return ScoringWeights.get_weights_for_profile(channel_profile, custom_weights)


def list_available_niches() -> list[str]:
    """
    Get list of all supported niche presets.

    Returns:
        List of niche preset keys
    """
    return list(ScoringWeights.NICHE_PRESETS.keys())


def preview_niche_weights(niche_preset: str) -> dict[str, float] | None:
    """
    Preview weights for a specific niche preset.

    Args:
        niche_preset: Niche preset key (e.g., "tech_educational")

    Returns:
        Weights dictionary or None if preset not found
    """
    return ScoringWeights.NICHE_PRESETS.get(niche_preset)
