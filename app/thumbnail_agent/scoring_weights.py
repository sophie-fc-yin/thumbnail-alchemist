"""Niche-specific scoring weight configurations for thumbnail selection.

Different YouTube niches prioritize different factors:
- Beauty/Lifestyle: Aesthetics matter most
- Tech/Education: Psychology and clarity
- Gaming: High energy and action
- Commentary: Authenticity and expression
"""

from __future__ import annotations

import json
import os
from typing import Any

from google.genai import Client


class ScoringWeights:
    """Default and niche-specific scoring weights."""

    # Default balanced weights (all computed from real data, no placeholders)
    DEFAULT = {
        "moment_importance": 0.20,  # ClickMoment: how "moment-ready" the frame is (audio+visual saliency)
        "creator_alignment": 0.25,  # How well it matches creator's brief/goals
        "aesthetic_quality": 0.15,  # Visual appeal (context-aware, includes brightness) - increased
        "psychology_score": 0.18,  # Psychological impact (context-aware)
        "editability": 0.20,  # Small-screen robustness (crop/zoom/clarity) - CRITICAL: boosted from 12% to 20%
        "face_quality": 0.08,  # Clear, engaging facial expression
        "composition": 0.02,  # Visual balance (head pose, centering)
        "technical_quality": 0.02,  # Image quality (eye/mouth clarity)
    }

    # Niche-specific weight presets
    NICHE_PRESETS = {
        # Beauty, fashion, lifestyle, vlogging
        "beauty_lifestyle": {
            "moment_importance": 0.18,
            "creator_alignment": 0.18,
            "aesthetic_quality": 0.30,  # ⬆️ Aesthetics critical
            "psychology_score": 0.18,
            "editability": 0.18,  # ⬆️ Boosted - still needs to work on mobile
            "face_quality": 0.12,
            "composition": 0.04,
            "technical_quality": 0.02,
        },
        # Tech reviews, tutorials, how-to
        "tech_educational": {
            "moment_importance": 0.18,
            "creator_alignment": 0.22,
            "aesthetic_quality": 0.15,  # Less critical
            "psychology_score": 0.22,  # ⬆️ Curiosity/clarity key
            "editability": 0.22,  # ⬆️ High - text overlays critical
            "face_quality": 0.10,
            "composition": 0.02,
            "technical_quality": 0.02,
        },
        # Gaming, let's plays, gaming news
        "gaming": {
            "moment_importance": 0.18,
            "creator_alignment": 0.18,
            "aesthetic_quality": 0.16,
            "psychology_score": 0.20,  # ⬆️ Hype and excitement
            "editability": 0.25,  # ⬆️ Highest - heavy text overlays, must survive mobile
            "face_quality": 0.12,  # Reactions important
            "composition": 0.02,
            "technical_quality": 0.02,
        },
        # Commentary, reaction, opinion videos
        "commentary": {
            "moment_importance": 0.18,
            "creator_alignment": 0.22,
            "aesthetic_quality": 0.15,
            "psychology_score": 0.18,
            "editability": 0.20,  # ⬆️ Boosted - needs to work with overlays
            "face_quality": 0.20,  # ⬆️ Expression is key
            "composition": 0.03,
            "technical_quality": 0.02,
        },
        # Cooking, recipes, food content
        "cooking_food": {
            "moment_importance": 0.17,
            "creator_alignment": 0.17,
            "aesthetic_quality": 0.28,  # ⬆️ Food must look good
            "psychology_score": 0.18,
            "editability": 0.20,  # ⬆️ Recipe text overlays common
            "face_quality": 0.08,  # Food can be focus
            "composition": 0.05,  # Plating matters
            "technical_quality": 0.02,
        },
        # Fitness, health, wellness
        "fitness_health": {
            "moment_importance": 0.18,
            "creator_alignment": 0.20,
            "aesthetic_quality": 0.25,  # ⬆️ Results/transformation
            "psychology_score": 0.20,  # Aspiration
            "editability": 0.20,  # ⬆️ Motivational text overlays - boosted
            "face_quality": 0.10,
            "composition": 0.03,
            "technical_quality": 0.02,
        },
        # Business, finance, motivation
        "business_finance": {
            "moment_importance": 0.18,
            "creator_alignment": 0.22,
            "aesthetic_quality": 0.17,
            "psychology_score": 0.20,  # ⬆️ Authority/credibility
            "editability": 0.22,  # ⬆️ Professional text overlays - boosted
            "face_quality": 0.10,  # Trust signals
            "composition": 0.02,
            "technical_quality": 0.02,
        },
        # Entertainment, comedy, sketches
        "entertainment": {
            "moment_importance": 0.18,
            "creator_alignment": 0.20,
            "aesthetic_quality": 0.17,
            "psychology_score": 0.22,  # ⬆️ Curiosity/surprise
            "editability": 0.20,  # ⬆️ Boosted - context text important
            "face_quality": 0.12,  # Expressions
            "composition": 0.02,
            "technical_quality": 0.02,
        },
        # Music, covers, performances
        "music": {
            "moment_importance": 0.16,
            "creator_alignment": 0.18,
            "aesthetic_quality": 0.30,  # ⬆️ Visual appeal
            "psychology_score": 0.18,
            "editability": 0.18,  # ⬆️ Boosted - needs to work on mobile
            "face_quality": 0.08,
            "composition": 0.06,  # ⬆️ Artistic composition
            "technical_quality": 0.02,
        },
        # News, journalism, documentary
        "news_journalism": {
            "moment_importance": 0.18,
            "creator_alignment": 0.25,  # ⬆️ Brief match critical
            "aesthetic_quality": 0.15,
            "psychology_score": 0.19,
            "editability": 0.22,  # ⬆️ Headlines critical - boosted
            "face_quality": 0.11,  # Authority
            "composition": 0.02,
            "technical_quality": 0.02,
        },
    }

    # Niche detection mapping (from channel profile niche to preset)
    NICHE_MAPPING = {
        # Frontend option values (exact matches)
        "educational / explainer": "tech_educational",
        "tech / product reviews": "tech_educational",
        "gaming": "gaming",
        "commentary / reaction": "commentary",
        "lifestyle / vlog": "beauty_lifestyle",
        "beauty / fashion": "beauty_lifestyle",
        "food / cooking": "cooking_food",
        "fitness / health": "fitness_health",
        "business / finance": "business_finance",
        "entertainment / comedy": "entertainment",
        "news / journalism": "news_journalism",
        "music / performance": "music",
        # Travel (maps to lifestyle - similar aesthetic priorities)
        "travel": "beauty_lifestyle",
        # Beauty & Lifestyle (alternate forms)
        "beauty": "beauty_lifestyle",
        "fashion": "beauty_lifestyle",
        "lifestyle": "beauty_lifestyle",
        "vlog": "beauty_lifestyle",
        "vlogging": "beauty_lifestyle",
        "makeup": "beauty_lifestyle",
        "skincare": "beauty_lifestyle",
        # Tech & Educational (alternate forms)
        "tech": "tech_educational",
        "technology": "tech_educational",
        "tech reviews": "tech_educational",
        "product reviews": "tech_educational",
        "tutorial": "tech_educational",
        "how-to": "tech_educational",
        "educational": "tech_educational",
        "education": "tech_educational",
        "explainer": "tech_educational",
        "science": "tech_educational",
        "programming": "tech_educational",
        "coding": "tech_educational",
        # Gaming (alternate forms)
        "games": "gaming",
        "let's play": "gaming",
        "esports": "gaming",
        "game reviews": "gaming",
        # Commentary (alternate forms)
        "commentary": "commentary",
        "reaction": "commentary",
        "opinion": "commentary",
        "discussion": "commentary",
        "podcast": "commentary",
        # Cooking & Food (alternate forms)
        "cooking": "cooking_food",
        "food": "cooking_food",
        "recipes": "cooking_food",
        "baking": "cooking_food",
        "culinary": "cooking_food",
        # Fitness & Health (alternate forms)
        "fitness": "fitness_health",
        "health": "fitness_health",
        "wellness": "fitness_health",
        "workout": "fitness_health",
        "yoga": "fitness_health",
        # Business & Finance (alternate forms)
        "business": "business_finance",
        "finance": "business_finance",
        "investing": "business_finance",
        "entrepreneurship": "business_finance",
        "motivation": "business_finance",
        "self-help": "business_finance",
        # Entertainment (alternate forms)
        "entertainment": "entertainment",
        "comedy": "entertainment",
        "funny": "entertainment",
        "sketches": "entertainment",
        "pranks": "entertainment",
        # Music (alternate forms)
        "music": "music",
        "covers": "music",
        "performance": "music",
        "singing": "music",
        "musician": "music",
        # News & Journalism (alternate forms)
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

        # Adjust based on content type (optional; not always provided)
        content_type = channel_profile.get("content_type")
        if content_type == "educational":
            # Educational content: boost psychology (clarity) and reduce aesthetics
            weights["psychology_score"] = min(weights["psychology_score"] + 0.05, 1.0)
            weights["aesthetic_quality"] = max(weights["aesthetic_quality"] - 0.05, 0.0)

        elif content_type == "entertainment":
            # Entertainment: boost psychology slightly (surprise/curiosity)
            weights["psychology_score"] = min(weights["psychology_score"] + 0.03, 1.0)

        elif content_type == "lifestyle":
            # Lifestyle: boost aesthetics
            weights["aesthetic_quality"] = min(weights["aesthetic_quality"] + 0.05, 1.0)

        # Adjust based on visual consistency preference
        visual_consistency = channel_profile.get("visual_consistency")
        if visual_consistency == "high":
            # High consistency: boost creator_alignment (brand match)
            weights["creator_alignment"] = min(weights["creator_alignment"] + 0.03, 1.0)
            # (No originality component in Phase-1 scoring)

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


async def determine_optimal_weights(
    brief: dict[str, Any],
    profile: dict[str, Any],
    model_name: str = "gemini-2.0-flash-exp",
) -> dict[str, float]:
    """
    Use Gemini to determine optimal scoring weights based on creative brief and channel profile.

    This analyzes the creator's content style, goals, and niche to return personalized
    scoring weights instead of using hardcoded niche presets.

    Args:
        brief: Creative brief with video_title, primary_message, tone, primary_goal
        profile: Channel profile with niche, personality, visual_style
        model_name: Gemini model to use (default: gemini-2.0-flash-exp)

    Returns:
        Dictionary of scoring weights (falls back to niche preset if agent fails)

    Example:
        >>> brief = {
        ...     "video_title": "Exploring New Orleans",
        ...     "primary_message": "Genuine reactions to NOLA",
        ...     "tone": "casual",
        ...     "primary_goal": "subscriber growth"
        ... }
        >>> profile = {"niche": "travel vlog"}
        >>> weights = await determine_optimal_weights(brief, profile)
        >>> weights["face_quality"]
        0.18  # Boosted for reaction-focused content
    """
    try:
        # Get Gemini API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[AGENT WEIGHTS] GEMINI_API_KEY not set, falling back to niche preset")
            return get_scoring_weights(profile)

        # Create client
        client = Client(api_key=api_key)

        prompt = f"""Analyze this creator's content and determine optimal scoring weights for thumbnail selection.

IMPORTANT: Treat creator-provided signals as STRONG guidance. When the creator specifies preferences, prioritize those heavily.

CREATOR SIGNALS (high priority):
- Video title: {brief.get('video_title', 'N/A')}
- Creative direction: {brief.get('primary_message', 'N/A')}
- Desired tone: {brief.get('tone', 'N/A')}
- Content niche: {profile.get('niche', 'N/A')}
- Visual style: {profile.get('visual_style', 'N/A')}

SYSTEM INFERENCES (secondary):
- Optimization goal: {brief.get('primary_goal', 'N/A')} (inferred from content)
- Personality traits: {profile.get('personality', 'N/A')} (inferred from tone)

Return ONLY a JSON object with weights and reasoning:
{{
  "weights": {{
    "moment_importance": 0.XX,
    "creator_alignment": 0.XX,
    "aesthetic_quality": 0.XX,
    "psychology_score": 0.XX,
    "editability": 0.XX,
    "face_quality": 0.XX,
    "composition": 0.XX,
    "technical_quality": 0.XX
  }},
  "reasoning": "2-3 sentence explanation of why these weights fit this creator's style and goals"
}}

Guidelines:
- PRIORITIZE creator's stated tone and visual style preferences
- If creator emphasizes authenticity/reactions: boost face_quality (18-22%)
- If creator emphasizes visuals/scenery: boost aesthetic_quality (25-30%)
- If creator wants energy/action: boost psychology_score (20-25%)
- Reaction/vlog content: boost face_quality (15-20%)
- Educational/how-to: boost psychology_score (20-25%)
- Aesthetic-focused: boost aesthetic_quality (25-30%)
- Creator alignment should be weighted highly (18-25%) to respect stated preferences
- All values 0.0-1.0, total must equal 1.0"""

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )

        # Parse response
        response_json = json.loads(response.text)

        # Extract weights and reasoning
        weights_json = response_json.get("weights", {})
        reasoning = response_json.get("reasoning", "No reasoning provided")

        # Validate weights
        required_keys = [
            "moment_importance",
            "creator_alignment",
            "aesthetic_quality",
            "psychology_score",
            "editability",
            "face_quality",
            "composition",
            "technical_quality",
        ]

        if not all(k in weights_json for k in required_keys):
            raise ValueError("Missing required weight keys")

        # Normalize to ensure sum = 1.0
        total = sum(weights_json.values())
        if abs(total - 1.0) > 0.01:
            weights_json = {k: v / total for k, v in weights_json.items()}

        print("[AGENT WEIGHTS] Determined custom weights based on brief + profile")
        print(f"[AGENT WEIGHTS] Reasoning: {reasoning}")
        return weights_json

    except Exception as e:
        print(f"[AGENT WEIGHTS] Failed to determine weights ({e}), falling back to niche preset")
        return get_scoring_weights(profile)
