"""Example usage of niche-specific scoring weights in thumbnail selection."""

from __future__ import annotations

from typing import Any

from app.thumbnail_agent.scoring_weights import ScoringWeights, get_scoring_weights


# Example 1: Automatic niche detection
def example_auto_niche_detection():
    """Example: Automatically detect niche and get weights."""
    channel_profile = {
        "channel_id": "UCxxxx",
        "niche": "tech reviews",  # ← Agent detects this
        "content_type": "educational",
        "audience_age": "18-24",
        "personality": ["informative", "energetic"],
    }

    # Get weights automatically
    weights = get_scoring_weights(channel_profile)

    print("Tech Reviews Channel Weights:")
    print(f"  Aesthetic Quality: {weights['aesthetic_quality']:.1%}")  # 15%
    print(f"  Psychology Score: {weights['psychology_score']:.1%}")  # 25% (high for curiosity)
    print(f"  Face Quality: {weights['face_quality']:.1%}")  # 15%

    return weights


# Example 2: Different niches, different weights
def example_niche_comparison():
    """Compare weights across different niches."""
    niches = {
        "Beauty Channel": {"niche": "beauty", "content_type": "lifestyle"},
        "Tech Channel": {"niche": "tech reviews", "content_type": "educational"},
        "Gaming Channel": {"niche": "gaming", "content_type": "entertainment"},
    }

    print("\n=== Niche Weight Comparison ===\n")

    for name, profile in niches.items():
        weights = get_scoring_weights(profile)
        print(f"{name}:")
        print(f"  Aesthetics: {weights['aesthetic_quality']:.1%}")
        print(f"  Psychology: {weights['psychology_score']:.1%}")
        print(f"  Face: {weights['face_quality']:.1%}")
        print()

    # Output:
    # Beauty Channel:
    #   Aesthetics: 30.0%  ← Highest
    #   Psychology: 20.0%
    #   Face: 15.0%
    #
    # Tech Channel:
    #   Aesthetics: 15.0%  ← Lower
    #   Psychology: 25.0%  ← Highest
    #   Face: 15.0%
    #
    # Gaming Channel:
    #   Aesthetics: 18.0%
    #   Psychology: 22.0%
    #   Face: 18.0%  ← Reactions important


# Example 3: Custom weight overrides
def example_custom_weights():
    """Example: Creator wants custom weights."""
    channel_profile = {
        "niche": "tech reviews",
        "content_type": "educational",
    }

    # Creator wants more emphasis on aesthetics than typical tech channel
    custom_overrides = {
        "aesthetic_quality": 0.25,  # Boost from 0.15 to 0.25
        "psychology_score": 0.20,  # Reduce from 0.25 to 0.20
    }

    weights = get_scoring_weights(channel_profile, custom_weights=custom_overrides)

    print("Custom Weights (Tech channel with aesthetic focus):")
    print(f"  Aesthetic Quality: {weights['aesthetic_quality']:.1%}")  # 25% (custom)
    print(f"  Psychology Score: {weights['psychology_score']:.1%}")  # 20% (custom)

    return weights


# Example 4: Full thumbnail scoring with niche weights
def example_full_thumbnail_scoring():
    """Example: Score a frame using niche-specific weights."""

    # Channel profile from Supabase
    channel_profile = {
        "niche": "beauty",
        "content_type": "lifestyle",
        "visual_consistency": "high",
    }

    # Get niche-specific weights
    weights = get_scoring_weights(channel_profile)

    # Frame analysis results (from Phase 2 of agent)
    frame_scores = {
        "creator_alignment": 0.92,
        "aesthetic_quality": 0.88,  # High aesthetic (beauty niche)
        "psychology_score": 0.85,
        "face_quality": 0.90,
        "originality": 0.75,
        "composition": 0.82,
        "technical_quality": 0.91,
    }

    # Calculate final score using niche weights
    final_score = sum(frame_scores[key] * weights[key] for key in weights.keys())

    print("\n=== Beauty Channel Frame Scoring ===")
    print("\nWeights:")
    for key, weight in weights.items():
        print(f"  {key}: {weight:.1%}")

    print("\nFrame Scores:")
    for key, score in frame_scores.items():
        contribution = score * weights[key]
        print(f"  {key}: {score:.2f} × {weights[key]:.2f} = {contribution:.3f}")

    print(f"\n🎯 Final Score: {final_score:.3f} ({final_score*100:.1f}%)")

    # For beauty channel, high aesthetic score gets 30% weight:
    # aesthetic_quality: 0.88 × 0.30 = 0.264 (26.4% of total!)

    return final_score


# Example 5: A/B test different weight strategies
def example_ab_test_weights():
    """Example: Compare frame scores with different weight strategies."""

    frame_scores = {
        "creator_alignment": 0.85,
        "aesthetic_quality": 0.90,  # Beautiful frame
        "psychology_score": 0.70,  # Lower psychology
        "face_quality": 0.88,
        "originality": 0.75,
        "composition": 0.85,
        "technical_quality": 0.92,
    }

    strategies = {
        "Beauty Focus (aesthetic 30%)": {"niche": "beauty"},
        "Tech Focus (psychology 25%)": {"niche": "tech reviews"},
        "Balanced (default)": {"niche": None},
    }

    print("\n=== Weight Strategy Comparison ===")
    print("Same frame, different niche weights:\n")

    for strategy_name, profile in strategies.items():
        weights = get_scoring_weights(profile)
        final_score = sum(frame_scores[key] * weights[key] for key in weights.keys())
        print(f"{strategy_name}: {final_score:.3f} ({final_score*100:.1f}%)")

    # Output shows how same frame scores differently by niche:
    # Beauty Focus (aesthetic 30%): 0.867 (86.7%) ← Highest (rewards aesthetics)
    # Tech Focus (psychology 25%): 0.842 (84.2%) ← Lower (penalizes low psychology)
    # Balanced (default): 0.851 (85.1%)


# Example 6: Explain weights to creator
def example_explain_weights():
    """Example: Generate explanation for creator."""

    channel_profile = {"niche": "tech reviews", "content_type": "educational"}

    weights = get_scoring_weights(channel_profile)
    explanation = ScoringWeights.explain_weights(weights, niche="Tech Reviews")

    print("\n" + explanation)

    # Output:
    # Scoring weights for Tech Reviews niche:
    #
    #   • Creator Alignment: 25.0%
    #     → High priority: Must match creator's brand and goals
    #   • Psychology Score: 25.0%
    #     → High priority: Engagement triggers drive clicks
    #   • Aesthetic Quality: 15.0%
    #   • Face Quality: 15.0%
    #   • Originality: 12.0%
    #   • Composition: 5.0%
    #   • Technical Quality: 3.0%


# Example 7: Dynamic weight adjustment based on metrics
def example_dynamic_adjustment():
    """Example: Adjust weights based on channel performance."""

    channel_profile = {
        "niche": "tech reviews",
        "avg_ctr": 0.08,  # Below average (avg is ~0.10)
    }

    weights = get_scoring_weights(channel_profile)

    # If CTR is low, boost psychology (curiosity triggers)
    if channel_profile["avg_ctr"] < 0.10:
        print("⚠️  CTR below average - boosting psychology weight")
        weights["psychology_score"] = min(weights["psychology_score"] + 0.05, 1.0)
        weights["aesthetic_quality"] = max(weights["aesthetic_quality"] - 0.05, 0.0)

        # Renormalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    print("\nAdjusted Weights for Low-CTR Channel:")
    print(f"  Psychology Score: {weights['psychology_score']:.1%}")  # Boosted
    print(f"  Aesthetic Quality: {weights['aesthetic_quality']:.1%}")  # Reduced

    return weights


# Example 8: Integration with thumbnail selection agent
async def example_agent_integration(
    frames: list[dict[str, Any]],
    channel_profile: dict[str, Any],
    creative_brief: dict[str, Any],
) -> dict[str, Any]:
    """
    Example: Full agent integration with niche-specific weights.

    This is how the actual agent would use the weights.
    """

    # Step 1: Get niche-specific weights
    weights = get_scoring_weights(
        channel_profile=channel_profile,
        custom_weights=creative_brief.get("custom_weights"),  # Optional override
    )

    print(f"\n📊 Using weights for {channel_profile.get('niche', 'default')} niche")

    # Step 2: Score each frame
    scored_frames = []
    for frame in frames:
        # Calculate component scores (from specialized models)
        component_scores = {
            "creator_alignment": calculate_creator_alignment(frame, creative_brief),
            "aesthetic_quality": calculate_aesthetic_quality(frame),
            "psychology_score": calculate_psychology_triggers(frame),
            "face_quality": calculate_face_quality(frame),
            "originality": calculate_originality(frame),
            "composition": calculate_composition(frame),
            "technical_quality": calculate_technical_quality(frame),
        }

        # Calculate final score using niche weights
        final_score = sum(component_scores[key] * weights[key] for key in weights.keys())

        scored_frames.append(
            {
                "frame_path": frame["frame_path"],
                "final_score": final_score,
                "component_scores": component_scores,
                "weights_used": weights,
            }
        )

    # Step 3: Rank and select best
    scored_frames.sort(key=lambda x: x["final_score"], reverse=True)
    best_frame = scored_frames[0]

    # Step 4: Return with explanation
    return {
        "selected_frame": best_frame["frame_path"],
        "final_score": best_frame["final_score"],
        "niche": channel_profile.get("niche"),
        "weights_used": weights,
        "reasoning": generate_reasoning(best_frame, weights),
        "alternatives": scored_frames[1:3],  # Top 3
    }


# Placeholder functions (would be implemented in actual agent)
def calculate_creator_alignment(frame, brief):
    return 0.90  # Placeholder


def calculate_aesthetic_quality(frame):
    return 0.85  # Placeholder


def calculate_psychology_triggers(frame):
    return 0.92  # Placeholder


def calculate_face_quality(frame):
    return 0.88  # Placeholder


def calculate_originality(frame):
    return 0.78  # Placeholder


def calculate_composition(frame):
    return 0.82  # Placeholder


def calculate_technical_quality(frame):
    return 0.91  # Placeholder


def generate_reasoning(frame, weights):
    return "Frame selected based on niche-specific criteria..."


# Run examples
if __name__ == "__main__":
    print("=" * 60)
    print("Niche-Specific Weight Examples")
    print("=" * 60)

    # Run all examples
    example_auto_niche_detection()
    example_niche_comparison()
    example_custom_weights()
    example_full_thumbnail_scoring()
    example_ab_test_weights()
    example_explain_weights()
    example_dynamic_adjustment()

    print("\n" + "=" * 60)
    print("✅ Examples complete!")
    print("=" * 60)
