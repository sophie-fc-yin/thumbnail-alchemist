"""Complete example of using the Thumbnail Selection Agent.

Demonstrates:
1. Loading frames from adaptive sampling results
2. Creating creative brief and channel profile
3. Running the thumbnail selector
4. Understanding the results
"""

from __future__ import annotations

import asyncio
import json

from app.thumbnail_agent import ThumbnailSelector


# ========================================================================
# EXAMPLE 1: Basic Usage
# ========================================================================
async def example_basic_usage():
    """Simplest usage of thumbnail selector."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Thumbnail Selection")
    print("=" * 70)

    # Mock frames (in production, these come from adaptive sampling)
    frames = [
        {
            "frame_number": 1,
            "timestamp": 5.2,
            "path": "/path/to/frame_5200ms.jpg",
            "url": "gs://bucket/projects/123/frames/frame_5200ms.jpg",
            "visual_analysis": {
                "dominant_emotion": "surprise",
                "expression_intensity": 0.82,
            },
            "moment_score": 0.75,
        },
        {
            "frame_number": 2,
            "timestamp": 12.8,
            "path": "/path/to/frame_12800ms.jpg",
            "url": "gs://bucket/projects/123/frames/frame_12800ms.jpg",
            "visual_analysis": {
                "dominant_emotion": "joy",
                "expression_intensity": 0.91,
            },
            "moment_score": 0.88,
        },
        {
            "frame_number": 3,
            "timestamp": 24.5,
            "path": "/path/to/frame_24500ms.jpg",
            "url": "gs://bucket/projects/123/frames/frame_24500ms.jpg",
            "visual_analysis": {
                "dominant_emotion": "serious",
                "expression_intensity": 0.65,
            },
            "moment_score": 0.72,
        },
    ]

    # Creative brief
    brief = {
        "video_title": "I Built This in 24 Hours!",
        "primary_message": "Surprising tech build challenge",
        "target_emotion": "surprise",
        "primary_goal": "maximize_ctr",
        "tone": "energetic",
    }

    # Channel profile
    profile = {
        "niche": "tech reviews",
        "personality": ["energetic", "informative"],
        "visual_style": "modern",
    }

    # Initialize selector
    selector = ThumbnailSelector()

    # Select best thumbnail
    result = await selector.select_best_thumbnail(
        frames=frames,
        creative_brief=brief,
        channel_profile=profile,
    )

    # Print results
    print("\n✅ SELECTED THUMBNAIL:")
    print(
        f"  Frame: #{result['selected_frame_number']} at {result['selected_frame_timestamp']:.1f}s"
    )
    print(f"  Path: {result['selected_frame_path']}")
    print(f"  Confidence: {result['confidence']:.0%}")

    print("\n💡 Creator Message:")
    print(f"  {result['creator_message']}")

    print("\n📝 Detailed Reasoning:")
    reasoning = result["reasoning"]
    print("\n  Summary:")
    print(f"    {reasoning['summary']}")
    print("\n  Visual Analysis:")
    print(f"    {reasoning['visual_analysis']}")
    print(f"\n  Niche Fit ({result.get('niche', 'N/A')}):")
    print(f"    {reasoning['niche_fit']}")
    print("\n  Goal Optimization:")
    print(f"    {reasoning['goal_optimization']}")
    print("\n  Psychology Triggers:")
    print(f"    {reasoning['psychology_triggers']}")

    print("\n⭐ Key Strengths:")
    for strength in result["key_strengths"]:
        print(f"  • {strength}")

    print("\n🔄 Comparative Analysis:")
    comp = result["comparative_analysis"]
    print(f"  Runner-up: {comp['runner_up']}")
    print(f"  Weaknesses avoided: {comp['weaknesses_avoided']}")

    print(f"\n💰 Cost: ${result['cost_usd']:.4f}")


# ========================================================================
# EXAMPLE 2: Beauty Channel (Different Niche)
# ========================================================================
async def example_beauty_channel():
    """Example for beauty channel with different priorities."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Beauty Channel Selection")
    print("=" * 70)

    frames = [
        {
            "frame_number": 1,
            "timestamp": 3.2,
            "path": "/path/to/beauty_frame1.jpg",
            "visual_analysis": {
                "dominant_emotion": "joy",
                "expression_intensity": 0.88,
            },
            "moment_score": 0.82,
        },
        {
            "frame_number": 2,
            "timestamp": 8.5,
            "path": "/path/to/beauty_frame2.jpg",
            "visual_analysis": {
                "dominant_emotion": "surprise",
                "expression_intensity": 0.75,
            },
            "moment_score": 0.78,
        },
    ]

    brief = {
        "video_title": "Glowy Makeup Routine ✨",
        "primary_message": "Dewy, fresh makeup look",
        "target_emotion": "aspiration",
        "primary_goal": "brand_building",
        "tone": "professional",
    }

    profile = {
        "niche": "beauty",
        "personality": ["warm", "approachable", "professional"],
        "visual_style": "instagram_worthy",
    }

    selector = ThumbnailSelector()
    result = await selector.select_best_thumbnail(
        frames=frames,
        creative_brief=brief,
        channel_profile=profile,
    )

    print("\n✅ SELECTED FOR BEAUTY CHANNEL:")
    print(f"  Frame: #{result['selected_frame_number']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"\n💡 {result['creator_message']}")
    print("\n📊 Quantitative Scores:")
    scores = result["quantitative_scores"]
    print(f"  Aesthetic (beauty-adjusted): {scores['aesthetic_score']:.2f}")
    print(f"  Psychology (brand-aligned): {scores['psychology_score']:.2f}")
    print(f"  Face Quality: {scores['face_quality']:.2f}")


# ========================================================================
# EXAMPLE 3: Comparing Multiple Goals
# ========================================================================
async def example_different_goals():
    """Show how same frames score differently with different goals."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Impact of Different Goals")
    print("=" * 70)

    frames = [
        {
            "frame_number": 1,
            "timestamp": 5.0,
            "path": "/path/to/frame1.jpg",
            "visual_analysis": {
                "dominant_emotion": "surprise",
                "expression_intensity": 0.90,
            },
            "moment_score": 0.85,
        },
        {
            "frame_number": 2,
            "timestamp": 15.0,
            "path": "/path/to/frame2.jpg",
            "visual_analysis": {
                "dominant_emotion": "serious",
                "expression_intensity": 0.70,
            },
            "moment_score": 0.75,
        },
    ]

    base_brief = {
        "video_title": "The Truth About Productivity Apps",
        "primary_message": "Honest review of popular apps",
        "target_emotion": "curiosity",
        "tone": "professional",
    }

    profile = {
        "niche": "tech reviews",
        "personality": ["analytical", "honest"],
    }

    selector = ThumbnailSelector()

    goals = ["maximize_ctr", "grow_subscribers", "brand_building"]

    results = {}
    for goal in goals:
        brief = base_brief.copy()
        brief["primary_goal"] = goal

        result = await selector.select_best_thumbnail(
            frames=frames,
            creative_brief=brief,
            channel_profile=profile,
        )
        results[goal] = result

    print("\n📊 SAME FRAMES, DIFFERENT GOALS:\n")
    for goal, result in results.items():
        print(f"{goal.upper()}:")
        print(f"  Selected: Frame #{result['selected_frame_number']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Reasoning: {result['reasoning'][:80]}...")
        print()


# ========================================================================
# EXAMPLE 4: Using Gemini 2.5 Pro (Higher Quality)
# ========================================================================
async def example_pro_model():
    """Example using Gemini 2.5 Pro for higher quality."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Gemini 2.5 Pro (Premium Quality)")
    print("=" * 70)

    frames = [
        {
            "frame_number": i,
            "timestamp": i * 3.0,
            "path": f"/path/to/frame_{i}.jpg",
            "visual_analysis": {
                "dominant_emotion": "surprise",
                "expression_intensity": 0.75 + (i * 0.03),
            },
            "moment_score": 0.70 + (i * 0.02),
        }
        for i in range(1, 6)
    ]

    brief = {
        "video_title": "My $10,000 Studio Setup Tour",
        "primary_message": "High-end production equipment showcase",
        "target_emotion": "aspiration",
        "primary_goal": "brand_building",
        "tone": "professional",
    }

    profile = {
        "niche": "tech reviews",
        "personality": ["premium", "professional", "detailed"],
    }

    # Use Pro model for better quality ($0.007 vs $0.0023)
    selector_pro = ThumbnailSelector(use_pro=True)

    result = await selector_pro.select_best_thumbnail(
        frames=frames,
        creative_brief=brief,
        channel_profile=profile,
    )

    print("\n✅ SELECTED WITH GEMINI 2.5 PRO:")
    print(f"  Model: {result['gemini_model']}")
    print(f"  Frame: #{result['selected_frame_number']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Cost: ${result['cost_usd']:.4f} (vs $0.0023 for Flash)")
    print("\n📝 Pro Reasoning:")
    print(f"  {result['reasoning']}")


# ========================================================================
# EXAMPLE 5: Full Pipeline with Real Data
# ========================================================================
async def example_full_pipeline():
    """Complete pipeline from adaptive sampling to thumbnail selection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Full Pipeline Integration")
    print("=" * 70)

    # Simulating results from adaptive sampling endpoint
    adaptive_sampling_result = {
        "project_id": "550e8400-e29b-41d4-a716-446655440000",
        "frames": [
            "gs://bucket/projects/550e8.../frames/frame_5200ms.jpg",
            "gs://bucket/projects/550e8.../frames/frame_8100ms.jpg",
            "gs://bucket/projects/550e8.../frames/frame_12400ms.jpg",
            "gs://bucket/projects/550e8.../frames/frame_18700ms.jpg",
            "gs://bucket/projects/550e8.../frames/frame_24200ms.jpg",
        ],
        "analysis_json_url": "gs://bucket/projects/550e8.../analysis/adaptive_sampling_analysis.json",
    }

    # In production, you'd load the analysis JSON from GCS
    # For demo, we'll mock it
    analysis_data = {
        "extracted_frames": [
            {
                "frame_number": 1,
                "timestamp": 5.2,
                "path": adaptive_sampling_result["frames"][0],
                "url": adaptive_sampling_result["frames"][0],
                "visual_analysis": {
                    "dominant_emotion": "surprise",
                    "expression_intensity": 0.82,
                },
                "moment_score": 0.88,
            },
            {
                "frame_number": 2,
                "timestamp": 8.1,
                "path": adaptive_sampling_result["frames"][1],
                "url": adaptive_sampling_result["frames"][1],
                "visual_analysis": {
                    "dominant_emotion": "joy",
                    "expression_intensity": 0.91,
                },
                "moment_score": 0.92,
            },
            # ... more frames
        ]
    }

    # Creator's input from Supabase
    creator_input = {
        "brief": {
            "video_title": "I Tried Coding For 100 Hours Straight",
            "primary_message": "Extreme coding challenge results",
            "target_emotion": "surprise",
            "primary_goal": "maximize_ctr",
            "tone": "energetic",
        },
        "profile": {
            "niche": "tech reviews",
            "personality": ["energetic", "relatable", "informative"],
            "visual_style": "modern",
        },
    }

    # Run thumbnail selector
    selector = ThumbnailSelector()
    result = await selector.select_best_thumbnail(
        frames=analysis_data["extracted_frames"],
        creative_brief=creator_input["brief"],
        channel_profile=creator_input["profile"],
    )

    # Save result
    output = {
        "project_id": adaptive_sampling_result["project_id"],
        "selected_thumbnail": {
            "frame_url": result["selected_frame_url"],
            "timestamp": result["selected_frame_timestamp"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "cost": result["cost_usd"],
        },
        "alternatives": [
            {
                "frame_number": frame["frame_number"],
                "score": frame["total_score"],
            }
            for frame in result["all_frame_scores"][:3]
        ],
    }

    print("\n✅ PIPELINE COMPLETE:")
    print(json.dumps(output, indent=2))


# ========================================================================
# EXAMPLE 6: Cost Analysis
# ========================================================================
async def example_cost_analysis():
    """Analyze costs at different scales."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Cost Analysis")
    print("=" * 70)

    # Simulate different frame counts
    frame_counts = [5, 10, 15, 20]

    selector = ThumbnailSelector()

    print("\n📊 Cost by Frame Count:")
    print(f"{'Frames':<10} {'Cost (Flash)':<15} {'Cost (Pro)':<15}")
    print("-" * 40)

    for count in frame_counts:
        cost_flash = selector._estimate_cost(count)

        selector_pro = ThumbnailSelector(use_pro=True)
        cost_pro = selector_pro._estimate_cost(count)

        print(f"{count:<10} ${cost_flash:<14.4f} ${cost_pro:<14.4f}")

    print("\n📊 Cost at Scale (10 frames):")
    volumes = [100, 1_000, 10_000, 100_000]

    print(f"\n{'Volume':<15} {'Flash Total':<15} {'Pro Total':<15}")
    print("-" * 45)

    for volume in volumes:
        flash_total = selector._estimate_cost(10) * volume
        pro_total = selector_pro._estimate_cost(10) * volume
        print(f"{volume:>10,}     ${flash_total:>10,.2f}     ${pro_total:>10,.2f}")


# ========================================================================
# RUN ALL EXAMPLES
# ========================================================================
async def main():
    """Run all examples."""
    await example_basic_usage()
    await example_beauty_channel()
    await example_different_goals()
    await example_pro_model()
    await example_full_pipeline()
    await example_cost_analysis()

    print("\n" + "=" * 70)
    print("✅ All Examples Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
