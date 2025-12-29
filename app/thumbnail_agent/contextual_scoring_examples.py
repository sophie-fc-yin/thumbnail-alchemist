"""Examples demonstrating context-aware aesthetic and psychology scoring.

Shows how scoring adapts to different niches and creative briefs.
"""

from __future__ import annotations

from app.thumbnail_agent.contextual_scoring import (
    ContextualScoringCriteria,
    get_contextual_scores,
)
from app.thumbnail_agent.scoring_weights import get_scoring_weights


# ========================================================================
# EXAMPLE 1: Beauty Channel vs Tech Channel - Same Frame
# ========================================================================
def example_same_frame_different_niches():
    """
    Show how the same frame scores differently for beauty vs tech channels.

    A softly-lit, warm-toned portrait will score:
    - High for beauty channel (matches aesthetic criteria)
    - Lower for tech channel (doesn't match clarity/sharpness needs)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Same Frame, Different Niches")
    print("=" * 70)

    # Frame with soft lighting, warm colors
    frame_features = {
        "lighting": ["soft", "warm", "ring_light"],
        "color_palette": ["warm_tones", "pastel"],
        "composition": ["centered_subject", "shallow_depth"],
        "polish_level": ["professional", "edited"],
        "visual_style": ["instagram_worthy"],
        # Tech-specific attributes missing
        "clarity": ["soft_focus"],  # Not ideal for tech
    }

    detected_triggers = [
        "aspiration",  # Good for beauty
        "emotional_contagion",  # Good for beauty
        "authority",  # Good for tech
    ]

    # ===== Beauty Channel =====
    beauty_profile = {"niche": "beauty", "content_type": "lifestyle"}
    beauty_brief = {"primary_goal": "brand_building", "tone": "professional"}

    beauty_criteria = ContextualScoringCriteria.get_aesthetic_criteria(beauty_profile, beauty_brief)
    beauty_psych = ContextualScoringCriteria.get_psychology_priorities(beauty_profile, beauty_brief)

    beauty_aesthetic = ContextualScoringCriteria.evaluate_aesthetic_alignment(
        frame_features, beauty_criteria
    )
    beauty_psychology = ContextualScoringCriteria.evaluate_psychology_alignment(
        detected_triggers, beauty_psych
    )

    print("\n🎨 BEAUTY CHANNEL:")
    print(f"  Aesthetic Score: {beauty_aesthetic['overall_score']:.2f}")
    print("  Why high? Soft lighting + warm tones + professional = beauty ideal")
    print(f"  Psychology Score: {beauty_psychology['overall_score']:.2f}")
    print("  Why high? Aspiration + emotional contagion = brand building")

    # ===== Tech Channel =====
    tech_profile = {"niche": "tech reviews", "content_type": "educational"}
    tech_brief = {"primary_goal": "maximize_ctr", "tone": "professional"}

    tech_criteria = ContextualScoringCriteria.get_aesthetic_criteria(tech_profile, tech_brief)
    tech_psych = ContextualScoringCriteria.get_psychology_priorities(tech_profile, tech_brief)

    tech_aesthetic = ContextualScoringCriteria.evaluate_aesthetic_alignment(
        frame_features, tech_criteria
    )
    tech_psychology = ContextualScoringCriteria.evaluate_psychology_alignment(
        detected_triggers, tech_psych
    )

    print("\n💻 TECH CHANNEL:")
    print(f"  Aesthetic Score: {tech_aesthetic['overall_score']:.2f}")
    print("  Why lower? Soft focus doesn't meet clarity requirement (weight 0.30)")
    print(f"  Psychology Score: {tech_psychology['overall_score']:.2f}")
    print("  Why lower? Missing curiosity_gap (top priority for CTR)")

    print("\n📊 VERDICT:")
    print("  Same frame, different scores based on niche-specific criteria")
    print(
        f"  Beauty: {beauty_aesthetic['overall_score']:.2f} aesthetic, {beauty_psychology['overall_score']:.2f} psychology"
    )
    print(
        f"  Tech: {tech_aesthetic['overall_score']:.2f} aesthetic, {tech_psychology['overall_score']:.2f} psychology"
    )


# ========================================================================
# EXAMPLE 2: Gaming Channel - Energy vs Calm
# ========================================================================
def example_gaming_energy_matters():
    """
    Show how gaming channels prioritize high energy aesthetics.

    Same composition, different energy levels → different scores.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Gaming Channel - Energy Matters")
    print("=" * 70)

    gaming_profile = {"niche": "gaming", "content_type": "entertainment"}
    gaming_brief = {"primary_goal": "maximize_ctr", "tone": "energetic"}

    criteria = ContextualScoringCriteria.get_aesthetic_criteria(gaming_profile, gaming_brief)

    # Frame 1: High energy
    high_energy_frame = {
        "lighting": ["dramatic", "RGB", "neon"],
        "color_palette": ["saturated", "bold", "neon"],
        "composition": ["dynamic", "action_focused"],
        "energy_level": ["high_energy", "intense"],
        "visual_style": ["cinematic", "epic"],
    }

    high_energy_score = ContextualScoringCriteria.evaluate_aesthetic_alignment(
        high_energy_frame, criteria
    )

    # Frame 2: Low energy (same composition, calm mood)
    low_energy_frame = {
        "lighting": ["natural", "soft"],
        "color_palette": ["muted", "desaturated"],
        "composition": ["static", "passive"],
        "energy_level": ["calm", "subdued"],
        "visual_style": ["realistic", "understated"],
    }

    low_energy_score = ContextualScoringCriteria.evaluate_aesthetic_alignment(
        low_energy_frame, criteria
    )

    print("\n⚡ HIGH ENERGY FRAME:")
    print(f"  Aesthetic Score: {high_energy_score['overall_score']:.2f}")
    print("  Matches: RGB lighting, saturated colors, dynamic composition")
    print("  Gaming criteria weight energy_level at 0.25 (25%)")

    print("\n😴 LOW ENERGY FRAME:")
    print(f"  Aesthetic Score: {low_energy_score['overall_score']:.2f}")
    print("  Penalized: Muted colors, calm mood, static composition")
    print("  Avoids: 'calm', 'subdued', 'passive' (gaming no-nos)")

    print("\n📊 VERDICT:")
    print("  Same face, same person - but energy level changes score dramatically")
    print(f"  High energy: {high_energy_score['overall_score']:.2f}")
    print(f"  Low energy: {low_energy_score['overall_score']:.2f}")


# ========================================================================
# EXAMPLE 3: Psychology Triggers Adapt to Goals
# ========================================================================
def example_psychology_adapts_to_goals():
    """
    Show how psychology trigger priorities change based on creator goals.

    Same detected triggers, different goals → different psychology scores.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Psychology Triggers Adapt to Creator Goals")
    print("=" * 70)

    # Same frame with these triggers detected
    detected_triggers = [
        "curiosity_gap",
        "authority",
        "authenticity",
        "surprise",
    ]

    tech_profile = {"niche": "tech reviews"}

    # ===== Goal 1: Maximize CTR =====
    ctr_brief = {"primary_goal": "maximize_ctr"}
    ctr_priorities = ContextualScoringCriteria.get_psychology_priorities(tech_profile, ctr_brief)
    ctr_score = ContextualScoringCriteria.evaluate_psychology_alignment(
        detected_triggers, ctr_priorities
    )

    print("\n🎯 GOAL: Maximize CTR")
    print(f"  Psychology Score: {ctr_score['overall_score']:.2f}")
    print("  Top priorities: curiosity_gap (✓), pattern_interrupt, surprise (✓)")
    print("  Detected triggers match CTR drivers")

    # ===== Goal 2: Grow Subscribers =====
    subscriber_brief = {"primary_goal": "grow_subscribers"}
    subscriber_priorities = ContextualScoringCriteria.get_psychology_priorities(
        tech_profile, subscriber_brief
    )
    subscriber_score = ContextualScoringCriteria.evaluate_psychology_alignment(
        detected_triggers, subscriber_priorities
    )

    print("\n👥 GOAL: Grow Subscribers")
    print(f"  Psychology Score: {subscriber_score['overall_score']:.2f}")
    print("  Top priorities: authority (✓), authenticity (✓), relatability, social_proof")
    print("  Detected triggers match loyalty builders")

    # ===== Goal 3: Brand Building =====
    brand_brief = {"primary_goal": "brand_building"}
    brand_priorities = ContextualScoringCriteria.get_psychology_priorities(
        tech_profile, brand_brief
    )
    brand_score = ContextualScoringCriteria.evaluate_psychology_alignment(
        detected_triggers, brand_priorities
    )

    print("\n🏢 GOAL: Brand Building")
    print(f"  Psychology Score: {brand_score['overall_score']:.2f}")
    print("  Top priorities: authenticity (✓), aspiration, authority (✓)")
    print("  Detected triggers match brand values")

    print("\n📊 VERDICT:")
    print("  Same triggers, different priorities based on creator goal:")
    print(f"  CTR: {ctr_score['overall_score']:.2f} (curiosity_gap valued high)")
    print(f"  Subscribers: {subscriber_score['overall_score']:.2f} (authority valued high)")
    print(f"  Brand: {brand_score['overall_score']:.2f} (authenticity valued high)")


# ========================================================================
# EXAMPLE 4: Tone Adjustments
# ========================================================================
def example_tone_affects_scoring():
    """
    Show how creator's tone preference adjusts aesthetic criteria.

    Professional vs Casual tone → different aesthetic priorities.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Tone Affects Aesthetic Criteria")
    print("=" * 70)

    commentary_profile = {"niche": "commentary"}

    # Same frame features
    frame_features = {
        "lighting": ["natural", "soft"],
        "color_palette": ["natural", "warm"],
        "composition": ["face_focused", "intimate"],
        "authenticity": ["genuine", "raw"],
        "polish_level": ["unpolished_ok"],
    }

    # ===== Professional Tone =====
    professional_brief = {"tone": "professional"}
    professional_criteria = ContextualScoringCriteria.get_aesthetic_criteria(
        commentary_profile, professional_brief
    )
    professional_score = ContextualScoringCriteria.evaluate_aesthetic_alignment(
        frame_features, professional_criteria
    )

    print("\n👔 PROFESSIONAL TONE:")
    print(f"  Aesthetic Score: {professional_score['overall_score']:.2f}")
    print("  Boosts: polish_level, clarity, clean_background")
    print("  Penalizes: raw, unpolished (conflicts with 'professional')")

    # ===== Casual Tone =====
    casual_brief = {"tone": "casual"}
    casual_criteria = ContextualScoringCriteria.get_aesthetic_criteria(
        commentary_profile, casual_brief
    )
    casual_score = ContextualScoringCriteria.evaluate_aesthetic_alignment(
        frame_features, casual_criteria
    )

    print("\n👕 CASUAL TONE:")
    print(f"  Aesthetic Score: {casual_score['overall_score']:.2f}")
    print("  Boosts: natural, authentic, relatable (matches 'raw' + 'genuine')")
    print("  Penalizes: overly_polished, corporate")

    print("\n📊 VERDICT:")
    print("  Same frame, tone preference changes evaluation:")
    print(f"  Professional tone: {professional_score['overall_score']:.2f} (wants polish)")
    print(f"  Casual tone: {casual_score['overall_score']:.2f} (values authenticity)")


# ========================================================================
# EXAMPLE 5: Full Integration with Scoring Weights
# ========================================================================
def example_full_integration():
    """
    Show complete integration: contextual scoring + niche weights → final score.

    This demonstrates the full pipeline.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Full Integration - Contextual Scores + Niche Weights")
    print("=" * 70)

    # Beauty channel example
    beauty_profile = {
        "niche": "beauty",
        "content_type": "lifestyle",
        "visual_consistency": "high",
    }
    beauty_brief = {
        "primary_goal": "brand_building",
        "tone": "professional",
        "visual_style": "instagram_worthy",
    }

    # Frame features
    frame_features = {
        "lighting": ["soft", "warm", "golden_hour"],
        "color_palette": ["pastel", "warm_tones", "complementary"],
        "composition": ["centered_subject", "shallow_depth", "clean_background"],
        "polish_level": ["professional", "edited", "polished"],
        "visual_style": ["instagram_worthy", "magazine_quality"],
    }

    detected_triggers = [
        "aspiration",
        "emotional_contagion",
        "social_proof",
    ]

    # Step 1: Get contextual scores
    contextual_scores = get_contextual_scores(
        frame_features, detected_triggers, beauty_profile, beauty_brief
    )

    # Step 2: Get niche-specific weights
    weights = get_scoring_weights(beauty_profile)

    # Step 3: Assume other component scores (from specialized models)
    all_scores = {
        "creator_alignment": 0.92,  # From brief matching
        "aesthetic_quality": contextual_scores["aesthetic_score"],  # ← Contextual
        "psychology_score": contextual_scores["psychology_score"],  # ← Contextual
        "face_quality": 0.90,  # From FER+ / MediaPipe
        "originality": 0.78,  # From CLIP similarity
        "composition": 0.85,  # From rule of thirds detector
        "technical_quality": 0.93,  # From sharpness/noise detector
    }

    # Step 4: Calculate final weighted score
    final_score = sum(all_scores[key] * weights[key] for key in weights.keys())

    print("\n🎨 BEAUTY CHANNEL FRAME SCORING:")
    print("\n1. CONTEXTUAL SCORES (adapted to beauty niche + brand goals):")
    print(f"   Aesthetic Quality: {contextual_scores['aesthetic_score']:.2f}")
    print("     → High: soft lighting + warm tones + professional = beauty ideal")
    print(f"   Psychology Score: {contextual_scores['psychology_score']:.2f}")
    print("     → High: aspiration + emotional contagion = brand building")

    print("\n2. NICHE-SPECIFIC WEIGHTS (beauty niche):")
    print("   Aesthetic Quality: 30% (highest for beauty)")
    print("   Psychology Score: 20%")
    print("   Creator Alignment: 20%")
    print("   Face Quality: 15%")

    print("\n3. FINAL WEIGHTED SCORE:")
    for key, score in all_scores.items():
        contribution = score * weights[key]
        marker = "← Contextual" if key in ["aesthetic_quality", "psychology_score"] else ""
        print(f"   {key}: {score:.2f} × {weights[key]:.2f} = {contribution:.3f} {marker}")

    print(f"\n🎯 FINAL SCORE: {final_score:.3f} ({final_score*100:.1f}%)")
    print("\nContextual scoring ensures:")
    print("  ✓ Aesthetic evaluated by beauty standards (not tech standards)")
    print("  ✓ Psychology triggers prioritized for brand building (not just CTR)")
    print("  ✓ Weights emphasize what matters for beauty niche (aesthetics 30%)")


# ========================================================================
# EXAMPLE 6: Side-by-Side Comparison
# ========================================================================
def example_side_by_side_comparison():
    """
    Compare how two different frames score for the same channel.

    Frame A: Perfect beauty aesthetic
    Frame B: Strong psychology, weaker aesthetic
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Side-by-Side Frame Comparison")
    print("=" * 70)

    beauty_profile = {"niche": "beauty", "content_type": "lifestyle"}
    beauty_brief = {"primary_goal": "maximize_ctr", "tone": "professional"}

    weights = get_scoring_weights(beauty_profile)

    # Frame A: Perfect aesthetic, weaker psychology
    frame_a_features = {
        "lighting": ["soft", "warm", "golden_hour"],
        "color_palette": ["pastel", "warm_tones"],
        "composition": ["centered_subject", "shallow_depth"],
        "polish_level": ["professional", "magazine_quality"],
    }
    frame_a_triggers = ["aspiration", "emotional_contagion"]  # Only 2 triggers

    frame_a_scores_ctx = get_contextual_scores(
        frame_a_features, frame_a_triggers, beauty_profile, beauty_brief
    )
    frame_a_all = {
        "creator_alignment": 0.90,
        "aesthetic_quality": frame_a_scores_ctx["aesthetic_score"],
        "psychology_score": frame_a_scores_ctx["psychology_score"],
        "face_quality": 0.92,
        "originality": 0.75,
        "composition": 0.88,
        "technical_quality": 0.95,
    }
    frame_a_final = sum(frame_a_all[k] * weights[k] for k in weights.keys())

    # Frame B: Weaker aesthetic, strong psychology
    frame_b_features = {
        "lighting": ["bright", "even"],
        "color_palette": ["vibrant", "high_contrast"],
        "composition": ["dynamic", "action_focused"],
        "polish_level": ["edited"],  # Not magazine quality
    }
    frame_b_triggers = [
        "curiosity_gap",
        "surprise",
        "pattern_interrupt",
        "aspiration",
    ]  # 4 triggers

    frame_b_scores_ctx = get_contextual_scores(
        frame_b_features, frame_b_triggers, beauty_profile, beauty_brief
    )
    frame_b_all = {
        "creator_alignment": 0.85,
        "aesthetic_quality": frame_b_scores_ctx["aesthetic_score"],
        "psychology_score": frame_b_scores_ctx["psychology_score"],
        "face_quality": 0.88,
        "originality": 0.82,
        "composition": 0.80,
        "technical_quality": 0.90,
    }
    frame_b_final = sum(frame_b_all[k] * weights[k] for k in weights.keys())

    print("\n🖼️  FRAME A: Perfect Beauty Aesthetic")
    print(f"  Aesthetic: {frame_a_scores_ctx['aesthetic_score']:.2f} (soft, warm, polished)")
    print(f"  Psychology: {frame_a_scores_ctx['psychology_score']:.2f} (only 2 triggers)")
    print(f"  Final Score: {frame_a_final:.3f}")

    print("\n🖼️  FRAME B: Strong Psychology")
    print(f"  Aesthetic: {frame_b_scores_ctx['aesthetic_score']:.2f} (less soft, more vibrant)")
    print(f"  Psychology: {frame_b_scores_ctx['psychology_score']:.2f} (4 CTR triggers)")
    print(f"  Final Score: {frame_b_final:.3f}")

    print("\n📊 VERDICT:")
    if frame_a_final > frame_b_final:
        print(f"  Frame A wins: {frame_a_final:.3f} > {frame_b_final:.3f}")
        print("  Why? Beauty niche weighs aesthetics at 30% (highest)")
        print("  Perfect beauty aesthetic outweighs stronger psychology")
    else:
        print(f"  Frame B wins: {frame_b_final:.3f} > {frame_a_final:.3f}")
        print("  Why? Goal is 'maximize_ctr' - psychology triggers boosted")
        print("  Strong CTR psychology outweighs perfect aesthetic")


# ========================================================================
# RUN ALL EXAMPLES
# ========================================================================
if __name__ == "__main__":
    example_same_frame_different_niches()
    example_gaming_energy_matters()
    example_psychology_adapts_to_goals()
    example_tone_affects_scoring()
    example_full_integration()
    example_side_by_side_comparison()

    print("\n" + "=" * 70)
    print("✅ All Contextual Scoring Examples Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Aesthetic criteria adapt to niche (beauty ≠ tech ≠ gaming)")
    print("  2. Psychology priorities adapt to creator goals (CTR vs subscribers)")
    print("  3. Tone adjustments fine-tune evaluation (professional vs casual)")
    print("  4. Contextual scores integrate with niche weights for final decision")
    print("  5. Same frame can score differently based on channel context")
    print("\n" + "=" * 70)
